# -*- coding: utf-8 -*-
"""
Simulator 模块：竞技场 - 实证排序
对所有候选动作进行暴力搜索，唯才是举
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

from ADA.utils.definitions import CandidateAction
from utils import get_logger

logger = get_logger("ADA.Simulator")


class Simulator:
    """
    仿真竞技场
    
    核心职责：
    1. 接收所有候选动作（Planner + Solver + Judger）
    2. 对每个候选进行暴力仿真
    3. 多维排序（安全性、过载消除、奖励、成本）
    4. 返回最优动作
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        max_workers: int = 1,
        rho_safety_threshold: float = 1.5,
        enable_parallel: bool = False,
        **kwargs
    ):
        """
        初始化 Simulator
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            max_workers: 并行仿真的最大线程数
            rho_safety_threshold: 安全阈值（超过此值视为不安全）
            enable_parallel: 是否启用并行仿真
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.max_workers = max_workers
        self.rho_safety_threshold = rho_safety_threshold
        # 出于与 Grid2Op / 后端线程安全性的兼容性考虑，默认关闭并行仿真。
        # 如果用户明确确认环境是线程安全的，可在初始化时手动将 enable_parallel=True。
        self.enable_parallel = enable_parallel
        
        logger.info(
            f"Simulator 初始化完成 (max_workers={max_workers}, parallel={enable_parallel})"
        )
    
    def select_best_action(
        self,
        observation: BaseObservation,
        candidates: List[CandidateAction],
        max_rho_before: Optional[float] = None
    ) -> Optional[CandidateAction]:
        """
        选择最优动作
        
        工作流程：
        1. 对每个候选动作进行仿真
        2. 评估仿真结果
        3. 多维排序
        4. 返回最优动作
        
        Args:
            observation: 当前观测
            candidates: 候选动作列表
            
        Returns:
            最优的 CandidateAction，如果没有安全动作则返回 None
        """
        if not candidates:
            logger.warning("Simulator: 没有候选动作")
            return None
        
        logger.info(f"Simulator: 开始评估 {len(candidates)} 个候选动作")
        
        # 1. 检查是否有高置信度结果（信任 Planner 的 Score=4 结果）
        # 优化：如果候选已经携带仿真结果且置信度高，可以跳过重复仿真
        candidates_to_simulate = []
        pre_evaluated = []
        
        for candidate in candidates:
            # 检查是否已有高置信度仿真结果（来自 Planner 的 ExpertInsight）
            if (candidate.simulation_result and 
                candidate.source == "Expert_Topo" and
                candidate.priority >= 4):  # Score=4 表示完美解决
                # 信任高置信度结果，但仍进行复核（可选）
                # 这里选择信任，跳过重复仿真
                logger.debug(f"信任高置信度候选: {candidate.description} (priority={candidate.priority})")
                pre_evaluated.append(candidate)
            else:
                candidates_to_simulate.append(candidate)
        
        # 2. 对需要仿真的候选进行仿真
        if candidates_to_simulate:
            if self.enable_parallel and len(candidates_to_simulate) > 1:
                evaluated_candidates = self._simulate_parallel(observation, candidates_to_simulate)
            else:
                evaluated_candidates = self._simulate_sequential(observation, candidates_to_simulate)
        else:
            evaluated_candidates = []
        
        # 合并结果
        evaluated_candidates = pre_evaluated + evaluated_candidates
        
        # 2. 过滤安全动作
        safe_candidates = [
            c for c in evaluated_candidates
            if c.simulation_result and c.simulation_result.get("is_safe", False)
        ]
        
        if not safe_candidates:
            logger.warning("Simulator: 没有安全动作")
            # 返回评估过的动作中最好的（即使不安全）
            if evaluated_candidates:
                best = self._sort_candidates(evaluated_candidates)[0]
                logger.warning(f"Simulator: 返回最佳不安全动作: {best}")
                return best
            return None
        
        # 3. 排序并返回最优（动态排序权重）
        sorted_candidates = self._sort_candidates(
            safe_candidates, 
            max_rho_before=max_rho_before
        )
        best = sorted_candidates[0]
        
        logger.info(
            f"Simulator: 选中动作 {best.source} "
            f"(rho_max={best.simulation_result.get('rho_max', 'N/A'):.3f}, "
            f"reward={best.simulation_result.get('reward', 'N/A'):.2f})"
        )
        
        return best
    
    def _simulate_parallel(
        self,
        observation: BaseObservation,
        candidates: List[CandidateAction]
    ) -> List[CandidateAction]:
        """并行仿真所有候选动作"""
        evaluated_candidates = []
        
        def simulate_single(candidate: CandidateAction) -> CandidateAction:
            """单个候选的仿真函数"""
            try:
                result = self._simulate_single_action(observation, candidate.action_obj)
                candidate.update_simulation_result(**result)
            except Exception as e:
                logger.error(f"仿真候选 {candidate.source} 失败: {e}")
                candidate.update_simulation_result(
                    is_safe=False,
                    rho_max=float('inf'),
                    reward=float('-inf'),
                    margin=float('-inf'),
                    exception=str(e)
                )
            return candidate
        
        # 使用线程池并行仿真
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_candidate = {
                executor.submit(simulate_single, candidate): candidate
                for candidate in candidates
            }
            
            for future in as_completed(future_to_candidate):
                try:
                    evaluated_candidate = future.result()
                    evaluated_candidates.append(evaluated_candidate)
                except Exception as e:
                    candidate = future_to_candidate[future]
                    logger.error(f"并行仿真异常 {candidate.source}: {e}")
                    evaluated_candidates.append(candidate)
        
        return evaluated_candidates
    
    def _simulate_sequential(
        self,
        observation: BaseObservation,
        candidates: List[CandidateAction]
    ) -> List[CandidateAction]:
        """顺序仿真所有候选动作"""
        evaluated_candidates = []
        
        for candidate in candidates:
            try:
                result = self._simulate_single_action(observation, candidate.action_obj)
                candidate.update_simulation_result(**result)
            except Exception as e:
                logger.error(f"仿真候选 {candidate.source} 失败: {e}")
                candidate.update_simulation_result(
                    is_safe=False,
                    rho_max=float('inf'),
                    reward=float('-inf'),
                    margin=float('-inf'),
                    exception=str(e)
                )
            evaluated_candidates.append(candidate)
        
        return evaluated_candidates
    
    def _simulate_single_action(
        self,
        observation: BaseObservation,
        action: BaseAction
    ) -> Dict[str, Any]:
        """
        仿真单个动作
        
        Args:
            observation: 当前观测
            action: 要仿真的动作
            
        Returns:
            仿真结果字典
        """
        try:
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=0)
            
            # 检查异常（修复 BUG：空异常列表不再视为失败）
            exception = sim_info.get('exception', None)
            exception_str = None
            if exception is not None:
                if isinstance(exception, list):
                    if len(exception) > 0:
                        exception_str = '; '.join([str(e) for e in exception])
                else:
                    exception_str = str(exception)
            
            # 如果有异常，标记为不安全
            if exception_str:
                return {
                    "is_safe": False,
                    "rho_max": float('inf'),
                    "reward": float(sim_reward) if sim_reward is not None else float('-inf'),
                    "margin": float('-inf'),
                    "exception": exception_str
                }
            
            # 潮流发散检查
            if np.any(np.isnan(sim_obs.rho)) or np.any(np.isinf(sim_obs.rho)):
                return {
                    "is_safe": False,
                    "rho_max": float('inf'),
                    "reward": float(sim_reward) if sim_reward is not None else float('-inf'),
                    "margin": float('-inf'),
                    "exception": "潮流发散 (NaN/Inf)"
                }
            
            # 计算指标
            max_rho_after = float(sim_obs.rho.max())
            overflow_count_after = int((sim_obs.rho > 1.0).sum())
            
            # 安全检查
            is_safe = (
                not sim_done and
                max_rho_after <= self.rho_safety_threshold and
                exception_str is None
            )
            
            # 计算安全裕度
            margin = 1.0 - max_rho_after if max_rho_after <= 1.0 else float('-inf')
            
            return {
                "is_safe": is_safe,
                "rho_max": max_rho_after,
                "reward": float(sim_reward) if sim_reward is not None else float('-inf'),
                "margin": margin,
                "overflow_count": overflow_count_after,
                "exception": exception_str
            }
            
        except Exception as e:
            logger.error(f"仿真过程异常: {e}", exc_info=True)
            return {
                "is_safe": False,
                "rho_max": float('inf'),
                "reward": float('-inf'),
                "margin": float('-inf'),
                "exception": str(e)
            }
    
    def _sort_candidates(
        self, 
        candidates: List[CandidateAction],
        max_rho_before: Optional[float] = None
    ) -> List[CandidateAction]:
        """
        多维排序候选动作（动态排序权重）
        
        排序规则（分层）：
        1. Tier 1 安全性: is_safe (True > False)
        2. Tier 2 消除过载: overflow_count (越小越好)
        3. Tier 3 最大负载率: rho_max (越小越好) - 安全状态下权重降低
        4. Tier 4 奖励: reward (越大越好) - 安全状态下权重提升
        5. Tier 5 成本: priority (越大越好，但通常不重要)
        
        Args:
            candidates: 候选动作列表
            max_rho_before: 执行前的最大负载率（用于动态调整权重）
            
        Returns:
            排序后的候选动作列表（最优的在前面）
        """
        # 判断是否处于安全状态（用于动态排序）
        is_safe_state = (
            max_rho_before is not None and 
            max_rho_before < 0.9 and
            all(
                c.simulation_result and c.simulation_result.get("is_safe", False)
                for c in candidates
            )
        )
        
        def sort_key(candidate: CandidateAction) -> tuple:
            """排序键函数（动态权重）"""
            result = candidate.simulation_result or {}
            
            # Tier 1: 安全性（True > False）
            is_safe = result.get("is_safe", False)
            safety_score = 1 if is_safe else 0
            
            # Tier 2: 过载线路数量（越小越好）
            overflow_count = result.get("overflow_count", float('inf'))
            
            # Tier 3: 最大负载率（越小越好）
            rho_max = result.get("rho_max", float('inf'))
            
            # Tier 4: 奖励（越大越好，取负号）
            reward = result.get("reward", float('-inf'))
            
            # Tier 5: 优先级（越大越好，取负号）
            priority = candidate.priority
            
            # 动态排序：安全状态下优先考虑 Reward
            if is_safe_state:
                # 安全状态：Reward 优先级高于 rho_max
                return (
                    -safety_score,      # 安全性
                    overflow_count,     # 过载数量
                    -reward,            # 奖励（优先级提升）
                    rho_max,            # 最大负载率（优先级降低）
                    -priority           # 优先级
                )
            else:
                # 危险状态：传统排序（安全性优先）
                return (
                    -safety_score,      # 安全性
                    overflow_count,     # 过载数量
                    rho_max,            # 最大负载率
                    -reward,            # 奖励
                    -priority           # 优先级
                )
        
        sorted_candidates = sorted(candidates, key=sort_key)
        return sorted_candidates
    
    def reset(self, observation: BaseObservation) -> None:
        """重置 Simulator 状态"""
        logger.debug("Simulator 重置")
        # 目前没有需要重置的状态
        pass
    
    def evaluate_action(
        self,
        observation: BaseObservation,
        action: BaseAction
    ) -> Dict[str, Any]:
        """
        评估单个动作（便捷方法）
        
        Args:
            observation: 当前观测
            action: 要评估的动作
            
        Returns:
            评估结果字典
        """
        return self._simulate_single_action(observation, action)

