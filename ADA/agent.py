# -*- coding: utf-8 -*-
"""
ADA (Adaptive Dispatch & Action) Agent v2.1
混合智能体：结合 Planner (专家规则)、Solver (数学优化) 和 Judger (LLM 融合)
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入工具模块
from utils import OpenAIChat, get_logger
from utils.embeddings import OpenAIEmbedding
from ADA.utils.definitions import CandidateAction

# 导入核心模块
from ADA.core import Planner, Solver, Simulator, Judger, Summarizer
from ADA.knowledgebase.service import KnowledgeService

logger = get_logger("ADA")


class ADA_Agent(BaseAgent):
    """
    ADA (Adaptive Dispatch & Action) Agent v2.1
    
    核心理念：混合智能与全量竞优
    - Planner (专家规则): 基于物理规则生成拓扑候选（不依赖 LLM）
    - Solver (数学优化): 基于凸优化生成调度候选（不依赖 LLM）
    - Judger (LLM 融合): 分析前两者并生成融合策略
    - Simulator (竞技场): 对所有候选进行暴力搜索，唯才是举
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: Optional[OpenAIChat] = None,
        name: str = "ADA_Agent",
        rho_danger: float = 0.95,
        rho_safe: float = 0.85,
        max_planner_candidates: int = 5,
        max_llm_candidates: int = 3,
        enable_knowledge_base: bool = True,
        **kwargs
    ):
        """
        初始化 ADA Agent
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            llm_client: LLM 客户端（用于 Judger 和 Summarizer）
            name: Agent 名称
            rho_danger: 危险阈值（超过此值进入危险模式）
            rho_safe: 安全阈值（低于此值进入安全模式）
            max_planner_candidates: Planner 返回的最大候选数
            max_llm_candidates: LLM 返回的最大候选数
            enable_knowledge_base: 是否启用知识库
            **kwargs: 其他参数
        """
        super().__init__(action_space)
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        self.max_planner_candidates = max_planner_candidates
        self.max_llm_candidates = max_llm_candidates
        
        # 拓扑震荡防护：记录最近执行的拓扑动作
        self.last_topology_action: Optional[Dict[str, Any]] = None
        self.topology_cooldown_steps: int = 0
        self.topology_cooldown_period: int = 3  # 拓扑动作后的冷却期
        
        # Summarizer 降频：仅在状态重大变化时触发
        self.last_summary_state: Optional[Dict[str, Any]] = None
        self.summary_cooldown_steps: int = 0
        
        # ------------------------------------------------------------------
        # 初始化 LLM 客户端
        # ------------------------------------------------------------------
        if llm_client is None:
            try:
                self.llm_client = OpenAIChat()
            except Exception as e:
                logger.warning(f"LLM 客户端初始化失败: {e}，Judger 和 Summarizer 将被禁用")
                self.llm_client = None
        else:
            self.llm_client = llm_client
        
        # ------------------------------------------------------------------
        # 初始化知识库（可选，供 Judger / Summarizer 使用）
        # ------------------------------------------------------------------
        self.knowledge_base = None
        if enable_knowledge_base and self.llm_client:
            try:
                embedding = OpenAIEmbedding()
                self.knowledge_base = KnowledgeService(
                    embedding_model=embedding,
                )
                logger.info("知识库已启用并接入 ADA Agent")
            except Exception as e:
                logger.warning(f"知识库初始化失败，将在无 RAG 模式下运行: {e}")
                self.knowledge_base = None
        
        # ------------------------------------------------------------------
        # 初始化核心模块
        # ------------------------------------------------------------------
        try:
            self.planner = Planner(
                action_space=action_space,
                observation_space=observation_space,
                max_candidates=max_planner_candidates,
                rho_danger=rho_danger,  # 显式传递 rho_danger 参数
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Planner 初始化失败: {e}")
            self.planner = None
        
        try:
            # Solver 需要 env 参数来读取线路电抗
            # 使用 pop 确保 env 从 kwargs 中移除，避免重复传递
            env = kwargs.pop("env", None)
            self.solver = Solver(
                action_space=action_space,
                observation_space=observation_space,
                env=env,
                rho_danger=rho_danger,
                rho_safe=rho_safe,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Solver 初始化失败: {e}")
            self.solver = None
        try:
            if self.llm_client:
                self.judger = Judger(
                    action_space=action_space,
                    observation_space=observation_space,
                    llm_client=self.llm_client,
                    max_candidates=max_llm_candidates,
                    **kwargs,
                )
            else:
                logger.warning("LLM 客户端不可用，Judger 将被禁用")
                self.judger = None
        except Exception as e:
            logger.warning(f"Judger 初始化失败: {e}")
            self.judger = None
        
        try:
            self.simulator = Simulator(
                action_space=action_space,
                observation_space=observation_space,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Simulator 初始化失败: {e}")
            self.simulator = None
        
        try:
            if self.llm_client:
                self.summarizer = Summarizer(
                    llm_client=self.llm_client,
                    knowledge_base=self.knowledge_base,
                    enable_learning=enable_knowledge_base,
                    **kwargs,
                )
            else:
                logger.debug("LLM 客户端不可用，Summarizer 将被禁用")
                self.summarizer = None
        except Exception as e:
            logger.warning(f"Summarizer 初始化失败: {e}")
            self.summarizer = None
        
        logger.info(f"ADA Agent '{name}' 初始化完成")
    
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """
        主决策方法
        
        工作流程：
        1. 快速通道：安全状态下直接返回 Do Nothing
        2. 状态门控：根据 rho 值决定调用哪些模块
        3. 并行获取 Planner 和 Solver 的候选集
        4. 按需调用 Judger 获取融合增强的候选集
        5. 将所有候选集扔进 Simulator 进行"大逃杀"仿真
        6. 执行胜出者，并触发 Summarizer 学习（降频）
        
        Args:
            observation: 当前观测
            reward: 当前奖励
            done: 是否结束
            
        Returns:
            选中的动作
        """
        try:
            # --- Phase 0: 快速通道 (Fast Path) ---
            max_rho = float(observation.rho.max())
            overflow_count = int((observation.rho > 1.0).sum())
            line_status = observation.line_status if hasattr(observation, 'line_status') else None
            
            # 如果电网非常安全，且无断线需要重连，直接 Do Nothing
            if max_rho < self.rho_safe:
                # 检查是否有断线需要恢复
                if line_status is not None and np.all(line_status):
                    # 所有线路都连通，直接返回 Do Nothing
                    logger.debug(f"快速通道: 电网安全 (rho={max_rho:.3f})，返回 Do Nothing")
                    return self.action_space({})
                # 如果有断线，继续执行 Planner 的恢复逻辑
            
            # 更新拓扑冷却期
            if self.topology_cooldown_steps > 0:
                self.topology_cooldown_steps -= 1
            
            # --- Phase 1: 状态门控 (State Gating) ---
            # 仅在危险时启用 Judger（节省 Token）
            use_judger = (
                max_rho > self.rho_danger and 
                self.llm_client is not None and 
                self.judger is not None
            )
            
            # 仅在预警/危险时调用 Solver 的危险模式
            use_solver_danger = max_rho > self.rho_danger
            
            # --- Phase 2: 并行生成候选集 (Candidate Generation) ---
            
            # 1. Planner (复刻 ExpertAgent): 基于物理规则生成拓扑候选
            planner_candidates: List[CandidateAction] = []
            if self.planner:
                try:
                    planner_candidates = self.planner.suggest_topologies(observation)
                    logger.debug(f"Planner 生成 {len(planner_candidates)} 个候选")
                except Exception as e:
                    logger.error(f"Planner 执行失败: {e}", exc_info=True)
                    self.planner.suggest_topologies(observation)
            else:
                logger.warning("Planner 未初始化，跳过拓扑候选生成")
            
            # 2. Solver (复刻 OptimCVXPY): 基于数学优化生成调度候选
            solver_candidates: List[CandidateAction] = []
            if self.solver:
                try:
                    # 仅在需要时调用 Solver（安全状态下 Solver 会自动返回 Do Nothing）
                    solver_candidates = self.solver.solve_dispatch(observation)
                    logger.debug(f"Solver 生成 {len(solver_candidates)} 个候选")
                except Exception as e:
                    logger.error(f"Solver 执行失败: {e}", exc_info=True)
            else:
                logger.warning("Solver 未初始化，跳过调度候选生成")
            
            # 3. KnowledgeBase: 获取历史参考（仅在需要 Judger 时调用）
            history_context = ""
            if use_judger and self.knowledge_base:
                try:
                    # 构建查询字符串
                    query = self._build_knowledge_query(observation)
                    history_context = self.knowledge_base.get_context_string(query)
                    logger.debug(f"知识库检索到 {len(history_context)} 字符的上下文")
                except Exception as e:
                    logger.warning(f"知识库检索失败: {e}")
            
            # --- Phase 3: LLM 融合增强 (Fusion & Enhancement) - 按需调用 ---
            
            # 4. Judger (LLM): 仅在危险状态或 Planner/Solver 冲突时调用
            llm_candidates: List[CandidateAction] = []
            if use_judger:
                # 检查是否需要 Judger：危险状态或方案冲突
                needs_judger = self._needs_judger_intervention(
                    planner_candidates, 
                    solver_candidates, 
                    max_rho
                )
                
                if needs_judger:
                    try:
                        llm_candidates = self.judger.generate_fused_actions(
                            observation=observation,
                            planner_candidates=planner_candidates,
                            solver_candidates=solver_candidates,
                            history_context=history_context
                        )
                        logger.debug(f"Judger 生成 {len(llm_candidates)} 个候选")
                    except Exception as e:
                        logger.error(f"Judger 执行失败: {e}", exc_info=True)
                else:
                    logger.debug("Judger: 跳过（状态安全或方案一致）")
            else:
                logger.debug("Judger: 跳过（状态门控）")
            
            # --- Phase 4: 仿真竞技场 (The Arena) ---
            
            # 5. 构建混合动作空间（改进的去重）
            all_candidates = self._deduplicate_candidates(
                planner_candidates + solver_candidates + llm_candidates
            )
            logger.info(f"去重后共有 {len(all_candidates)} 个候选动作")
            
            # 6. 拓扑震荡防护：过滤反向拓扑动作
            if self.topology_cooldown_steps > 0 and self.last_topology_action:
                all_candidates = self._filter_topology_oscillation(
                    all_candidates, 
                    self.last_topology_action
                )
            
            # 7. Simulator: 暴力搜索与择优（动态排序权重）
            best_action: Optional[BaseAction] = None
            best_record: Optional[CandidateAction] = None
            
            if self.simulator and all_candidates:
                try:
                    best_record = self.simulator.select_best_action(
                        observation=observation,
                        candidates=all_candidates,
                        max_rho_before=max_rho  # 传递当前状态用于动态排序
                    )
                    if best_record and best_record.is_valid():
                        best_action = best_record.action_obj
                        logger.info(f"Simulator 选中动作: {best_record}")
                except Exception as e:
                    logger.error(f"Simulator 执行失败: {e}", exc_info=True)
            
            # 降级处理：改进的降级策略
            if best_action is None:
                best_action, best_record = self._fallback_strategy(
                    observation, 
                    solver_candidates, 
                    planner_candidates,
                    max_rho
                )
            
            # 记录拓扑动作（用于震荡防护）
            if best_record:
                self._update_topology_tracking(best_record)
            
            # --- Phase 5: 闭环学习 (Learning) - 降频调用 ---
            
            # 8. Summarizer: 仅在状态重大变化时记录经验
            if best_record and self.summarizer and self.llm_client:
                should_summarize = self._should_summarize(
                    observation, 
                    best_record, 
                    reward
                )
                if should_summarize:
                    try:
                        self.summarizer.summarize(
                            observation=observation,
                            best_action=best_record,
                            reward=reward
                        )
                        self.summary_cooldown_steps = 10  # 冷却 10 步
                    except Exception as e:
                        logger.warning(f"Summarizer 执行失败: {e}")
            
            return best_action
            
        except Exception as e:
            logger.error(f"ADA Agent act() 执行失败: {e}", exc_info=True)
            # 发生任何错误时，返回 Do Nothing
            return self.action_space({})
    
    def _deduplicate_candidates(self, candidates: List[CandidateAction]) -> List[CandidateAction]:
        """
        去重候选动作（改进版：使用规范化字符串描述）
        
        Args:
            candidates: 候选动作列表
            
        Returns:
            去重后的候选动作列表
        """
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            if not candidate.is_valid():
                continue
            
            # 使用动作的规范化字符串描述进行去重（更稳定）
            try:
                action_dict = candidate.action_obj.as_dict()
                # 规范化：移除 None 值，排序键
                normalized_str = self._normalize_action_dict(action_dict)
                action_key = hash(normalized_str)
            except Exception:
                # 降级：使用字符串表示
                action_key = hash(str(candidate.action_obj))
            
            if action_key not in seen:
                seen.add(action_key)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _normalize_action_dict(self, action_dict: Dict[str, Any]) -> str:
        """规范化动作字典为字符串（用于去重）"""
        # 移除 None 值
        cleaned = {k: v for k, v in action_dict.items() if v is not None}
        
        # 对列表和字典进行排序（如果可能）
        def normalize_value(v):
            if isinstance(v, dict):
                return {k: normalize_value(v) for k, v in sorted(v.items())}
            elif isinstance(v, list):
                return [normalize_value(item) for item in v]
            elif isinstance(v, np.ndarray):
                return v.tolist()
            else:
                return v
        
        cleaned = {k: normalize_value(v) for k, v in cleaned.items()}
        
        # 排序键并转换为 JSON 字符串
        return json.dumps(cleaned, sort_keys=True)
    
    def _needs_judger_intervention(
        self,
        planner_candidates: List[CandidateAction],
        solver_candidates: List[CandidateAction],
        max_rho: float
    ) -> bool:
        """
        判断是否需要 Judger 介入
        
        条件：
        1. 危险状态（max_rho > rho_danger）
        2. Planner 和 Solver 的建议发生严重冲突
        3. Planner 和 Solver 都没有提供安全候选
        
        Args:
            planner_candidates: Planner 候选
            solver_candidates: Solver 候选
            max_rho: 最大负载率
            
        Returns:
            如果需要 Judger 介入则返回 True
        """
        # 条件1：危险状态
        if max_rho <= self.rho_danger:
            return False
        
        # 条件2：检查是否有安全候选
        has_safe_candidate = False
        for cand in planner_candidates + solver_candidates:
            if cand.simulation_result and cand.simulation_result.get("is_safe", False):
                has_safe_candidate = True
                break
        
        # 如果没有安全候选，需要 Judger
        if not has_safe_candidate:
            return True
        
        # 条件3：检查方案冲突（简化：如果 Planner 和 Solver 都提供了候选，且类型不同）
        planner_has_topo = any(
            "Topo" in c.source or "topology" in c.description.lower() 
            for c in planner_candidates
        )
        solver_has_dispatch = any(
            "Dispatch" in c.source or "redispatch" in c.description.lower()
            for c in solver_candidates
        )
        
        # 如果两者都有候选但类型不同，可能需要融合
        if planner_has_topo and solver_has_dispatch:
            return True
        
        return False
    
    def _filter_topology_oscillation(
        self,
        candidates: List[CandidateAction],
        last_action: Dict[str, Any]
    ) -> List[CandidateAction]:
        """
        过滤可能导致拓扑震荡的动作
        
        Args:
            candidates: 候选动作列表
            last_action: 上一步执行的拓扑动作信息
            
        Returns:
            过滤后的候选动作列表
        """
        filtered = []
        last_action_type = last_action.get("type", "")
        last_sub_id = last_action.get("substation_id", -1)
        
        for candidate in candidates:
            action_dict = candidate.action_obj.as_dict()
            
            # 检查是否是反向拓扑操作
            is_reverse = False
            if 'set_bus' in action_dict and action_dict['set_bus']:
                set_bus_data = action_dict['set_bus']
                if isinstance(set_bus_data, dict) and 'substations_id' in set_bus_data:
                    subs_data = set_bus_data['substations_id']
                    if isinstance(subs_data, list) and len(subs_data) > 0:
                        sub_id, _ = subs_data[0]
                        # 如果是对同一个变电站的操作，可能是反向操作
                        if sub_id == last_sub_id and last_action_type == "set_bus":
                            is_reverse = True
            
            # 在冷却期内，禁止反向操作（除非是恢复操作）
            if is_reverse and self.topology_cooldown_steps > 0:
                logger.debug(f"过滤反向拓扑动作: {candidate.description}")
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _update_topology_tracking(self, best_record: CandidateAction) -> None:
        """更新拓扑动作跟踪（用于震荡防护）"""
        action_dict = best_record.action_obj.as_dict()
        
        # 检查是否是拓扑动作
        if 'set_bus' in action_dict and action_dict['set_bus']:
            set_bus_data = action_dict['set_bus']
            if isinstance(set_bus_data, dict) and 'substations_id' in set_bus_data:
                subs_data = set_bus_data['substations_id']
                if isinstance(subs_data, list) and len(subs_data) > 0:
                    sub_id, _ = subs_data[0]
                    self.last_topology_action = {
                        "type": "set_bus",
                        "substation_id": sub_id,
                        "step": 0  # 可以记录步数
                    }
                    self.topology_cooldown_steps = self.topology_cooldown_period
        elif 'set_line_status' in action_dict and action_dict['set_line_status']:
            line_status_data = action_dict['set_line_status']
            if isinstance(line_status_data, list) and len(line_status_data) > 0:
                line_id, _ = line_status_data[0]
                self.last_topology_action = {
                    "type": "set_line_status",
                    "line_id": line_id,
                    "step": 0
                }
                self.topology_cooldown_steps = self.topology_cooldown_period
    
    def _fallback_strategy(
        self,
        observation: BaseObservation,
        solver_candidates: List[CandidateAction],
        planner_candidates: List[CandidateAction],
        max_rho: float
    ) -> Tuple[BaseAction, Optional[CandidateAction]]:
        """
        改进的降级策略
        
        Args:
            observation: 当前观测
            solver_candidates: Solver 候选
            planner_candidates: Planner 候选
            max_rho: 最大负载率
            
        Returns:
            (best_action, best_record) 元组
        """
        # 极度危险时：尝试"牺牲局部保全局"，但必须经过 Simulator 仿真验证
        if max_rho > 1.2:
            logger.warning("极度危险状态，尝试牺牲局部保全局策略（带仿真校验）")
            
            # 尝试断开最严重的过载线路
            overflow_mask = observation.rho > 1.0
            if overflow_mask.any():
                worst_line_id = int(np.argmax(observation.rho))
                try:
                    sacrifice_action = self.action_space(
                        {"set_line_status": [(worst_line_id, -1)]}
                    )
                    # 如果有 Simulator，则先仿真验证该拓扑动作是否安全
                    if self.simulator is not None:
                        sim_result = self.simulator.evaluate_action(
                            observation, sacrifice_action
                        )
                        is_safe = bool(sim_result.get("is_safe", False))
                        if is_safe:
                            logger.warning(
                                f"降级策略：仿真验证通过，断开最严重过载线路 {worst_line_id}"
                            )
                            return sacrifice_action, CandidateAction(
                                source="Fallback_Sacrifice",
                                action_obj=sacrifice_action,
                                description=f"牺牲线路 {worst_line_id} 保全局",
                                priority=-1,
                            )
                        else:
                            logger.warning(
                                f"降级策略仿真判定为不安全，放弃断开线路 {worst_line_id}，"
                                f"原因: {sim_result.get('exception', 'rho 或 done 不满足安全条件')}"
                            )
                    else:
                        # 没有 Simulator 时，保守起见不执行断线动作
                        logger.warning(
                            "Simulator 不可用，跳过牺牲断线策略以避免自杀式拓扑操作"
                        )
                except Exception as e:
                    logger.error(f"牺牲策略失败: {e}")
        
        # 降级：使用 Solver 的第一个候选
        if solver_candidates:
            logger.warning("使用 Solver 降级方案")
            return solver_candidates[0].action_obj, solver_candidates[0]
        
        # 最后降级：Do Nothing
        logger.warning("使用 Do Nothing 降级方案")
        return self.action_space({}), None
    
    def _should_summarize(
        self,
        observation: BaseObservation,
        best_record: CandidateAction,
        reward: float
    ) -> bool:
        """
        判断是否应该触发 Summarizer（降频调用）
        
        仅在以下情况触发：
        1. 状态发生重大变化（从危险转为安全，或反之）
        2. Reward 突降（可能表示动作效果不佳）
        3. 冷却期已过
        
        Args:
            observation: 当前观测
            best_record: 选中的动作
            reward: 当前奖励
            
        Returns:
            是否应该触发 Summarizer
        """
        # 冷却期检查
        if self.summary_cooldown_steps > 0:
            self.summary_cooldown_steps -= 1
            return False
        
        # 状态变化检查
        max_rho = float(observation.rho.max())
        current_state = {
            "max_rho": max_rho,
            "is_danger": max_rho > self.rho_danger,
            "reward": reward
        }
        
        if self.last_summary_state is None:
            self.last_summary_state = current_state
            return True  # 第一次调用
        
        # 检查状态是否发生重大变化
        last_is_danger = self.last_summary_state.get("is_danger", False)
        current_is_danger = current_state["is_danger"]
        
        # 状态转换：危险 <-> 安全
        if last_is_danger != current_is_danger:
            self.last_summary_state = current_state
            return True
        
        # Reward 突降（下降超过 20%）
        last_reward = self.last_summary_state.get("reward", 0.0)
        if last_reward > 0 and reward < last_reward * 0.8:
            self.last_summary_state = current_state
            return True
        
        # 其他情况：不触发（降频）
        return False
    
    def _build_knowledge_query(self, observation: BaseObservation) -> str:
        """
        构建知识库查询字符串
        
        Args:
            observation: 当前观测
            
        Returns:
            查询字符串
        """
        # 提取关键信息：过载线路、负载率等
        rho = observation.rho
        overloaded_lines = np.where(rho >= 1.0)[0]
        
        query_parts = []
        if len(overloaded_lines) > 0:
            query_parts.append(f"过载线路: {overloaded_lines.tolist()}")
            query_parts.append(f"最大负载率: {np.max(rho):.3f}")
        
        return " ".join(query_parts) if query_parts else "电网安全状态"
    
    def reset(self, observation: BaseObservation) -> None:
        """
        重置 Agent 状态（在每个 episode 开始时调用）
        
        Args:
            observation: 初始观测
        """
        logger.info("ADA Agent 重置")
        
        # 重置各模块的内部状态
        if self.planner:
            self.planner.reset(observation)
        if self.solver:
            self.solver.reset(observation)
        if self.simulator:
            self.simulator.reset(observation)
        if self.judger:
            self.judger.reset(observation)
        if self.summarizer:
            self.summarizer.reset()
        
        # 重置内部状态
        self.last_topology_action = None
        self.topology_cooldown_steps = 0
        self.last_summary_state = None
        self.summary_cooldown_steps = 0
    
    def load(self, path: Optional[str]) -> None:
        """
        加载 Agent 状态
        
        Args:
            path: 加载路径
        """
        if path is None:
            return
        
        logger.info(f"从 {path} 加载 ADA Agent")
        # TODO: 实现加载逻辑
        # if self.knowledge_base:
        #     self.knowledge_base.load(path)
    
    def save(self, path: Optional[str]) -> None:
        """
        保存 Agent 状态
        
        Args:
            path: 保存路径
        """
        if path is None:
            return
        
        logger.info(f"保存 ADA Agent 到 {path}")
        # TODO: 实现保存逻辑
        # if self.knowledge_base:
        #     self.knowledge_base.save(path)

