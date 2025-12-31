# -*- coding: utf-8 -*-
"""
混合智能体 (Hybrid Agent / NeuroSymbolic Agent)

分层混合控制架构：
- Layer 1 (Muscle/肌肉): OptimCVXPY - 负责数值优化（再调度、切负荷）
- Layer 2 (Brain/大脑): LLM-Topology - 负责拓扑调整（母线分裂）

设计哲学：
- 正常情况下使用 OptimCVXPY（快速、精确）
- 当 OptimCVXPY 失败或无法解决过载时，激活 LLM 进行拓扑调整
"""

from typing import Optional
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入 OptimCVXPY
import sys
import os
from pathlib import Path

# 添加 example/OptimCVXPY 到路径
current_file = Path(__file__).parent
project_root = current_file.parent.parent
optimcvxpy_path = project_root / "example" / "OptimCVXPY"
if str(optimcvxpy_path) not in sys.path:
    sys.path.insert(0, str(optimcvxpy_path))

try:
    from optimCVXPY import OptimCVXPY
except ImportError:
    # 如果导入失败，尝试从其他路径导入
    try:
        # 尝试从 l2rpn_baselines 导入
        from l2rpn_baselines.OptimCVXPY.optimCVXPY import OptimCVXPY
    except ImportError:
        OptimCVXPY = None
        logger.warning("无法导入 OptimCVXPY，请确保已安装或路径正确")

# 导入本地模块
from .topology_prompter import TopologyPrompter
from .topology_parser import TopologyParser

from utils import OpenAIChat, get_logger
logger = get_logger("HybridAgent")


class HybridAgent(BaseAgent):
    """
    混合智能体（神经符号混合架构）
    
    结合了：
    1. OptimCVXPY（数值优化器）- 处理连续变量
    2. LLM-Topology（拓扑专家）- 处理离散变量
    
    控制流程：
    1. 快速检查：如果 rho < 0.85，直接返回 OptimCVXPY 的结果
    2. 尝试优化：调用 OptimCVXPY 得到动作
    3. 模拟验证：检查优化后的状态是否安全
    4. LLM 介入：如果优化失败或仍然过载，激活 LLM 进行拓扑调整
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        env,
        llm_client: Optional[OpenAIChat] = None,
        rho_safe: float = 0.85,
        rho_danger: float = 0.95,
        rho_llm_threshold: float = 1.05,  # 超过此阈值才激活 LLM
        **optimizer_kwargs
    ):
        """
        初始化混合智能体
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            env: Grid2Op 环境（用于 OptimCVXPY）
            llm_client: LLM 客户端（如果为 None，则只使用 OptimCVXPY）
            rho_safe: 安全阈值（低于此值认为安全）
            rho_danger: 危险阈值（高于此值认为危险）
            rho_llm_threshold: LLM 激活阈值（优化后仍超过此值才激活 LLM）
            **optimizer_kwargs: 传递给 OptimCVXPY 的其他参数
        """
        super().__init__(action_space)
        
        self.observation_space = observation_space
        self.env = env
        self.llm_client = llm_client
        self.rho_safe = rho_safe
        self.rho_danger = rho_danger
        self.rho_llm_threshold = rho_llm_threshold
        
        # 初始化 OptimCVXPY（Layer 1: Muscle）
        if OptimCVXPY is None:
            logger.error("无法导入 OptimCVXPY，混合智能体将无法工作")
            self.optimizer = None
        else:
            try:
                self.optimizer = OptimCVXPY(
                    action_space,
                    env,
                    rho_safe=rho_safe,
                    rho_danger=rho_danger,
                    **optimizer_kwargs
                )
                logger.info("OptimCVXPY 初始化成功")
            except Exception as e:
                logger.error(f"OptimCVXPY 初始化失败: {e}", exc_info=True)
                self.optimizer = None
        
        # 初始化 LLM 组件（Layer 2: Brain）
        if llm_client is None:
            logger.warning("LLM 客户端为 None，将只使用 OptimCVXPY")
            self.topology_prompter = None
            self.topology_parser = None
        else:
            self.topology_prompter = TopologyPrompter()
            self.topology_parser = TopologyParser()
            logger.info("LLM 拓扑组件初始化成功")
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "optimizer_only": 0,
            "llm_activated": 0,
            "llm_success": 0,
            "llm_failed": 0,
            "simulation_failures": 0
        }
        
        logger.info(f"HybridAgent 初始化完成 (rho_safe={rho_safe}, rho_danger={rho_danger}, rho_llm_threshold={rho_llm_threshold})")
    
    def reset(self, observation: BaseObservation):
        """重置智能体状态"""
        self.stats = {k: 0 for k in self.stats}
        
        # 重置 OptimCVXPY
        if self.optimizer is not None:
            try:
                self.optimizer.reset(observation)
            except Exception as e:
                logger.warning(f"OptimCVXPY 重置失败: {e}")
        
        logger.info("HybridAgent 已重置")
    
    def act(
        self,
        observation: BaseObservation,
        reward: float = 0.0,
        done: bool = False
    ) -> BaseAction:
        """
        主决策函数
        
        控制流程：
        1. 快速检查：如果安全，直接使用 OptimCVXPY
        2. 尝试优化：调用 OptimCVXPY
        3. 模拟验证：检查优化结果
        4. LLM 介入：如果优化失败或仍然过载，激活 LLM
        """
        self.stats["total_steps"] += 1
        
        # 1. 快速检查：如果非常安全，直接使用 OptimCVXPY
        max_rho = float(observation.rho.max())
        if max_rho < self.rho_safe:
            if self.optimizer is not None:
                self.stats["optimizer_only"] += 1
                return self.optimizer.act(observation, reward, done)
            else:
                return self.action_space({})
        
        # 2. 尝试数值优化（OptimCVXPY）
        if self.optimizer is None:
            logger.warning("OptimCVXPY 不可用，返回空动作")
            return self.action_space({})
        
        opt_action = self.optimizer.act(observation, reward, done)
        
        # 3. 模拟验证优化结果
        sim_success, sim_obs, sim_info = self._simulate_action(opt_action, observation)
        
        if not sim_success:
            # 模拟失败（可能是动作非法）
            self.stats["simulation_failures"] += 1
            logger.warning("OptimCVXPY 动作模拟失败，尝试 LLM 拓扑调整")
            llm_action = self._try_llm_topology(observation)
            if llm_action is not None:
                return llm_action
            # LLM 也失败，返回优化器动作（作为保底）
            return opt_action
        
        # 检查优化后的负载率
        max_rho_after = float(sim_obs.rho.max())
        
        # 4. 判断是否需要 LLM 介入
        if max_rho_after > self.rho_llm_threshold:
            # 优化后仍然过载，激活 LLM
            logger.info(f"OptimCVXPY 无法解决过载 (Max Rho: {max_rho_after:.2%} > {self.rho_llm_threshold:.2%})，激活 LLM 拓扑调整")
            self.stats["llm_activated"] += 1
            
            llm_action = self._try_llm_topology(observation)
            if llm_action is not None:
                # 验证 LLM 动作
                llm_sim_success, llm_sim_obs, llm_sim_info = self._simulate_action(llm_action, observation)
                if llm_sim_success:
                    llm_max_rho = float(llm_sim_obs.rho.max())
                    # 如果 LLM 动作比优化器动作更好，使用 LLM 动作
                    if llm_max_rho < max_rho_after:
                        logger.info(f"LLM 拓扑调整有效 (Max Rho: {llm_max_rho:.2%} < {max_rho_after:.2%})")
                        self.stats["llm_success"] += 1
                        return llm_action
                    else:
                        logger.warning(f"LLM 拓扑调整无效 (Max Rho: {llm_max_rho:.2%} >= {max_rho_after:.2%})，使用优化器动作")
                        self.stats["llm_failed"] += 1
                else:
                    logger.warning("LLM 动作模拟失败，使用优化器动作")
                    self.stats["llm_failed"] += 1
            else:
                logger.warning("LLM 无法生成有效动作，使用优化器动作")
                self.stats["llm_failed"] += 1
        
        # 5. 使用优化器动作
        self.stats["optimizer_only"] += 1
        return opt_action
    
    def _simulate_action(
        self,
        action: BaseAction,
        observation: BaseObservation
    ) -> tuple[bool, Optional[BaseObservation], dict]:
        """
        模拟动作（不实际执行）
        
        Returns:
            (success, sim_obs, sim_info)
            - success: 是否成功
            - sim_obs: 模拟后的观测（如果成功）
            - sim_info: 模拟信息（包含异常等）
        """
        try:
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=0)
            
            # 检查异常
            exception = sim_info.get('exception', None)
            if exception is not None:
                return False, None, sim_info
            
            # 检查潮流发散
            if np.any(np.isnan(sim_obs.rho)) or np.any(np.isinf(sim_obs.rho)):
                return False, None, sim_info
            
            return True, sim_obs, sim_info
            
        except Exception as e:
            logger.warning(f"动作模拟异常: {e}")
            return False, None, {"exception": str(e)}
    
    def _try_llm_topology(
        self,
        observation: BaseObservation
    ) -> Optional[BaseAction]:
        """
        尝试使用 LLM 进行拓扑调整
        
        Args:
            observation: 当前观测
            
        Returns:
            Grid2Op Action，如果失败则返回 None
        """
        # 检查 LLM 是否可用
        if self.llm_client is None or self.topology_prompter is None or self.topology_parser is None:
            return None
        
        # 1. 找到最严重的过载线路
        worst_line_id = self._find_worst_overloaded_line(observation)
        if worst_line_id is None:
            logger.warning("未找到过载线路，无法使用 LLM 拓扑调整")
            return None
        
        # 2. 构建提示词
        try:
            messages = self.topology_prompter.build_prompt(observation, worst_line_id)
        except Exception as e:
            logger.error(f"构建提示词失败: {e}", exc_info=True)
            return None
        
        # 3. 调用 LLM
        try:
            system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            llm_history = [msg for msg in messages if msg["role"] not in ["system", "user"]]
            
            llm_response = self.llm_client.chat(
                prompt=user_prompt,
                history=llm_history,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}", exc_info=True)
            return None
        
        # 4. 解析 LLM 响应
        try:
            action = self.topology_parser.parse_bus_splitting(
                llm_response,
                observation,
                self.action_space
            )
            return action
        except Exception as e:
            logger.error(f"解析 LLM 响应失败: {e}", exc_info=True)
            return None
    
    def _find_worst_overloaded_line(
        self,
        observation: BaseObservation
    ) -> Optional[int]:
        """
        找到最严重的过载线路
        
        Returns:
            线路ID，如果没有过载线路则返回 None
        """
        overflow_mask = observation.rho > 1.0
        if not np.any(overflow_mask):
            return None
        
        overflow_indices = np.where(overflow_mask)[0]
        overflow_rhos = observation.rho[overflow_indices]
        
        # 找到负载率最高的线路
        worst_idx = np.argmax(overflow_rhos)
        worst_line_id = int(overflow_indices[worst_idx])
        
        return worst_line_id
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats["total_steps"] > 0:
            stats["optimizer_only_rate"] = stats["optimizer_only"] / stats["total_steps"]
            stats["llm_activation_rate"] = stats["llm_activated"] / stats["total_steps"]
            if stats["llm_activated"] > 0:
                stats["llm_success_rate"] = stats["llm_success"] / stats["llm_activated"]
            else:
                stats["llm_success_rate"] = 0.0
        return stats

