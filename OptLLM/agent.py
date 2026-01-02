# -*- coding: utf-8 -*-
"""
OptAgent (OptLLM) - 混合智能优化代理

将 OptimCVXPY（凸优化）作为坚实的数学基座，负责处理连续变量（如调度、削减、储能）；
将 LLM 作为高级指挥官，负责处理非凸问题（如拓扑调整）、动态参数调整以及在多条行动路径中进行评估决策。

核心工作流：
1. 动作增强与优选 (Action Ensemble & Selection)：并行生成"纯优化动作"和"LLM增强动作"，通过仿真模拟择优录取。
2. 动态参数配置 (Dynamic Meta-Optimization)：LLM 根据当前电网场景（如过载严重程度、拥塞位置），动态调整优化器的超参数。
"""

from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import json
import copy

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入 OptimCVXPY
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 example/OptimCVXPY 到路径，以便直接导入 optimCVXPY 模块
optimcvxpy_path = project_root / "example" / "OptimCVXPY"
if str(optimcvxpy_path) not in sys.path:
    sys.path.insert(0, str(optimcvxpy_path))

try:
    from optimCVXPY import OptimCVXPY
    _HAS_OPTIMCVXPY = True
except ImportError:
    _HAS_OPTIMCVXPY = False
    OptimCVXPY = None

from .summarizer import StateSummarizer
from .prompts import PromptManager
from .parser import ActionParser

from utils import OpenAIChat, get_logger

logger = get_logger("OptAgent")


class OptAgent(BaseAgent):
    """
    混合智能优化代理 (OptLLM)
    
    架构：
    - Layer 1 (Base): OptimCVXPY - 处理连续变量优化
    - Layer 2 (Enhancer): LLM - 处理拓扑调整和参数调优
    - Layer 3 (Selector): Simulation - 评估并选择最佳动作
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        env,
        llm_client: OpenAIChat,
        llm_trigger_rho: float = 0.95,  # 超过此阈值才激活 LLM
        rho_safe: float = 0.85,
        rho_danger: float = 0.95,
        **optimizer_kwargs
    ):
        """
        初始化混合智能代理
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            env: Grid2Op 环境（用于 OptimCVXPY）
            llm_client: LLM 客户端
            llm_trigger_rho: LLM 激活阈值（默认 0.95）
            rho_safe: 安全阈值（默认 0.85）
            rho_danger: 危险阈值（默认 0.95）
            **optimizer_kwargs: 传递给 OptimCVXPY 的其他参数
        """
        super().__init__(action_space)
        
        if not _HAS_OPTIMCVXPY:
            raise RuntimeError("无法导入 OptimCVXPY，OptAgent 需要 OptimCVXPY 作为基础求解器")
        
        self.observation_space = observation_space
        self.env = env
        self.llm_client = llm_client
        self.llm_trigger_rho = llm_trigger_rho
        self.rho_safe = rho_safe
        self.rho_danger = rho_danger
        
        # 检查环境是否支持削减操作
        self.supports_curtailment = action_space.supports_type("curtail")
        
        # 初始化 OptimCVXPY（Layer 1: Base Solver）
        try:
            self.optim_agent = OptimCVXPY(
                action_space,
                env,
                rho_safe=rho_safe,
                rho_danger=rho_danger,
                **optimizer_kwargs
            )
            logger.info("OptimCVXPY 初始化成功")
        except Exception as e:
            logger.error(f"OptimCVXPY 初始化失败: {e}", exc_info=True)
            raise
        
        # 初始化 LLM 组件（Layer 2: Enhancer）
        self.summarizer = StateSummarizer()
        self.prompt_manager = PromptManager()
        self.parser = ActionParser()
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "optimizer_only": 0,  # 仅使用优化器（安全状态）
            "llm_activated": 0,   # LLM 被激活的次数
            "workflow1_count": 0, # Workflow 1 (拓扑增强) 使用次数
            "workflow2_count": 0, # Workflow 2 (参数调优) 使用次数
            "simulation_count": 0, # 仿真评估次数
            "best_action_selected": {}  # 记录每种策略被选中的次数
        }
        
        logger.info(f"OptAgent 初始化完成 (llm_trigger_rho={llm_trigger_rho})")
    
    def reset(self, observation: BaseObservation):
        """重置智能体状态"""
        self.stats["total_steps"] = 0
        self.stats["optimizer_only"] = 0
        self.stats["llm_activated"] = 0
        self.stats["workflow1_count"] = 0
        self.stats["workflow2_count"] = 0
        self.stats["simulation_count"] = 0
        self.stats["best_action_selected"] = {}
        
        if hasattr(self.optim_agent, 'reset'):
            self.optim_agent.reset(observation)
        
        logger.info("OptAgent 已重置")
    
    def act(
        self,
        observation: BaseObservation,
        reward: float = 0.0,
        done: bool = False
    ) -> BaseAction:
        """
        主决策函数
        
        流程：
        1. 快速过滤：如果 rho < llm_trigger_rho，直接使用优化器
        2. 危险状态：进入混合智能流程
           - Workflow 1: 并行生成动作（纯优化、LLM增强）
           - Workflow 2: LLM 参数调优
        3. 仿真评估：对所有候选动作进行仿真，选择最佳动作
        """
        self.stats["total_steps"] += 1
        
        max_rho = float(observation.rho.max())
        
        # --- 第一层：快速过滤 ---
        if max_rho < self.llm_trigger_rho:
            # 安全状态：仅使用优化器（通常是 Safe Recovery 模式）
            self.stats["optimizer_only"] += 1
            action = self.optim_agent.act(observation, reward, done)
            return action
        
        # --- 第二层：候选动作生成（危险状态）---
        self.stats["llm_activated"] += 1
        candidates = {}
        
        # 保存当前参数以便恢复
        original_params = self._capture_optim_params()
        
        # 1. 基准动作 (Standard Optimizer)
        try:
            action_base = self.optim_agent.act(observation, reward, done)
            candidates["base_optim"] = action_base
        except Exception as e:
            logger.warning(f"基准优化器失败: {e}")
            candidates["base_optim"] = self.action_space({})
        
        # 2. LLM 参数调优动作 (Workflow 2)
        tuned_params = self._get_llm_tuned_params(observation)
        if tuned_params:
            self.stats["workflow2_count"] += 1
            try:
                self._apply_optim_params(tuned_params)
                # 重新运行优化器
                action_tuned = self.optim_agent.act(observation, reward, done)
                candidates["tuned_optim"] = action_tuned
            except Exception as e:
                logger.warning(f"参数调优优化器失败: {e}")
            finally:
                # 恢复参数
                self._apply_optim_params(original_params)
        
        # 3. LLM 拓扑增强动作 (Workflow 1)
        action_topo = self._get_llm_topology_action(observation, candidates.get("base_optim"))
        if action_topo:
            self.stats["workflow1_count"] += 1
            # 组合动作：拓扑 + 优化器调度
            # Grid2Op 允许在同一个 step 混合 redispatch 和 set_bus
            base_action = candidates.get("base_optim", self.action_space({}))
            combined_action = self._combine_actions(base_action, action_topo)
            candidates["llm_topo"] = combined_action
        
        # --- 第三层：仿真评估与选择 ---
        if len(candidates) == 0:
            logger.warning("没有生成任何候选动作，返回空动作")
            return self.action_space({})
        
        best_action_name, best_action = self._evaluate_and_select(observation, candidates)
        
        # 更新统计
        self.stats["best_action_selected"][best_action_name] = \
            self.stats["best_action_selected"].get(best_action_name, 0) + 1
        
        logger.info(f"Step {self.stats['total_steps']}: 选择策略 '{best_action_name}' (Max Rho: {max_rho:.2%})")
        
        return best_action
    
    def _capture_optim_params(self) -> Dict[str, float]:
        """捕获当前优化器参数"""
        # OptimCVXPY 的属性是 cp.Parameter，需要获取 .value 来提取实际的 float 值
        def get_param_value(param):
            """安全获取参数值，兼容 Parameter 和 float"""
            if hasattr(param, 'value'):
                # 是 cp.Parameter，访问 .value 获取实际值
                return float(param.value)
            else:
                # 已经是 float 或其他数值类型
                return float(param)
        
        return {
            "margin_th_limit": get_param_value(self.optim_agent.margin_th_limit),
            "penalty_curtailment": get_param_value(self.optim_agent.penalty_curtailment),
            "penalty_redispatching": get_param_value(self.optim_agent.penalty_redispatching),
            "penalty_storage": get_param_value(self.optim_agent.penalty_storage),
        }
    
    def _apply_optim_params(self, params: Dict[str, float]):
        """动态修改 OptimCVXPY 实例的属性"""
        if "margin_th_limit" in params:
            self.optim_agent.margin_th_limit = params["margin_th_limit"]
        if "penalty_curtailment" in params:
            self.optim_agent.penalty_curtailment = params["penalty_curtailment"]
        if "penalty_redispatching" in params:
            self.optim_agent.penalty_redispatching = params["penalty_redispatching"]
        if "penalty_storage" in params:
            self.optim_agent.penalty_storage = params["penalty_storage"]
    
    def _get_llm_tuned_params(self, observation: BaseObservation) -> Optional[Dict[str, float]]:
        """
        Workflow 2: LLM 动态参数调优
        
        根据当前电网场景，让 LLM 调整优化器参数
        """
        try:
            # 生成状态摘要
            state_summary_dict = self.summarizer.summarize(observation)
            state_summary_text = self.summarizer.format_summary(state_summary_dict)
            
            # 构建参数调优 Prompt
            messages = self.prompt_manager.build_tuning_prompt(state_summary_text)
            
            # LLM 推理
            system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            
            llm_response = self.llm_client.chat(
                prompt=user_prompt,
                history=[],
                system_prompt=system_prompt
            )
            
            # 解析参数配置
            config = self.parser.extract_tuning_config(llm_response)
            
            if config:
                logger.info(f"LLM 参数调优: {config}")
                return config
            else:
                logger.warning("无法解析 LLM 参数调优响应")
                return None
                
        except Exception as e:
            logger.error(f"LLM 参数调优失败: {e}", exc_info=True)
            return None
    
    def _get_llm_topology_action(
        self,
        observation: BaseObservation,
        base_action: Optional[BaseAction]
    ) -> Optional[BaseAction]:
        """
        Workflow 1: LLM 拓扑增强
        
        将优化器的动作作为 context，让 LLM 建议拓扑调整
        """
        try:
            # 生成状态摘要
            state_summary_dict = self.summarizer.summarize(observation)
            state_summary_text = self.summarizer.format_summary(state_summary_dict)
            
            # 描述优化器的动作
            base_action_desc = self._describe_action(base_action) if base_action else "无动作"
            
            # 构建拓扑增强 Prompt
            messages = self.prompt_manager.build_enhancement_prompt(
                state_summary_text,
                base_action_desc
            )
            
            # LLM 推理
            system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            
            llm_response = self.llm_client.chat(
                prompt=user_prompt,
                history=[],
                system_prompt=system_prompt
            )
            
            # 解析拓扑动作
            action = self.parser.parse_topology_action(llm_response, self.action_space)
            
            if action:
                logger.info(f"LLM 拓扑增强: 生成拓扑动作")
                return action
            else:
                logger.warning("无法解析 LLM 拓扑动作")
                return None
                
        except Exception as e:
            logger.error(f"LLM 拓扑增强失败: {e}", exc_info=True)
            return None
    
    def _describe_action(self, action: BaseAction) -> str:
        """描述动作（用于 LLM context）"""
        desc_parts = []
        
        # 重调度
        if hasattr(action, 'redispatch') and action.redispatch is not None:
            redisp = action.redispatch
            if np.any(np.abs(redisp) > 0.01):
                redisp_sum = float(np.sum(np.abs(redisp)))
                desc_parts.append(f"重调度总量: {redisp_sum:.1f} MW")
        
        # 削减
        if hasattr(action, 'curtail_mw') and action.curtail_mw is not None:
            curt = action.curtail_mw
            # 某些环境下可能全为 -1，需过滤有效值
            if np.any(curt > 0):
                curt_sum = float(np.sum(curt[curt > 0]))
                desc_parts.append(f"削减总量: {curt_sum:.1f} MW")
        
        # 储能 - [增强检查]
        # 必须先检查环境是否有储能设备，避免在无储能环境中访问属性
        if self.action_space.n_storage > 0:
            if hasattr(action, 'storage_p') and action.storage_p is not None:
                storage = action.storage_p
                if np.any(np.abs(storage) > 0.01):
                    storage_sum = float(np.sum(np.abs(storage)))
                    desc_parts.append(f"储能调节: {storage_sum:.1f} MW")
        
        if not desc_parts:
            return "优化器建议：无动作（或动作很小）"
        
        return "优化器建议: " + ", ".join(desc_parts)
    
    def _combine_actions(self, action1: BaseAction, action2: BaseAction) -> BaseAction:
        """组合两个动作（拓扑 + 调度）"""
        # 创建新动作
        combined = self.action_space({})
        
        # 1. 复制调度 (Redispatching)
        # Grid2Op 环境通常都支持 redispatching，但检查属性是否存在更安全
        if hasattr(action1, 'redispatch') and action1.redispatch is not None:
            combined.redispatch = action1.redispatch.copy()
        
        # 2. 复制削减 (Curtailment)
        # 检查环境是否支持削减操作（类似 storage 的检查）
        if self.supports_curtailment:
            if hasattr(action1, 'curtail_mw') and action1.curtail_mw is not None:
                combined.curtail_mw = action1.curtail_mw.copy()
        
        # 3. 复制储能 (Storage) - [关键修复点]
        # 必须先检查环境是否有储能设备，否则赋值会抛出 IllegalAction 异常
        if self.action_space.n_storage > 0:
            if hasattr(action1, 'storage_p') and action1.storage_p is not None:
                combined.storage_p = action1.storage_p.copy()
        
        # 4. 添加 action2 的拓扑动作
        if hasattr(action2, 'set_bus') and action2.set_bus is not None:
            combined.set_bus = action2.set_bus.copy()
        if hasattr(action2, 'set_line_status') and action2.set_line_status is not None:
            combined.set_line_status = action2.set_line_status.copy()
        if hasattr(action2, 'change_bus') and action2.change_bus is not None:
            combined.change_bus = action2.change_bus.copy()
        
        return combined
    
    def _evaluate_and_select(
        self,
        obs: BaseObservation,
        candidates: Dict[str, BaseAction]
    ) -> Tuple[str, BaseAction]:
        """
        仿真评估与择优选择
        
        对所有候选动作进行仿真，根据安全性、有效性和成本选择最佳动作
        """
        best_score = -float('inf')
        best_act = self.action_space({})  # Default do nothing
        best_name = "do_nothing"
        
        for name, action in candidates.items():
            self.stats["simulation_count"] += 1
            
            try:
                # 使用 Grid2Op 的模拟功能
                sim_obs, _, sim_done, sim_info = obs.simulate(action, time_step=0)
                
                # 评分逻辑
                score = self._calculate_score(obs, sim_obs, sim_done, sim_info, action)
                
                logger.debug(f"策略 '{name}': 得分 {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_act = action
                    best_name = name
                    
            except Exception as e:
                logger.warning(f"策略 '{name}' 仿真失败: {e}")
                continue
        
        return best_name, best_act
    
    def _calculate_score(
        self,
        obs_before: BaseObservation,
        obs_after: BaseObservation,
        done: bool,
        sim_info: Dict,
        action: BaseAction
    ) -> float:
        """
        计算动作评分
        
        评分指标：
        1. 安全性 (Safety): 权重最高，rho < 1.0 且未导致 Game Over
        2. 有效性 (Recovery): rho 下降了多少
        3. 成本 (Cost): 动作的经济成本（调度 > 拓扑）
        """
        # 检查异常
        exception = sim_info.get('exception', None)
        if exception is not None:
            # 严重错误：潮流发散/电网解列
            if isinstance(exception, list):
                err_strs = [str(e) for e in exception]
                err_msg = '; '.join(err_strs)
            else:
                err_msg = str(exception)
            
            if any(keyword in err_msg for keyword in ["Divergence", "non connected", "SolverFactor", "unbounded"]):
                return -1000.0  # 严重错误，分数极低
            return -500.0  # 普通错误
        
        # 检查潮流发散
        if np.any(np.isnan(obs_after.rho)) or np.any(np.isinf(obs_after.rho)):
            return -1000.0
        
        # 安全检查
        max_rho_before = float(obs_before.rho.max())
        max_rho_after = float(obs_after.rho.max())
        
        if done:
            return -1000.0  # 导致游戏结束
        
        if max_rho_after > 1.5:
            return -800.0  # 极度过载
        
        # 评分计算
        score = 0.0
        
        # 1. 安全性（权重最高）
        if max_rho_after < 1.0:
            # 安全：给予高分
            safety_score = 100.0 - max_rho_after * 50.0  # rho越小分越高
        else:
            # 仍然过载：分数很低
            safety_score = -max_rho_after * 100.0
        
        score += safety_score * 2.0  # 安全性权重 x2
        
        # 2. 有效性（改善程度）
        if max_rho_before > 1.0:
            # 原本过载，检查改善
            improvement = max_rho_before - max_rho_after
            recovery_score = improvement * 50.0  # 每降低 0.01 rho 得 0.5 分
        else:
            # 原本安全，检查是否恶化
            if max_rho_after > max_rho_before + 0.10:
                recovery_score = -50.0  # 大幅恶化
            else:
                recovery_score = 0.0  # 保持安全
        
        score += recovery_score
        
        # 3. 成本（惩罚）
        cost = 0.0
        
        # 重调度成本
        if hasattr(action, 'redispatch') and action.redispatch is not None:
            redisp = action.redispatch
            cost += float(np.sum(np.abs(redisp))) * 0.03  # 每 MW 0.03 成本
        
        # 削减成本
        if hasattr(action, 'curtail_mw') and action.curtail_mw is not None:
            curt = action.curtail_mw
            if np.any(curt > 0):
                cost += float(np.sum(curt[curt > 0])) * 0.1  # 每 MW 0.1 成本
        
        # 拓扑动作成本（较低）
        if hasattr(action, 'set_bus') and action.set_bus is not None:
            if np.any(action.set_bus != 0):
                cost += 1.0  # 拓扑动作固定成本 1.0
        
        score -= cost  # 成本是惩罚项
        
        return score
