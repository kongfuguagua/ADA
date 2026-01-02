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

from typing import Optional, Dict, Any, Tuple
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入 OptimCVXPY
import sys
import os
from pathlib import Path

# 添加项目根目录到路径，以便导入 utils 模块
current_file = Path(__file__).parent
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 example/OptimCVXPY 到路径，以便直接导入 optimCVXPY 模块
optimcvxpy_path = project_root / "example" / "OptimCVXPY"
if str(optimcvxpy_path) not in sys.path:
    sys.path.insert(0, str(optimcvxpy_path))

from optimCVXPY import OptimCVXPY

# 导入本地模块
from .topology_prompter import TopologyPrompter
from .topology_parser import TopologyParser
from .summarizer import StateSummarizer

from utils import OpenAIChat, get_logger
logger = get_logger("HybridAgent")


class HybridAgent(BaseAgent):
    """
    混合智能体（神经符号混合架构，参考 OptLLM 重构）
    
    结合了：
    1. OptimCVXPY（数值优化器）- 处理连续变量（再调度、切负荷、储能）
    2. LLM-Topology（拓扑专家）- 处理离散变量（母线分裂、线路开关）
    3. LLM-Parameter Tuning（参数调优）- 动态调整优化器参数
    
    控制流程（重构后，参考 OptLLM）：
    1. 快速检查：如果 rho < rho_safe，直接返回 OptimCVXPY 的结果
    2. 候选动作生成：
       - 基准优化动作（标准 OptimCVXPY）
       - LLM 参数调优动作（动态调整优化器参数后重新优化）
       - LLM 拓扑增强动作（拓扑调整 + 优化器调度组合）
    3. 仿真评估：对所有候选动作进行仿真，根据安全性、有效性和成本选择最佳动作
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
        
        # 检查环境是否支持削减操作
        self.supports_curtailment = action_space.supports_type("curtail")
        
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
            self.summarizer = None
        else:
            self.topology_prompter = TopologyPrompter()
            self.topology_parser = TopologyParser()
            self.summarizer = StateSummarizer()
            logger.info("LLM 拓扑组件初始化成功")
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "optimizer_only": 0,
            "llm_activated": 0,
            "llm_success": 0,
            "llm_failed": 0,
            "simulation_failures": 0,
            "param_tuning_count": 0,  # 参数调优次数
            "action_combination_count": 0,  # 动作组合次数
        }
        
        # 动作缓存（用于提高效率）
        self.topology_cache = {}
        
        logger.info(f"HybridAgent 初始化完成 (rho_safe={rho_safe}, rho_danger={rho_danger}, rho_llm_threshold={rho_llm_threshold})")
    
    def reset(self, observation: BaseObservation):
        """重置智能体状态"""
        self.stats = {k: 0 for k in self.stats}
        
        # 清空动作缓存
        self.topology_cache = {}
        
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
        主决策函数（重构后，参考 OptLLM）
        
        控制流程：
        1. 快速检查：如果安全，直接使用 OptimCVXPY
        2. 候选动作生成：
           - 基准优化动作
           - LLM 参数调优动作（可选）
           - LLM 拓扑增强动作（可选）
        3. 仿真评估：对所有候选动作进行仿真，选择最佳动作
        """
        self.stats["total_steps"] += 1
        
        max_rho = float(observation.rho.max())
        
        # 1. 快速检查：如果非常安全，直接返回优化器动作
        if max_rho < self.rho_safe:
            if self.optimizer is None:
                return self.action_space({})
            try:
                action = self.optimizer.act(observation, reward, done)
                self.stats["optimizer_only"] += 1
                return action
            except Exception as e:
                logger.error(f"优化器计算出错: {e}")
                return self.action_space({})
        
        # 2. 危险状态：进入混合智能流程
        if self.optimizer is None:
            logger.warning("OptimCVXPY 不可用，返回空动作")
            return self.action_space({})
        
        candidates = {}
        
        # 保存当前参数以便恢复
        original_params = self._capture_optim_params()
        
        # 2.1 基准动作 (Standard Optimizer)
        try:
            action_base = self.optimizer.act(observation, reward, done)
            candidates["base_optim"] = action_base
        except Exception as e:
            logger.warning(f"基准优化器失败: {e}")
            candidates["base_optim"] = self.action_space({})
        
        # 2.2 LLM 参数调优动作（如果 LLM 可用）
        if self.llm_client is not None and self.summarizer is not None:
            tuned_params = self._get_llm_tuned_params(observation)
            if tuned_params:
                self.stats["param_tuning_count"] += 1
                try:
                    self._apply_optim_params(tuned_params)
                    # 重新运行优化器
                    action_tuned = self.optimizer.act(observation, reward, done)
                    candidates["tuned_optim"] = action_tuned
                except Exception as e:
                    logger.warning(f"参数调优优化器失败: {e}")
                finally:
                    # 恢复参数
                    self._apply_optim_params(original_params)
        
        # 2.3 LLM 拓扑增强动作（如果 LLM 可用）
        if self.llm_client is not None and max_rho > self.rho_llm_threshold:
            self.stats["llm_activated"] += 1
            action_topo = self._try_llm_topology(observation)
            if action_topo:
                # 组合动作：拓扑 + 优化器调度
                base_action = candidates.get("base_optim", self.action_space({}))
                combined_action = self._combine_actions(base_action, action_topo)
                candidates["llm_topo"] = combined_action
                self.stats["action_combination_count"] += 1
        
        # 3. 仿真评估与选择
        if len(candidates) == 0:
            logger.warning("没有生成任何候选动作，返回空动作")
            return self.action_space({})
        
        best_action_name, best_action = self._evaluate_and_select(observation, candidates)
        
        # 更新统计（注意：optimizer_only 在快速检查时已经统计，这里不再重复统计）
        if best_action_name == "tuned_optim":
            pass  # 已在上面统计 param_tuning_count
        elif best_action_name == "llm_topo":
            self.stats["llm_success"] += 1
        elif best_action_name == "base_optim":
            # 只有在危险状态下选择 base_optim 时才统计（快速检查时已统计）
            if max_rho >= self.rho_safe:
                self.stats["optimizer_only"] += 1
        
        logger.debug(f"Step {self.stats['total_steps']}: 选择策略 '{best_action_name}' (Max Rho: {max_rho:.2%})")
        
        return best_action
    
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
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=2)
            
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
        尝试使用 LLM 进行拓扑调整（带缓存机制）
        
        Args:
            observation: 当前观测
            
        Returns:
            Grid2Op Action，如果失败则返回 None
        """
        # 检查 LLM 是否可用
        if self.llm_client is None or self.topology_prompter is None or self.topology_parser is None:
            return None
        
        # 1. 找到最严重的过载线路（使用配置的 rho_danger 作为阈值）
        worst_line_id = self._find_worst_overloaded_line(observation, threshold=self.rho_danger)
        if worst_line_id is None:
            logger.warning("未找到过载线路，无法使用 LLM 拓扑调整")
            return None
        
        # 2. 检查缓存（使用过载线路ID作为缓存键）
        cache_key = worst_line_id
        if cache_key in self.topology_cache:
            logger.info(f"命中拓扑缓存: Line {worst_line_id}")
            cached_action = self.topology_cache[cache_key]
            # 验证缓存的动作在当前是否仍然合法（简单检查）
            try:
                # 尝试模拟缓存的动作
                sim_success, _, _ = self._simulate_action(cached_action, observation)
                if sim_success:
                    return cached_action
                else:
                    # 缓存的动作已失效，删除缓存
                    del self.topology_cache[cache_key]
                    logger.info(f"缓存动作已失效，重新生成")
            except Exception:
                # 缓存的动作已失效，删除缓存
                del self.topology_cache[cache_key]
                logger.info(f"缓存动作验证失败，重新生成")
        
        # 3. 构建提示词
        try:
            messages = self.topology_prompter.build_prompt(observation, worst_line_id)
        except Exception as e:
            logger.error(f"构建提示词失败: {e}", exc_info=True)
            return None
        
        # 4. 调用 LLM
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
        
        # 5. 解析 LLM 响应
        try:
            action = self.topology_parser.parse_bus_splitting(
                llm_response,
                observation,
                self.action_space
            )
            
            # 6. 如果解析成功，存入缓存
            if action is not None:
                self.topology_cache[cache_key] = action
                return action
            return None
        except Exception as e:
            logger.error(f"解析 LLM 响应失败: {e}", exc_info=True)
            return None
    
    def _capture_optim_params(self) -> Dict[str, float]:
        """捕获当前优化器参数"""
        if self.optimizer is None:
            return {}
        
        def get_param_value(param):
            """安全获取参数值，兼容 Parameter 和 float"""
            if hasattr(param, 'value'):
                # 是 cp.Parameter，访问 .value 获取实际值
                return float(param.value)
            else:
                # 已经是 float 或其他数值类型
                return float(param)
        
        params = {}
        if hasattr(self.optimizer, 'margin_th_limit'):
            params["margin_th_limit"] = get_param_value(self.optimizer.margin_th_limit)
        if hasattr(self.optimizer, 'penalty_curtailment'):
            params["penalty_curtailment"] = get_param_value(self.optimizer.penalty_curtailment)
        if hasattr(self.optimizer, 'penalty_redispatching'):
            params["penalty_redispatching"] = get_param_value(self.optimizer.penalty_redispatching)
        if hasattr(self.optimizer, 'penalty_storage'):
            params["penalty_storage"] = get_param_value(self.optimizer.penalty_storage)
        
        return params
    
    def _apply_optim_params(self, params: Dict[str, float]):
        """动态修改 OptimCVXPY 实例的属性"""
        if self.optimizer is None:
            return
        
        if "margin_th_limit" in params:
            self.optimizer.margin_th_limit = params["margin_th_limit"]
        if "penalty_curtailment" in params:
            self.optimizer.penalty_curtailment = params["penalty_curtailment"]
        if "penalty_redispatching" in params:
            self.optimizer.penalty_redispatching = params["penalty_redispatching"]
        if "penalty_storage" in params:
            self.optimizer.penalty_storage = params["penalty_storage"]
    
    def _get_llm_tuned_params(self, observation: BaseObservation) -> Optional[Dict[str, float]]:
        """
        LLM 动态参数调优
        
        根据当前电网场景，让 LLM 调整优化器参数
        """
        if self.llm_client is None or self.summarizer is None:
            return None
        
        try:
            # 生成状态摘要
            state_summary_dict = self.summarizer.summarize(observation)
            state_summary_text = self.summarizer.format_summary(state_summary_dict)
            
            # 构建参数调优 Prompt
            messages = self._build_tuning_prompt(state_summary_text)
            
            # LLM 推理
            system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            
            llm_response = self.llm_client.chat(
                prompt=user_prompt,
                history=[],
                system_prompt=system_prompt
            )
            
            # 解析参数配置（使用 topology_parser 的解析方法，如果支持的话）
            # 或者使用简单的 JSON 解析
            import json
            import re
            
            # 尝试提取 JSON
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(llm_response)
            for match in matches:
                try:
                    config = json.loads(match)
                    if "margin_th_limit" in config or "penalty_curtailment" in config:
                        # 规范化配置
                        normalized = {}
                        if "margin_th_limit" in config:
                            normalized["margin_th_limit"] = float(config["margin_th_limit"])
                        if "penalty_curtailment" in config:
                            normalized["penalty_curtailment"] = float(config["penalty_curtailment"])
                        if "penalty_redispatch" in config:
                            normalized["penalty_redispatching"] = float(config["penalty_redispatch"])
                        elif "penalty_redispatching" in config:
                            normalized["penalty_redispatching"] = float(config["penalty_redispatching"])
                        if "penalty_storage" in config:
                            normalized["penalty_storage"] = float(config["penalty_storage"])
                        
                        if normalized:
                            logger.info(f"LLM 参数调优: {normalized}")
                            return normalized
                except json.JSONDecodeError:
                    continue
            
            logger.warning("无法解析 LLM 参数调优响应")
            return None
                
        except Exception as e:
            logger.error(f"LLM 参数调优失败: {e}", exc_info=True)
            return None
    
    def _build_tuning_prompt(self, state_summary: str) -> list:
        """构建参数调优 Prompt"""
        system_prompt = """你是电网优化器参数调优专家。

## 你的任务
根据当前电网状态，动态调整优化器的超参数，以提高求解成功率和效果。

## 可调参数

1. **margin_th_limit** (热极限安全裕度)
   - 范围: 0.5 ~ 1.0
   - 默认: 0.9
   - **调优策略**:
     - 如果优化器无解 (INFEASIBLE)，**提高**此值（如 0.95 或 0.98）以放宽约束
     - 如果当前过载严重 (Max Rho > 110%)，可以提高到 1.0 以允许满载运行
     - 如果 AC/DC 误差导致模拟失败，**降低**此值（如 0.85）以留出更多缓冲

2. **penalty_curtailment** (切负荷惩罚)
   - 范围: 0.001 ~ 10.0
   - 默认: 0.1
   - **调优策略**:
     - 如果过载严重且优化器无解，**降低**此值（如 0.01 或 0.001）以允许更多切负荷
     - 如果系统安全，**提高**此值（如 1.0 或 10.0）以禁止切负荷

3. **penalty_redispatch** (再调度惩罚)
   - 范围: 0.01 ~ 1.0
   - 默认: 0.03
   - 通常保持默认值，除非需要优先使用再调度

4. **penalty_storage** (储能惩罚)
   - 范围: 0.1 ~ 1.0
   - 默认: 0.3
   - 通常保持默认值

## 输出格式

请直接输出 JSON，不要包含 Markdown 代码块：

{
    "margin_th_limit": 0.95,
    "penalty_curtailment": 0.01,
    "penalty_redispatch": 0.03,
    "penalty_storage": 0.3,
    "reasoning": "简要说明调优理由"
}"""
        
        user_content = f"""当前电网状态：

{state_summary}

请根据上述状态，输出优化器参数配置（JSON 格式）："""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
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
        
        # 储能
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
        if hasattr(action1, 'redispatch') and action1.redispatch is not None:
            combined.redispatch = action1.redispatch.copy()
        
        # 2. 复制削减 (Curtailment)
        # 检查环境是否支持削减操作（类似 storage 的检查）
        if self.supports_curtailment:
            if hasattr(action1, 'curtail_mw') and action1.curtail_mw is not None:
                combined.curtail_mw = action1.curtail_mw.copy()
        
        # 3. 复制储能 (Storage)
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
    
    def _find_worst_overloaded_line(
        self,
        observation: BaseObservation,
        threshold: Optional[float] = None
    ) -> Optional[int]:
        """
        找到最严重的过载线路
        
        Args:
            observation: 电网观测
            threshold: 过载阈值（如果未指定，默认使用 rho_danger）
        
        Returns:
            线路ID，如果没有过载线路则返回 None
        """
        # 如果未指定，默认使用 dangerous 阈值，而不是硬编码的 1.0
        if threshold is None:
            threshold = self.rho_danger
        
        overflow_mask = observation.rho > threshold
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

