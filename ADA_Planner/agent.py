# -*- coding: utf-8 -*-
"""
ADA_Planner Agent 核心实现 (Fix: Logic Bug & Auto-Correction)
"""

import sys
import re
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 简化导入逻辑
try:
    from .formatters import ObservationFormatter
    from .parser import ActionParser
    from .prompts import PromptManager
except ImportError:
    from ADA_Planner.formatters import ObservationFormatter
    from ADA_Planner.parser import ActionParser
    from ADA_Planner.prompts import PromptManager

# ExpertInsight 是可选的（简化导入逻辑）
try:
    from .analysis import ExpertInsightService
    _HAS_EXPERT_INSIGHT = True
except ImportError:
    try:
        from ADA_Planner.analysis import ExpertInsightService
        _HAS_EXPERT_INSIGHT = True
    except ImportError:
        _HAS_EXPERT_INSIGHT = False
        ExpertInsightService = None

# 导入 ADA 工具
from utils import OpenAIChat, get_logger

logger = get_logger("ADA_Planner")


class ADA_Planner(BaseAgent):
    """
    ADA_Planner Baseline Agent (Enhanced with Expert Insight)
    修复了模拟验证逻辑 BUG，并增加了自动修正（Auto-Correction）功能。
    集成了 ExpertInsightService，实现"符号引导，神经执行"的设计理念。
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        max_react_steps: int = 3,
        name: str = "ADA_Planner",
        rho_danger: float = 0.92, # 降低默认阈值以实现预防性调度
        grid_name: str = "IEEE14",  # 用于 ExpertInsight
        use_expert_insight: bool = True,  # 是否启用专家洞察
        **kwargs
    ):
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_react_steps = max_react_steps
        self.rho_danger = rho_danger
        self.use_expert_insight = use_expert_insight and _HAS_EXPERT_INSIGHT
        
        self.formatter = ObservationFormatter()
        self.parser = ActionParser()
        self.prompt_manager = PromptManager()
        
        # Expert Insight Service（专家洞察服务）
        self.expert_insight = None
        if self.use_expert_insight:
            try:
                self.expert_insight = ExpertInsightService(
                    action_space,
                    observation_space,
                    grid_name=grid_name,
                    **kwargs
                )
                logger.info("ExpertInsightService 已启用")
            except Exception as e:
                logger.warning(f"ExpertInsightService 初始化失败: {e}，ExpertInsight 将被禁用")
                self.use_expert_insight = False
        
        self.current_step = 0
        self.react_history = []
        self.env_info = None
        self.current_insight_report = None  # 存储当前专家洞察报告
        self.last_action = None  # 防震荡：记录上一步动作
        self.last_action_description = ""  # 上一步动作的描述
        
        self.stats = {
            "total_steps": 0,
            "react_loops": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "do_nothing_count": 0,
            "sanitized_count": 0, # 统计自动修正次数
            "expert_insight_calls": 0,  # 统计专家洞察调用次数
            "trusted_expert_actions": 0,  # 信任直通车：跳过二次模拟的次数
        }
        
        logger.info(f"ADA_Planner '{name}' 初始化完成 (v3.0 Expert-Augmented)")
    
    def reset(self, observation: BaseObservation):
        self.current_step = 0
        self.react_history = []
        self.last_action = None
        self.last_action_description = ""
        if self.env_info is None:
            self.env_info = self._extract_env_info(observation)
            self.prompt_manager.set_env_info(self.env_info)
        self.stats = {k: 0 for k in self.stats}
        logger.info(f"ADA_Planner 已重置")
    
    def act(self, observation: BaseObservation, reward: float = 0.0, done: bool = False) -> BaseAction:
        self.current_step += 1
        self.stats["total_steps"] += 1
        self.current_insight_report = None  # 重置当前洞察报告
        
        # 0. 启发式策略
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        if overflow_count == 0 and max_rho <= self.rho_danger:
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 1. 准备 ADA_Planner
        obs_text = self.formatter.format(observation)
        
        # 1.5. 专家洞察（Expert Insight）- 关键步骤
        expert_insight_text = ""
        if self.use_expert_insight and self._is_danger(observation):
            logger.info("Danger detected. Consulting ExpertInsight...")
            try:
                insight_report = self.expert_insight.generate_insight(observation)
                self.current_insight_report = insight_report  # 存储供 ActionParser 使用
                self.stats["expert_insight_calls"] += 1
                
                # === 策略一：信任直通车 ===
                # 如果 Expert 给出了 Score=4 的完美方案，直接执行，跳过 LLM
                trusted_action = self._check_trust_pass_through(insight_report, observation)
                if trusted_action is not None:
                    logger.info(f"Step {self.current_step}: 触发信任直通车 - Expert Score=4，直接执行，跳过 LLM")
                    self.stats["trusted_expert_actions"] += 1
                    self.stats["successful_actions"] += 1
                    self.react_history = []
                    self.last_action = trusted_action
                    self.last_action_description = self._describe_action_for_history(trusted_action)
                    return trusted_action
                
                # 将报告转化为自然语言
                expert_insight_text = self._format_expert_insight(insight_report)
                logger.info(f"ExpertInsight generated {len(insight_report.get('solutions', []))} solutions")
            except Exception as e:
                logger.error(f"ExpertInsight failed: {e}")
                traceback.print_exc()
                # ExpertInsight 内部已有降级逻辑，如果失败则返回空字符串，让LLM自己决策
                expert_insight_text = ""
                self.current_insight_report = None
        # 如果没有过载或 ExpertInsight 不可用，expert_insight_text 保持为空字符串
        
        is_first_call = len(self.react_history) == 0
        history = self.prompt_manager.build(
            obs_text, 
            self.react_history if not is_first_call else None, 
            physics_hint=expert_insight_text,
            expert_insight=self.current_insight_report,  # 传递原始报告
            last_action=self.last_action_description  # 防震荡：传递上一步动作
        )
        
        # 2. ADA_Planner 循环
        for react_step in range(self.max_react_steps):
            self.stats["react_loops"] += 1
            
            try:
                # LLM 调用
                prompt = history[-1]["content"] if history[-1]["role"] == "user" else ""
                llm_history = [msg for msg in history if msg["role"] != "system" and msg != history[-1]]
                system_prompt = next((msg["content"] for msg in history if msg["role"] == "system"), None)
                
                llm_response = self.llm_client.chat(prompt=prompt, history=llm_history, system_prompt=system_prompt)
                
                # 提取和解析
                action_text = self.parser.extract_action_from_response(llm_response)
                if not action_text:
                    self.stats["failed_actions"] += 1
                    return self.action_space({})
                
                try:
                    # 传递当前洞察报告给解析器（用于 execute_expert_solution）
                    raw_action = self.parser.parse(
                        action_text, 
                        self.action_space,
                        expert_insight_report=self.current_insight_report
                    )
                    
                    # === 新增：动作自动修正 (Auto-Correction) ===
                    # 自动处理爬坡率限制，避免因为数值超限导致动作非法
                    sanitized_action, correction_msg = self._sanitize_action(raw_action, observation)
                    if correction_msg:
                        logger.info(f"Step {self.current_step}: 动作已自动修正 -> {correction_msg}")
                        self.stats["sanitized_count"] += 1
                    
                    # === 信任直通车：如果Expert Score=4，跳过二次模拟 ===
                    should_trust_expert = self._should_trust_expert_action(
                        action_text, 
                        self.current_insight_report
                    )
                    
                    if should_trust_expert:
                        logger.info(f"Step {self.current_step}: 信任Expert方案（Score=4），跳过二次模拟")
                        self.stats["trusted_expert_actions"] += 1
                        self.stats["successful_actions"] += 1
                        self.react_history = []
                        # 记录动作用于防震荡
                        self.last_action = sanitized_action
                        self.last_action_description = self._describe_action_for_history(sanitized_action)
                        return sanitized_action
                    
                    # 模拟验证（非信任路径）
                    is_safe, reason = self._simulate_action(sanitized_action, observation)
                    
                    if is_safe:
                        logger.info(f"Step {self.current_step}: 动作安全 (Step {react_step+1})")
                        self.stats["successful_actions"] += 1
                        self.react_history = []
                        # 记录动作用于防震荡
                        self.last_action = sanitized_action
                        self.last_action_description = self._describe_action_for_history(sanitized_action)
                        return sanitized_action
                    else:
                        # 反馈错误
                        feedback_msg = f"模拟警告: {reason}"
                        if correction_msg:
                            feedback_msg += f" (注: 原始动作已被修正: {correction_msg})"
                            
                        self.react_history.append({"role": "assistant", "content": llm_response})
                        self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, feedback_msg)
                        history = self.prompt_manager.build(
                            obs_text, 
                            self.react_history, 
                            physics_hint=expert_insight_text,
                            expert_insight=self.current_insight_report,
                            last_action=self.last_action_description
                        )
                        continue
                        
                except ValueError as e:
                    # 处理解析错误（包括 execute_expert_solution 需要专家报告的情况）
                    error_msg = str(e)
                    if "execute_expert_solution" in error_msg and "需要专家洞察报告" in error_msg:
                        # 如果 LLM 尝试使用 execute_expert_solution 但没有专家报告，提示它使用其他方法
                        error_msg = f"{error_msg}。当前没有可用的专家方案，请使用 redispatch() 或 set_line_status() 手动生成动作。"
                    
                    self.react_history.append({"role": "assistant", "content": llm_response})
                    self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, f"格式错误: {error_msg}")
                    history = self.prompt_manager.build(
                        obs_text, 
                        self.react_history, 
                        physics_hint=expert_insight_text,
                        expert_insight=self.current_insight_report,
                        last_action=self.last_action_description
                    )
                    continue
                    
            except Exception as e:
                logger.error(f"ReAct 异常: {e}")
                traceback.print_exc()
                return self.action_space({})
        
        logger.warning(f"Step {self.current_step}: ReAct 耗尽步数，返回 do_nothing")
        self.stats["failed_actions"] += 1
        self.react_history = []
        return self.action_space({})

    def _sanitize_action(self, action: BaseAction, observation: BaseObservation) -> tuple[BaseAction, str]:
        """
        自动修正动作：
        1. 检查 Redispatch 是否超过爬坡率，如果超过则截断。
        2. 剔除对不可调度发电机的操作。
        """
        correction_details = []
        action_dict = action.as_dict()
        modified = False
        
        # 1. 修正 Redispatch
        if 'redispatch' in action_dict and action_dict['redispatch'] is not None:
            new_redispatch = []
            redispatch_data = action_dict['redispatch']
            
            # 转换为 list of (id, amount)
            items = []
            if isinstance(redispatch_data, dict):
                items = [(int(k), float(v)) for k, v in redispatch_data.items()]
            elif hasattr(redispatch_data, '__iter__') and not isinstance(redispatch_data, str):
                try:
                    # 处理 numpy array 或 list
                    # 注意: grid2op 的 redispatch 数组通常包含所有发电机，非零即为操作
                    for i, val in enumerate(redispatch_data):
                        if abs(val) > 1e-6:
                            items.append((i, float(val)))
                except:
                    pass

            for gen_id, amount in items:
                # 检查是否可调度
                if hasattr(observation, 'gen_redispatchable') and not observation.gen_redispatchable[gen_id]:
                    correction_details.append(f"忽略发电机 {gen_id} (不可调度)")
                    modified = True
                    continue
                
                # 检查爬坡率 (Ramp Rate)
                clamped_amount = amount
                if hasattr(observation, 'gen_max_ramp_up') and hasattr(observation, 'gen_max_ramp_down'):
                    max_up = float(observation.gen_max_ramp_up[gen_id])
                    max_down = float(observation.gen_max_ramp_down[gen_id])
                    
                    if amount > max_up:
                        clamped_amount = max_up
                        correction_details.append(f"Gen {gen_id} +{amount:.1f}->+{max_up:.1f} (爬坡限制)")
                        modified = True
                    elif amount < -max_down:
                        clamped_amount = -max_down
                        correction_details.append(f"Gen {gen_id} {amount:.1f}->-{max_down:.1f} (爬坡限制)")
                        modified = True
                
                new_redispatch.append((gen_id, clamped_amount))
            
            if modified:
                # 重新创建动作
                # 注意：我们不能直接修改 action 对象，最好创建一个新的
                new_action = self.action_space({})
                new_action.redispatch = new_redispatch
                # 复制其他属性 (如 set_line_status)
                if 'set_line_status' in action_dict and action_dict['set_line_status'] is not None:
                     # 这是一个简化处理，通常 parser 一次只生成一种类型的动作
                     # 如果混合了动作，这里需要更复杂的复制逻辑。
                     # 简单起见，假设 ActionParser 主要生成 redispatch 或 topology
                     pass 
                
                # 如果原动作还有 set_line_status，也得保留
                # Grid2Op 的 action 更新比较繁琐，这里采用 "叠加" 方式
                if 'set_line_status' in action_dict:
                     # 这是一个 tricky 的地方，直接修改 action 可能不生效或报错
                     # 最稳妥的是：如果仅仅修改了 redispatch，我们就在原 action 上覆盖
                     action.redispatch = new_redispatch
                     
                return action, "; ".join(correction_details)

        return action, ""

    def _simulate_action(self, action: BaseAction, observation: BaseObservation) -> tuple[bool, str]:
        """
        模拟验证 (修复了 BUG：空异常列表不再视为失败)
        """
        try:
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=0)
            
            # === BUG FIX START ===
            # 检查异常：Grid2Op 在成功时返回 'exception': [] (空列表)
            # 旧代码错误地认为 if exception is not None 就一定是有错
            exception = sim_info.get('exception', None)
            if exception is not None:
                if isinstance(exception, list):
                    if len(exception) > 0:
                        # 真正的异常发生
                        err_strs = [str(e) for e in exception]
                        return False, f"动作不合法: {'; '.join(err_strs)}"
                    # else: 空列表 = 成功！不要返回 False！
                else:
                    # 单个异常对象
                    return False, f"动作不合法: {str(exception)}"
            # === BUG FIX END ===

            # 潮流发散检查
            if np.any(np.isnan(sim_obs.rho)) or np.any(np.isinf(sim_obs.rho)):
                return False, "模拟失败：潮流发散 (NaN/Inf)"
            
            # 1. 安全检查 (熔断机制)
            max_rho_after = float(sim_obs.rho.max())
            if sim_done:
                return False, "动作导致游戏结束 (Game Over)"
            if max_rho_after > 1.5:
                return False, f"动作导致极度过载 ({max_rho_after:.2%})"
            
            # 2. 缓解策略 (Mitigation)
            max_rho_before = float(observation.rho.max())
            overflow_before = (observation.rho > 1.0).sum()
            overflow_after = (sim_obs.rho > 1.0).sum()
            
            if overflow_before > 0:
                # 只要有过载，任何改善都是好的
                if overflow_after < overflow_before:
                    return True, f"有效缓解: 过载线路 {overflow_before}->{overflow_after}"
                if max_rho_after < max_rho_before - 0.005: # 哪怕降低 0.5%
                    return True, f"有效缓解: Max Rho {max_rho_before:.2%}->{max_rho_after:.2%}"
                if max_rho_after >= max_rho_before:
                    return False, f"无效动作: 负载率未下降 ({max_rho_before:.2%} -> {max_rho_after:.2%})"
            
            else:
                # 原本安全
                if overflow_after > 0:
                    return False, f"动作导致新过载 ({max_rho_after:.2%})"
                if max_rho_after > max_rho_before + 0.10: # 放宽一点限制，允许适度上升
                    return False, "动作导致负载率大幅上升"
            
            return True, "验证通过"

        except Exception as e:
            return False, f"模拟过程出错: {str(e)}"

    def _is_danger(self, observation: BaseObservation) -> bool:
        """
        判断当前状态是否危险（需要专家介入）
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            如果存在过载则返回 True
        """
        overflow_count = int((observation.rho > 1.0).sum())
        return overflow_count > 0
    
    def _format_expert_insight(self, insight_report: Dict[str, Any]) -> str:
        """
        将专家洞察报告格式化为结构化文本（关键优化：精简Prompt）
        
        Args:
            insight_report: 专家洞察报告字典
            
        Returns:
            格式化的文本字符串（结构化格式）
        """
        if insight_report.get("status") == "SAFE":
            return ""
        
        lines = []
        lines.append("【Expert Insight】")
        
        critical_line = insight_report.get("critical_line", -1)
        critical_rho = insight_report.get("critical_rho", 0.0)
        lines.append(f"Critical: Line {critical_line} (rho={critical_rho:.2%})")
        lines.append("")
        
        solutions = insight_report.get("solutions", [])
        if not solutions:
            lines.append("No solutions found. Try redispatch().")
            return "\n".join(lines)
        
        # 结构化输出（关键优化：节省Token，同时突出成本信息）
        for i, solution in enumerate(solutions):
            sol_type = solution.get("type", "Unknown")
            score = solution.get("score", 0)
            sub_id = solution.get("substation_id", -1)
            side_effects = solution.get("side_effects", {})
            cost = solution.get("cost_estimation", {})
            
            # 紧凑格式：[ID, Type, Sub, Score, SideEffects, Cost]
            side_effect_str = ""
            if side_effects.get("worsened_lines"):
                worsened = side_effects["worsened_lines"]
                side_effect_str = f", Worsens: {worsened[:3]}"  # 最多显示3个
            if side_effects.get("risk_level") == "high":
                side_effect_str += ", Risk: HIGH"
            
            # === 策略四：突出成本信息 ===
            cost_val = cost.get('estimated_cost', 0.0) if cost else 0.0
            cost_str = f", Cost: {cost_val:.2f}" if cost_val > 0 else ", Cost: 0.00"
            
            if sol_type == "Topology Action":
                lines.append(f"[{i}] Topo Sub{sub_id} Score{score}/4{side_effect_str}{cost_str}")
            elif sol_type == "Redispatch":
                gen_up = solution.get("gen_up", [])
                gen_down = solution.get("gen_down", [])
                gen_str = ""
                if gen_down:
                    gen_str += f"Down:{gen_down[:2]}"
                if gen_up:
                    gen_str += f" Up:{gen_up[:2]}" if gen_str else f"Up:{gen_up[:2]}"
                lines.append(f"[{i}] Redispatch {gen_str}{cost_str}")
        
        lines.append("")
        lines.append("Note: Score 4=perfect, 3=good. Use execute_expert_solution(i) for Topo actions.")
        lines.append("⚠️ 成本优化：在安全的前提下（Score >= 3），优先选择成本最低的方案（Cost 值越小越好）。")
        lines.append("Consider side effects and cost when choosing between solutions.")
        
        return "\n".join(lines)
    
    def _check_trust_pass_through(
        self,
        insight_report: Dict[str, Any],
        observation: BaseObservation
    ) -> Optional[BaseAction]:
        """
        策略一：信任直通车
        检查 Expert 是否给出了 Score=4 的完美方案，如果是则直接执行，跳过 LLM
        
        Args:
            insight_report: 专家洞察报告
            observation: 当前观测
            
        Returns:
            如果应该信任则返回动作对象，否则返回 None
        """
        if not insight_report or insight_report.get("status") != "DANGER":
            return None
        
        solutions = insight_report.get("solutions", [])
        if not solutions:
            return None
        
        # 查找 Score=4 的完美方案
        for solution in solutions:
            score = solution.get("score", 0)
            if score == 4:
                # 额外检查：确保没有高风险副作用
                side_effects = solution.get("side_effects", {})
                risk_level = side_effects.get("risk_level", "low")
                
                if risk_level != "high":
                    # 获取动作对象
                    action_obj = solution.get("action_object")
                    if action_obj is not None:
                        # 进行基本的合法性检查（但不进行模拟验证，因为 Expert 已经验证过了）
                        try:
                            # 快速检查：确保动作对象有效
                            action_dict = action_obj.as_dict()
                            if action_dict:
                                logger.info(f"信任直通车：执行 Expert Score=4 方案（变电站 {solution.get('substation_id', 'N/A')}）")
                                return action_obj
                        except Exception as e:
                            logger.warning(f"信任直通车：动作对象检查失败: {e}")
                            # 如果检查失败，降级到 LLM 处理
                            return None
        
        return None
    
    def _should_trust_expert_action(
        self, 
        action_text: str, 
        insight_report: Optional[Dict[str, Any]]
    ) -> bool:
        """
        判断是否应该信任Expert方案（信任直通车：Score=4时跳过二次模拟）
        注意：这个方法现在主要用于 LLM 选择了 execute_expert_solution 的情况
        
        Args:
            action_text: LLM生成的动作文本
            insight_report: 专家洞察报告
            
        Returns:
            如果应该信任则返回True
        """
        if not insight_report:
            return False
        
        # 检查是否是execute_expert_solution
        expert_match = re.search(r'execute_expert_solution\s*\(\s*(\d+)\s*\)', action_text, re.IGNORECASE)
        if not expert_match:
            return False
        
        solution_idx = int(expert_match.group(1))
        solutions = insight_report.get("solutions", [])
        
        if solution_idx < 0 or solution_idx >= len(solutions):
            return False
        
        solution = solutions[solution_idx]
        score = solution.get("score", 0)
        
        # 只有Score=4的完美方案才信任
        if score == 4:
            # 额外检查：确保没有高风险副作用
            side_effects = solution.get("side_effects", {})
            if side_effects.get("risk_level") != "high":
                return True
        
        return False
    
    def _describe_action_for_history(self, action: BaseAction) -> str:
        """
        描述动作用于历史记录（防震荡）
        
        Args:
            action: Grid2Op 动作对象
            
        Returns:
            动作描述字符串
        """
        action_dict = action.as_dict()
        
        if 'set_bus' in action_dict and action_dict['set_bus'] is not None:
            set_bus_data = action_dict['set_bus']
            if isinstance(set_bus_data, dict) and 'substations_id' in set_bus_data:
                subs_data = set_bus_data['substations_id']
                if isinstance(subs_data, list) and len(subs_data) > 0:
                    sub_id, _ = subs_data[0]
                    return f"Topology change on substation {sub_id}"
        
        if 'set_line_status' in action_dict and action_dict['set_line_status'] is not None:
            line_status_data = action_dict['set_line_status']
            if isinstance(line_status_data, list) and len(line_status_data) > 0:
                line_id, status = line_status_data[0]
                status_str = "connected" if status > 0 else "disconnected"
                return f"Line {line_id} {status_str}"
        
        if 'redispatch' in action_dict and action_dict['redispatch'] is not None:
            redispatch_data = action_dict['redispatch']
            if isinstance(redispatch_data, list) and len(redispatch_data) > 0:
                gen_ids = [str(g) for g, _ in redispatch_data[:3]]
                return f"Redispatch on generators {', '.join(gen_ids)}"
        
        return "Unknown action"
    
    def _extract_env_info(self, observation):
        # 保持原样，省略以节省空间
        env_info = {
            "n_gen": int(observation.n_gen),
            "n_line": int(observation.n_line),
            "n_sub": int(observation.n_sub),
        }
        # 简单提取发电机 Ramp 信息
        gen_info = []
        if hasattr(observation, 'gen_max_ramp_up'):
            for i in range(observation.n_gen):
                gen_info.append({
                    "gen_id": i,
                    "max_ramp_up": float(observation.gen_max_ramp_up[i]),
                    "max_ramp_down": float(observation.gen_max_ramp_down[i]),
                    "redispatchable": bool(observation.gen_redispatchable[i]) if hasattr(observation, 'gen_redispatchable') else True
                })
        env_info["generators"] = gen_info
        return env_info