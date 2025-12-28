# -*- coding: utf-8 -*-
"""
ReAct Agent 核心实现 (Fix: Logic Bug & Auto-Correction)
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import numpy as np # 确保导入 numpy

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

try:
    from .formatters import ObservationFormatter
    from .parser import ActionParser
    from .prompts import PromptManager
except ImportError:
    from ReAct_Baseline.formatters import ObservationFormatter
    from ReAct_Baseline.parser import ActionParser
    from ReAct_Baseline.prompts import PromptManager

# 尝试导入 ADA 的工具
try:
    from ADA import OpenAIChat, get_logger
except ImportError:
    try:
        from ADA.utils.llm import OpenAIChat
        from ADA.utils.logger import get_logger
    except ImportError:
        import logging
        OpenAIChat = None
        def get_logger(name): return logging.getLogger(name)

logger = get_logger("ReActAgent")


class ReActAgent(BaseAgent):
    """
    ReAct Baseline Agent (Enhanced)
    修复了模拟验证逻辑 BUG，并增加了自动修正（Auto-Correction）功能。
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        max_react_steps: int = 3,
        name: str = "ReActAgent",
        rho_danger: float = 0.92, # 降低默认阈值以实现预防性调度
        **kwargs
    ):
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_react_steps = max_react_steps
        self.rho_danger = rho_danger
        
        self.formatter = ObservationFormatter()
        self.parser = ActionParser()
        self.prompt_manager = PromptManager()
        
        self.current_step = 0
        self.react_history = []
        self.env_info = None
        
        self.stats = {
            "total_steps": 0,
            "react_loops": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "do_nothing_count": 0,
            "sanitized_count": 0, # 统计自动修正次数
        }
        
        logger.info(f"ReActAgent '{name}' 初始化完成 (v2.0 Fix)")
    
    def reset(self, observation: BaseObservation):
        self.current_step = 0
        self.react_history = []
        if self.env_info is None:
            self.env_info = self._extract_env_info(observation)
            self.prompt_manager.set_env_info(self.env_info)
        self.stats = {k: 0 for k in self.stats}
        logger.info(f"ReActAgent 已重置")
    
    def act(self, observation: BaseObservation, reward: float = 0.0, done: bool = False) -> BaseAction:
        self.current_step += 1
        self.stats["total_steps"] += 1
        
        # 0. 启发式策略
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        if overflow_count == 0 and max_rho <= self.rho_danger:
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 1. 准备 ReAct
        obs_text = self.formatter.format(observation)
        is_first_call = len(self.react_history) == 0
        history = self.prompt_manager.build(obs_text, self.react_history if not is_first_call else None)
        
        # 2. ReAct 循环
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
                    raw_action = self.parser.parse(action_text, self.action_space)
                    
                    # === 新增：动作自动修正 (Auto-Correction) ===
                    # 自动处理爬坡率限制，避免因为数值超限导致动作非法
                    sanitized_action, correction_msg = self._sanitize_action(raw_action, observation)
                    if correction_msg:
                        logger.info(f"Step {self.current_step}: 动作已自动修正 -> {correction_msg}")
                        self.stats["sanitized_count"] += 1
                    
                    # 模拟验证
                    is_safe, reason = self._simulate_action(sanitized_action, observation)
                    
                    if is_safe:
                        logger.info(f"Step {self.current_step}: 动作安全 (Step {react_step+1})")
                        self.stats["successful_actions"] += 1
                        self.react_history = [] 
                        return sanitized_action
                    else:
                        # 反馈错误
                        feedback_msg = f"模拟警告: {reason}"
                        if correction_msg:
                            feedback_msg += f" (注: 原始动作已被修正: {correction_msg})"
                            
                        self.react_history.append({"role": "assistant", "content": llm_response})
                        self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, feedback_msg)
                        history = self.prompt_manager.build(obs_text, self.react_history)
                        continue
                        
                except ValueError as e:
                    self.react_history.append({"role": "assistant", "content": llm_response})
                    self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, f"格式错误: {e}")
                    history = self.prompt_manager.build(obs_text, self.react_history)
                    continue
                    
            except Exception as e:
                logger.error(f"ReAct 异常: {e}")
                import traceback
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