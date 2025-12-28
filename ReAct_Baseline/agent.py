# -*- coding: utf-8 -*-
"""
ReAct Agent 核心实现
基于 ReAct (Reasoning + Acting) 范式的电网调度智能体
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
import logging

# 添加项目根目录到路径（以便导入 ADA 的 utils）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加当前目录到路径，以便直接运行时可以导入同目录的模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入本地模块（支持直接运行和作为模块导入）
try:
    # 如果作为模块导入，使用相对导入
    from .formatters import ObservationFormatter
    from .parser import ActionParser
    from .prompts import PromptManager
except ImportError:
    # 如果直接运行，使用绝对导入
    try:
        from ReAct_Baseline.formatters import ObservationFormatter
        from ReAct_Baseline.parser import ActionParser
        from ReAct_Baseline.prompts import PromptManager
    except ImportError:
        from formatters import ObservationFormatter
        from parser import ActionParser
        from prompts import PromptManager

# 导入 ADA 的工具（LLM 和日志）
try:
    # 优先从 ADA 根模块导入（更简洁）
    from ADA import OpenAIChat, get_logger
except ImportError:
    # 如果失败，尝试从 ADA.utils 导入
    try:
        from ADA.utils.llm import OpenAIChat
        from ADA.utils.logger import get_logger
    except ImportError:
        # 如果还是失败，尝试从 ReAct 目录导入
        try:
            from ReAct.utils.llm import OpenAIChat
            from ReAct.utils.logger import get_logger
        except ImportError:
            # 如果还是无法导入，使用标准库
            import logging
            OpenAIChat = None
            def get_logger(name):
                return logging.getLogger(name)

logger = get_logger("ReActAgent")


class ReActAgent(BaseAgent):
    """
    ReAct Baseline Agent
    
    基于 ReAct (Reasoning + Acting) 范式的电网调度智能体。
    不依赖数学优化求解器，完全依赖 LLM 的推理能力。
    
    工作流程：
    1. Observe: 获取环境观测并转换为文本
    2. Think: LLM 分析当前状态
    3. Act: LLM 生成文本动作指令
    4. Execute: 解析动作为 Grid2Op Action，在模拟环境中验证
    5. 如果安全: 输出该动作
    6. 如果不安全/非法: 生成错误反馈，返回第 2 步，让 LLM 重新思考
    
    参考设计文档: ReAct/readme.md
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        max_react_steps: int = 3,
        name: str = "ReActAgent",
        rho_danger: float = 1.0,
        **kwargs
    ):
        """
        初始化 ReAct Agent
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间（未使用，但 Grid2Op 要求）
            llm_client: LLM 客户端（OpenAIChat 实例）
            max_react_steps: ReAct 循环的最大重试次数
            name: 智能体名称
            rho_danger: 危险阈值，当最大负载率超过此值时才调用 LLM（默认 1.0，即有过载时）
            **kwargs: 其他参数
        """
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_react_steps = max_react_steps
        self.rho_danger = rho_danger  # 启发式策略：只有超过此阈值才调用 LLM
        
        # 初始化组件
        self.formatter = ObservationFormatter()
        self.parser = ActionParser()
        self.prompt_manager = PromptManager()
        
        # 内部状态
        self.current_step = 0
        self.react_history = []  # ReAct 循环历史
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "react_loops": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "do_nothing_count": 0,
            "heuristic_do_nothing_count": 0,  # 启发式策略导致的 do_nothing
        }
        
        logger.info(f"ReActAgent '{name}' 初始化完成")
        logger.info(f"  Max ReAct Steps: {max_react_steps}")
        logger.info(f"  Rho Danger Threshold: {rho_danger} (启发式策略：超过此值才调用 LLM)")
    
    def reset(self, observation: BaseObservation):
        """
        重置智能体状态（在每个 episode 开始时调用）
        
        Args:
            observation: 初始观测
        """
        self.current_step = 0
        self.react_history = []
        
        # 重置统计信息
        self.stats = {
            "total_steps": 0,
            "react_loops": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "do_nothing_count": 0,
            "heuristic_do_nothing_count": 0,
        }
        
        logger.info(f"ReActAgent 已重置 (episode start)")
    
    def act(
        self,
        observation: BaseObservation,
        reward: float = 0.0,
        done: bool = False
    ) -> BaseAction:
        """
        智能体的主要方法：根据观测返回动作
        
        启发式策略（类似 OptimCVXPY 和 ExpertAgent）：
        - 正常时（无过载或负载率低）：直接返回 do_nothing，不调用 LLM
        - 超限后（有过载或负载率高）：调用 LLM 进行 ReAct 循环
        
        实现 ReAct 循环：
        1. 启发式判断：检查是否有过载
        2. 转换观测为文本
        3. 构建 Prompt（包含历史上下文）
        4. ReAct 循环（最多 max_react_steps 次）：
           a. 调用 LLM 生成 Thought 和 Action
           b. 解析动作为 Grid2Op Action
           c. 在模拟环境中验证动作
           d. 如果安全: 返回动作
           e. 如果不安全/非法: 生成反馈，继续循环
        
        Args:
            observation: 当前观测
            reward: 上一步的奖励（未使用）
            done: 是否结束（未使用）
        
        Returns:
            Grid2Op Action
        """
        self.current_step += 1
        self.stats["total_steps"] += 1
        
        # 0. 启发式策略：检查是否有过载或负载率过高
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        # 如果没有过载且最大负载率低于危险阈值，直接返回 do_nothing
        if overflow_count == 0 and max_rho <= self.rho_danger:
            logger.debug(f"Step {self.current_step}: 启发式策略 - 无过载且负载率安全 (max_rho={max_rho:.2%}), 直接返回 do_nothing")
            self.stats["heuristic_do_nothing_count"] += 1
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 1. 转换观测为文本
        obs_text = self.formatter.format(observation)
        
        # 2. 构建初始 Prompt（包含历史上下文）
        # 第一次调用时包含 few-shot examples，后续调用只包含历史
        is_first_call = len(self.react_history) == 0
        history = self.prompt_manager.build(
            observation_text=obs_text,
            history=self.react_history if not is_first_call else None
        )
        
        # 3. ReAct 循环（最多 max_react_steps 次）
        for react_step in range(self.max_react_steps):
            self.stats["react_loops"] += 1
            
            try:
                # 3a. 调用 LLM 生成响应
                logger.debug(f"Step {self.current_step}, ReAct Loop {react_step + 1}: 调用 LLM")
                
                # 从 history 中提取 system_prompt、prompt 和 llm_history
                # history 格式: [{"role": "system", "content": ...}, ..., {"role": "user", "content": ...}]
                system_prompt = None
                prompt = ""
                llm_history = []
                
                # 遍历消息，分离 system、user 和 assistant
                for msg in history:
                    if msg["role"] == "system":
                        system_prompt = msg["content"]
                    elif msg["role"] == "assistant":
                        llm_history.append(msg)
                    elif msg["role"] == "user":
                        llm_history.append(msg)
                
                # 最后一个 user 消息作为 prompt，其他作为 history
                if llm_history and llm_history[-1]["role"] == "user":
                    prompt = llm_history[-1]["content"]
                    llm_history = llm_history[:-1]  # 移除最后一个 user 消息
                elif llm_history:
                    # 如果没有 user 消息，使用最后一个消息的内容
                    prompt = llm_history[-1].get("content", "")
                    llm_history = llm_history[:-1]
                
                llm_response = self.llm_client.chat(
                    prompt=prompt,
                    history=llm_history if llm_history else None,
                    system_prompt=system_prompt
                )
                
                logger.debug(f"LLM 响应: {llm_response[:200]}...")
                
                # 3b. 从响应中提取动作指令
                action_text = self.parser.extract_action_from_response(llm_response)
                
                if not action_text:
                    logger.warning(f"无法从 LLM 响应中提取动作，返回 do_nothing")
                    self.stats["failed_actions"] += 1
                    return self.action_space({})
                
                logger.debug(f"提取的动作文本: {action_text}")
                
                # 3c. 解析动作为 Grid2Op Action
                try:
                    grid_action = self.parser.parse(action_text, self.action_space)
                except ValueError as e:
                    # 格式错误，加入历史，重试
                    error_msg = f"动作格式错误: {str(e)}。请检查参数格式是否正确。"
                    logger.warning(f"动作解析失败: {error_msg}")
                    
                    # 添加 LLM 的响应到历史（作为 assistant 消息）
                    self.react_history.append({
                        "role": "assistant",
                        "content": llm_response
                    })
                    
                    # 添加错误反馈到历史
                    self.react_history = self.prompt_manager.add_observation_feedback(
                        self.react_history,
                        error_msg
                    )
                    
                    # 更新 history 用于下次 LLM 调用
                    history = self.prompt_manager.build(
                        observation_text=obs_text,
                        history=self.react_history
                    )
                    continue
                
                # 3d. 在模拟环境中验证动作
                logger.debug(f"模拟验证动作: {action_text}")
                is_safe, reason = self._simulate_action(grid_action, observation)
                
                if is_safe:
                    # 动作安全，返回
                    logger.info(f"Step {self.current_step}: 动作安全，执行 {action_text}")
                    self.stats["successful_actions"] += 1
                    
                    # 检查是否为 do_nothing
                    if action_text.strip().lower().startswith("do_nothing"):
                        self.stats["do_nothing_count"] += 1
                    
                    # 清空历史（为下一步准备）
                    self.react_history = []
                    return grid_action
                else:
                    # 动作不安全，加入历史，让 LLM 重新思考
                    error_msg = f"模拟显示该动作会导致危险: {reason}。请尝试其他策略。"
                    logger.warning(f"动作不安全: {error_msg}")
                    
                    # 添加 LLM 的响应到历史（作为 assistant 消息）
                    self.react_history.append({
                        "role": "assistant",
                        "content": llm_response
                    })
                    
                    # 添加错误反馈到历史
                    self.react_history = self.prompt_manager.add_observation_feedback(
                        self.react_history,
                        error_msg
                    )
                    
                    # 更新 history 用于下次 LLM 调用
                    history = self.prompt_manager.build(
                        observation_text=obs_text,
                        history=self.react_history
                    )
                    continue
                    
            except Exception as e:
                # LLM 调用或其他异常
                logger.error(f"ReAct 循环异常: {e}")
                import traceback
                traceback.print_exc()
                self.stats["failed_actions"] += 1
                # 如果异常，返回 do_nothing
                return self.action_space({})
        
        # 如果超过步数仍未找到解，返回 do_nothing
        logger.warning(f"Step {self.current_step}: ReAct 循环达到最大步数，返回 do_nothing")
        self.stats["failed_actions"] += 1
        self.react_history = []  # 清空历史
        return self.action_space({})
    
    def _validate_action_before_simulate(
        self,
        action: BaseAction,
        observation: BaseObservation
    ) -> tuple[bool, str]:
        """
        在模拟前验证动作的合法性
        
        Args:
            action: Grid2Op Action 对象
            observation: 当前观测
        
        Returns:
            (is_valid, reason) 元组
        """
        # 检查 redispatch 动作
        if hasattr(action, 'redispatch') and action.redispatch is not None:
            # 安全地检查 redispatch 是否非空
            try:
                redispatch_list = list(action.redispatch) if hasattr(action.redispatch, '__iter__') else []
            except (TypeError, ValueError):
                redispatch_list = []
            
            if len(redispatch_list) > 0:
                for gen_id, amount in redispatch_list:
                    # 检查发电机 ID 是否有效
                    if gen_id < 0 or gen_id >= observation.n_gen:
                        return False, f"发电机 ID {gen_id} 无效（有效范围: 0-{observation.n_gen-1}）"
                    
                    # 检查发电机是否可调度
                    if hasattr(observation, 'gen_redispatchable') and not observation.gen_redispatchable[gen_id]:
                        return False, f"发电机 {gen_id} 不可调度（不是可调度发电机）"
                    
                    # 检查再调度量是否在允许范围内
                    if hasattr(observation, 'gen_p') and hasattr(observation, 'gen_pmax') and hasattr(observation, 'gen_pmin'):
                        current_p = float(observation.gen_p[gen_id])
                        max_p = float(observation.gen_pmax[gen_id])
                        min_p = float(observation.gen_pmin[gen_id])
                        new_p = current_p + amount
                        
                        if new_p > max_p:
                            return False, f"再调度后发电机 {gen_id} 出力 {new_p:.2f} MW 超过最大值 {max_p:.2f} MW（当前: {current_p:.2f} MW，调整量: {amount:.2f} MW）"
                        if new_p < min_p:
                            return False, f"再调度后发电机 {gen_id} 出力 {new_p:.2f} MW 低于最小值 {min_p:.2f} MW（当前: {current_p:.2f} MW，调整量: {amount:.2f} MW）"
        
        # 检查 set_line_status 动作
        if hasattr(action, 'set_line_status') and action.set_line_status is not None:
            # 安全地检查 set_line_status 是否非空
            try:
                line_status_list = list(action.set_line_status) if hasattr(action.set_line_status, '__iter__') else []
            except (TypeError, ValueError):
                line_status_list = []
            
            if len(line_status_list) > 0:
                for line_id, status in line_status_list:
                    # 检查线路 ID 是否有效
                    if line_id < 0 or line_id >= observation.n_line:
                        return False, f"线路 ID {line_id} 无效（有效范围: 0-{observation.n_line-1}）"
                    
                    # 检查线路冷却时间
                    if hasattr(observation, 'time_before_cooldown_line'):
                        cooldown = int(observation.time_before_cooldown_line[line_id])
                        if cooldown > 0:
                            return False, f"线路 {line_id} 正在冷却中（剩余 {cooldown} 步），无法改变状态"
                    
                    # 检查是否尝试对已断开线路执行断开操作（或对已连接线路执行连接操作）
                    if hasattr(observation, 'line_status'):
                        is_connected = bool(observation.line_status[line_id])
                        if status == 1 and is_connected:
                            return False, f"线路 {line_id} 已经连接，无需再次连接"
                        if status == -1 and not is_connected:
                            return False, f"线路 {line_id} 已经断开，无需再次断开"
        
        return True, ""
    
    def _simulate_action(
        self,
        action: BaseAction,
        observation: BaseObservation
    ) -> tuple[bool, str]:
        """
        在模拟环境中验证动作是否安全
        
        Args:
            action: Grid2Op Action 对象
            observation: 当前观测
        
        Returns:
            (is_safe, reason) 元组
            - is_safe: 是否安全
            - reason: 如果不安全，说明原因
        """
        # 首先验证动作合法性
        is_valid, validation_error = self._validate_action_before_simulate(action, observation)
        if not is_valid:
            return False, validation_error
        
        try:
            # 使用 observation.simulate() 模拟动作
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(
                action,
                time_step=0
            )
            
            # 检查是否有异常（更详细的错误信息）
            exception = sim_info.get('exception', None)
            if exception is not None:
                # 如果异常是列表，尝试提取更多信息
                if isinstance(exception, list):
                    if len(exception) > 0:
                        # 尝试获取异常类型和消息
                        error_msg = f"模拟异常: {exception}"
                    else:
                        # 空列表可能表示动作不合法
                        error_msg = "动作不合法：可能是参数超出范围、违反物理约束或操作不允许（例如对冷却中的线路操作）"
                else:
                    error_msg = f"模拟异常: {exception}"
                return False, error_msg
            
            # 检查是否导致游戏结束
            if sim_done:
                return False, "动作导致游戏结束（系统崩溃）"
            
            # 检查是否导致过载
            max_rho_after = float(sim_obs.rho.max())
            overflow_count_after = int((sim_obs.rho > 1.0).sum())
            
            # 获取当前状态
            max_rho_before = float(observation.rho.max())
            overflow_count_before = int((observation.rho > 1.0).sum())
            
            # 判断是否安全
            # 1. 如果动作后仍有过载，不安全
            if overflow_count_after > 0:
                return False, f"动作后仍有 {overflow_count_after} 条线路过载（最大负载率: {max_rho_after:.2%}）"
            
            # 2. 如果动作后最大负载率显著增加（超过 5%），视为不安全
            if max_rho_after > max_rho_before + 0.05:
                return False, f"动作后最大负载率从 {max_rho_before:.2%} 增加到 {max_rho_after:.2%}，风险增加"
            
            # 3. 如果动作后最大负载率仍然很高（> 95%），视为不安全
            if max_rho_after > 0.95:
                return False, f"动作后最大负载率 {max_rho_after:.2%} 仍然很高（> 95%），接近过载"
            
            # 动作安全
            return True, f"动作安全（最大负载率: {max_rho_after:.2%}）"
            
        except Exception as e:
            # 模拟失败，视为不安全
            error_msg = str(e)
            # 提供更友好的错误信息
            if "illegal" in error_msg.lower() or "invalid" in error_msg.lower():
                return False, f"动作不合法: {error_msg}"
            return False, f"模拟失败: {error_msg}"
    
    def load(self, path: Optional[str]):
        """加载智能体状态（如果需要）"""
        if path is None:
            return
        # TODO: 实现加载逻辑
        logger.info(f"从 {path} 加载智能体状态")
    
    def save(self, path: Optional[str]):
        """保存智能体状态（如果需要）"""
        if path is None:
            return
        # TODO: 实现保存逻辑
        logger.info(f"保存智能体状态到 {path}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_steps = max(1, self.stats["total_steps"])
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_actions"] / total_steps
            ),
            "avg_react_loops_per_step": (
                self.stats["react_loops"] / total_steps
            ),
            "heuristic_do_nothing_rate": (
                self.stats["heuristic_do_nothing_count"] / total_steps
            ),
        }

