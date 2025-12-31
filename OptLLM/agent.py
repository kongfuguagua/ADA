# -*- coding: utf-8 -*-
"""
OptAgent (Optimization-Augmented ReAct Agent) - 重构版
专注于优化器参数配置，移除手动动作解析的混合模式
"""

from typing import Optional, List, Dict, Any
import logging
import numpy as np
import json

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 使用标准相对导入（作为包的一部分）
from .summarizer import StateSummarizer
from .prompts import PromptManager
from .parser import ActionParser
from .analysis import OptimizationService

_HAS_OPTIMIZATION_SERVICE = True

from utils import OpenAIChat, get_logger

logger = get_logger("OptAgent")


class OptAgent(BaseAgent):
    """
    OptAgent (Optimization-Augmented ReAct Agent) - 重构版
    
    专注于让 LLM 配置优化器参数，优化器负责计算具体动作。
    不再支持手动动作解析，简化逻辑，提高效率。
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        env=None,
        max_retry_steps: int = 2,  # 减少重试次数，提高效率
        name: str = "OptAgent",
        rho_safe: float = 0.85,  # 安全阈值
        **kwargs
    ):
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_retry_steps = max_retry_steps
        self.rho_safe = rho_safe
        
        # 初始化组件
        self.summarizer = StateSummarizer() if StateSummarizer else None
        self.prompt_manager = PromptManager() if PromptManager else None
        self.parser = ActionParser() if ActionParser else None
        
        # 初始化优化服务
        self.opt_service = None
        if not _HAS_OPTIMIZATION_SERVICE:
            logger.warning("OptimizationService 导入失败，无法使用优化器")
        elif env is None:
            logger.warning("env 参数为 None，无法初始化 OptimizationService")
        else:
            try:
                self.opt_service = OptimizationService(action_space, env, **kwargs)
                logger.info("OptimizationService 初始化成功")
            except Exception as e:
                logger.error(f"OptimizationService 初始化失败: {e}", exc_info=True)
                self.opt_service = None
        
        # 状态跟踪
        self.current_step = 0
        self.last_feedback = None
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "optimizer_calls": 0,
            "optimizer_success": 0,
            "optimizer_infeasible": 0,
            "simulation_failures": 0,
            "do_nothing_count": 0,
        }
        
        logger.info(f"OptAgent '{name}' 初始化完成（重构版）")
    
    def reset(self, observation: BaseObservation):
        """重置智能体状态"""
        self.current_step = 0
        self.last_feedback = None
        self.stats = {k: 0 for k in self.stats}
        
        # 重置优化服务
        if self.opt_service is not None:
            try:
                self.opt_service.reset(observation)
            except Exception as e:
                logger.warning(f"优化服务重置失败: {e}")
        
        logger.info("OptAgent 已重置")
    
    def act(
        self, 
        observation: BaseObservation, 
        reward: float = 0.0, 
        done: bool = False
    ) -> BaseAction:
        """
        主决策函数
        
        专注于优化器参数配置，不再支持手动动作解析
        """
        self.current_step += 1
        self.stats["total_steps"] += 1
        
        # 1. 快速启发式检查（无需 LLM）
        max_rho = float(observation.rho.max())
        if max_rho < self.rho_safe:
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 2. 检查优化器是否可用
        if self.opt_service is None:
            logger.warning("优化器不可用，返回空动作")
            return self.action_space({})
        
        # 3. 生成状态摘要（大幅减少 Token 消耗）
        state_summary_dict = self.summarizer.summarize(observation, self.last_feedback)
        state_summary_text = self.summarizer.format_summary(state_summary_dict)
        
        # 4. LLM 决策循环（限制重试次数）
        for retry_step in range(self.max_retry_steps):
            try:
                # 4.1 构建 Prompt（极简格式）
                messages = self.prompt_manager.build(state_summary_text, self.last_feedback)
                
                # 4.2 LLM 推理
                system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                llm_history = [msg for msg in messages if msg["role"] not in ["system", "user"]]
                
                llm_response = self.llm_client.chat(
                    prompt=user_prompt,
                    history=llm_history,
                    system_prompt=system_prompt
                )
                
                # 4.3 解析优化器配置（优先 JSON 模式）
                config = self.parser.extract_optimization_config_json(llm_response)
                if config is None:
                    logger.warning(f"无法解析 LLM 响应，尝试文本解析: {llm_response[:200]}")
                    config = self.parser.extract_optimization_config(llm_response)
                
                if config is None:
                    logger.error(f"无法解析优化器配置，返回空动作")
                    self.last_feedback = "无法解析配置，请输出有效的 JSON 格式"
                    continue
                
                # 4.4 调用优化器
                self.stats["optimizer_calls"] += 1
                result = self.opt_service.solve_with_config(observation, config)
                
                if result["status"] == "SUCCESS":
                    # 4.5 安全模拟检查（AC/DC 误差验证）
                    action = result["action"]
                    is_safe, reason, error_type = self._simulate_action(action, observation)
                    
                    if is_safe:
                        logger.info(
                            f"Step {self.current_step}: 优化成功 "
                            f"(策略: {config.get('strategy_description', 'N/A')}, "
                            f"重试: {retry_step + 1})"
                        )
                        self.stats["optimizer_success"] += 1
                        self.last_feedback = None  # 成功，清除反馈
                        return action
                    else:
                        # 模拟失败，根据错误类型生成智能反馈
                        self.stats["simulation_failures"] += 1
                        
                        if error_type == "DIVERGENCE":
                            # 严重错误：潮流发散/电网解列
                            self.last_feedback = (
                                f"模拟失败: {reason}。\n"
                                f"【系统危急】检测到电压崩溃或电网解列。常规调整已失效。\n"
                                f"请立即切换到 SURVIVAL (生存模式)，不惜一切代价消除过载。"
                            )
                        else:
                            # 普通 AC/DC 误差（如过载）
                            max_rho = float(observation.rho.max())
                            if max_rho > 1.1:
                                # 严重过载，建议切换到更激进的模式
                                self.last_feedback = (
                                    f"优化成功但模拟不安全: {reason}。\n"
                                    f"当前最大负载率 {max_rho:.2%} 过高。建议切换到 EMERGENCY 或 SURVIVAL 模式。"
                                )
                            else:
                                # 普通越限
                                self.last_feedback = (
                                    f"优化成功但模拟不安全: {reason}。\n"
                                    f"建议：切换到 CAUTIOUS 模式以留出更多安全裕度。"
                                )
                        logger.warning(f"Step {self.current_step}: {self.last_feedback}")
                        continue
                
                elif result["status"] == "INFEASIBLE":
                    # 优化器无解
                    self.stats["optimizer_infeasible"] += 1
                    self.last_feedback = (
                        f"优化无解: {result.get('reason', '未知原因')}。"
                        f"请放宽 margin_th_limit 或降低 penalty_curtailment"
                    )
                    logger.warning(f"Step {self.current_step}: {self.last_feedback}")
                    continue
                
                else:
                    # 其他错误
                    self.last_feedback = f"优化器错误: {result.get('reason', '未知错误')}"
                    logger.error(f"Step {self.current_step}: {self.last_feedback}")
                    continue
                    
            except Exception as e:
                logger.error(f"Step {self.current_step}: 决策过程异常: {e}", exc_info=True)
                self.last_feedback = f"系统异常: {str(e)}"
                continue
        
        # 5. 所有重试都失败，返回空动作
        logger.warning(f"Step {self.current_step}: 所有重试都失败，返回空动作")
        return self.action_space({})
    
    def _simulate_action(
        self, 
        action: BaseAction, 
        observation: BaseObservation
    ) -> tuple[bool, str, str]:
        """
        模拟验证动作（检查 AC/DC 误差）
        
        Returns:
            (is_safe, reason, error_type)
            error_type: "NONE", "VIOLATION", "DIVERGENCE"
        """
        try:
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=0)
            
            # 检查异常
            exception = sim_info.get('exception', None)
            if exception is not None:
                if isinstance(exception, list):
                    if len(exception) > 0:
                        err_strs = [str(e) for e in exception]
                        err_msg = '; '.join(err_strs)
                        # 检查是否为严重错误
                        if any(keyword in err_msg for keyword in ["Divergence", "non connected", "SolverFactor", "unbounded"]):
                            return False, f"动作不合法: {err_msg}", "DIVERGENCE"
                        return False, f"动作不合法: {err_msg}", "VIOLATION"
                else:
                    err_msg = str(exception)
                    if any(keyword in err_msg for keyword in ["Divergence", "non connected", "SolverFactor", "unbounded"]):
                        return False, f"动作不合法: {err_msg}", "DIVERGENCE"
                    return False, f"动作不合法: {err_msg}", "VIOLATION"
            
            # 潮流发散检查
            if np.any(np.isnan(sim_obs.rho)) or np.any(np.isinf(sim_obs.rho)):
                return False, "模拟失败：潮流发散 (NaN/Inf)", "DIVERGENCE"
            
            # 安全检查
            max_rho_after = float(sim_obs.rho.max())
            if sim_done:
                return False, "动作导致游戏结束", "VIOLATION"
            if max_rho_after > 1.5:
                return False, f"动作导致极度过载 ({max_rho_after:.2%})", "VIOLATION"
            
            # 缓解策略检查
            max_rho_before = float(observation.rho.max())
            overflow_before = (observation.rho > 1.0).sum()
            overflow_after = (sim_obs.rho > 1.0).sum()
            
            if overflow_before > 0:
                # 原本过载，检查是否改善
                if overflow_after < overflow_before:
                    return True, f"有效缓解: 过载线路 {overflow_before}->{overflow_after}", "NONE"
                if max_rho_after < max_rho_before - 0.005:
                    return True, f"有效缓解: Max Rho {max_rho_before:.2%}->{max_rho_after:.2%}", "NONE"
                if max_rho_after >= max_rho_before:
                    return False, f"无效动作: 负载率未下降 ({max_rho_before:.2%} -> {max_rho_after:.2%})", "VIOLATION"
            else:
                # 原本安全
                if overflow_after > 0:
                    return False, f"动作导致新过载 ({max_rho_after:.2%})", "VIOLATION"
                if max_rho_after > max_rho_before + 0.10:
                    return False, "动作导致负载率大幅上升", "VIOLATION"
            
            return True, "验证通过", "NONE"
            
        except Exception as e:
            err_msg = str(e)
            # 检查是否为严重错误（潮流发散/解列）
            if any(keyword in err_msg for keyword in ["Divergence", "non connected", "SolverFactor", "unbounded", "BackendError"]):
                return False, f"潮流计算发散/解列: {err_msg}", "DIVERGENCE"
            return False, f"模拟过程出错: {err_msg}", "VIOLATION"
