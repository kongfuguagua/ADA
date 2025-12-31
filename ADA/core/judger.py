# -*- coding: utf-8 -*-
"""
Judger 模块：LLM 融合中枢
分析 Planner 和 Solver 的建议，生成融合策略
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

from ADA.utils.definitions import CandidateAction
from ADA.utils.parser import ActionParser
from ADA.utils.formatters import ObservationFormatter
from ADA.utils.prompts import PromptManager
from utils import OpenAIChat, get_logger

logger = get_logger("ADA.Judger")


class Judger:
    """
    LLM 融合中枢
    
    核心职责：
    1. 接收 Planner 和 Solver 的建议
    2. 结合 RAG 检索的历史上下文
    3. 使用 LLM 生成融合策略
    4. 解析 LLM 输出为 Grid2Op Action
    5. 返回 CandidateAction 列表
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        max_candidates: int = 3,
        **kwargs
    ):
        """
        初始化 Judger
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            llm_client: LLM 客户端
            max_candidates: 最大返回候选数
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_candidates = max_candidates
        
        # 初始化工具
        self.parser = ActionParser()
        self.formatter = ObservationFormatter()
        self.prompt_manager = PromptManager()
        
        logger.info(f"Judger 初始化完成 (max_candidates={max_candidates})")
    
    def generate_fused_actions(
        self,
        observation: BaseObservation,
        planner_candidates: List[CandidateAction],
        solver_candidates: List[CandidateAction],
        history_context: str = ""
    ) -> List[CandidateAction]:
        """
        生成融合动作
        
        工作流程：
        1. 格式化当前观测
        2. 构建融合提示词（包含 Planner、Solver 方案和历史参考）
        3. 调用 LLM 生成融合策略（带去重逻辑）
        4. 解析 LLM 输出为 Grid2Op Action
        5. 包装为 CandidateAction
        
        Args:
            observation: 当前观测
            planner_candidates: Planner 生成的候选动作
            solver_candidates: Solver 生成的候选动作
            history_context: 历史参考上下文
            
        Returns:
            CandidateAction 列表（来源：LLM_Fusion）
        """
        candidates = []
        
        # 如果 LLM 不可用，返回空列表
        if self.llm_client is None:
            logger.warning("Judger: LLM 客户端不可用，跳过融合候选生成")
            return candidates
        
        # 用于本次生成周期的去重
        seen_actions_str = set()
        
        try:
            # 1. 格式化当前观测（精简模式：仅输出关键子图）
            observation_text = self.formatter.format(observation, compact=True)
            
            # 2. 构建融合提示词
            messages = self.prompt_manager.build_fusion_prompt(
                observation_text=observation_text,
                planner_candidates=planner_candidates,
                solver_candidates=solver_candidates,
                history_context=history_context
            )
            
            # 3. 循环生成（允许重试以避免重复）
            attempts = 0
            max_attempts = self.max_candidates * 2  # 允许一定的重试次数
            
            while len(candidates) < self.max_candidates and attempts < max_attempts:
                attempts += 1
                try:
                    # 调用 LLM
                    llm_response = self.llm_client.chat(
                        prompt=messages[-1]["content"],
                        history=messages[:-1] if len(messages) > 1 else None,
                        system_prompt=messages[0]["content"] if messages else None
                    )
                    
                    # 4. 提取动作指令
                    action_text = self.parser.extract_action_from_response(llm_response)
                    if not action_text:
                        logger.debug(f"Judger: 第 {attempts} 次尝试无法提取动作指令")
                        continue
                    
                    # === 修复：去重逻辑 ===
                    # 简单的字符串去重，去除空格影响
                    normalized_text = action_text.replace(" ", "")
                    if normalized_text in seen_actions_str:
                        logger.debug(f"Judger: 跳过重复生成的动作: {action_text}")
                        continue
                    
                    # 5. 解析为 Grid2Op Action
                    try:
                        action_obj = self.parser.parse(action_text, self.action_space)
                    except ValueError as e:
                        logger.warning(f"Judger: 动作解析失败: {action_text}, 错误: {e}")
                        continue
                    
                    # 记录已生成的动作
                    seen_actions_str.add(normalized_text)
                    
                    # 6. 包装为 CandidateAction
                    candidate = CandidateAction(
                        source="LLM_Fusion",
                        action_obj=action_obj,
                        description=f"LLM 融合策略: {action_text}",
                        priority=2  # LLM 融合的优先级中等
                    )
                    candidates.append(candidate)
                    
                    logger.debug(f"Judger: 生成有效候选: {action_text}")
                    
                except Exception as e:
                    logger.warning(f"Judger: 生成过程出错: {e}")
                    continue
            
            logger.info(f"Judger: 成功生成 {len(candidates)} 个不重复的融合候选 (尝试次数: {attempts})")
            
        except Exception as e:
            logger.error(f"Judger: 生成融合动作失败: {e}", exc_info=True)
        
        return candidates
    
    def reset(self, observation: BaseObservation) -> None:
        """重置 Judger 状态"""
        logger.debug("Judger 重置")
        # 目前没有需要重置的状态
        pass

