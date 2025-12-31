# -*- coding: utf-8 -*-
"""
Summarizer 模块：经验总结与入库
将实证结果转化为经验，存入知识库
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Observation import BaseObservation

from ADA.utils.definitions import CandidateAction
from ADA.utils.formatters import ObservationFormatter
from utils import OpenAIChat, get_logger

logger = get_logger("ADA.Summarizer")


class Summarizer:
    """
    经验总结器
    
    核心职责：
    1. 记录最终选中的动作及其效果
    2. 使用 LLM 生成结构化经验总结
    3. 存入知识库供后续检索
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIChat] = None,
        knowledge_base=None,
        enable_learning: bool = True,
        **kwargs
    ):
        """
        初始化 Summarizer
        
        Args:
            llm_client: LLM 客户端（用于生成经验总结）
            knowledge_base: 知识库服务（可选）
            enable_learning: 是否启用学习（存储经验）
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.enable_learning = enable_learning
        
        # 初始化格式化器
        self.formatter = ObservationFormatter()
        
        logger.info(f"Summarizer 初始化完成 (enable_learning={enable_learning})")
    
    def summarize(
        self,
        observation: BaseObservation,
        best_action: CandidateAction,
        reward: float = 0.0
    ) -> None:
        """
        总结经验并存入知识库
        
        Args:
            observation: 当前观测
            best_action: 最终选中的动作
            reward: 执行后的奖励
        """
        if not self.enable_learning:
            return
        
        if self.knowledge_base is None:
            logger.debug("Summarizer: 知识库不可用，跳过经验存储")
            return
        
        if self.llm_client is None:
            logger.debug("Summarizer: LLM 客户端不可用，跳过经验总结")
            return
        
        try:
            # 1. 提取场景特征
            scenario_features = self._extract_scenario_features(observation, best_action)
            
            # 2. 使用 LLM 生成经验总结
            experience_text = self._generate_experience_summary(
                observation=observation,
                best_action=best_action,
                reward=reward,
                scenario_features=scenario_features
            )
            
            if not experience_text:
                logger.warning("Summarizer: LLM 生成经验总结失败")
                return
            
            # 3. 构建元数据
            metadata = {
                "source": best_action.source,
                "reward": float(reward),
                "rho_max_before": float(observation.rho.max()),
                "rho_max_after": best_action.simulation_result.get("rho_max", 0.0) if best_action.simulation_result else 0.0,
                "is_safe": best_action.simulation_result.get("is_safe", False) if best_action.simulation_result else False,
                "action_description": best_action.description,
            }
            
            # 4. 存入知识库
            try:
                # 尝试使用知识库的 add_knowledge 方法
                if hasattr(self.knowledge_base, 'add_knowledge'):
                    # 如果知识库支持类型参数，使用它
                    if hasattr(self.knowledge_base, 'KnowledgeType'):
                        from enum import Enum
                        # 尝试导入或定义知识类型
                        try:
                            # 假设知识库有 AK (Action Knowledge) 类型
                            knowledge_type = getattr(self.knowledge_base, 'KnowledgeType', None)
                            if knowledge_type and hasattr(knowledge_type, 'AK'):
                                self.knowledge_base.add_knowledge(
                                    content=experience_text,
                                    knowledge_type=knowledge_type.AK,
                                    metadata=metadata
                                )
                            else:
                                # 降级：直接调用，不指定类型
                                self.knowledge_base.add_knowledge(
                                    content=experience_text,
                                    metadata=metadata
                                )
                        except Exception as e:
                            logger.warning(f"Summarizer: 知识库存储失败（类型问题）: {e}")
                            # 降级：使用简单的存储方法
                            self._simple_store(experience_text, metadata)
                    else:
                        # 降级：直接调用，不指定类型
                        self.knowledge_base.add_knowledge(
                            content=experience_text,
                            metadata=metadata
                        )
                else:
                    # 降级：使用简单的存储方法
                    self._simple_store(experience_text, metadata)
                
                logger.debug(f"Summarizer: 经验已存入知识库")
                
            except Exception as e:
                logger.warning(f"Summarizer: 知识库存储失败: {e}")
                # 降级：使用简单的存储方法
                self._simple_store(experience_text, metadata)
        
        except Exception as e:
            logger.error(f"Summarizer: 经验总结失败: {e}", exc_info=True)
    
    def _extract_scenario_features(
        self,
        observation: BaseObservation,
        best_action: CandidateAction
    ) -> Dict[str, Any]:
        """
        提取场景特征（用于知识库检索）
        
        Args:
            observation: 当前观测
            best_action: 选中的动作
            
        Returns:
            场景特征字典
        """
        features = {
            "max_rho": float(observation.rho.max()),
            "overflow_count": int((observation.rho > 1.0).sum()),
            "overflow_lines": [],
            "action_source": best_action.source,
        }
        
        # 提取过载线路信息
        overflow_mask = observation.rho > 1.0
        if overflow_mask.any():
            overflow_indices = observation.rho[overflow_mask].argsort()[::-1][:3]  # Top-3
            features["overflow_lines"] = [
                {
                    "line_id": int(idx),
                    "rho": float(observation.rho[idx])
                }
                for idx in overflow_indices
            ]
        
        return features
    
    def _generate_experience_summary(
        self,
        observation: BaseObservation,
        best_action: CandidateAction,
        reward: float,
        scenario_features: Dict[str, Any]
    ) -> Optional[str]:
        """
        使用 LLM 生成经验总结
        
        Args:
            observation: 当前观测
            best_action: 选中的动作
            reward: 执行后的奖励
            scenario_features: 场景特征
            
        Returns:
            经验总结文本
        """
        try:
            # 构建提示词
            prompt = self._build_summary_prompt(
                observation=observation,
                best_action=best_action,
                reward=reward,
                scenario_features=scenario_features
            )
            
            # 调用 LLM
            summary = self.llm_client.chat(
                prompt=prompt,
                system_prompt="你是一个电网调度经验总结专家。请简洁地总结成功案例，便于后续检索和参考。"
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summarizer: LLM 生成经验总结失败: {e}", exc_info=True)
            # 降级：生成简单的文本总结
            return self._simple_summary(observation, best_action, reward, scenario_features)
    
    def _build_summary_prompt(
        self,
        observation: BaseObservation,
        best_action: CandidateAction,
        reward: float,
        scenario_features: Dict[str, Any]
    ) -> str:
        """构建总结提示词"""
        lines = []
        
        lines.append("请总结以下电网调度经验：")
        lines.append("")
        lines.append("=== 场景 ===")
        lines.append(f"最大负载率: {scenario_features['max_rho']:.2%}")
        lines.append(f"过载线路数: {scenario_features['overflow_count']}")
        
        if scenario_features['overflow_lines']:
            lines.append("过载线路:")
            for line_info in scenario_features['overflow_lines']:
                lines.append(f"  - 线路 {line_info['line_id']}: {line_info['rho']:.2%}")
        
        lines.append("")
        lines.append("=== 采取的动作 ===")
        lines.append(f"来源: {best_action.source}")
        lines.append(f"描述: {best_action.description}")
        
        if best_action.simulation_result:
            sim_result = best_action.simulation_result
            lines.append(f"预期效果:")
            lines.append(f"  - 最大负载率: {sim_result.get('rho_max', 'N/A'):.2%}")
            lines.append(f"  - 安全性: {'安全' if sim_result.get('is_safe') else '不安全'}")
            lines.append(f"  - 奖励: {sim_result.get('reward', 'N/A')}")
        
        lines.append("")
        lines.append(f"=== 执行结果 ===")
        lines.append(f"奖励: {reward:.2f}")
        
        lines.append("")
        lines.append("请用1-2句话总结这个经验，格式：")
        lines.append("场景：[场景描述]；动作：[动作描述]；效果：[效果描述]")
        
        return "\n".join(lines)
    
    def _simple_summary(
        self,
        observation: BaseObservation,
        best_action: CandidateAction,
        reward: float,
        scenario_features: Dict[str, Any]
    ) -> str:
        """生成简单的文本总结（降级方案）"""
        lines = []
        
        lines.append(f"场景：最大负载率 {scenario_features['max_rho']:.2%}")
        if scenario_features['overflow_count'] > 0:
            lines.append(f"，{scenario_features['overflow_count']} 条过载线路")
        
        lines.append(f"；动作：{best_action.description}（来源：{best_action.source}）")
        
        if best_action.simulation_result:
            rho_after = best_action.simulation_result.get('rho_max', 0.0)
            lines.append(f"；效果：负载率降至 {rho_after:.2%}，奖励 {reward:.2f}")
        
        return "".join(lines)
    
    def _simple_store(self, content: str, metadata: Dict[str, Any]) -> None:
        """简单的存储方法（降级方案）"""
        # 如果知识库有简单的存储方法，使用它
        if hasattr(self.knowledge_base, 'add_document') or hasattr(self.knowledge_base, 'add'):
            try:
                if hasattr(self.knowledge_base, 'add_document'):
                    self.knowledge_base.add_document(content, metadata=metadata)
                elif hasattr(self.knowledge_base, 'add'):
                    self.knowledge_base.add(content, metadata=metadata)
            except Exception as e:
                logger.warning(f"Summarizer: 简单存储也失败: {e}")
        else:
            logger.debug("Summarizer: 知识库不支持存储，跳过")
    
    def reset(self) -> None:
        """重置 Summarizer 状态"""
        logger.debug("Summarizer 重置")
        # 目前没有需要重置的状态
        pass

