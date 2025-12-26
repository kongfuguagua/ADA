# -*- coding: utf-8 -*-
"""
ADA 工具模块

包含：
- const: 数据契约定义
- interact: Agent 接口定义
- logger: 日志系统
- llm: LLM 服务
- embeddings: Embedding 服务
"""

from .const import (
    # 枚举类型
    VariableType,
    KnowledgeType,
    FeedbackType,
    
    # 数据结构
    VariableDefinition,
    SolverAlgorithmMeta,
    PhysicalMetrics,
    KnowledgeItem,
    EnvironmentState,
    OptimizationProblem,
    Solution,
    Feedback,
    AugmentationStep,
    ToolAction,
    ExecutionTrace,
)

from .interact import (
    # 基础服务接口
    BaseTool,
    BaseSimulator,
    BaseVectorStore,
    
    # Agent 接口
    BasePlanner,
    BaseSolverStrategy,
    BaseSolver,
    BaseJudger,
    BaseSummarizer,
)

from .logger import get_logger
from .json_utils import safe_json_dumps, safe_json_dump, convert_to_serializable

from .llm import BaseLLM, OpenAIChat
from .embeddings import BaseEmbeddings, OpenAIEmbedding

__all__ = [
    # 枚举
    'VariableType',
    'KnowledgeType', 
    'FeedbackType',
    
    # 数据结构
    'VariableDefinition',
    'SolverAlgorithmMeta',
    'PhysicalMetrics',
    'KnowledgeItem',
    'EnvironmentState',
    'OptimizationProblem',
    'Solution',
    'Feedback',
    'AugmentationStep',
    'ToolAction',
    'ExecutionTrace',
    
    # 接口
    'BaseLLM',
    'BaseEmbeddings',
    'BaseTool',
    'BaseSimulator',
    'BaseVectorStore',
    'BasePlanner',
    'BaseSolverStrategy',
    'BaseSolver',
    'BaseJudger',
    'BaseSummarizer',
    
    # 实现
    'OpenAIChat',
    'OpenAIEmbedding',
    
    # 工具函数
    'get_logger',
    'safe_json_dumps',
    'safe_json_dump',
    'convert_to_serializable',
]
