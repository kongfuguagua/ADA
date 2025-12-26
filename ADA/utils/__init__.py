# -*- coding: utf-8 -*-
"""
ADA 通用工具模块
包含数据契约、接口定义、日志工具等
"""

from .const import (
    # 枚举类型
    AgentRole,
    KnowledgeType,
    FeedbackType,
    # 数据契约
    EnvironmentState,
    KnowledgeItem,
    VariableDefinition,
    OptimizationProblem,
    SolverAlgorithmMeta,
    Solution,
    PhysicalMetrics,
    Feedback,
    ExecutionTrace,
    ToolAction,
    AugmentationStep,
)

from .interact import (
    # 基础接口
    BaseLLM,
    BaseVectorStore,
    BaseTool,
    # Agent 接口
    BasePlanner,
    BaseSolverStrategy,
    BaseSolver,
    BaseJudger,
    BaseSummarizer,
)

from .logger import (
    get_logger,
    ADALogger,
)

__all__ = [
    # 枚举
    'AgentRole',
    'KnowledgeType', 
    'FeedbackType',
    # 数据契约
    'EnvironmentState',
    'KnowledgeItem',
    'VariableDefinition',
    'OptimizationProblem',
    'SolverAlgorithmMeta',
    'Solution',
    'PhysicalMetrics',
    'Feedback',
    'ExecutionTrace',
    'ToolAction',
    'AugmentationStep',
    # 基础接口
    'BaseLLM',
    'BaseVectorStore',
    'BaseTool',
    # Agent 接口
    'BasePlanner',
    'BaseSolverStrategy',
    'BaseSolver',
    'BaseJudger',
    'BaseSummarizer',
    # 日志
    'get_logger',
    'ADALogger',
]

