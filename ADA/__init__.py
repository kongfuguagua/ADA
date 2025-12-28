# -*- coding: utf-8 -*-
"""
ADA - Agile Dispatch Agent
知识驱动的复杂系统敏捷调度智能体框架

主要导出：
- ADAgent: 主智能体类
- 工具模块: utils (LLM, logger, embeddings 等)
- 配置模块: config (SystemConfig, LLMConfig)
- 核心组件: Planner, Solver, Judger, Summarizer
"""

# 主智能体
from .ADAgent import ADAgent

# 工具模块（最常用，优先导出）
from .utils import (
    # LLM 和 Embeddings
    OpenAIChat,
    OpenAIEmbedding,
    BaseLLM,
    BaseEmbeddings,
    
    # 日志
    get_logger,
    
    # 数据结构
    EnvironmentState,
    OptimizationProblem,
    Solution,
    Feedback,
    FeedbackType,
    
    # 工具函数
    safe_json_dumps,
    safe_json_dump,
    convert_to_serializable,
)

# 配置模块
from .config import (
    SystemConfig,
    LLMConfig,
)

# 核心组件（按需导入，避免循环依赖）
# 如果需要使用，可以：
# from ADA.Planner import PlannerAgent
# from ADA.Solver import SolverAgent
# from ADA.Judger import JudgerAgent
# from ADA.Summarizer import SummarizerAgent

__all__ = [
    # 主智能体
    'ADAgent',
    
    # LLM 和 Embeddings
    'OpenAIChat',
    'OpenAIEmbedding',
    'BaseLLM',
    'BaseEmbeddings',
    
    # 日志
    'get_logger',
    
    # 数据结构
    'EnvironmentState',
    'OptimizationProblem',
    'Solution',
    'Feedback',
    'FeedbackType',
    
    # 配置
    'SystemConfig',
    'LLMConfig',
    
    # 工具函数
    'safe_json_dumps',
    'safe_json_dump',
    'convert_to_serializable',
]

__version__ = "1.0.0"

