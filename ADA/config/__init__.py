# -*- coding: utf-8 -*-
"""
ADA 全局配置模块
统一管理系统配置、LLM接口、各组件参数

使用方式:
    from config import LLMConfig, SystemConfig
    
    llm_config = LLMConfig()  # 每次创建新实例
    sys_config = SystemConfig()
"""

from .llm_config import LLMConfig
from .system_config import SystemConfig

__all__ = [
    'LLMConfig',
    'SystemConfig',
]
