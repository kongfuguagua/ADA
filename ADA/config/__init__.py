# -*- coding: utf-8 -*-
"""
ADA 全局配置模块
统一管理系统配置、LLM接口、各组件参数
"""

from .llm_config import LLMConfig, get_llm_config
from .system_config import SystemConfig, get_system_config

__all__ = [
    'LLMConfig',
    'get_llm_config', 
    'SystemConfig',
    'get_system_config'
]

