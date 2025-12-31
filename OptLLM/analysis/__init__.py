# -*- coding: utf-8 -*-
"""
OARA Analysis Module
包含优化服务等分析工具
"""

# 支持相对导入和直接导入
try:
    from .optimization_service import OptimizationService
except ImportError:
    # 直接导入（当直接运行或作为独立模块时）
    from optimization_service import OptimizationService

__all__ = [
    "OptimizationService",
]

