# -*- coding: utf-8 -*-
"""
Solver 求解智能体模块
负责问题特征分析和算法匹配求解

主要组件：
- SolverAgent: 求解智能体主入口
- AlgorithmMatcher: 算法匹配器
- ProblemFeatureExtractor: 特征提取器
- BaseAlgorithm: 算法基类

使用示例：
    from Solver import SolverAgent
    
    solver = SolverAgent()
    solution = solver.solve(problem)
"""

# 主入口
from .solver import SolverAgent

# 特征提取
from .feature import ProblemFeatureExtractor, ProblemFeatures

# 算法匹配
from .matcher import AlgorithmMatcher

# 算法基类
from .Template.base import BaseAlgorithm

# 具体算法
from .Template import (
    ConvexOptimizer,
    GurobiOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
    GradientOptimizer,
)

__all__ = [
    # 主入口
    'SolverAgent',
    
    # 特征
    'ProblemFeatureExtractor',
    'ProblemFeatures',
    
    # 匹配器
    'AlgorithmMatcher',
    
    # 算法
    'BaseAlgorithm',
    'ConvexOptimizer',
    'GurobiOptimizer',
    'PSOOptimizer',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'GradientOptimizer',
]
