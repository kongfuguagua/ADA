# -*- coding: utf-8 -*-
"""
评分基类
定义评分器的通用接口
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, Solution


class BaseReward(ABC):
    """
    评分基类
    所有评分器都应继承此类
    """
    
    def __init__(self, weight: float = 1.0):
        """
        初始化评分器
        
        Args:
            weight: 评分权重
        """
        self.weight = weight
    
    @abstractmethod
    def forward(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算评分
        
        Args:
            problem: 优化问题
            solution: 求解结果
        
        Returns:
            (评分, 详细信息)
        """
        pass
    
    def __call__(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """可调用接口"""
        return self.forward(problem, solution)
    
    @property
    def name(self) -> str:
        """评分器名称"""
        return self.__class__.__name__

