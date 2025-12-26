# -*- coding: utf-8 -*-
"""
求解算法基类
定义所有优化算法的通用接口
"""

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, Solution, SolverAlgorithmMeta
from utils.interact import BaseSolverStrategy
from utils.logger import get_logger

logger = get_logger("Algorithm")


class BaseAlgorithm(BaseSolverStrategy):
    """
    优化算法基类
    所有具体算法都应继承此类
    """
    
    def __init__(self):
        self._solving_time: float = 0.0
        self._convergence_curve: List[float] = []
    
    @property
    @abstractmethod
    def meta(self) -> SolverAlgorithmMeta:
        """算法元信息"""
        pass
    
    @property
    def name(self) -> str:
        """算法名称"""
        return self.meta.name
    
    @abstractmethod
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """
        算法具体实现（子类必须实现）
        
        Args:
            problem: 优化问题
        
        Returns:
            包含 {variables: Dict, objective: float, feasible: bool, message: str}
        """
        pass
    
    def solve(self, problem: OptimizationProblem) -> Solution:
        """
        求解优化问题
        
        Args:
            problem: 优化问题
        
        Returns:
            求解结果
        """
        logger.info(f"使用 {self.name} 求解")
        
        self._convergence_curve = []
        start_time = time.time()
        
        try:
            result = self._solve_impl(problem)
            self._solving_time = time.time() - start_time
            
            return Solution(
                is_feasible=result.get("feasible", False),
                algorithm_used=self.name,
                decision_variables=result.get("variables", {}),
                objective_value=result.get("objective", float('inf')),
                solving_time=self._solving_time,
                convergence_curve=self._convergence_curve,
                solver_message=result.get("message", "")
            )
        except Exception as e:
            self._solving_time = time.time() - start_time
            logger.error(f"求解失败: {e}")
            
            return Solution(
                is_feasible=False,
                algorithm_used=self.name,
                solving_time=self._solving_time,
                solver_message=f"求解异常: {str(e)}"
            )
    
    def get_capability_vector(self) -> List[float]:
        """
        获取算法能力向量 ψ(A)
        
        Returns:
            5 维能力向量 [凸处理, 非凸处理, 约束处理, 速度, 全局最优性]
        """
        caps = self.meta.capabilities
        return [
            caps.get("convex_handling", 0.5),
            caps.get("non_convex_handling", 0.5),
            caps.get("constraint_handling", 0.5),
            caps.get("speed", 0.5),
            caps.get("global_optimality", 0.5),
        ]
    
    def record_convergence(self, value: float) -> None:
        """记录收敛曲线点"""
        self._convergence_curve.append(value)
    
    @staticmethod
    def parse_objective_function(problem: OptimizationProblem):
        """
        解析目标函数为可调用函数
        
        Args:
            problem: 优化问题
        
        Returns:
            目标函数 callable
        """
        code = problem.objective_function_code
        if not code:
            # 使用简单的默认函数
            return lambda x: sum(x.values()) if isinstance(x, dict) else sum(x)
        
        # 创建安全的执行环境
        import math
        safe_globals = {
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "math": math,
            **{f"x{i}": 0 for i in range(100)},  # 预定义变量
        }
        
        def objective(x: Dict[str, float]) -> float:
            local_vars = {**safe_globals, **x}
            try:
                return eval(code, {"__builtins__": {}}, local_vars)
            except Exception:
                return float('inf')
        
        return objective
