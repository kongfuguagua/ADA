# -*- coding: utf-8 -*-
"""
凸优化求解器
使用 scipy.optimize 实现
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, SolverAlgorithmMeta
from .base import BaseAlgorithm


class ConvexOptimizer(BaseAlgorithm):
    """
    凸优化求解器
    适用于凸目标函数和凸约束的问题
    """
    
    def __init__(self, method: str = "SLSQP", max_iter: int = 1000):
        """
        初始化凸优化器
        
        Args:
            method: scipy 优化方法 (SLSQP, trust-constr, etc.)
            max_iter: 最大迭代次数
        """
        super().__init__()
        self.method = method
        self.max_iter = max_iter
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="ConvexOptimizer",
            description="基于 scipy.optimize 的凸优化求解器，适用于光滑凸问题",
            capabilities={
                "convex_handling": 1.0,      # 凸问题处理能力强
                "non_convex_handling": 0.3,  # 非凸问题可能陷入局部最优
                "constraint_handling": 0.8,  # 支持等式和不等式约束
                "speed": 0.9,                # 求解速度快
                "global_optimality": 0.4,    # 仅保证局部最优
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """凸优化求解实现"""
        try:
            from scipy.optimize import minimize, Bounds
            import numpy as np
        except ImportError:
            return {
                "feasible": False,
                "message": "scipy 未安装，请运行: pip install scipy"
            }
        
        # 获取变量信息
        var_names = problem.get_variable_names()
        n_vars = len(var_names)
        
        if n_vars == 0:
            return {"feasible": False, "message": "无决策变量"}
        
        # 构建边界
        bounds_dict = problem.get_variable_bounds()
        lower_bounds = [bounds_dict[v][0] for v in var_names]
        upper_bounds = [bounds_dict[v][1] for v in var_names]
        
        # 处理无穷边界
        lower_bounds = [b if b != float('-inf') else -1e10 for b in lower_bounds]
        upper_bounds = [b if b != float('inf') else 1e10 for b in upper_bounds]
        
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # 构建目标函数
        obj_func = self.parse_objective_function(problem)
        
        def scipy_objective(x):
            x_dict = {var_names[i]: x[i] for i in range(n_vars)}
            value = obj_func(x_dict)
            self.record_convergence(value)
            return value if problem.is_minimization else -value
        
        # 初始点
        x0 = np.array([
            (lower_bounds[i] + upper_bounds[i]) / 2 
            for i in range(n_vars)
        ])
        
        # 求解
        result = minimize(
            scipy_objective,
            x0,
            method=self.method,
            bounds=bounds,
            options={"maxiter": self.max_iter}
        )
        
        # 构建结果
        variables = {var_names[i]: float(result.x[i]) for i in range(n_vars)}
        objective_value = result.fun if problem.is_minimization else -result.fun
        
        return {
            "feasible": result.success,
            "variables": variables,
            "objective": objective_value,
            "message": result.message if hasattr(result, 'message') else str(result)
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 ConvexOptimizer:")
    
    # 创建测试问题: min x1^2 + x2^2
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        objective_function_code="x1**2 + x2**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
        ]
    )
    
    optimizer = ConvexOptimizer()
    solution = optimizer.solve(problem)
    
    print(f"\n求解结果:")
    print(f"  可行: {solution.is_feasible}")
    print(f"  变量: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value}")
    print(f"  求解时间: {solution.solving_time:.4f}s")
    print(f"  收敛曲线长度: {len(solution.convergence_curve)}")
    
    # 测试能力向量
    print(f"\n算法能力向量: {optimizer.get_capability_vector()}")

