# -*- coding: utf-8 -*-
"""
梯度下降优化算法
适用于可微的连续优化问题
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, SolverAlgorithmMeta
from .base import BaseAlgorithm


class GradientOptimizer(BaseAlgorithm):
    """
    梯度下降优化器
    使用数值梯度进行优化
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        momentum: float = 0.9
    ):
        """
        初始化梯度优化器
        
        Args:
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            momentum: 动量系数
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.momentum = momentum
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="GradientOptimizer",
            description="梯度下降优化器，适用于可微的连续优化问题",
            capabilities={
                "convex_handling": 0.9,
                "non_convex_handling": 0.4,  # 容易陷入局部最优
                "constraint_handling": 0.3,  # 约束处理能力弱
                "speed": 0.8,
                "global_optimality": 0.3,
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """梯度下降求解实现"""
        var_names = problem.get_variable_names()
        n_vars = len(var_names)
        
        if n_vars == 0:
            return {"feasible": False, "message": "无决策变量"}
        
        # 获取边界
        bounds_dict = problem.get_variable_bounds()
        lower_bounds = [bounds_dict[v][0] for v in var_names]
        upper_bounds = [bounds_dict[v][1] for v in var_names]
        
        # 处理无穷边界
        lower_bounds = [b if b != float('-inf') else -100 for b in lower_bounds]
        upper_bounds = [b if b != float('inf') else 100 for b in upper_bounds]
        
        # 构建目标函数
        obj_func = self.parse_objective_function(problem)
        
        def evaluate(x: List[float]) -> float:
            x_dict = {var_names[i]: x[i] for i in range(n_vars)}
            value = obj_func(x_dict)
            return value if problem.is_minimization else -value
        
        def compute_gradient(x: List[float], eps: float = 1e-5) -> List[float]:
            """数值梯度计算"""
            grad = []
            f_x = evaluate(x)
            for i in range(n_vars):
                x_plus = x[:]
                x_plus[i] += eps
                f_plus = evaluate(x_plus)
                grad.append((f_plus - f_x) / eps)
            return grad
        
        # 初始点
        x = [
            (lower_bounds[i] + upper_bounds[i]) / 2
            for i in range(n_vars)
        ]
        
        # 动量
        velocity = [0.0] * n_vars
        
        best_x = x[:]
        best_value = evaluate(x)
        self.record_convergence(best_value)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 计算梯度
            grad = compute_gradient(x)
            
            # 更新速度和位置
            for i in range(n_vars):
                velocity[i] = self.momentum * velocity[i] - self.learning_rate * grad[i]
                x[i] += velocity[i]
                # 边界约束
                x[i] = max(lower_bounds[i], min(upper_bounds[i], x[i]))
            
            # 评估
            value = evaluate(x)
            
            if value < best_value:
                best_x = x[:]
                best_value = value
            
            self.record_convergence(best_value)
            
            # 收敛检查
            grad_norm = sum(g**2 for g in grad) ** 0.5
            if grad_norm < self.tolerance:
                break
        
        # 构建结果
        variables = {var_names[i]: best_x[i] for i in range(n_vars)}
        objective_value = best_value if problem.is_minimization else -best_value
        
        return {
            "feasible": True,
            "variables": variables,
            "objective": objective_value,
            "message": f"梯度下降完成，迭代 {iteration + 1} 次"
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 GradientOptimizer:")
    
    # 创建二次问题
    problem = OptimizationProblem(
        objective_function_latex=r"\min (x_1-2)^2 + (x_2-3)^2",
        objective_function_code="(x1-2)**2 + (x2-3)**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
        ]
    )
    
    optimizer = GradientOptimizer(learning_rate=0.1, max_iter=500)
    solution = optimizer.solve(problem)
    
    print(f"\n求解结果:")
    print(f"  可行: {solution.is_feasible}")
    print(f"  变量: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value:.6f}")
    print(f"  求解时间: {solution.solving_time:.4f}s")
    print(f"  (最优解应接近 (2, 3)，最优值为 0)")

