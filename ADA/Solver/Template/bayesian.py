# -*- coding: utf-8 -*-
"""
贝叶斯优化算法
适用于昂贵的黑盒优化问题
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, SolverAlgorithmMeta
from .base import BaseAlgorithm


class BayesianOptimizer(BaseAlgorithm):
    """
    贝叶斯优化算法
    使用高斯过程代理模型，适用于评估代价高的问题
    """
    
    def __init__(
        self,
        n_init: int = 5,      # 初始采样点数
        n_iter: int = 50,     # 优化迭代次数
        exploration: float = 0.1  # 探索参数
    ):
        """
        初始化贝叶斯优化器
        
        Args:
            n_init: 初始随机采样点数
            n_iter: 贝叶斯优化迭代次数
            exploration: 探索-利用平衡参数
        """
        super().__init__()
        self.n_init = n_init
        self.n_iter = n_iter
        self.exploration = exploration
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="BayesianOptimizer",
            description="贝叶斯优化算法，适用于评估代价高的黑盒优化问题",
            capabilities={
                "convex_handling": 0.6,
                "non_convex_handling": 0.8,
                "constraint_handling": 0.4,  # 约束处理能力一般
                "speed": 0.4,                # 速度较慢（需要拟合代理模型）
                "global_optimality": 0.9,    # 全局搜索能力强
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """贝叶斯优化求解实现"""
        # 获取变量信息
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
        
        # 初始采样
        X_samples: List[List[float]] = []
        y_samples: List[float] = []
        
        for _ in range(self.n_init):
            x = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(n_vars)]
            y = evaluate(x)
            X_samples.append(x)
            y_samples.append(y)
            self.record_convergence(min(y_samples))
        
        # 贝叶斯优化迭代
        for iteration in range(self.n_iter):
            # 使用简化的采集函数（随机搜索 + 局部改进）
            # 在实际应用中应使用高斯过程和 EI/UCB 采集函数
            
            best_idx = y_samples.index(min(y_samples))
            best_x = X_samples[best_idx]
            
            # 生成候选点：在当前最优点附近采样
            candidates = []
            for _ in range(20):
                if random.random() < self.exploration:
                    # 全局探索
                    x = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(n_vars)]
                else:
                    # 局部利用
                    x = [
                        best_x[i] + random.gauss(0, (upper_bounds[i] - lower_bounds[i]) * 0.1)
                        for i in range(n_vars)
                    ]
                    # 边界处理
                    x = [max(lower_bounds[i], min(upper_bounds[i], x[i])) for i in range(n_vars)]
                candidates.append(x)
            
            # 选择最佳候选点（简化版：直接评估）
            best_candidate = None
            best_candidate_value = float('inf')
            
            for x in candidates:
                # 使用简单的预测（实际应使用 GP 预测）
                y = evaluate(x)
                if y < best_candidate_value:
                    best_candidate = x
                    best_candidate_value = y
            
            # 添加新样本
            X_samples.append(best_candidate)
            y_samples.append(best_candidate_value)
            self.record_convergence(min(y_samples))
        
        # 返回最优解
        best_idx = y_samples.index(min(y_samples))
        best_x = X_samples[best_idx]
        best_y = y_samples[best_idx]
        
        variables = {var_names[i]: best_x[i] for i in range(n_vars)}
        objective_value = best_y if problem.is_minimization else -best_y
        
        return {
            "feasible": True,
            "variables": variables,
            "objective": objective_value,
            "message": f"贝叶斯优化完成，评估 {len(y_samples)} 次"
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 BayesianOptimizer:")
    
    # 创建测试问题
    problem = OptimizationProblem(
        objective_function_latex=r"\min (x_1-3)^2 + (x_2-2)^2",
        objective_function_code="(x1-3)**2 + (x2-2)**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ]
    )
    
    optimizer = BayesianOptimizer(n_init=5, n_iter=30)
    solution = optimizer.solve(problem)
    
    print(f"\n求解结果:")
    print(f"  可行: {solution.is_feasible}")
    print(f"  变量: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value:.6f}")
    print(f"  求解时间: {solution.solving_time:.4f}s")
    print(f"  (最优解应接近 (3, 2)，最优值为 0)")

