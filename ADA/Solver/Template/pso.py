# -*- coding: utf-8 -*-
"""
粒子群优化算法 (PSO)
适用于非凸、黑盒优化问题
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, SolverAlgorithmMeta
from .base import BaseAlgorithm


class PSOOptimizer(BaseAlgorithm):
    """
    粒子群优化算法
    适用于连续非凸优化问题
    """
    
    def __init__(
        self, 
        n_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,      # 惯性权重
        c1: float = 1.5,     # 个体学习因子
        c2: float = 1.5      # 社会学习因子
    ):
        """
        初始化 PSO 优化器
        
        Args:
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        super().__init__()
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="PSOOptimizer",
            description="粒子群优化算法，适用于连续非凸优化问题",
            capabilities={
                "convex_handling": 0.7,      # 凸问题也能处理
                "non_convex_handling": 0.9,  # 非凸问题处理能力强
                "constraint_handling": 0.5,  # 约束处理通过惩罚函数
                "speed": 0.6,                # 中等速度
                "global_optimality": 0.8,    # 较好的全局搜索能力
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """PSO 求解实现"""
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
        
        # 初始化粒子群
        particles = []
        velocities = []
        for _ in range(self.n_particles):
            particle = [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(n_vars)
            ]
            velocity = [
                random.uniform(-1, 1) * (upper_bounds[i] - lower_bounds[i]) * 0.1
                for i in range(n_vars)
            ]
            particles.append(particle)
            velocities.append(velocity)
        
        # 初始化个体最优和全局最优
        p_best = [p[:] for p in particles]
        p_best_scores = [evaluate(p) for p in particles]
        
        g_best_idx = p_best_scores.index(min(p_best_scores))
        g_best = p_best[g_best_idx][:]
        g_best_score = p_best_scores[g_best_idx]
        
        # 迭代优化
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # 更新速度
                for d in range(n_vars):
                    r1 = random.random()
                    r2 = random.random()
                    
                    velocities[i][d] = (
                        self.w * velocities[i][d] +
                        self.c1 * r1 * (p_best[i][d] - particles[i][d]) +
                        self.c2 * r2 * (g_best[d] - particles[i][d])
                    )
                
                # 更新位置
                for d in range(n_vars):
                    particles[i][d] += velocities[i][d]
                    # 边界处理
                    particles[i][d] = max(lower_bounds[d], min(upper_bounds[d], particles[i][d]))
                
                # 评估
                score = evaluate(particles[i])
                
                # 更新个体最优
                if score < p_best_scores[i]:
                    p_best[i] = particles[i][:]
                    p_best_scores[i] = score
                    
                    # 更新全局最优
                    if score < g_best_score:
                        g_best = particles[i][:]
                        g_best_score = score
            
            self.record_convergence(g_best_score)
        
        # 构建结果
        variables = {var_names[i]: g_best[i] for i in range(n_vars)}
        objective_value = g_best_score if problem.is_minimization else -g_best_score
        
        return {
            "feasible": True,
            "variables": variables,
            "objective": objective_value,
            "message": f"PSO 完成，迭代 {self.max_iter} 次"
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 PSOOptimizer:")
    
    # 创建非凸测试问题: Rastrigin 函数
    # f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    problem = OptimizationProblem(
        objective_function_latex=r"\min 10n + \sum(x_i^2 - 10\cos(2\pi x_i))",
        objective_function_code="10*2 + (x1**2 - 10*math.cos(2*3.14159*x1)) + (x2**2 - 10*math.cos(2*3.14159*x2))",
        variables=[
            VariableDefinition(name="x1", lower_bound=-5.12, upper_bound=5.12),
            VariableDefinition(name="x2", lower_bound=-5.12, upper_bound=5.12),
        ]
    )
    
    optimizer = PSOOptimizer(n_particles=50, max_iter=100)
    solution = optimizer.solve(problem)
    
    print(f"\n求解结果:")
    print(f"  可行: {solution.is_feasible}")
    print(f"  变量: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value:.6f}")
    print(f"  求解时间: {solution.solving_time:.4f}s")
    print(f"  收敛曲线长度: {len(solution.convergence_curve)}")
    print(f"  (全局最优在 (0,0)，最优值为 0)")

