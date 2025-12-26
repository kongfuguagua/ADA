# -*- coding: utf-8 -*-
"""
遗传算法 (GA)
适用于离散/组合优化问题
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, SolverAlgorithmMeta, VariableType
from .base import BaseAlgorithm


class GeneticOptimizer(BaseAlgorithm):
    """
    遗传算法
    适用于离散、组合优化问题
    """
    
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1
    ):
        """
        初始化遗传算法
        
        Args:
            population_size: 种群大小
            max_generations: 最大代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_ratio: 精英保留比例
        """
        super().__init__()
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="GeneticOptimizer",
            description="遗传算法，适用于离散和组合优化问题",
            capabilities={
                "convex_handling": 0.5,
                "non_convex_handling": 0.8,
                "constraint_handling": 0.6,
                "speed": 0.5,
                "global_optimality": 0.7,
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """遗传算法求解实现"""
        var_names = problem.get_variable_names()
        n_vars = len(var_names)
        
        if n_vars == 0:
            return {"feasible": False, "message": "无决策变量"}
        
        # 获取边界和类型
        bounds_dict = problem.get_variable_bounds()
        var_types = {v.name: v.type for v in problem.variables}
        
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
        
        def create_individual() -> List[float]:
            """创建个体"""
            individual = []
            for i in range(n_vars):
                if var_types[var_names[i]] == VariableType.BINARY:
                    individual.append(float(random.randint(0, 1)))
                elif var_types[var_names[i]] == VariableType.INTEGER:
                    individual.append(float(random.randint(int(lower_bounds[i]), int(upper_bounds[i]))))
                else:
                    individual.append(random.uniform(lower_bounds[i], upper_bounds[i]))
            return individual
        
        def mutate(individual: List[float]) -> List[float]:
            """变异操作"""
            mutated = individual[:]
            for i in range(n_vars):
                if random.random() < self.mutation_rate:
                    if var_types[var_names[i]] == VariableType.BINARY:
                        mutated[i] = 1.0 - mutated[i]
                    elif var_types[var_names[i]] == VariableType.INTEGER:
                        mutated[i] = float(random.randint(int(lower_bounds[i]), int(upper_bounds[i])))
                    else:
                        # 高斯变异
                        mutated[i] += random.gauss(0, (upper_bounds[i] - lower_bounds[i]) * 0.1)
                        mutated[i] = max(lower_bounds[i], min(upper_bounds[i], mutated[i]))
            return mutated
        
        def crossover(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
            """交叉操作"""
            if random.random() > self.crossover_rate:
                return parent1[:], parent2[:]
            
            # 单点交叉
            point = random.randint(1, n_vars - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        
        def select(population: List[List[float]], scores: List[float]) -> List[float]:
            """锦标赛选择"""
            tournament_size = 3
            selected_indices = random.sample(range(len(population)), tournament_size)
            best_idx = min(selected_indices, key=lambda i: scores[i])
            return population[best_idx][:]
        
        # 初始化种群
        population = [create_individual() for _ in range(self.population_size)]
        scores = [evaluate(ind) for ind in population]
        
        best_individual = population[scores.index(min(scores))][:]
        best_score = min(scores)
        self.record_convergence(best_score)
        
        # 进化
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        
        for generation in range(self.max_generations):
            # 精英保留
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
            new_population = [population[i][:] for i in sorted_indices[:n_elite]]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                parent1 = select(population, scores)
                parent2 = select(population, scores)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            scores = [evaluate(ind) for ind in population]
            
            # 更新最优
            gen_best_idx = scores.index(min(scores))
            if scores[gen_best_idx] < best_score:
                best_individual = population[gen_best_idx][:]
                best_score = scores[gen_best_idx]
            
            self.record_convergence(best_score)
        
        # 构建结果
        variables = {var_names[i]: best_individual[i] for i in range(n_vars)}
        objective_value = best_score if problem.is_minimization else -best_score
        
        return {
            "feasible": True,
            "variables": variables,
            "objective": objective_value,
            "message": f"遗传算法完成，进化 {self.max_generations} 代"
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 GeneticOptimizer:")
    
    # 创建混合整数问题
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1 + 2x_2 + 3x_3",
        objective_function_code="x1 + 2*x2 + 3*x3",
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="x3", type=VariableType.INTEGER, lower_bound=0, upper_bound=5),
        ]
    )
    
    optimizer = GeneticOptimizer(population_size=30, max_generations=50)
    solution = optimizer.solve(problem)
    
    print(f"\n求解结果:")
    print(f"  可行: {solution.is_feasible}")
    print(f"  变量: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value:.6f}")
    print(f"  求解时间: {solution.solving_time:.4f}s")
    print(f"  (最优解应为 x1=0, x2=0, x3=0，最优值为 0)")

