# -*- coding: utf-8 -*-
"""
Solver 模块功能测试
测试各类优化问题的求解能力

运行方式:
    conda activate ada
    cd /Users/yangsiyu/Desktop/ADA/ADA
    python -m Solver.test.demo
"""

import sys
import math
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Solver import SolverAgent, ProblemFeatureExtractor, AlgorithmMatcher
from utils.const import OptimizationProblem, VariableDefinition, VariableType


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """打印小节"""
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)


def test_feature_extraction():
    """测试特征提取功能"""
    print_header("测试1: 特征提取 (Feature Extraction)")
    
    extractor = ProblemFeatureExtractor()
    
    # 1.1 线性问题
    print_section("1.1 线性规划问题")
    lp = OptimizationProblem(
        objective_function_latex=r"\min 3x_1 + 2x_2 + 5x_3",
        objective_function_code="3*x1 + 2*x2 + 5*x3",
        constraints_latex=[r"x_1 + x_2 \leq 10", r"2x_2 + x_3 \leq 15"],
        constraints_code=["x1 + x2 <= 10", "2*x2 + x3 <= 15"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=100),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=100),
            VariableDefinition(name="x3", lower_bound=0, upper_bound=100),
        ]
    )
    features = extractor.extract(lp)
    print(f"特征向量: {[f'{v:.3f}' for v in features.to_vector()]}")
    print(f"  [非凸性, 非线性, 约束紧迫度, 离散性, 规模]")
    print(f"详细: {features.to_dict()}")
    
    # 1.2 非凸问题（三角函数）
    print_section("1.2 非凸问题（三角函数）")
    trig = OptimizationProblem(
        objective_function_latex=r"\min \sin(x_1) \cdot \cos(x_2) + x_1^2",
        objective_function_code="math.sin(x1) * math.cos(x2) + x1**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=-3.14, upper_bound=3.14),
            VariableDefinition(name="x2", lower_bound=-3.14, upper_bound=3.14),
        ]
    )
    features = extractor.extract(trig)
    print(f"特征向量: {[f'{v:.3f}' for v in features.to_vector()]}")
    print(f"有三角函数: {features.has_trigonometric_terms}")
    print(f"有二次项: {features.has_quadratic_terms}")
    
    # 1.3 混合整数问题
    print_section("1.3 混合整数问题")
    mip = OptimizationProblem(
        objective_function_latex=r"\min \sum_{i} c_i x_i + \sum_j d_j y_j",
        objective_function_code="x1 + 2*x2 + 3*y1 + 4*y2",
        constraints_latex=[r"\sum x_i + \sum y_j = 5"],
        constraints_code=["x1 + x2 + y1 + y2 == 5"],
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="y1", type=VariableType.INTEGER, lower_bound=0, upper_bound=10),
            VariableDefinition(name="y2", type=VariableType.INTEGER, lower_bound=0, upper_bound=10),
        ]
    )
    features = extractor.extract(mip)
    print(f"特征向量: {[f'{v:.3f}' for v in features.to_vector()]}")
    print(f"离散性分数: {features.discreteness_score:.3f}")
    print(f"有二元变量: {features.has_binary_variables}")
    print(f"有整数变量: {features.has_integer_variables}")


def test_algorithm_matching():
    """测试算法匹配功能"""
    print_header("测试2: 算法匹配 (Algorithm Matching)")
    
    solver = SolverAgent()
    print(f"已注册算法: {solver.list_algorithms()}")
    
    # 2.1 凸问题 - 应该选择 ConvexOptimizer 或 GurobiOptimizer
    print_section("2.1 凸二次问题")
    convex = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2 + x_3^2",
        objective_function_code="x1**2 + x2**2 + x3**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x3", lower_bound=-10, upper_bound=10),
        ]
    )
    scores = solver.get_match_scores(convex)
    print("匹配评分排名:")
    for i, (name, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    # 2.2 非凸问题 - 应该选择 PSO 或 Genetic
    print_section("2.2 非凸多峰问题")
    nonconvex = OptimizationProblem(
        objective_function_latex=r"\min \sin(x_1) + \cos(x_2) + \sin(x_3)",
        objective_function_code="math.sin(x1) + math.cos(x2) + math.sin(x3)",
        variables=[
            VariableDefinition(name="x1", lower_bound=-5, upper_bound=5),
            VariableDefinition(name="x2", lower_bound=-5, upper_bound=5),
            VariableDefinition(name="x3", lower_bound=-5, upper_bound=5),
        ]
    )
    scores = solver.get_match_scores(nonconvex)
    print("匹配评分排名:")
    for i, (name, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    # 2.3 混合整数问题 - 应该选择 Gurobi 或 Genetic
    print_section("2.3 混合整数问题")
    mip = OptimizationProblem(
        objective_function_latex=r"\min x_1 + 2x_2 + 3x_3",
        objective_function_code="x1 + 2*x2 + 3*x3",
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="x3", type=VariableType.INTEGER, lower_bound=0, upper_bound=5),
        ]
    )
    scores = solver.get_match_scores(mip)
    print("匹配评分排名:")
    for i, (name, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        print(f"  {i+1}. {name}: {score:.4f}")


def test_solving():
    """测试求解功能"""
    print_header("测试3: 优化问题求解")
    
    solver = SolverAgent()
    
    # 3.1 简单二次问题 min x^2 + y^2
    print_section("3.1 简单二次问题: min x² + y²")
    print("理论最优解: x=0, y=0, 目标值=0")
    
    problem1 = OptimizationProblem(
        objective_function_latex=r"\min x^2 + y^2",
        objective_function_code="x**2 + y**2",
        variables=[
            VariableDefinition(name="x", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="y", lower_bound=-10, upper_bound=10),
        ]
    )
    
    solution = solver.solve(problem1)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: x={solution.decision_variables.get('x', 'N/A'):.6f}, "
          f"y={solution.decision_variables.get('y', 'N/A'):.6f}")
    print(f"目标值: {solution.objective_value:.6f}")
    print(f"求解时间: {solution.solving_time:.4f}s")
    print(f"可行: {solution.is_feasible}")
    
    # 3.2 Rosenbrock 函数（非凸）
    print_section("3.2 Rosenbrock 函数（非凸）")
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print("理论最优解: x=1, y=1, 目标值=0")
    
    problem2 = OptimizationProblem(
        objective_function_latex=r"\min (1-x)^2 + 100(y-x^2)^2",
        objective_function_code="(1-x)**2 + 100*(y-x**2)**2",
        variables=[
            VariableDefinition(name="x", lower_bound=-5, upper_bound=5),
            VariableDefinition(name="y", lower_bound=-5, upper_bound=5),
        ]
    )
    
    solution = solver.solve(problem2)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: x={solution.decision_variables.get('x', 'N/A'):.6f}, "
          f"y={solution.decision_variables.get('y', 'N/A'):.6f}")
    print(f"目标值: {solution.objective_value:.6f}")
    print(f"求解时间: {solution.solving_time:.4f}s")
    
    # 3.3 Rastrigin 函数（多峰非凸）
    print_section("3.3 Rastrigin 函数（多峰非凸）")
    print("f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))")
    print("理论最优解: x=0, y=0, 目标值=0")
    
    problem3 = OptimizationProblem(
        objective_function_latex=r"\min 20 + x^2 + y^2 - 10(\cos(2\pi x) + \cos(2\pi y))",
        objective_function_code="20 + x**2 + y**2 - 10*(math.cos(2*3.14159*x) + math.cos(2*3.14159*y))",
        variables=[
            VariableDefinition(name="x", lower_bound=-5.12, upper_bound=5.12),
            VariableDefinition(name="y", lower_bound=-5.12, upper_bound=5.12),
        ]
    )
    
    solution = solver.solve(problem3)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: x={solution.decision_variables.get('x', 'N/A'):.6f}, "
          f"y={solution.decision_variables.get('y', 'N/A'):.6f}")
    print(f"目标值: {solution.objective_value:.6f}")
    print(f"求解时间: {solution.solving_time:.4f}s")
    
    # 3.4 带约束的优化问题
    print_section("3.4 带约束的优化问题")
    print("min x + 2y, s.t. x + y >= 1, x,y ∈ [0,10]")
    print("理论最优解: x=1, y=0, 目标值=1")
    
    problem4 = OptimizationProblem(
        objective_function_latex=r"\min x + 2y",
        objective_function_code="x + 2*y",
        constraints_latex=[r"x + y \geq 1"],
        constraints_code=["x + y >= 1"],
        variables=[
            VariableDefinition(name="x", lower_bound=0, upper_bound=10),
            VariableDefinition(name="y", lower_bound=0, upper_bound=10),
        ]
    )
    
    solution = solver.solve(problem4)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: x={solution.decision_variables.get('x', 'N/A'):.6f}, "
          f"y={solution.decision_variables.get('y', 'N/A'):.6f}")
    print(f"目标值: {solution.objective_value:.6f}")
    print(f"求解时间: {solution.solving_time:.4f}s")
    
    # 3.5 0-1 背包问题
    print_section("3.5 0-1 背包问题")
    print("物品: 价值=[60,100,120], 重量=[10,20,30]")
    print("背包容量: 50")
    print("理论最优: 选择物品2和3，总价值=220")
    
    problem5 = OptimizationProblem(
        objective_function_latex=r"\max 60x_1 + 100x_2 + 120x_3",
        objective_function_code="60*x1 + 100*x2 + 120*x3",
        constraints_latex=[r"10x_1 + 20x_2 + 30x_3 \leq 50"],
        constraints_code=["10*x1 + 20*x2 + 30*x3 <= 50"],
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="x3", type=VariableType.BINARY),
        ],
        is_minimization=False  # 最大化问题
    )
    
    solution = solver.solve(problem5)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: x1={solution.decision_variables.get('x1', 'N/A')}, "
          f"x2={solution.decision_variables.get('x2', 'N/A')}, "
          f"x3={solution.decision_variables.get('x3', 'N/A')}")
    print(f"目标值（总价值）: {solution.objective_value:.0f}")
    print(f"求解时间: {solution.solving_time:.4f}s")


def test_specific_algorithm():
    """测试指定算法求解"""
    print_header("测试4: 指定算法求解")
    
    solver = SolverAgent()
    
    problem = OptimizationProblem(
        objective_function_latex=r"\min x^2 + y^2",
        objective_function_code="x**2 + y**2",
        variables=[
            VariableDefinition(name="x", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="y", lower_bound=-10, upper_bound=10),
        ]
    )
    
    algorithms_to_test = ["ConvexOptimizer", "PSOOptimizer", "GeneticOptimizer"]
    
    print("同一问题使用不同算法求解:")
    print(f"问题: min x² + y², 理论最优值=0\n")
    
    for algo_name in algorithms_to_test:
        try:
            solution = solver.solve_with_algorithm(problem, algo_name)
            print(f"{algo_name}:")
            print(f"  结果: x={solution.decision_variables.get('x', 0):.6f}, "
                  f"y={solution.decision_variables.get('y', 0):.6f}")
            print(f"  目标值: {solution.objective_value:.6f}")
            print(f"  时间: {solution.solving_time:.4f}s")
        except Exception as e:
            print(f"{algo_name}: 错误 - {e}")


def test_large_scale():
    """测试较大规模问题"""
    print_header("测试5: 较大规模问题")
    
    solver = SolverAgent()
    
    # 10维问题
    print_section("5.1 10维球函数 min Σxi²")
    
    n = 10
    variables = [
        VariableDefinition(name=f"x{i}", lower_bound=-10, upper_bound=10)
        for i in range(n)
    ]
    
    obj_code = " + ".join([f"x{i}**2" for i in range(n)])
    
    problem = OptimizationProblem(
        objective_function_latex=r"\min \sum_{i=1}^{10} x_i^2",
        objective_function_code=obj_code,
        variables=variables
    )
    
    solution = solver.solve(problem)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"目标值: {solution.objective_value:.6f} (理论最优=0)")
    print(f"求解时间: {solution.solving_time:.4f}s")
    
    # 20维问题
    print_section("5.2 20维球函数")
    
    n = 20
    variables = [
        VariableDefinition(name=f"x{i}", lower_bound=-10, upper_bound=10)
        for i in range(n)
    ]
    
    obj_code = " + ".join([f"x{i}**2" for i in range(n)])
    
    problem = OptimizationProblem(
        objective_function_latex=r"\min \sum_{i=1}^{20} x_i^2",
        objective_function_code=obj_code,
        variables=variables
    )
    
    solution = solver.solve(problem)
    print(f"选择算法: {solver.get_selected_algorithm()}")
    print(f"目标值: {solution.objective_value:.6f} (理论最优=0)")
    print(f"求解时间: {solution.solving_time:.4f}s")


def main():
    """运行所有测试"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "       Solver 模块功能测试".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        test_feature_extraction()
        test_algorithm_matching()
        test_solving()
        test_specific_algorithm()
        test_large_scale()
        
        print("\n" + "=" * 70)
        print("  ✅ 所有测试完成!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

