# -*- coding: utf-8 -*-
"""
Solver Agent 主入口
负责协调特征提取、算法匹配和求解执行
"""

from typing import List, Optional

from utils.const import OptimizationProblem, Solution
from utils.interact import BaseSolver, BaseSolverStrategy
from utils.logger import get_logger
from config import SystemConfig

from .feature import ProblemFeatureExtractor
from .matcher import AlgorithmMatcher
from .Template import (
    ConvexOptimizer,
    GurobiOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
    GradientOptimizer,
)

logger = get_logger("Solver")


class SolverAgent(BaseSolver):
    """
    求解智能体
    
    核心职责：
    1. 特征提取 - 分析优化问题的数学特征
    2. 算法匹配 - 选择最适合的求解算法
    3. 执行求解 - 运行选定算法并返回结果
    
    使用示例:
        solver = SolverAgent()
        solution = solver.solve(problem)
    """
    
    # 默认算法列表
    DEFAULT_ALGORITHMS = [
        ConvexOptimizer,
        GurobiOptimizer,
        PSOOptimizer,
        BayesianOptimizer,
        GeneticOptimizer,
        GradientOptimizer,
    ]
    
    def __init__(
        self,
        algorithms: List[BaseSolverStrategy] = None,
        llm=None,
        timeout: float = None,
        use_llm_features: bool = False
    ):
        """
        初始化 Solver
        
        Args:
            algorithms: 可用算法列表（None 则使用默认算法）
            llm: LLM 服务（用于辅助特征分析）
            timeout: 求解超时时间
            use_llm_features: 是否使用 LLM 辅助特征提取
        """
        config = SystemConfig()
        self.timeout = timeout or config.solver_timeout
        self.use_llm_features = use_llm_features
        
        # 初始化组件
        self._matcher = AlgorithmMatcher()
        self._feature_extractor = ProblemFeatureExtractor(llm=llm)
        
        # 注册算法
        if algorithms:
            for algo in algorithms:
                self._matcher.register(algo)
        else:
            self._register_default_algorithms()
        
        # 记录最后选择的算法
        self._selected_algorithm: str = ""
    
    def _register_default_algorithms(self) -> None:
        """注册默认算法"""
        for AlgoClass in self.DEFAULT_ALGORITHMS:
            try:
                self._matcher.register(AlgoClass())
            except Exception as e:
                logger.warning(f"注册算法 {AlgoClass.__name__} 失败: {e}")
    
    def register_strategy(self, strategy: BaseSolverStrategy) -> None:
        """注册求解策略"""
        self._matcher.register(strategy)
        logger.info(f"注册算法: {strategy.meta.name}")
    
    def solve(self, problem: OptimizationProblem) -> Solution:
        """
        求解优化问题
        
        流程：问题验证 -> 特征提取 -> 算法匹配 -> 执行求解
        
        Args:
            problem: 优化问题
        
        Returns:
            求解结果
        
        Raises:
            ValueError: 问题定义不完整或无效
        """
        # 0. 问题验证（鲁棒性检查）
        validation_error = self._validate_problem(problem)
        if validation_error:
            logger.warning(f"问题定义验证失败: {validation_error}")
            # 返回一个失败的 Solution，而不是抛出异常
            return Solution(
                is_feasible=False,
                algorithm_used="None",
                decision_variables={},
                objective_value=float('inf'),
                solving_time=0.0,
                solver_message=f"问题定义验证失败: {validation_error}"
            )
        
        logger.info("开始求解",
                   variables=len(problem.variables),
                   constraints=len(problem.constraints_latex))
        
        # 1. 特征提取
        if self.use_llm_features:
            features = self._feature_extractor.extract_with_llm(problem)
        else:
            features = self._feature_extractor.extract(problem)
        
        logger.debug("特征提取完成", features=features.to_dict())
        
        # 2. 算法匹配
        best_algo = self._matcher.match(features)
        self._selected_algorithm = best_algo.meta.name
        logger.info(f"选择算法: {self._selected_algorithm}")
        
        # 3. 执行求解
        try:
            solution = best_algo.solve(problem)
        except Exception as e:
            logger.error(f"求解器执行失败: {e}")
            return Solution(
                is_feasible=False,
                algorithm_used=self._selected_algorithm,
                decision_variables={},
                objective_value=float('inf'),
                solving_time=0.0,
                solver_message=f"求解器异常: {str(e)}"
            )
        
        logger.info("求解完成",
                   feasible=solution.is_feasible,
                   objective=solution.objective_value,
                   time=f"{solution.solving_time:.4f}s")
        
        return solution
    
    def _validate_problem(self, problem: OptimizationProblem) -> str:
        """
        验证问题定义的合理性
        
        Args:
            problem: 优化问题
        
        Returns:
            错误描述（如果有问题），否则返回空字符串
        """
        # 检查目标函数
        if not problem.objective_function_latex or len(problem.objective_function_latex.strip()) < 3:
            return "目标函数定义不完整或为空"
        
        # 检查变量
        if len(problem.variables) == 0:
            return "问题没有定义任何变量"
        
        # 检查变量名称唯一性
        var_names = [var.name for var in problem.variables]
        if len(var_names) != len(set(var_names)):
            return "变量名称重复"
        
        # 检查变量边界合理性
        for var in problem.variables:
            if var.lower_bound > var.upper_bound:
                return f"变量 {var.name} 的下界大于上界"
            if var.lower_bound == var.upper_bound and len(problem.constraints_latex) == 0:
                # 如果所有变量都是固定值且没有约束，问题可能过于简单
                pass
        
        # 如果只有一个变量且没有约束，可能是解析失败
        if len(problem.variables) == 1 and len(problem.constraints_latex) == 0:
            var = problem.variables[0]
            # 检查是否有合理的边界
            if var.lower_bound == float('-inf') and var.upper_bound == float('inf'):
                return "问题定义过于简单（可能解析失败）：只有一个无界变量且无约束"
        
        return ""
    
    def solve_with_algorithm(
        self,
        problem: OptimizationProblem,
        algorithm_name: str
    ) -> Solution:
        """
        使用指定算法求解（跳过自动匹配）
        
        Args:
            problem: 优化问题
            algorithm_name: 算法名称
        
        Returns:
            求解结果
        
        Raises:
            ValueError: 算法不存在
        """
        algo = self._matcher.get_algorithm(algorithm_name)
        if algo is None:
            available = self._matcher.list_algorithms()
            raise ValueError(f"算法 '{algorithm_name}' 不存在，可用算法: {available}")
        
        self._selected_algorithm = algorithm_name
        logger.info(f"使用指定算法: {algorithm_name}")
        
        return algo.solve(problem)
    
    def get_selected_algorithm(self) -> str:
        """获取选中的算法名称"""
        return self._selected_algorithm
    
    def list_algorithms(self) -> List[str]:
        """列出所有可用算法"""
        return self._matcher.list_algorithms()
    
    def get_algorithm_descriptions(self) -> str:
        """获取所有算法的描述"""
        return self._matcher.get_algorithm_info()
    
    def get_match_scores(self, problem: OptimizationProblem) -> dict:
        """
        获取问题与各算法的匹配评分（调试用）
        
        Args:
            problem: 优化问题
        
        Returns:
            算法名称到评分的映射
        """
        features = self._feature_extractor.extract(problem)
        return self._matcher.compute_scores(features.to_vector())


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition, VariableType
    
    print("=" * 60)
    print("测试 SolverAgent")
    print("=" * 60)
    
    # 创建 Solver
    solver = SolverAgent()
    print(f"\n已注册算法: {solver.list_algorithms()}")
    
    # 测试1: 凸二次问题
    print("\n" + "-" * 40)
    print("测试1: 凸二次问题 min x1² + x2²")
    print("-" * 40)
    
    convex_problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        objective_function_code="x1**2 + x2**2",
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
        ]
    )
    
    # 查看匹配评分
    scores = solver.get_match_scores(convex_problem)
    print("匹配评分:")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    solution = solver.solve(convex_problem)
    print(f"\n选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value:.6f}")
    
    # 测试2: 非凸问题
    print("\n" + "-" * 40)
    print("测试2: 非凸问题 min sin(x1) + cos(x2)")
    print("-" * 40)
    
    nonconvex_problem = OptimizationProblem(
        objective_function_latex=r"\min \sin(x_1) + \cos(x_2)",
        objective_function_code="math.sin(x1) + math.cos(x2)",
        variables=[
            VariableDefinition(name="x1", lower_bound=-3.14, upper_bound=3.14),
            VariableDefinition(name="x2", lower_bound=-3.14, upper_bound=3.14),
        ]
    )
    
    scores = solver.get_match_scores(nonconvex_problem)
    print("匹配评分:")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    solution = solver.solve(nonconvex_problem)
    print(f"\n选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value:.6f}")
    
    # 测试3: 混合整数问题
    print("\n" + "-" * 40)
    print("测试3: 混合整数问题 min x1 + 2*x2")
    print("-" * 40)
    
    mip_problem = OptimizationProblem(
        objective_function_latex=r"\min x_1 + 2x_2",
        objective_function_code="x1 + 2*x2",
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.INTEGER, lower_bound=0, upper_bound=5),
        ]
    )
    
    scores = solver.get_match_scores(mip_problem)
    print("匹配评分:")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    solution = solver.solve(mip_problem)
    print(f"\n选择算法: {solver.get_selected_algorithm()}")
    print(f"求解结果: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value:.6f}")
    
    # 测试4: 指定算法求解
    print("\n" + "-" * 40)
    print("测试4: 指定算法求解")
    print("-" * 40)
    
    solution = solver.solve_with_algorithm(convex_problem, "PSOOptimizer")
    print(f"使用 PSO 求解凸问题:")
    print(f"  结果: {solution.decision_variables}")
    print(f"  目标值: {solution.objective_value:.6f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

