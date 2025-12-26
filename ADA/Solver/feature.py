# -*- coding: utf-8 -*-
"""
问题特征提取模块
将优化问题映射到特征空间，用于算法匹配
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from utils.const import OptimizationProblem, VariableType
from utils.logger import get_logger

logger = get_logger("Feature")


@dataclass
class ProblemFeatures:
    """
    优化问题特征
    对应论文公式中的特征向量 φ
    所有特征均归一化到 [0, 1] 区间
    """
    
    # ===== 规模特征 =====
    variable_count: int = 0
    constraint_count: int = 0
    
    # ===== 核心特征 (均为 [0-1] 分数) =====
    non_convexity_score: float = 0.0    # 非凸性程度
    non_linearity_score: float = 0.0    # 非线性程度
    constraint_stiffness: float = 0.0   # 约束紧迫度
    discreteness_score: float = 0.0     # 离散性程度
    scale_score: float = 0.0            # 规模复杂度
    
    # ===== 结构特征 (辅助判断) =====
    has_quadratic_terms: bool = False
    has_exponential_terms: bool = False
    has_trigonometric_terms: bool = False
    has_integer_variables: bool = False
    has_binary_variables: bool = False
    
    def to_vector(self) -> List[float]:
        """
        转换为 5 维特征向量
        用于计算算法对齐评分
        
        Returns:
            [非凸性, 非线性, 约束紧迫度, 离散性, 规模]
        """
        return [
            self.non_convexity_score,
            self.non_linearity_score,
            self.constraint_stiffness,
            self.discreteness_score,
            self.scale_score,
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于日志和调试）"""
        return {
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
            "non_convexity_score": round(self.non_convexity_score, 3),
            "non_linearity_score": round(self.non_linearity_score, 3),
            "constraint_stiffness": round(self.constraint_stiffness, 3),
            "discreteness_score": round(self.discreteness_score, 3),
            "scale_score": round(self.scale_score, 3),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProblemFeatures':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class ProblemFeatureExtractor:
    """
    问题特征提取器
    分析优化问题的数学特征，输出归一化的特征向量
    """
    
    # ===== 模式定义 =====
    
    # 非线性模式及其权重
    NONLINEAR_PATTERNS = {
        r'\*\*\s*[2-9]': 0.3,       # 幂次 x**2
        r'\^[2-9]': 0.3,            # 幂次 x^2
        r'sin\s*\(': 0.5,           # 三角函数
        r'cos\s*\(': 0.5,
        r'tan\s*\(': 0.5,
        r'exp\s*\(': 0.4,           # 指数函数
        r'log\s*\(': 0.4,           # 对数函数
        r'sqrt\s*\(': 0.3,          # 平方根
        r'\w+\s*\*\s*\w+': 0.2,     # 变量乘积
    }
    
    # 非凸模式及其权重
    NONCONVEX_PATTERNS = {
        r'sin\s*\(': 0.6,           # 三角函数（非凸）
        r'cos\s*\(': 0.6,
        r'-\s*\w+\s*\*\*\s*2': 0.7, # 负的平方项
        r'max\s*\(': 0.4,           # max 函数
        r'min\s*\(': 0.4,           # min 函数
        r'abs\s*\(': 0.3,           # 绝对值
        r'\*\*\s*[3-9]': 0.5,       # 高次幂
        r'\*\*\s*\(': 0.6,          # 变量幂次
    }
    
    def __init__(self, llm=None):
        """
        初始化特征提取器
        
        Args:
            llm: LLM 服务（可选，用于辅助分析）
        """
        self.llm = llm
    
    def extract(self, problem: OptimizationProblem) -> ProblemFeatures:
        """
        提取问题特征
        
        Args:
            problem: 优化问题
        
        Returns:
            问题特征（所有分数归一化到 [0,1]）
        """
        features = ProblemFeatures()
        
        # 1. 基础统计
        features.variable_count = len(problem.variables)
        features.constraint_count = len(problem.constraints_latex)
        
        # 2. 分析变量类型 -> 离散性
        features.discreteness_score = self._analyze_discreteness(problem, features)
        
        # 3. 分析表达式 -> 非凸性、非线性
        expr = self._get_all_expressions(problem)
        features.non_linearity_score = self._analyze_nonlinearity(expr, features)
        features.non_convexity_score = self._analyze_nonconvexity(expr, features)
        
        # 4. 分析约束 -> 约束紧迫度
        features.constraint_stiffness = self._analyze_constraint_stiffness(problem, features)
        
        # 5. 分析规模 -> 规模复杂度
        features.scale_score = self._analyze_scale(features)
        
        logger.debug("特征提取完成", features=features.to_dict())
        
        return features
    
    def _get_all_expressions(self, problem: OptimizationProblem) -> str:
        """获取所有表达式的合并字符串"""
        parts = [
            problem.objective_function_latex,
            problem.objective_function_code,
            " ".join(problem.constraints_latex),
            " ".join(problem.constraints_code),
        ]
        return " ".join(parts).lower()
    
    def _analyze_discreteness(
        self,
        problem: OptimizationProblem,
        features: ProblemFeatures
    ) -> float:
        """
        分析离散性程度
        
        Returns:
            离散性分数 [0, 1]
        """
        if not problem.variables:
            return 0.0
        
        integer_count = 0
        binary_count = 0
        
        for var in problem.variables:
            if var.type == VariableType.INTEGER:
                features.has_integer_variables = True
                integer_count += 1
            elif var.type == VariableType.BINARY:
                features.has_binary_variables = True
                binary_count += 1
        
        # 二元变量权重更高
        total = len(problem.variables)
        score = (binary_count * 1.0 + integer_count * 0.7) / total
        
        return min(1.0, score)
    
    def _analyze_nonlinearity(self, expr: str, features: ProblemFeatures) -> float:
        """
        分析非线性程度
        
        Returns:
            非线性分数 [0, 1]
        """
        score = 0.0
        
        for pattern, weight in self.NONLINEAR_PATTERNS.items():
            matches = re.findall(pattern, expr, re.IGNORECASE)
            if matches:
                score += weight * min(len(matches), 3)  # 限制单模式贡献
                
                # 更新结构特征
                if 'sin' in pattern or 'cos' in pattern or 'tan' in pattern:
                    features.has_trigonometric_terms = True
                elif 'exp' in pattern:
                    features.has_exponential_terms = True
                elif '**' in pattern or '^' in pattern:
                    features.has_quadratic_terms = True
        
        return min(1.0, score)
    
    def _analyze_nonconvexity(self, expr: str, features: ProblemFeatures) -> float:
        """
        分析非凸性程度
        
        Returns:
            非凸性分数 [0, 1]
            0 = 凸问题
            1 = 高度非凸
        """
        score = 0.0
        
        for pattern, weight in self.NONCONVEX_PATTERNS.items():
            matches = re.findall(pattern, expr, re.IGNORECASE)
            if matches:
                score += weight * min(len(matches), 2)
        
        # 如果有三角函数，非凸性至少为 0.5
        if features.has_trigonometric_terms:
            score = max(score, 0.5)
        
        return min(1.0, score)
    
    def _analyze_constraint_stiffness(
        self,
        problem: OptimizationProblem,
        features: ProblemFeatures
    ) -> float:
        """
        分析约束紧迫度
        
        考虑：
        - 约束数量与变量数量的比率
        - 变量边界的紧度
        - 等式约束的存在
        
        Returns:
            约束紧迫度分数 [0, 1]
        """
        if features.variable_count == 0:
            return 0.0
        
        score = 0.0
        
        # 1. 约束密度
        constraint_ratio = features.constraint_count / features.variable_count
        score += min(0.4, constraint_ratio * 0.2)
        
        # 2. 边界紧度
        tight_bounds = 0
        for var in problem.variables:
            lb, ub = var.lower_bound, var.upper_bound
            if lb != float('-inf') and ub != float('inf'):
                range_size = ub - lb
                if range_size < 10:
                    tight_bounds += 1
                elif range_size < 100:
                    tight_bounds += 0.5
        
        if problem.variables:
            score += 0.3 * (tight_bounds / len(problem.variables))
        
        # 3. 等式约束（更严格）
        all_constraints = " ".join(problem.constraints_latex + problem.constraints_code)
        equality_count = len(re.findall(r'[^<>!]=', all_constraints))
        if features.constraint_count > 0:
            score += 0.3 * min(1.0, equality_count / features.constraint_count)
        
        return min(1.0, score)
    
    def _analyze_scale(self, features: ProblemFeatures) -> float:
        """
        分析规模复杂度
        
        Returns:
            规模分数 [0, 1]
        """
        # 变量数归一化（假设 100 为大规模）
        var_score = min(1.0, features.variable_count / 100)
        
        # 约束数归一化
        con_score = min(1.0, features.constraint_count / 50)
        
        # 综合（变量权重更高）
        return 0.7 * var_score + 0.3 * con_score
    
    def extract_with_llm(self, problem: OptimizationProblem) -> ProblemFeatures:
        """
        使用 LLM 辅助提取特征（更准确但更慢）
        
        Args:
            problem: 优化问题
        
        Returns:
            问题特征
        """
        from .prompt import SolverPrompts
        
        # 先用规则提取基础特征
        features = self.extract(problem)
        
        if not self.llm:
            return features
        
        # 用 LLM 验证和补充
        prompt = SolverPrompts.build_feature_analysis_prompt(
            objective_function=problem.objective_function_latex,
            constraints=str(problem.constraints_latex),
            variables=str([v.name for v in problem.variables])
        )
        
        try:
            response = self.llm.chat(prompt)
            
            # 解析 JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str.strip())
            
            # 更新特征（取 LLM 和规则的较大值）
            if "non_convexity_score" in data:
                features.non_convexity_score = max(
                    features.non_convexity_score,
                    float(data["non_convexity_score"])
                )
            if "non_linearity_score" in data:
                features.non_linearity_score = max(
                    features.non_linearity_score,
                    float(data["non_linearity_score"])
                )
            if "constraint_stiffness" in data:
                features.constraint_stiffness = max(
                    features.constraint_stiffness,
                    float(data["constraint_stiffness"])
                )
            
            logger.debug("LLM 特征分析完成", llm_data=data)
            
        except Exception as e:
            logger.warning(f"LLM 特征分析失败: {e}")
        
        return features


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("=" * 60)
    print("测试 ProblemFeatureExtractor")
    print("=" * 60)
    
    extractor = ProblemFeatureExtractor()
    
    # 测试1: 线性问题
    print("\n" + "-" * 40)
    print("测试1: 线性规划问题")
    print("-" * 40)
    
    linear_problem = OptimizationProblem(
        objective_function_latex=r"\min 2x_1 + 3x_2",
        objective_function_code="2*x1 + 3*x2",
        constraints_latex=[r"x_1 + x_2 \leq 10", r"x_1 \geq 0"],
        constraints_code=["x1 + x2 <= 10", "x1 >= 0"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ]
    )
    
    features = extractor.extract(linear_problem)
    print(f"特征: {features.to_dict()}")
    print(f"特征向量: {features.to_vector()}")
    
    # 测试2: 凸二次问题
    print("\n" + "-" * 40)
    print("测试2: 凸二次问题")
    print("-" * 40)
    
    quadratic_problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        objective_function_code="x1**2 + x2**2",
        constraints_latex=[r"x_1 + x_2 = 1"],
        constraints_code=["x1 + x2 == 1"],
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
        ]
    )
    
    features = extractor.extract(quadratic_problem)
    print(f"特征: {features.to_dict()}")
    print(f"特征向量: {features.to_vector()}")
    
    # 测试3: 非凸问题（三角函数）
    print("\n" + "-" * 40)
    print("测试3: 非凸问题（三角函数）")
    print("-" * 40)
    
    nonconvex_problem = OptimizationProblem(
        objective_function_latex=r"\min \sin(x_1) + \cos(x_2)",
        objective_function_code="sin(x1) + cos(x2)",
        variables=[
            VariableDefinition(name="x1", lower_bound=-3.14, upper_bound=3.14),
            VariableDefinition(name="x2", lower_bound=-3.14, upper_bound=3.14),
        ]
    )
    
    features = extractor.extract(nonconvex_problem)
    print(f"特征: {features.to_dict()}")
    print(f"特征向量: {features.to_vector()}")
    print(f"有三角函数: {features.has_trigonometric_terms}")
    
    # 测试4: 混合整数问题
    print("\n" + "-" * 40)
    print("测试4: 混合整数问题")
    print("-" * 40)
    
    mip_problem = OptimizationProblem(
        objective_function_latex=r"\min \sum c_i x_i",
        objective_function_code="sum(c[i] * x[i] for i in range(n))",
        constraints_latex=[r"\sum x_i = 1"],
        constraints_code=["sum(x) == 1"],
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="x3", type=VariableType.INTEGER, lower_bound=0, upper_bound=10),
        ]
    )
    
    features = extractor.extract(mip_problem)
    print(f"特征: {features.to_dict()}")
    print(f"特征向量: {features.to_vector()}")
    print(f"有整数变量: {features.has_integer_variables}")
    print(f"有二元变量: {features.has_binary_variables}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
