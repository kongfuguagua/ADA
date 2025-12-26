# -*- coding: utf-8 -*-
"""
Solver Prompt 模板库
包含特征分析、算法选择等提示模板
"""


class SolverPrompts:
    """Solver 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个优化算法专家。你的任务是：
1. 分析优化问题的数学特征
2. 选择最适合的求解算法
3. 解释算法选择的理由

你需要考虑问题的凸性、规模、约束类型等因素。"""

    # ============= 特征分析提示 =============
    FEATURE_ANALYSIS_PROMPT = """## 任务
分析以下优化问题的数学特征。

## 优化问题
目标函数: {objective_function}
约束条件: {constraints}
变量定义: {variables}

## 分析要求
请判断以下特征：
1. 问题是否是凸优化问题？
2. 目标函数是否是线性的？
3. 非线性程度如何（0-1）？
4. 约束的紧迫程度如何（0-1）？
5. 是否包含整数/二元变量？

请用 JSON 格式回答：
```json
{{
    "is_convex": true/false,
    "is_linear": true/false,
    "non_linearity_score": 0.0-1.0,
    "constraint_stiffness": 0.0-1.0,
    "has_integer_variables": true/false,
    "analysis_rationale": "分析理由"
}}
```
"""

    # ============= 算法选择提示 =============
    ALGORITHM_SELECTION_PROMPT = """## 任务
根据问题特征选择最合适的优化算法。

## 问题特征
{features}

## 可用算法
{algorithms}

## 选择要求
请选择最适合的算法，并解释理由。

请用 JSON 格式回答：
```json
{{
    "selected_algorithm": "算法名称",
    "selection_rationale": "选择理由",
    "expected_performance": "预期性能"
}}
```
"""

    @classmethod
    def build_feature_analysis_prompt(
        cls,
        objective_function: str,
        constraints: str,
        variables: str
    ) -> str:
        """构建特征分析提示"""
        return cls.FEATURE_ANALYSIS_PROMPT.format(
            objective_function=objective_function,
            constraints=constraints,
            variables=variables
        )
    
    @classmethod
    def build_algorithm_selection_prompt(
        cls,
        features: str,
        algorithms: str
    ) -> str:
        """构建算法选择提示"""
        return cls.ALGORITHM_SELECTION_PROMPT.format(
            features=features,
            algorithms=algorithms
        )

