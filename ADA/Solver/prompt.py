# -*- coding: utf-8 -*-
"""
Solver Prompt 模板库
包含特征分析、算法选择等提示模板
"""


class SolverPrompts:
    """Solver 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个优化算法专家，专门分析电力系统优化问题的数学特征。

核心职责：
1. 分析优化问题的数学特征（凸性、非线性、约束类型、规模等）
2. 评估问题的求解难度和算法适应性
3. 提供准确的特征评分（0-1区间）

关键要求：
- 必须仔细分析目标函数和约束的数学性质
- 考虑变量维度对求解复杂度的影响
- 评估约束的紧致程度和可行性
- 输出严格符合JSON格式的分析结果"""

    # ============= 特征分析提示 =============
    FEATURE_ANALYSIS_PROMPT = """## 任务
分析以下优化问题的数学特征，用于算法匹配。

## 优化问题
目标函数: {objective_function}
约束条件: {constraints}
变量定义: {variables}

## 分析维度

### 1. 非凸性程度 (non_convexity_score: 0.0-1.0)
- 0.0: 完全凸问题（线性或二次凸函数）
- 0.3: 轻微非凸（如max函数、分段线性）
- 0.7: 中等非凸（如非凸二次项）
- 1.0: 高度非凸（多峰、非凸约束）

判断依据：
- 目标函数是否包含max/min、绝对值等非凸操作
- 约束是否包含非凸集合（如整数约束、逻辑约束）

### 2. 非线性程度 (non_linearity_score: 0.0-1.0)
- 0.0: 完全线性
- 0.3: 轻微非线性（如二次项、简单非线性）
- 0.7: 中等非线性（如多项式、分式）
- 1.0: 高度非线性（如指数、三角函数、复杂组合）

判断依据：
- 目标函数和约束中的非线性项占比
- 非线性项的复杂度

### 3. 约束紧迫度 (constraint_stiffness: 0.0-1.0)
- 0.0: 约束非常宽松，可行域大
- 0.5: 约束适中，有合理的可行域
- 1.0: 约束非常紧，可行域很小或接近不可行

判断依据：
- 变量边界是否接近（如[0, 100] vs [50, 55]）
- 等式约束的数量和复杂度
- 约束之间的冲突程度

## 输出要求
请用严格JSON格式回答（仅JSON，无其他文字）：
{{
    "non_convexity_score": 0.0,
    "non_linearity_score": 0.0,
    "constraint_stiffness": 0.0,
    "analysis": "详细分析理由，说明每个评分的依据"
}}

注意：所有分数必须是0.0到1.0之间的浮点数。"""

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
