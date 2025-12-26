# -*- coding: utf-8 -*-
"""
Judger Prompt 模板库
包含评估、诊断等提示模板
"""


class JudgerPrompts:
    """Judger 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个优化解评估专家。你的任务是：
1. 评估优化解的质量和可行性
2. 检查解是否满足所有约束
3. 诊断问题并给出改进建议

你需要从物理可行性和逻辑合理性两个维度进行评估。"""

    # ============= 逻辑评估提示 =============
    LOGICAL_EVALUATION_PROMPT = """## 任务
评估以下优化解的逻辑合理性。

## 优化问题
目标函数: {objective_function}
约束条件: {constraints}
建模理由: {modeling_rationale}

## 求解结果
算法: {algorithm}
决策变量: {decision_variables}
目标值: {objective_value}

## 评估要求
请从以下方面评估：
1. 解是否在变量边界内？
2. 解是否满足约束条件？
3. 目标值是否合理？
4. 解是否具有实际意义？

请用 JSON 格式回答：
```json
{{
    "is_logical": true/false,
    "score": 0.0-1.0,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "comment": "总体评价"
}}
```
"""

    # ============= 错误诊断提示 =============
    ERROR_DIAGNOSIS_PROMPT = """## 任务
诊断优化求解失败的原因。

## 优化问题
目标函数: {objective_function}
约束条件: {constraints}
变量定义: {variables}

## 求解结果
算法: {algorithm}
求解状态: {solver_status}
错误信息: {error_message}

## 物理指标（如有）
{physical_metrics}

## 诊断要求
请分析失败原因：
1. 是模型定义问题（Planner 责任）还是算法问题（Solver 责任）？
2. 具体是什么导致了失败？
3. 如何修复？

请用 JSON 格式回答：
```json
{{
    "error_source": "planner/solver",
    "error_type": "错误类型",
    "root_cause": "根本原因",
    "suggested_fix": "修复建议",
    "confidence": 0.0-1.0
}}
```
"""

    # ============= 反馈生成提示 =============
    FEEDBACK_GENERATION_PROMPT = """## 任务
生成给 {target_agent} 的反馈信息。

## 问题诊断
错误来源: {error_source}
错误类型: {error_type}
根本原因: {root_cause}

## 反馈要求
请生成清晰、可操作的反馈：
1. 明确指出问题所在
2. 给出具体的修复建议
3. 提供参考信息

反馈内容:
"""

    @classmethod
    def build_logical_evaluation_prompt(
        cls,
        objective_function: str,
        constraints: str,
        modeling_rationale: str,
        algorithm: str,
        decision_variables: str,
        objective_value: str
    ) -> str:
        """构建逻辑评估提示"""
        return cls.LOGICAL_EVALUATION_PROMPT.format(
            objective_function=objective_function,
            constraints=constraints,
            modeling_rationale=modeling_rationale,
            algorithm=algorithm,
            decision_variables=decision_variables,
            objective_value=objective_value
        )
    
    @classmethod
    def build_error_diagnosis_prompt(
        cls,
        objective_function: str,
        constraints: str,
        variables: str,
        algorithm: str,
        solver_status: str,
        error_message: str,
        physical_metrics: str = "无"
    ) -> str:
        """构建错误诊断提示"""
        return cls.ERROR_DIAGNOSIS_PROMPT.format(
            objective_function=objective_function,
            constraints=constraints,
            variables=variables,
            algorithm=algorithm,
            solver_status=solver_status,
            error_message=error_message,
            physical_metrics=physical_metrics
        )

