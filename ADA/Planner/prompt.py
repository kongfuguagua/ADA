# -*- coding: utf-8 -*-
"""
Planner Prompt 模板库
包含状态增广、问题建模等提示模板
"""


class PlannerPrompts:
    """Planner 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个专业的优化问题建模专家。你的任务是：
1. 理解用户的调度需求
2. 通过调用工具获取必要的信息
3. 将需求转化为精确的数学优化问题

你需要输出标准的优化问题定义，包括：
- 目标函数（最小化或最大化）
- 决策变量及其边界
- 约束条件

请使用 LaTeX 格式表达数学公式。"""

    # ============= 状态增广提示 =============
    STATE_AUGMENTATION_PROMPT = """## 当前任务
分析当前状态，决定是否需要调用工具获取更多信息。

## 当前状态
{current_state}

## 可用工具
{tool_descriptions}

## 已有的动作知识（参考）
{action_knowledge}

## 历史工具调用
{tool_history}

## 指令
请分析当前状态，判断：
1. 是否有足够的信息来建立优化模型？
2. 如果信息不足，需要调用哪个工具来获取什么信息？

如果信息充足，请回复 "FINISH"。
如果需要调用工具，请按以下格式回复：
```json
{{
    "thought": "我需要获取...信息，因为...",
    "tool_name": "工具名称",
    "tool_params": {{"参数名": "参数值"}}
}}
```
"""

    # ============= 问题建模提示 =============
    FORMULATION_PROMPT = """## 当前任务
基于收集到的信息，建立数学优化模型。

## 环境状态
{environment_state}

## 收集到的信息
{augmented_state}

## 相关任务知识（参考历史成功案例）
{task_knowledge}

{retry_context}

## 输出要求
请输出完整的优化问题定义，使用以下 JSON 格式：
```json
{{
    "objective_function_latex": "目标函数的 LaTeX 表达式",
    "objective_function_code": "目标函数的 Python 代码",
    "is_minimization": true,
    "constraints_latex": ["约束1的 LaTeX 表达式", "约束2..."],
    "constraints_code": ["约束1的 Python 代码", "约束2..."],
    "variables": [
        {{
            "name": "变量名",
            "type": "continuous/binary/integer",
            "lower_bound": 下界,
            "upper_bound": 上界,
            "description": "变量含义"
        }}
    ],
    "parameters": {{"参数名": 数值}},
    "modeling_rationale": "建模理由说明"
}}
```
"""

    # ============= 重试上下文模板 =============
    RETRY_CONTEXT_TEMPLATE = """## ⚠️ 上次尝试失败
失败类型: {feedback_type}
错误诊断: {diagnosis}
建议修复: {suggested_fix}

请根据以上反馈修正你的建模。"""

    # ============= 工具调用结果处理 =============
    TOOL_RESULT_TEMPLATE = """## 工具调用结果
工具: {tool_name}
输入: {tool_input}
输出: {tool_output}

请基于此结果更新你对问题的理解。"""

    @classmethod
    def build_augmentation_prompt(
        cls,
        current_state: str,
        tool_descriptions: str,
        action_knowledge: str,
        tool_history: str
    ) -> str:
        """构建状态增广提示"""
        return cls.STATE_AUGMENTATION_PROMPT.format(
            current_state=current_state,
            tool_descriptions=tool_descriptions,
            action_knowledge=action_knowledge,
            tool_history=tool_history
        )
    
    @classmethod
    def build_formulation_prompt(
        cls,
        environment_state: str,
        augmented_state: str,
        task_knowledge: str,
        feedback=None
    ) -> str:
        """构建问题建模提示"""
        retry_context = ""
        if feedback and feedback.needs_retry():
            retry_context = cls.RETRY_CONTEXT_TEMPLATE.format(
                feedback_type=feedback.feedback_type.value,
                diagnosis=feedback.diagnosis,
                suggested_fix=feedback.suggested_fix
            )
        
        return cls.FORMULATION_PROMPT.format(
            environment_state=environment_state,
            augmented_state=augmented_state,
            task_knowledge=task_knowledge,
            retry_context=retry_context
        )


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 PlannerPrompts:")
    
    # 测试状态增广提示
    prompt = PlannerPrompts.build_augmentation_prompt(
        current_state="负载: 100MW, 发电: 120MW",
        tool_descriptions="- weather_forecast: 获取天气预报\n- power_flow: 计算潮流",
        action_knowledge="当负载不确定时，建议先调用天气预报工具",
        tool_history="暂无"
    )
    print("状态增广提示:")
    print(prompt[:500] + "...")
    
    print("\n" + "="*50)
    
    # 测试问题建模提示
    prompt = PlannerPrompts.build_formulation_prompt(
        environment_state="负载: 100MW",
        augmented_state="天气: 晴天，温度: 25°C",
        task_knowledge="历史案例：最小化发电成本",
        feedback=None
    )
    print("问题建模提示:")
    print(prompt[:500] + "...")

