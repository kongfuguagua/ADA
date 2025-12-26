# -*- coding: utf-8 -*-
"""
Summarizer Prompt 模板库
包含知识提炼、经验总结等提示模板
"""


class SummarizerPrompts:
    """Summarizer 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个知识提炼专家。你的任务是：
1. 分析成功的调度案例
2. 提炼可复用的知识模式
3. 生成结构化的知识条目

你需要从工具使用模式和问题建模模式两个维度提炼知识。"""

    # ============= 动作知识提炼提示 =============
    ACTION_KNOWLEDGE_PROMPT = """## 任务
从以下成功案例中提炼动作知识（工具使用模式）。

## 案例信息
用户指令: {user_instruction}
最终评分: {score}

## 工具调用链
{tool_chain}

## 提炼要求
请总结：
1. 在什么情况下应该使用这些工具？
2. 工具调用的最佳顺序是什么？
3. 有什么需要注意的要点？

请用以下格式输出知识条目：
```
当[场景描述]时，建议[工具调用策略]。
注意事项：[要点]
```
"""

    # ============= 任务知识提炼提示 =============
    TASK_KNOWLEDGE_PROMPT = """## 任务
从以下成功案例中提炼任务知识（问题建模模式）。

## 案例信息
用户指令: {user_instruction}
最终评分: {score}

## 问题建模
目标函数: {objective_function}
约束条件: {constraints}
建模理由: {modeling_rationale}

## 求解结果
算法: {algorithm}
目标值: {objective_value}

## 提炼要求
请总结：
1. 这类问题应该如何建模？
2. 目标函数应该包含哪些成分？
3. 需要考虑哪些约束？

请用以下格式输出知识条目：
```
对于[问题类型]，建议采用[建模方法]。
目标函数：[描述]
关键约束：[描述]
```
"""

    # ============= MCTS 节点评估提示 =============
    MCTS_EVALUATION_PROMPT = """## 任务
评估以下决策路径的价值。

## 决策路径
{decision_path}

## 最终结果
评分: {score}
是否成功: {success}

## 评估要求
请评估这条决策路径的价值（0-1）：
- 1.0: 非常有价值，应该强化
- 0.5: 一般，可以保留
- 0.0: 无价值，应该避免

评估结果:
"""

    @classmethod
    def build_action_knowledge_prompt(
        cls,
        user_instruction: str,
        score: float,
        tool_chain: str
    ) -> str:
        """构建动作知识提炼提示"""
        return cls.ACTION_KNOWLEDGE_PROMPT.format(
            user_instruction=user_instruction,
            score=score,
            tool_chain=tool_chain
        )
    
    @classmethod
    def build_task_knowledge_prompt(
        cls,
        user_instruction: str,
        score: float,
        objective_function: str,
        constraints: str,
        modeling_rationale: str,
        algorithm: str,
        objective_value: str
    ) -> str:
        """构建任务知识提炼提示"""
        return cls.TASK_KNOWLEDGE_PROMPT.format(
            user_instruction=user_instruction,
            score=score,
            objective_function=objective_function,
            constraints=constraints,
            modeling_rationale=modeling_rationale,
            algorithm=algorithm,
            objective_value=objective_value
        )

