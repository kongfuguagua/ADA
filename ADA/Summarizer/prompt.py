# -*- coding: utf-8 -*-
"""
Summarizer Prompt 模板库
包含知识提炼、经验总结等提示模板
"""


class SummarizerPrompts:
    """Summarizer 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个知识提炼专家，专门从成功的调度案例中提炼可复用的知识模式。

核心职责：
1. 分析成功的调度案例，识别关键成功因素
2. 提炼工具使用模式和问题建模模式
3. 生成结构化的知识条目，便于后续复用

关键要求：
- 重点关注导致成功的决策模式
- 提炼通用的、可迁移的知识
- 区分动作知识（工具使用）和任务知识（建模方法）
- 输出清晰、可执行的知识条目"""

    # ============= 动作知识提炼提示 =============
    ACTION_KNOWLEDGE_PROMPT = """## 任务
从以下成功案例中提炼动作知识（工具使用模式），用于指导未来的工具调用决策。

## 案例信息
用户指令: {user_instruction}
最终评分: {score}（评分越高表示越成功）

## 工具调用链
{tool_chain}

## 提炼要求
请分析工具调用链，提炼以下内容：

1. **触发条件**：在什么场景/情况下应该使用这些工具？
   - 用户指令的特征
   - 环境状态的特征
   - 信息缺失的情况

2. **调用策略**：工具调用的最佳顺序和组合是什么？
   - 哪些工具应该先调用？
   - 哪些工具可以并行调用？
   - 工具之间的依赖关系

3. **关键要点**：使用这些工具时需要注意什么？
   - 参数设置的关键点
   - 可能出现的错误和应对方法
   - 工具输出的解读方法

## 输出格式
请用清晰、结构化的格式输出知识条目：

**场景识别**：[描述什么情况下适用此策略]

**工具调用策略**：
1. 第一步：[工具名] - [目的] - [关键参数]
2. 第二步：[工具名] - [目的] - [关键参数]
...

**注意事项**：
- [要点1]
- [要点2]
...

**适用条件**：[明确说明此知识条目的适用边界]
"""

    # ============= 任务知识提炼提示 =============
    TASK_KNOWLEDGE_PROMPT = """## 任务
从以下成功案例中提炼任务知识（问题建模模式），用于指导未来的问题建模。

## 案例信息
用户指令: {user_instruction}
最终评分: {score}（评分越高表示建模越成功）

## 问题建模
目标函数: {objective_function}
约束条件: {constraints}
建模理由: {modeling_rationale}

## 求解结果
算法: {algorithm}
目标值: {objective_value}

## 提炼要求
请分析这个成功的建模案例，提炼以下内容：

1. **问题类型识别**：这是什么类型的优化问题？
   - 目标是什么（成本最小、安全最大化等）
   - 主要约束类型（平衡约束、边界约束、安全约束等）

2. **建模方法**：这类问题应该如何建模？
   - 目标函数的结构和组成
   - 关键决策变量的定义和维度
   - 约束条件的完整集合

3. **关键成功因素**：为什么这个建模方法成功了？
   - 目标函数设计的关键点
   - 约束条件的完整性和准确性
   - 参数设置的合理性

4. **可复用模式**：哪些部分可以应用到类似问题？
   - 通用的建模框架
   - 可调整的参数和变量
   - 需要注意的边界情况

## 输出格式
请用清晰、结构化的格式输出知识条目：

**问题类型**：[描述这类问题的特征]

**建模框架**：
- 目标函数结构：[描述目标函数的组成和形式]
- 决策变量：[变量名、维度、含义]
- 关键约束：
  1. [约束1名称]：[约束的数学表达和物理意义]
  2. [约束2名称]：[约束的数学表达和物理意义]
  ...

**参数设置**：
- [参数名]：[参数的含义和典型取值范围]
- ...

**建模要点**：
- [要点1：为什么这样建模]
- [要点2：需要注意什么]
- ...

**适用条件**：[明确说明此建模方法的适用场景和限制]
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

