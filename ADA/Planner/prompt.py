# -*- coding: utf-8 -*-
"""
Planner Prompt 模板库
包含状态增广、问题建模等提示模板
"""


class PlannerPrompts:
    """Planner 提示模板集合"""
    
    # ============= 系统提示 =============
    SYSTEM_PROMPT = """你是一个专业的优化问题建模专家，专门处理电力系统调度优化问题。

核心职责：
1. 理解用户的调度需求
2. 通过调用工具获取必要的信息（负载、发电机参数、线路信息等）
3. 将需求转化为精确的数学优化问题

关键要求：
- 必须输出严格符合JSON格式的问题定义
- 所有变量必须明确定义维度（标量、向量、矩阵）
- 约束条件必须完整且可执行
- 参数必须包含所有必要的数值数据
- 目标函数和约束的代码必须可以直接运行

输出格式：必须使用纯JSON格式，不要添加任何解释性文字。"""

    # ============= 状态增广提示 =============
    STATE_AUGMENTATION_PROMPT = """## 当前任务
分析当前状态，决定是否需要调用工具获取更多信息来建立优化模型。

## 当前状态
{current_state}

## 可用工具
{tool_descriptions}

## 已有的动作知识（参考历史成功案例）
{action_knowledge}

## 历史工具调用
{tool_history}

## 建模所需的关键信息
建立电力系统优化模型通常需要：
1. **发电机信息**：数量、出力上下限、成本系数
2. **负载信息**：当前负载值、负载分布
3. **线路信息**：线路数量、负载率、容量限制
4. **网络拓扑**：功率流关系（如需要）
5. **运行约束**：安全裕度、备用要求等

## 决策规则
请分析当前状态，判断：
1. 是否已收集到所有必要的建模信息？
2. 如果信息不足，需要调用哪个工具获取什么信息？
3. 工具调用的优先级是什么？

## 输出格式
如果信息充足，请回复：
```
FINISH
```

如果需要调用工具，请严格按以下JSON格式回复（不要添加markdown代码块标记）：
{{
    "thought": "我需要获取...信息，因为...",
    "tool_name": "工具名称",
    "tool_params": {{"参数名": "参数值"}}
}}

注意：JSON格式必须完全正确，可以直接解析。"""

    # ============= 问题建模提示 =============
    FORMULATION_PROMPT = """## 当前任务
基于收集到的信息，建立精确的数学优化模型。

## ⚠️ 重要：环境特征信息（必须参考）
{environment_features}

**请务必基于上述环境特征信息来确定：**
- 决策变量的数量和维度（如：n_g个发电机 → n_g个决策变量 p_0, p_1, ..., p_{{n_g-1}}）
- 约束条件的类型和数量
- 参数的具体数值范围

## 环境状态信息
{environment_state}

## 收集到的增广信息
{augmented_state}

## 相关任务知识（参考历史成功案例）
{task_knowledge}

{retry_context}

## 建模要求（重要！）

### 1. 变量定义（重要！）
- **向量变量必须拆分为多个标量变量**：如果变量是向量（如p是n_g维向量），必须定义为n_g个独立的标量变量（如p_0, p_1, ..., p_{{n_g-1}}）
- 每个变量必须有唯一的名称、类型、边界和描述
- 边界值必须基于实际物理约束（如：发电机出力不能为负，通常≥0）
- 变量名建议使用有意义的名称（如gen_power_0, gen_power_1等）

示例：如果p是6维向量（6个发电机的出力），应该定义为：
```json
"variables": [
    {{"name": "p_0", "type": "continuous", "lower_bound": 20.0, "upper_bound": 60.0, "description": "发电机0的出力(MW)"}},
    {{"name": "p_1", "type": "continuous", "lower_bound": 30.0, "upper_bound": 90.0, "description": "发电机1的出力(MW)"}},
    ...
]
```

### 2. 目标函数
- LaTeX表达式必须使用正确的数学符号（可以使用向量符号，如\\sum_{{i=1}}^{{n_g}}）
- **Python代码必须使用拆分后的变量名**（如p_0, p_1等，而不是p[0], p[1]）
- Python代码必须可以直接执行，使用变量名和参数名
- 确保所有参数都在parameters中定义
- 代码中可以使用列表推导式，但最终要返回标量值

示例：
```python
def objective(p_0, p_1, p_2, c, lambda_val, rho, theta):
    # 将标量变量组织成列表
    p = [p_0, p_1, p_2]
    cost = sum(c[i] * p[i] for i in range(len(p))])
    penalty = lambda_val * sum(max(0, rho[j] - theta) for j in range(len(rho)))
    return cost + penalty
```

### 3. 约束条件
- 每个约束必须同时提供LaTeX和Python代码
- **Python代码必须使用拆分后的变量名**（如p_0, p_1等）
- Python代码必须返回约束值（等式约束返回0，不等式约束返回≤0的值）
- 必须包含：
  - **功率平衡约束**：\sum p_i = \sum load_k
  - **变量边界约束**：已在变量定义中，但可在约束中再次明确
  - **物理约束**：线路负载率≤1、发电机出力限制等
- 约束代码应该检查所有相关变量

示例：
```python
def power_balance(p_0, p_1, p_2, load):
    p = [p_0, p_1, p_2]
    return sum(p) - sum(load)  # 等式约束，返回0表示满足
```

### 4. 参数定义
- 必须包含所有在目标函数和约束中使用的参数
- 列表参数用数组表示：{{"c": [0.05, 0.06, ...]}}
- 字典参数用对象表示：{{"gen_limits": {{"lower": [...], "upper": [...]}}}}

### 5. JSON格式要求
- 必须输出纯JSON，不要添加markdown代码块标记外的任何文字
- 所有字符串中的反斜杠必须转义（如 \\min 而不是 \min）
- 确保JSON语法完全正确（无尾随逗号、引号匹配等）

## 输出格式（严格遵循）

```json
{{
    "objective_function_latex": "\\\\min \\\\sum_{{i=1}}^{{n_g}} c_i p_i + \\\\lambda \\\\sum_{{j}} \\\\max(0, \\\\rho_j - \\\\theta)",
    "objective_function_code": "def objective(p, rho, c, lambda_val, theta):\\n    cost = sum(c[i] * p[i] for i in range(len(p)))\\n    penalty = lambda_val * sum(max(0, rho[j] - theta) for j in range(len(rho)))\\n    return cost + penalty",
    "is_minimization": true,
    "constraints_latex": [
        "\\\\sum_{{i=1}}^{{n_g}} p_i = \\\\sum_{{k=1}}^{{n_l}} d_k",
        "p_i^{{ \\\\min }} \\\\leq p_i \\\\leq p_i^{{ \\\\max }}, \\\\forall i"
    ],
    "constraints_code": [
        "def power_balance(p, load):\\n    return sum(p) - sum(load)",
        "def gen_limits(p, p_min, p_max):\\n    return [p[i] - p_max[i] for i in range(len(p))] + [p_min[i] - p[i] for i in range(len(p))]"
    ],
    "variables": [
        {{
            "name": "p_0",
            "type": "continuous",
            "lower_bound": 20.0,
            "upper_bound": 60.0,
            "description": "发电机0的出力(MW)"
        }},
        {{
            "name": "p_1",
            "type": "continuous",
            "lower_bound": 30.0,
            "upper_bound": 90.0,
            "description": "发电机1的出力(MW)"
        }}
    ],
    "parameters": {{
        "c": [0.05, 0.06, 0.07],
        "load": [100.0, 120.0],
        "lambda_val": 100.0,
        "theta": 0.9,
        "p_min": [20.0, 30.0],
        "p_max": [60.0, 90.0]
    }},
    "modeling_rationale": "最小化发电成本，同时惩罚高负载线路，确保系统安全运行"
}}
```

## 验证清单（输出前检查）
- [ ] JSON格式完全正确，可以通过json.loads()解析
- [ ] **向量变量已拆分为多个标量变量**（如p是6维向量→p_0, p_1, ..., p_5）
- [ ] 所有变量都有唯一的名称和明确的描述
- [ ] 所有参数都在parameters中定义
- [ ] Python代码使用拆分后的变量名（p_0, p_1等），可以直接执行
- [ ] 约束条件完整（功率平衡、边界约束、物理约束）
- [ ] 边界值符合物理意义（如出力≥0）
- [ ] 变量数量与问题规模匹配（如n_g个发电机→n_g个变量）

现在请输出优化问题定义（仅JSON，无其他文字）："""

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
        environment_features: str = "",
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
            environment_features=environment_features,
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

