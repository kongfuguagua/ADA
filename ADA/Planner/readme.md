# Planner Agent 设计规范

## 1. 核心职责

Planner 是系统的"大脑"，负责解决**动态场景解析向物理约束转化的歧义性**问题（论文 2.1 节）。它不直接求解，而是通过主动调用工具链消除环境不确定性，生成严格的数学规划模型。

## 2. 论文需求响应表

| 论文需求 | 实现状态 | 实现位置 |
|---------|---------|---------|
| 主动状态增广 (Active State Augmentation) | ✅ 已实现 | `core.py: _augment_state()` |
| 动作知识检索 (AK Retrieval) | ✅ 已实现 | `core.py: _get_action_knowledge()` |
| 任务知识检索 (TK Retrieval) | ✅ 已实现 | `core.py: _get_task_knowledge()` |
| 工具调用链建模 | ✅ 已实现 | `core.py: _tool_chain` |
| 数学规划建模 | ✅ 已实现 | `core.py: _formulate_problem()` |
| Self-Correction (错误修正) | ✅ 已实现 | `core.py: plan(retry_feedback)` |
| 状态演化链 $x_0 \to x_n$ | ✅ 已实现 | `core.py: _augment_state()` |

## 3. 核心逻辑：主动状态增广

过程建模为链式推导：$x_0 \xrightarrow{t_0} x_1 \dots \xrightarrow{t_n} x_n$

```
┌─────────────────────────────────────────────────────────────────┐
│                    Planner 工作流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 感知 (Perception)                                           │
│     └── 接收原始环境状态 x₀                                      │
│                                                                 │
│  2. 检索 (Retrieval)                                            │
│     └── 从 KnowledgeBase 检索动作知识 AK                         │
│                                                                 │
│  3. 增广 (Augmentation) [核心循环]                               │
│     ├── 基于 AK 决策是否需要调用工具                              │
│     ├── 执行工具调用 tᵢ (分析工具，非环境交互)                    │
│     └── 更新状态向量 xᵢ                                          │
│                                                                 │
│  4. 建模 (Formulation)                                          │
│     ├── 检索任务知识 TK                                          │
│     ├── 基于最终状态 xₙ 和 TK 生成优化问题                        │
│     └── 输出 OptimizationProblem                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 模块架构

```
Planner/
├── __init__.py          # 模块导出
├── readme.md            # 本文档
├── core.py              # 【主入口】PlannerAgent 类
├── prompt.py            # 【Prompt】所有 LLM 提示模板
│
└── tools/               # 【分析工具】为 Planner 决策服务
    ├── __init__.py      # 工具导出
    └── registry.py      # 工具注册表 + 分析工具实现
```

### 工具职责说明

**Planner 工具** (`tools/`) 是为 **Planner 决策服务** 的分析工具：
- 输出：统计数据、规则检测结果、趋势分析等
- 目的：帮助 Planner 理解环境状态，做出建模决策
- 特点：更少的原始数据，更多的分析结论

**区别于** `env/tools.py` 的环境交互工具：
- 输出：原始数据、执行结果
- 目的：与 Grid2Op 环境进行交互
- 特点：发送命令、获取数据

## 5. 接口定义

### 输入

```python
class EnvironmentState(BaseModel):
    user_instruction: str           # 用户指令
    real_data: Dict[str, Any]       # 实时数据
    extra_context: Dict[str, Any]   # 额外上下文
```

### 输出

```python
class OptimizationProblem(BaseModel):
    objective_function_latex: str   # 目标函数 LaTeX
    objective_function_code: str    # 目标函数代码
    constraints_latex: List[str]    # 约束条件 LaTeX
    constraints_code: List[str]     # 约束条件代码
    variables: List[VariableDefinition]  # 变量定义
    parameters: Dict[str, float]    # 参数
    modeling_rationale: str         # 建模理由
```

## 6. 异常处理

若收到 Judger 的 `MODEL_ERROR` 反馈，Planner 需：
1. 读取反馈中的 `diagnosis` 和 `suggested_fix`
2. 在下一次 Prompt 中增加"修正约束"的指令
3. 进行 Self-Correction

## 7. 使用示例

```python
from Planner import PlannerAgent
from utils.const import EnvironmentState

# 创建 Planner
planner = PlannerAgent(llm=llm, kb=kb)

# 规划
state = EnvironmentState(
    user_instruction="优化发电调度，最小化成本",
    real_data={"load": 120.0}
)
problem = planner.plan(state)

print(f"目标函数: {problem.objective_function_latex}")
print(f"变量数: {len(problem.variables)}")
```
