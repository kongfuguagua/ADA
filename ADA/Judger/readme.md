# Judger Agent 设计规范

## 1. 核心职责

Judger 是系统的"质检员"，负责建立**物理安全与逻辑合理的融合评价体系**，并进行**故障溯源**。它是实现端到端闭环优化的关键（论文 2.3 节）。

## 2. 论文需求响应表

| 论文需求 | 实现状态 | 实现位置 |
|---------|---------|---------|
| 物理仿真评估 R_ENV | ✅ 已实现 | `Reward/phy_reward.py` |
| LLM-as-a-Judge 评估 R_LLM | ✅ 已实现 | `Reward/llm_reward.py` |
| 混合评分 R = α·R_ENV + (1-α)·R_LLM | ✅ 已实现 | `Debug/feedback.py` |
| 故障溯源 (Model Error vs Solver Error) | ✅ 已实现 | `Debug/feedback.py: diagnose_with_llm()` |
| 结构化反馈生成 | ✅ 已实现 | `Debug/feedback.py: FeedbackGenerator` |
| 硬约束违规检测 | ✅ 已实现 | `Reward/phy_reward.py` |
| 问题定义完整性检查 | ✅ 已实现 | `Debug/feedback.py: _check_problem_completeness()` |

## 3. 核心逻辑：混合评分与反馈

```
┌─────────────────────────────────────────────────────────────────┐
│                    Judger 工作流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: OptimizationProblem + Solution                           │
│                                                                 │
│  1. 物理仿真 (Physical Simulation)                              │
│     ├── 检查变量边界约束                                         │
│     ├── 验证等式/不等式约束                                      │
│     ├── 检测硬约束违规（电压越限等）                              │
│     └── 计算物理评分 R_ENV ∈ [0, 1]                             │
│                                                                 │
│  2. 逻辑评估 (LLM-as-a-Judge)                                   │
│     ├── 评估解的可解释性                                         │
│     ├── 检查逻辑合理性                                           │
│     └── 计算逻辑评分 R_LLM ∈ [0, 1]                             │
│                                                                 │
│  3. 综合打分                                                    │
│     └── R = α · R_ENV + (1-α) · R_LLM                          │
│                                                                 │
│  4. 故障溯源 (若失败)                                           │
│     ├── MODEL_ERROR → Planner 责任（约束漏写/错写）              │
│     ├── SOLVER_ERROR → Solver 责任（陷入局部最优）               │
│     └── 生成结构化反馈 Feedback                                  │
│                                                                 │
│  输出: Feedback                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 模块架构

```
Judger/
├── __init__.py          # 模块导出
├── readme.md            # 本文档
├── core.py              # 【主入口】JudgerAgent 类
├── prompt.py            # 【Prompt】所有 LLM 提示模板
│
├── Reward/              # 【评分模块】
│   ├── __init__.py
│   ├── base_reward.py   # 评分基类
│   ├── phy_reward.py    # 物理评分（仿真环境）
│   └── llm_reward.py    # 逻辑评分（LLM-as-a-Judge）
│
└── Debug/               # 【诊断模块】
    ├── __init__.py
    └── feedback.py      # 反馈生成 + 故障诊断
```

## 5. 接口定义

### 输入

```python
# OptimizationProblem (来自 Planner)
# Solution (来自 Solver)
```

### 输出

```python
class Feedback(BaseModel):
    feedback_type: FeedbackType  # PASSED / MODEL_ERROR / SOLVER_ERROR / RUNTIME_ERROR
    score: float                  # 综合评分 [0, 1]
    physical_metrics: Dict        # 物理指标
    diagnosis: str                # 诊断信息
    error_source: str             # 错误来源 (planner/solver)
    suggested_fix: str            # 修复建议
```

### FeedbackType 枚举

```python
class FeedbackType(str, Enum):
    PASSED = "passed"              # 通过评估
    MODEL_ERROR = "model_error"    # 模型定义错误（Planner 责任）
    SOLVER_ERROR = "solver_error"  # 求解器错误（Solver 责任）
    RUNTIME_ERROR = "runtime_error"  # 运行时错误
```

## 6. 评分公式

综合评分：
$$R = \alpha \cdot W \cdot R_{ENV} + (1-\alpha) \cdot R_{LLM}$$

其中：
- $\alpha$：物理评分权重（默认 0.7）
- $W$：约束满足度加权矩阵
- $R_{ENV}$：物理仿真评分
- $R_{LLM}$：LLM 逻辑评分

## 7. 故障溯源逻辑

| 错误现象 | 诊断结果 | 责任归属 |
|---------|---------|---------|
| 约束无法满足 | 约束定义不合理 | Planner (MODEL_ERROR) |
| 变量越界 | 边界设置错误 | Planner (MODEL_ERROR) |
| 目标值异常 | 目标函数定义错误 | Planner (MODEL_ERROR) |
| 收敛到局部最优 | 算法选择不当 | Solver (SOLVER_ERROR) |
| 求解超时 | 问题规模过大 | Solver (SOLVER_ERROR) |
| 数值不稳定 | 算法实现问题 | Solver (SOLVER_ERROR) |

## 8. 使用示例

```python
from Judger import JudgerAgent
from utils.const import OptimizationProblem, Solution

# 创建 Judger
judger = JudgerAgent(simulator=simulator, llm=llm)

# 评估
feedback = judger.evaluate(problem, solution)

print(f"评估结果: {feedback.feedback_type}")
print(f"综合评分: {feedback.score:.4f}")

if feedback.needs_retry():
    print(f"诊断: {feedback.diagnosis}")
    print(f"建议: {feedback.suggested_fix}")
```
