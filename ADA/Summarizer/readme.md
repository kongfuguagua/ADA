# Summarizer Agent 设计规范

## 1. 核心职责

Summarizer 是系统的"进化引擎"，负责实现**基于动态知识演化的自适应搜索**（论文 2.3 节）。它不参与实时调度，而在幕后通过经验回溯更新知识库（TK/AK），持续剪枝无效搜索路径。

## 2. 论文需求响应表

| 论文需求 | 实现状态 | 实现位置 |
|---------|---------|---------|
| 经验回放 (Experience Replay) | ✅ 已实现 | `core.py: _trace_buffer` |
| MCTS 驱动的路径识别 | ✅ 已实现 | `core.py: MCTSNode, _update_mcts_tree()` |
| UCB 准则选择高价值路径 | ✅ 已实现 | `core.py: MCTSNode.ucb_score()` |
| 动作知识 (AK) 提炼与更新 | ✅ 已实现 | `knowledge_updater.py: extract_action_knowledge()` |
| 任务知识 (TK) 提炼与更新 | ✅ 已实现 | `knowledge_updater.py: extract_task_knowledge()` |
| 最小入库分数阈值 | ✅ 已实现 | `knowledge_updater.py: min_score_threshold` |
| 批量分析历史轨迹 | ✅ 已实现 | `core.py: _batch_analysis()` |

## 3. 核心逻辑：MCTS 驱动的知识更新

```
┌─────────────────────────────────────────────────────────────────┐
│                  Summarizer 工作流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: ExecutionTrace (完整调度轨迹)                             │
│                                                                 │
│  1. 经验回放 (Experience Replay)                                │
│     └── 收集轨迹 E = {(R_before, R_after, tool_i, ...)}        │
│                                                                 │
│  2. MCTS 搜索树更新                                             │
│     ├── 将工具调用序列建模为搜索树                               │
│     ├── 利用 UCB 准则识别高价值决策路径 N*                       │
│     └── 反向传播更新节点值                                       │
│                                                                 │
│  3. 知识提炼                                                    │
│     ├── AK 更新: 归纳成功的工具调用模式                          │
│     └── TK 更新: 归纳高分的问题建模模板                          │
│                                                                 │
│  4. 知识入库                                                    │
│     ├── 检查分数是否超过阈值                                     │
│     └── 调用 KnowledgeService 写入向量数据库                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 模块架构

```
Summarizer/
├── __init__.py           # 模块导出
├── readme.md             # 本文档
├── core.py               # 【主入口】SummarizerAgent + MCTS 实现
├── prompt.py             # 【Prompt】知识提炼提示模板
└── knowledge_updater.py  # 【知识更新】调用 KnowledgeService
```

## 5. MCTS 节点结构

```python
class MCTSNode:
    state: str              # 状态描述
    parent: MCTSNode        # 父节点
    children: List[MCTSNode]  # 子节点列表
    visits: int             # 访问次数
    value: float            # 累计价值
    action: str             # 导致此状态的动作
    
    def ucb_score(self, c=1.414) -> float:
        """UCB 评分 = exploitation + exploration"""
        return self.value / self.visits + c * sqrt(ln(parent.visits) / self.visits)
```

## 6. 知识类型定义

### 动作知识 (Action Knowledge, AK)

- **用途**：指导 Planner 的工具调用决策
- **内容**：成功的工具调用模式、调用顺序、适用场景
- **示例**：
  ```
  当检测到线路负载率 > 0.9 时，应先调用 overflow_analysis 工具，
  然后调用 action_simulation 评估再调度方案。
  ```

### 任务知识 (Task Knowledge, TK)

- **用途**：指导 Planner 的问题建模
- **内容**：成功的建模模板、目标函数形式、约束条件组合
- **示例**：
  ```
  对于电网调度成本优化问题，目标函数应包含发电成本项和惩罚项，
  约束条件应包含功率平衡约束、线路容量约束和发电机出力约束。
  ```

## 7. 接口定义

### 输入

```python
class ExecutionTrace(BaseModel):
    trace_id: str
    environment: EnvironmentState
    problem: OptimizationProblem
    solution: Solution
    feedback: Feedback
    tool_chain: List[AugmentationStep]  # 工具调用链
    attempt_count: int
```

### 输出

无直接输出，通过 KnowledgeService 更新知识库。

## 8. 配置参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| exploration_constant | MCTS 探索系数 | 1.414 |
| min_score_threshold | 最小入库分数阈值 | 0.7 |
| batch_analysis_size | 批量分析触发数量 | 10 |

## 9. 使用示例

```python
from Summarizer import SummarizerAgent
from utils.const import ExecutionTrace

# 创建 Summarizer
summarizer = SummarizerAgent(kb=kb, llm=llm)

# 总结轨迹
summarizer.summarize(trace)

# 获取统计信息
stats = summarizer.get_statistics()
print(f"知识库条目: {stats['knowledge_count']}")
print(f"最佳工具序列: {stats['best_sequence']}")

# 导出 MCTS 树
tree = summarizer.export_mcts_tree()
```
