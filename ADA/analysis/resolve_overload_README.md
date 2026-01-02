# `resolve_overload` 核心逻辑详解

## 目录
- [方法概述](#方法概述)
- [输入输出](#输入输出)
- [核心逻辑流程](#核心逻辑流程)
- [关键概念](#关键概念)
- [决策流程详解](#决策流程详解)
- [关键参数配置](#关键参数配置)
- [示例场景](#示例场景)

---

## 方法概述

`resolve_overload` 是 ExpertInsightService 的核心方法，完全复刻了 ExpertAgent (L2RPN Baselines) 的过载处理逻辑。该方法通过多阶段决策机制，在电网出现过载时（`rho >= 1.0`）寻找最优的拓扑重构动作。

### 核心特点
1. **多阶段决策**：从最优解到妥协方案，再到兜底机制
2. **智能截断**：找到完美解或关键解时立即停止搜索
3. **副作用最小化**：在关键时刻选择副作用最小的动作
4. **参考拓扑恢复**：当直接方案失败时，尝试恢复参考拓扑

---

## 输入输出

### 输入参数

```python
def resolve_overload(
    self, 
    observation: BaseObservation,      # 当前电网观测状态
    sub_2nodes: Set[int],              # 已分裂为2个节点的变电站集合
    lines_disconnected: Set[int]       # 已断开的线路集合
) -> Dict[str, Any]
```

- **observation**: 包含当前电网状态（线路负载率 `rho`、过载时间步 `timestep_overflow` 等）
- **sub_2nodes**: 记录哪些变电站已经被分裂（用于后续恢复逻辑）
- **lines_disconnected**: 记录哪些线路已断开（当前未直接使用）

### 输出结果

#### 成功找到动作
```python
{
    "status": "DANGER",                    # 状态：存在过载
    "action": BaseAction,                  # 推荐的动作
    "score": int,                          # 动作评分 (0-4)
    "efficacy": float,                     # 动作效能值
    "sub_id_to_split": int,                # 要分裂的变电站ID (>=0) 或 -1
    "sub_id_to_discard": int (可选),       # 要恢复的变电站ID (兜底逻辑)
    "description": str                     # 动作描述
}
```

#### 无过载
```python
{
    "status": "SAFE",
    "action": None
}
```

#### 未找到有效动作
```python
{
    "status": "DANGER",
    "action": None,
    "score": 0
}
```

---

## 核心逻辑流程

### 整体流程图

```
开始
  ↓
[阶段1] 获取并排序过载线路
  ├─ 无过载 → 返回 SAFE
  └─ 有过载 → 继续
  ↓
[阶段2] 初始化追踪变量
  ↓
[阶段3] 循环处理 Top N 过载 (最多 max_overloads_at_a_time 个)
  ├─ 对每个过载线路：
  │   ├─ 创建模拟器
  │   ├─ 调用 expert_operator 获取候选动作
  │   ├─ 评估并更新最佳方案
  │   ├─ [截断检查] Score 4 或 (Score 3 + Critical) → 立即停止
  │   └─ [妥协逻辑] 关键时刻寻找副作用最小方案
  ↓
[阶段4] 低分兜底逻辑 (Score <= 1)
  ├─ 尝试恢复参考拓扑（已考虑的变电站）
  └─ 尝试恢复参考拓扑（其他已分裂的变电站）
  ↓
[阶段5] 封装并返回结果
```

---

## 关键概念

### 1. 过载线路排序 (`_get_ranked_overloads`)

过载线路按以下规则排序：

1. **筛选条件**：`rho >= 1.0` 的线路
2. **排序规则**：按 `rho` 值降序排列
3. **优先级分组**：
   - **Critical 过载**：`timestep_overflow[line] == NB_TIMESTEP_OVERFLOW_ALLOWED`
     - 这些线路即将达到最大允许过载时间，必须立即处理
   - **Not Critical 过载**：其他过载线路
   - **最终顺序**：Critical 在前，Not Critical 在后

### 2. 动作评分系统 (Topology Simulated Score)

评分范围：**0-4**

| Score | 含义 | 说明 |
|-------|------|------|
| **4** | 完美解决 | 解决所有过载，且不产生新过载 |
| **3** | 解决目标过载 | 解决目标过载，但可能产生其他过载 |
| **2** | 部分缓解 | 缓解目标过载，但未完全解决 |
| **1** | 副作用方案 | 解决目标过载，但恶化其他线路 |
| **0** | 无效方案 | 无法解决目标过载 |

### 3. Efficacy (效能值)

- **定义**：动作对电网状态的改善程度
- **用途**：在相同 Score 的动作中，选择 Efficacy 最高的
- **计算**：由 `expert_operator` 内部计算，基于 `MinMargin_reward`

### 4. Critical vs Not Critical

- **Critical**：`timestep_overflow[line] == NB_TIMESTEP_OVERFLOW_ALLOWED`
  - 该线路已达到最大允许过载时间
  - 必须立即处理，否则会导致线路断开
- **Not Critical**：其他过载线路
  - 还有时间缓冲，可以稍后处理

### 5. 并行线路处理 (`_additional_lines_to_cut`)

针对 IEEE118 等特定电网，某些线路是并行运行的，需要同时考虑：

- **IEEE118_R2**: 线路对 `(22,23)`, `(33,35)`, `(34,32)`
- **IEEE118**: 线路对 `(135,136)`, `(149,147)`, `(148,146)`

当处理其中一个线路时，会自动考虑其配对线路。

---

## 决策流程详解

### 阶段1：过载检测与排序

```python
ltc_list = self._get_ranked_overloads(observation)
```

**逻辑**：
1. 筛选所有 `rho >= 1.0` 的线路
2. 按 `rho` 降序排序
3. Critical 过载优先于 Not Critical 过载

**输出**：排序后的过载线路列表 `ltc_list`

---

### 阶段2：初始化追踪变量

```python
best_solution = None          # 当前最佳动作
score_best = 0                # 当前最佳评分
efficacy_best = -999.0        # 当前最佳效能
sub_id_to_split = -1          # 要分裂的变电站ID
ltc_already_considered = []   # 已考虑的过载线路
counter_tested = 0            # 已测试的过载数量
```

**关键判断**：
- `is_many_overloads = (len(ltc_list) > timesteps_allowed)`
  - 如果过载数量超过允许的时间步数，系统会采用更激进的策略

---

### 阶段3：主循环 - 处理 Top N 过载

#### 3.1 循环条件

```python
for ltc in ltc_list:
    if counter_tested >= self.max_overloads_at_a_time:  # 默认 3
        break
```

**限制**：最多处理 `max_overloads_at_a_time` 个过载（默认 3 个）

#### 3.2 Critical 判断

```python
is_critical = (observation.timestep_overflow[ltc] == timesteps_allowed)
```

#### 3.3 并行线路处理

```python
additional_cuts, lines_considered = self._additional_lines_to_cut(ltc)
ltc_already_considered.extend(lines_considered)
```

针对 IEEE118 等电网，自动处理并行线路对。

#### 3.4 专家系统模拟

```python
simulator = Grid2opSimulation(...)
ranked_combinations, expert_results, actions = expert_operator(simulator, ...)
```

**expert_operator** 会：
1. 生成多个候选拓扑重构方案
2. 对每个方案进行模拟评估
3. 计算 Score 和 Efficacy
4. 返回排序后的结果

#### 3.5 结果评估与更新

```python
# 获取本次模拟的最佳结果
new_score_best = expert_results['Topology simulated score'].max()
best_candidates = expert_results[expert_results['Topology simulated score'] == new_score_best]
idx_best = pd.to_numeric(best_candidates["Efficacity"]).idxmax()
```

**更新条件**：
```python
if (new_score_best > score_best) and (new_score_best >= 3):
    # 更新最佳方案
    best_solution = actions[idx_best]
    score_best = new_score_best
    efficacy_best = ...
    sub_id_to_split = ...
```

**关键点**：只有 Score >= 3 的方案才会被考虑更新全局最佳方案。

#### 3.6 截断逻辑 (Termination)

```python
# 1. 完美解决 (Score 4) → 立即停止
if score_best == 4:
    break

# 2. 解决关键过载 (Score 3 + Critical) → 立即停止
if (score_best == 3) and is_critical:
    break
```

**设计理念**：
- Score 4：完美解，无需继续搜索
- Score 3 + Critical：已解决最紧急的问题，可以停止

#### 3.7 妥协逻辑 (Least Worsened)

```python
if is_critical or (is_many_overloads and score_best == 0):
    idx_compromise = self._get_action_with_least_worsened_lines(expert_results, ltc_list)
    if idx_compromise is not None:
        # 更新为妥协方案
        best_solution = actions[idx_compromise]
        score_best = 1  # 妥协方案通常是 Score 1
        
        if is_critical:  # 关键时刻找到救命稻草，直接停止
            break
```

**触发条件**：
1. **关键时刻**：当前过载是 Critical
2. **多过载且无解**：过载很多且当前最佳方案 Score = 0

**选择策略** (`_get_action_with_least_worsened_lines`)：
1. 筛选所有 Score = 1 的候选动作
2. 计算每个动作的副作用：
   - `existing_worsened`：原本过载的线路中，被恶化的数量
   - `new_worsened`：新产生的过载线路数量
3. 优先选择：
   - 最小化 `existing_worsened`
   - 其次最小化 `new_worsened`

---

### 阶段4：低分兜底逻辑 (Score <= 1)

当主循环结束后，如果 `score_best <= 1`，说明没有找到好的直接解决方案，此时尝试恢复参考拓扑。

#### 4.1 尝试恢复已考虑的变电站

```python
if 'expert_results' in locals() and self._is_valid_result(expert_results):
    subs_expert_results = expert_results["Substation ID"].tolist()
    fallback_action, sub_id_to_discard = self._try_out_reference_topologies(
        simulator, score_best, efficacy_best, is_critical, 
        sub_2nodes, subs_expert_results, subs_in_cooldown
    )
```

**逻辑**：
- 尝试恢复专家系统已考虑过的变电站（这些变电站之前被分裂过）
- 恢复意味着将分裂的变电站合并回参考拓扑（全1拓扑）

#### 4.2 尝试恢复其他已分裂的变电站

```python
if fallback_action is None:
    subs_to_try = sub_2nodes - set(subs_expert_results)
    if subs_to_try:
        fallback_action, sub_id_to_discard = self._try_out_reference_topologies(
            simulator, score_best, efficacy_best, is_critical, 
            subs_to_try, list(subs_to_try), subs_in_cooldown
        )
```

**逻辑**：
- 如果第一步失败，尝试恢复其他已分裂的变电站
- 这些变电站是专家系统未直接考虑的

#### 4.3 恢复动作的评估条件

在 `_compute_score_on_new_combinations` 中，恢复动作的接受条件：

```python
improve_score = (new_best_score >= 3)                    # 显著改善
improve_critical = (new_best_score == 1) and (new_eff >= best_eff) and is_critical  # 关键时刻
improve_general = (new_best_score >= best_score) and (new_eff >= best_eff) and (not is_line_cut)  # 一般改善
```

**设计理念**：
- 恢复操作通常不会产生新过载（因为恢复的是原始拓扑）
- 在某些情况下，恢复可能比分裂更安全

---

### 阶段5：结果封装

```python
if best_solution:
    result = {
        "status": "DANGER",
        "action": best_solution,
        "score": score_best,
        "efficacy": efficacy_best,
        "sub_id_to_split": sub_id_to_split,  # >= 0 表示要分裂，-1 表示恢复
        "description": f"Expert Action (Score: {score_best})"
    }
    if sub_id_to_discard_from_fallback is not None:
        result["sub_id_to_discard"] = sub_id_to_discard_from_fallback
    return result
```

**关键字段**：
- `sub_id_to_split >= 0`：表示要分裂的变电站，Planner 需要将其加入 `sub_2nodes`
- `sub_id_to_split == -1`：表示是恢复操作，不需要更新 `sub_2nodes`
- `sub_id_to_discard`：如果存在，表示要恢复的变电站，Planner 需要将其从 `sub_2nodes` 移除

---

## 关键参数配置

### ExpertAgent 配置

```python
self.config = {
    "totalnumberofsimulatedtopos": 25,           # 总模拟拓扑数
    "numberofsimulatedtopospernode": 5,          # 每个节点模拟拓扑数
    "maxUnusedLines": 2,                         # 最大未使用线路数
    "ratioToReconsiderFlowDirection": 0.75,     # 重新考虑流向比例
    "ratioToKeepLoop": 0.25,                     # 保持环路比例
    "ThersholdMinPowerOfLoop": 0.1,              # 环路最小功率阈值
    "ThresholdReportOfLine": 0.2                  # 线路报告阈值
}
```

### 关键阈值

```python
self.reward_type = "MinMargin_reward"            # 奖励类型
self.threshold_powerFlow_safe = 0.95             # 安全功率流阈值
self.max_overloads_at_a_time = 3                 # 每次最多处理的过载数
```

---

## 示例场景

### 场景1：完美解决 (Score 4)

**初始状态**：
- 线路 10 过载：`rho[10] = 1.2`
- 线路 20 过载：`rho[20] = 1.1`

**处理流程**：
1. 排序：`ltc_list = [10, 20]`（按 rho 降序）
2. 处理线路 10：
   - 专家系统生成多个候选动作
   - 找到 Score 4 的动作（分裂变电站 5）
   - **截断**：`score_best == 4` → 立即停止
3. 返回：`{action: 分裂变电站5, score: 4}`

**结果**：所有过载被解决，无新过载产生。

---

### 场景2：关键过载处理 (Score 3 + Critical)

**初始状态**：
- 线路 10 过载且 Critical：`rho[10] = 1.3`, `timestep_overflow[10] == NB_TIMESTEP_OVERFLOW_ALLOWED`
- 线路 20 过载：`rho[20] = 1.1`

**处理流程**：
1. 排序：`ltc_list = [10, 20]`（Critical 在前）
2. 处理线路 10（Critical）：
   - 找到 Score 3 的动作（分裂变电站 5）
   - 更新：`score_best = 3`
   - **截断**：`score_best == 3 and is_critical` → 立即停止
3. 返回：`{action: 分裂变电站5, score: 3}`

**结果**：关键过载被解决，但可能产生其他过载。

---

### 场景3：妥协方案 (Least Worsened)

**初始状态**：
- 线路 10 过载且 Critical：`rho[10] = 1.3`
- 多个其他过载线路

**处理流程**：
1. 处理线路 10：
   - 专家系统未找到 Score >= 3 的方案
   - 触发妥协逻辑：`is_critical == True`
   - 在 Score 1 的方案中，选择副作用最小的：
     - 方案A：恶化 2 条现有过载线路，新增 1 条过载
     - 方案B：恶化 1 条现有过载线路，新增 2 条过载
     - **选择方案A**（优先最小化现有过载恶化）
2. 返回：`{action: 方案A, score: 1}`

**结果**：解决了目标过载，但产生副作用。

---

### 场景4：兜底恢复 (Reference Topology Fallback)

**初始状态**：
- 多个过载线路
- 变电站 5 已被分裂（`sub_2nodes = {5}`）
- 主循环未找到好方案（`score_best = 0`）

**处理流程**：
1. 主循环结束，`score_best = 0`
2. 触发兜底逻辑：
   - 尝试恢复变电站 5 的参考拓扑
   - 模拟验证：恢复后所有 `rho < 0.95`
   - 接受：`improve_general == True`
3. 返回：`{action: 恢复变电站5, score: 1, sub_id_to_discard: 5}`

**结果**：通过恢复原始拓扑，改善了电网状态。

---

## 总结

`resolve_overload` 方法通过**多阶段决策机制**，在复杂电网过载场景中寻找最优解：

1. **优先策略**：寻找完美解（Score 4）或关键解（Score 3 + Critical）
2. **妥协策略**：关键时刻选择副作用最小的方案（Least Worsened）
3. **兜底策略**：尝试恢复参考拓扑，避免电网进一步恶化

该方法完全复刻了 ExpertAgent 的逻辑，确保了决策的一致性和可靠性。

