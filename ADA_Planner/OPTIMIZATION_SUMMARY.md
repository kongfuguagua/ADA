# ADA_Planner 系统优化总结

## 优化目标

根据架构分析，Planner（LLM + Expert）跑不过纯 ExpertAgent 的核心原因在于：**引入了不确定性，却未建立足够的"安全底座"**。

本次优化的目标是：**Baseline = ExpertAgent，Upper Bound = LLM**。即：最差也要和 Expert 一样好，最好能比 Expert 更好。

## 优化策略实施

### 策略一：信任直通车 (Trust Pass-through) ✅

**问题**：Expert 已经算出了满分答案（Score=4），却还要问一遍 LLM，LLM 可能会自作聪明地修改或解析错误。

**解决方案**：
- 在 `act()` 函数中，Expert 生成洞察后立即检查是否有 Score=4 的完美方案
- 如果存在 Score=4 且无高风险副作用的方案，**直接执行，跳过 LLM**
- 这确保了在危急时刻，Planner 的表现**下限**就是 ExpertAgent

**实现位置**：
- `ADA_Planner/agent.py`:
  - 新增 `_check_trust_pass_through()` 方法
  - 在 `act()` 方法中，Expert 生成洞察后立即调用此方法

**效果**：
- 确保在危急时刻，Planner 的表现**下限**就是 ExpertAgent
- 绝不会因为 LLM 的"胡言乱语"导致崩溃
- 统计信息：`stats["trusted_expert_actions"]` 记录跳过 LLM 的次数

---

### 策略二：LLM 定位转向"长效优化"与"复杂决策" ✅

**问题**：ExpertAgent 的弱点是**短视**（Greedy，只看当前一步）。LLM 的强项是**常识推理**和**多步规划**。

**解决方案**：
1. **区分两种场景**：
   - **场景 A (危急)**：Expert 有方案。Prompt: "Expert 建议执行 X。除非你能找到明显更好的方案，否则请输出 `execute_expert_solution(0)`。"
   - **场景 B (预警)**：Expert 无方案或负载率在 0.85-0.95 之间。Prompt: "当前线路 X 负载率 92%，虽然未过载，但请建议一个温和的 Redispatch 策略来降低风险。"

2. **让 LLM 处理 Expert 解决不了的问题**：
   - Expert Score < 3 时：需要 LLM 介入，尝试**组合动作**（例如：Expert 的拓扑 + 少量 Redispatch）
   - 预防性调度：当 `rho` 在 0.85-0.95 之间时，让 LLM 介入，通过 Redispatch 提前降低负载

**实现位置**：
- `ADA_Planner/prompts.py`:
  - `build()` 方法中根据 `expert_insight` 状态动态调整 System Prompt
  - 新增 `_is_preventive_scenario()` 方法判断预警场景
  - 区分危急场景、复杂决策场景、预警场景三种 Prompt

**效果**：
- LLM 专注于处理 Expert 解决不了的问题
- 在非过载状态下，利用 LLM 进行低成本的 Redispatch 优化，赚取比 ExpertAgent 更高的 Reward

---

### 策略三：修正 Parser 与 Action Space 映射 ✅

**问题**：Parser 是目前的短板。LLM 容易产生格式错误，导致有效动作被判定为无效。

**解决方案**：
1. **强制结构化输出**：
   - 支持 JSON 格式输出，包含 `action_type`, `params`, `reasoning`
   - JSON 格式更不容易出错，解析更可靠

2. **动作空间掩码 (Action Masking)**：
   - 在 Prompt 中明确列出**"禁止操作列表"**（如冷却中的线路、已达最大出力的发电机）
   - 显式告诉 LLM 这些约束，避免建议非法动作

**实现位置**：
- `ADA_Planner/parser.py`:
  - 新增 `_extract_from_json()` 方法，支持从 JSON 格式提取动作
  - `extract_action_from_response()` 方法优先使用 JSON 格式提取
- `ADA_Planner/prompts.py`:
  - 在 System Prompt 中添加 JSON 格式说明
  - 添加"动作空间掩码"部分，明确列出禁止操作列表

**效果**：
- 减少解析错误，提高动作执行成功率
- LLM 明确知道哪些动作是禁止的，避免生成非法动作

---

### 策略四：Reward 最大化机制 ✅

**问题**：ExpertAgent 只关注活下来（Survival）。要提高 Reward（通常与运营成本、线路损耗有关）。

**解决方案**：
1. **成本估算**：
   - 在 `ExpertInsight` 中计算每个方案的 **Cost**（操作成本、风险成本、运营成本）
   - 将 Top-3 方案及其 Cost 喂给 LLM

2. **成本优化决策**：
   - 在 Prompt 中强调：**在安全的前提下（Score >= 3），优先选择成本最低的方案**
   - 让 LLM 扮演"经济师"的角色，在安全的前提下挑选最省钱的方案

**实现位置**：
- `ADA_Planner/analysis/expert_insight.py`:
  - 优化 `_estimate_cost()` 方法，更准确地估算成本
  - 考虑操作成本、风险成本、运营效率
- `ADA_Planner/agent.py`:
  - `_format_expert_insight()` 方法中突出显示成本信息
- `ADA_Planner/prompts.py`:
  - 在 System Prompt 中添加成本优化说明

**效果**：
- LLM 在安全的前提下选择最省钱的方案
- 长期 Reward 更高，运营效率更高

---

## 优化路线图

### 第一阶段（固本）：✅ 已完成
- 实现"信任直通车"。只要 Expert 说有完美解，无条件执行。
- **这能让你的成功率立刻追平 ExpertAgent。**

### 第二阶段（增强）：✅ 已完成
- 修复 Parser，支持 JSON 结构化输出，确保 LLM 的输出能被 100% 正确解析。
- 解决"有解但解析挂了"的问题。

### 第三阶段（超越）：✅ 已完成
- 引入"预防性调度"。在非过载状态下（Score=Safe），利用 LLM 进行低成本的 Redispatch 优化，赚取比 ExpertAgent 更高的 Reward。
- 实现成本优化机制，让 LLM 在安全前提下选择最省钱的方案。

---

## 关键改进点总结

| 维度 | 优化前 | 优化后 |
| --- | --- | --- |
| **决策确定性** | LLM 可能解析失败、产生幻觉 | Score=4 时直接执行，跳过 LLM |
| **执行路径** | Expert -> Prompt -> LLM -> Parser -> Action | Score=4 时：Expert -> Action（直通车） |
| **异常处理** | 依赖 try...except，很多有效动作被判定为无效 | JSON 格式 + 动作空间掩码，减少解析错误 |
| **成本优化** | 只关注安全，不考虑成本 | 在安全前提下选择成本最低的方案 |

---

## 预期效果

1. **稳定性提升**：信任直通车确保在危急时刻的表现下限 = ExpertAgent
2. **成功率提升**：JSON 格式 + 动作空间掩码减少解析错误
3. **Reward 提升**：成本优化机制让 LLM 选择更经济的方案
4. **智能增强**：LLM 专注于处理 Expert 解决不了的问题（组合动作、预防性调度）

---

## 使用说明

所有优化已自动集成到 `ADA_Planner` 中，无需额外配置。系统会根据场景自动选择最优策略：

- **危急场景 + Expert Score=4**：信任直通车，直接执行
- **危急场景 + Expert Score<3**：LLM 进行组合优化
- **预警场景（0.85-0.95）**：LLM 进行预防性调度
- **安全场景（<0.85）**：do_nothing

---

## 统计信息

系统会记录以下统计信息（在 `agent.stats` 中）：
- `trusted_expert_actions`: 信任直通车触发次数
- `expert_insight_calls`: 专家洞察调用次数
- `sanitized_count`: 动作自动修正次数
- `successful_actions`: 成功执行的动作数
- `failed_actions`: 失败的动作数

通过这些统计信息，可以评估优化效果。

