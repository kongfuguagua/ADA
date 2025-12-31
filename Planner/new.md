# Grid2Op Expert-Augmented Planner Agent (EARA) 设计文档

## 1. 核心设计理念 (Core Philosophy)

**"Symbolic Guidance, Neural Execution" (符号引导，神经执行)**

将 `ExpertAgent` 中经过验证的、基于规则和数学的求解能力，**完全剥离**并封装为一个高阶工具 `ExpertInsight`。在 Planner 循环启动时，强制执行一次专家诊断，将诊断结果作为**强提示 (Strong Prompt)** 注入给 LLM。LLM 不再需要从零猜测物理规律，而是从专家提供的“候选方案”中进行决策和微调。

---

## 2. 系统架构 (System Architecture)

### 2.1 模块交互图

1. **Environment**: 输出 `Observation`。
2. **Trigger**: 检测到危险状态 -> 启动 Agent 流程。
3. **ExpertInsight Engine (封装的原 ExpertAgent)**:
* 输入: `Observation`
* 执行: 影响图构建 -> 候选拓扑生成 -> 模拟打分。
* 输出: `InsightReport` (包含 Top-3 推荐动作及其物理依据)。


4. **Prompt Manager**: 将 `InsightReport` 转化为自然语言（例如：“专家强烈建议对变电站 14 进行节点分裂，因为这能消除线路 12 的 98% 负载”）。
5. **LLM Core**: 接收 Observation + Expert Insight，进行 Planner 推理。
6. **Action Execution**: LLM 输出最终动作。

---

## 3. 核心模块：ExpertInsight Service

我们需要创建一个新的服务模块 `expert_insight.py`，它基本复用 `ExpertAgent` 的核心算法，但**不直接执行动作**，而是**返回分析数据**。

### 3.1 伪代码实现 (Pseudocode)

```python
class ExpertInsightService:
    def __init__(self, env):
        # 复用 ExpertAgent 的配置
        self.config = {
            "totalnumberofsimulatedtopos": 25,
            "numberofsimulatedtopospernode": 5
            # ...
        }
        # 初始化底层的数学模拟器 (来自 ExpertAgent)
        self.simulator = Grid2opSimulation(..., param_options=self.config)

    def generate_insight(self, observation) -> dict:
        """
        核心方法：生成专家洞察
        """
        # 1. 识别过载 (Ranked Overloads)
        overloaded_lines = self._get_ranked_overloads(observation)
        if not overloaded_lines:
            return {"status": "SAFE", "suggestion": "Do Nothing or Maintenance"}

        target_line = overloaded_lines[0] # 聚焦最严重的过载
        
        # 2. 运行 ExpertAgent 的核心搜索逻辑 (expert_operator)
        # 这会计算影响图，并模拟 Top-N 个拓扑变化
        # [Symbolic Calculation happens here]
        ranked_topologies, results, actions = expert_operator(
            self.simulator, 
            ltc=[target_line] # Line To Cut/Cure
        )
        
        # 3. 提炼 Top-K 方案
        top_k_solutions = []
        if results is not None and not results.empty:
            # 筛选得分高的方案 (Score >= 3 表示能解决问题)
            best_candidates = results[results['Topology simulated score'] >= 3].head(3)
            
            for idx, row in best_candidates.iterrows():
                action_obj = actions[idx]
                sub_id = int(row['Substation ID'])
                score = row['Topology simulated score']
                
                solution = {
                    "type": "Topology Action",
                    "substation_id": sub_id,
                    "description": f"Split Substation {sub_id} (Score: {score}/4)",
                    "expected_outcome": "Solves overload completely",
                    "action_object": action_obj, # 保存原始动作对象供后续解析
                    "action_code": f"change_bus({sub_id}, ...)" # 转化为 LLM 可读的代码
                }
                top_k_solutions.append(solution)
        
        # 4. 生成备选方案 (如果拓扑无法解决，建议 redispatch)
        if not top_k_solutions:
            # 使用简单的灵敏度分析推荐再调度 (Sensitivity Analysis)
            redispatch_hint = self._analyze_sensitivity(observation, target_line)
            top_k_solutions.append(redispatch_hint)

        return {
            "status": "DANGER",
            "critical_line": target_line,
            "solutions": top_k_solutions
        }

    def _analyze_sensitivity(self, observation, line_id):
        # 简化的灵敏度分析 (当 Expert System 找不到拓扑解时作为保底)
        # 返回: {"type": "Redispatch", "gen_up": [1, 2], "gen_down": [3, 4]}
        pass

```

---

## 4. Planner Agent 的深度改造

### 4.1 修改 `agent.py`: 引入专家介入

```python
class Planner(BaseAgent):
    def __init__(self, ...):
        # ... 初始化 ...
        self.expert_insight = ExpertInsightService(env)  # 初始化专家服务

    def act(self, observation, ...):
        # 1. 检查状态
        if self._is_danger(observation):
            # === 关键步骤：在启动 LLM 前，先咨询专家 ===
            logging.info("Danger detected. Consulting ExpertInsight...")
            
            # 获取专家报告 (耗时操作，但值得)
            insight_report = self.expert_insight.generate_insight(observation)
            
            # 将报告注入 Prompt
            prompt = self.prompt_manager.build(
                observation, 
                history=self.Planner_history,
                expert_insight=insight_report  # 新增参数
            )
            
            # 2. LLM 思考与决策 (Planner Loop)
            # LLM 现在会在 Prompt 中看到："Expert suggests splitting Substation 14..."
            action = self._run_Planner_loop(prompt)
            return action
            
        else:
            return self._do_maintenance(observation) # 安全时的维护逻辑

```

### 4.2 修改 `prompts.py`: 专家提示模板

我们需要专门设计一段 Prompt，教 LLM 如何利用专家意见。

```python
SYSTEM_PROMPT = """
你是一个电网调度指挥官。你有一个强大的技术顾问团队（Expert System）。
当系统出现紧急情况（如过载）时，技术顾问会给你提供【专家建议 (Expert Insight)】。

你的职责是：
1. **优先采纳**：如果专家提供了得分为 3 或 4 的高置信度方案（通常是拓扑调整），你应该优先生成对应的指令执行。
2. **审查与兜底**：如果专家说“无有效拓扑方案”，你需要利用你的物理知识，尝试再调度（Redispatching）或其他手段。
3. **格式转换**：将专家的自然语言建议转化为标准的 Action 代码（如 `set_bus(...)` 或 `redispatch(...)`）。

【专家建议示例】
[Expert Insight]: 
- Critical Overload on Line 5.
- Top Recommendation: Split Substation 14. (Score: 4/4 - Solves all overloads).
- Action Code Hint: set_bus(14, [1, 1, 2, 1, ...])

【你的反应】
Thought: 线路 5 严重过载。专家系统强烈建议对变电站 14 进行节点分裂（Split），评分满分，说明能完全解决问题。我将采纳此建议。
Action: set_bus(14, [1, 1, 2, 1, ...])
"""

```

### 4.3 动作解析的挑战 (Action Parsing)

ExpertAgent 生成的是复杂的拓扑动作（`set_bus` 向量），很难直接用简单的 `change_bus(sub_id, line_id)` 描述。

**解决方案：引用式动作 (Reference Action)**
为了避免让 LLM 输出长长的 `[1, 1, 2, ...]` 向量（容易出错），我们允许 LLM 直接引用专家方案。

* **Prompt**: "You can execute expert solution directly by using command: `execute_expert_solution(index)`"
* **Action Parser**:
```python
if "execute_expert_solution" in action_text:
    idx = int(extract_arg(action_text))
    # 直接从缓存的 insight_report 中取出那个复杂的 Action 对象返回
    return self.current_insight_report['solutions'][idx]['action_object']

```



这种方式**零错误率**，完美复用了 ExpertAgent 的计算结果。

---

## 5. 实现路线图 (Implementation Roadmap)

1. **Phase 1: 移植 ExpertAgent 核心**
* 将 `ExpertAgent/expertAgent.py` 中的数学逻辑（`Grid2opSimulation`, `expert_operator`）剥离出来，去掉 `act` 方法，改为 `get_analysis()` 形式的纯函数。
* 创建 `ADA/Planner/tools/expert_insight.py`。


2. **Phase 2: 定义通信协议**
* 确定 `InsightReport` 的字典结构（必须包含：过载描述、Top-K 方案描述、方案对应的 Grid2Op Action 对象）。


3. **Phase 3: 改造 Planner Loop**
* 修改 `Planner.act`，在检测到过载时调用 ExpertInsight。
* 更新 `PromptManager`，支持渲染 Insight 文本。
* 更新 `ActionParser`，支持 `execute_expert_solution(i)` 这种“快捷指令”。