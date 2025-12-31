# ADA (Adaptive Dispatch & Action) 系统设计文档 v2.1

## 1. 设计宗旨与核心理念

ADA 的目标是构建一个**神经符号（Neuro-Symbolic）**混合智能体，通过多源策略竞争与实证仿真，解决电网控制中的稳定性与泛化性平衡问题。

**核心原则：**

1. **混合智能与全量竞优 (Hybrid Intelligence & Tournament Selection)**：
* **Planner (符号/规则)**：**完全复刻** `ExpertAgent` 的核心算法（如基于灵敏度的节点分裂、状态增广、拓扑搜索）。不依赖 LLM，提供基于物理规则的离散拓扑动作。
* **Solver (数学/优化)**：**完全复刻** `OptimCVXPY` 的凸优化逻辑。利用 DC 潮流模型求解连续变量（重调度/削减/储能），提供基于数学优化的动作。
* **Judger (神经/融合)**：LLM 作为高层智脑。它分析 Planner 和 Solver 的输出，结合知识库（RAG）的历史经验，试图发现单一方法无法覆盖的**融合策略**（例如：Planner 的拓扑 + Solver 的重调度）。
* **Simulator (实证/裁判)**：**核心决策关卡**。它不再只是验证 LLM，而是接收 **{Planner 候选集 + Solver 候选集 + LLM 增强集}** 构成的完整动作空间。对该空间进行**暴力搜索/全排序仿真**，纯粹依据仿真结果（安全性、Reward）择优执行。


2. **严格的输入规范 (Strict Specification)**：
* LLM 的输出必须经过 `parser.py` 的严格匹配和规范化，转化为标准 Grid2Op 动作对象，杜绝幻觉产生的非法动作。


3. **闭环进化 (Closed-Loop Evolution)**：
* **Summarizer** 负责复盘。它将 Simulator 选出的**最优策略**（无论是来自 Expert、Solver 还是 LLM）与当前场景上下文结合，生成结构化经验存入知识库，实现持续学习。



---

## 2. 系统架构概览

```mermaid
graph TD
    Obs[Grid2Op Observation] --> Planner[Module: Planner (Expert Logic)]
    Obs --> Solver[Module: Solver (Convex Opt)]
    Obs --> KB[Module: KnowledgeBase (RAG)]
    
    %% 原始候选生成
    Planner -->|Topology Candidates| ActionPool
    Solver -->|Dispatch Candidate| ActionPool
    
    %% LLM 融合增强
    Planner -->|Topology Info| Judger
    Solver -->|Dispatch Info| Judger
    KB -->|History Context| Judger
    Obs -->|Context| Judger
    
    Judger[Module: Judger (LLM Strategy)] -->|Fused/Enhanced Actions| ActionPool
    
    %% 仿真竞技场
    ActionPool{Combined Action Space} -->|Brute-Force Simulation| Simulator[Module: Simulator (The Arena)]
    
    %% 执行与学习
    Simulator -->|Best Action| Executor[Execute Action]
    Executor -->|Result| Summarizer[Module: Summarizer]
    Summarizer -->|Update| KB

```

---

## 3. 详细模块设计

### 3.1 基础数据结构 (Shared Definitions)

定义统一的数据包，用于在模块间流转。

* **`CandidateAction` 类**：
* `source`: 来源标识 ("Expert_Topo", "Math_Dispatch", "LLM_Fusion", "LLM_Recovery")
* `action_obj`: Grid2Op BaseAction 对象 (核心载体)
* `description`: 动作的自然语言/代码描述
* `simulation_result`: (由 Simulator 填充) `dict(is_safe, rho_max, reward, margin)`



### 3.2 Module: Planner (拓扑专家 - 物理规则)

**职责**：基于电网物理规则和图算法，通过**状态增广**生成拓扑候选集。

**核心逻辑**：

1. **算法复刻**：完整实现 `ExpertAgent` (L2RPN Baseline) 的逻辑。
* 计算过载线路的敏感度分析。
* 构建影响图 (Influence Graph)。
* 执行节点分裂 (Bus Splitting) 和线路开关 (Line Switching) 的搜索。


2. **状态增广**：不局限于单一动作，而是生成 Top-N 个可能的拓扑变更方案。
3. **独立性**：此模块纯 Python 实现，**不调用 LLM**。

**输出**：`List[CandidateAction]` (Source: "Expert_Topo")

### 3.3 Module: Solver (优化专家 - 数学计算)

**职责**：基于凸优化解决功率流平衡问题，处理连续变量。

**核心逻辑**：

1. **算法复刻**：完整实现 `OptimCVXPY` 的逻辑。
2. **建模求解**：
* 基于 DC 潮流模型构建优化问题。
* 目标函数：最小化过载 (Danger Mode) 或 最小化成本/最大化余量 (Safe Mode)。
* 约束条件：线路热极限、发电机爬坡、储能限制。


3. **独立性**：此模块纯 Python + CVXPY 实现，**不调用 LLM**。

**输出**：`List[CandidateAction]` (通常为 1 个, Source: "Math_Dispatch")

### 3.4 Module: KnowledgeBase (记忆库)

**职责**：提供历史上的成功案例作为参考。

**核心逻辑**：

1. **存储结构**：JSON / Vector DB。
2. **Key (Embedding)**：过载线路 ID、过载程度 (Rho)、拓扑指纹、发电机出力分布。
3. **Value**：场景描述、采取的动作（Action Code）、执行效果（Reward, Survival Steps）。
4. **检索策略**：基于相似度检索 Top-K 历史场景，优先返回 Reward 高且存活时间长的案例。

### 3.5 Module: Judger (融合中枢 - LLM)

**职责**：作为“战略指挥官”，利用推理能力融合多方信息，生成**更优的组合策略**。

**输入**：

1. **Planner 方案**：Top-N 拓扑动作及其预期效果。
2. **Solver 方案**：重调度计划及其成本。
3. **KB 历史**：类似场景下别人是怎么做的。
4. **当前观测**：过载详情。

**思考与生成 (Chain of Thought)**：

1. **分析差异**：Planner 解决了 A 问题但忽略了 B？Solver 成本太高？
2. **策略融合**：
* *组合模式*：是否可以将 Planner 的【节点分裂】与 Solver 的【重调度】叠加？
* *历史迁移*：历史记录显示这里应该断开 Line X，但 Planner 没提，我补充这个动作。


3. **规范化输出**：
* LLM 生成建议文本 -> `parser.py` 解析匹配 -> 生成 `Grid2Op Action` 对象。
* **注意**：LLM 生成的动作如果解析失败，直接丢弃，不进入仿真池。



**输出**：`List[CandidateAction]` (Source: "LLM_Fusion")

### 3.6 Module: Simulator (竞技场 - 实证排序)

**职责**：**暴力搜索与择优**。这是系统最核心的过滤器。

**核心逻辑**：

1. **构建动作池 (Action Pool)**：Planner\Solver\LLM全部结果入池


2. **去重**：移除重复的动作对象。
3. **全量仿真 (Brute-Force Simulation)**：
* 对 Pool 中的**每一个**动作执行 `obs.simulate(action)`。


4. **多维排序 (Hierarchical Sorting)**：
1. **安全性 (Safety)**: 必须无异常、无发散、无解列。
2. **过载消除 (Rho)**: 优先选择 `Max Rho` 最低的。
3. **稳定性 (Margin)**: 在消除过载的前提下，选择安全裕度最大的。
4. **奖励 (Reward)**: Grid2Op 返回的 Reward 最大化。
5. **成本 (Cost)**: 操作代价最小（如：优先拓扑，次选重调度）。



**输出**：返回排序第一的 `Best Actions`。

### 3.7 Module: Summarizer (学习者)

**职责**：将实证结果转化为经验。

**逻辑**：

1. **输入**：当前场景 + 最终被 Simulator 选中的 Best Action (及其来源) + 仿真结果。
2. **总结**：LLM 生成简短分析。“场景：Line 14 重度过载。策略：采用了 [来源: LLM_Fusion] 的方案，结合了 Planner 的拓扑和 Solver 的重调度，成功将 Rho 降至 0.8。”
3. **入库**：调用 `KnowledgeBase.add_experience`。

---

## 4. 核心工作流伪代码 (For Coder-AI)

```python
class ADA_Agent(BaseAgent):
    def act(self, observation, reward, done):
        # --- Phase 1: 并行生成候选集 (Candidate Generation) ---
        
        # 1. Planner (复刻 ExpertAgent): 基于物理规则生成拓扑候选
        planner_candidates = self.planner.suggest_topologies(observation) 
        # type: List[CandidateAction], source="Expert_Topo"
        
        # 2. Solver (复刻 OptimCVXPY): 基于数学优化生成调度候选
        solver_candidate = self.solver.solve_dispatch(observation)
        # type: List[CandidateAction], source="Math_Dispatch"
        
        # 3. KnowledgeBase: 获取历史参考
        history_context = self.kb.query(observation)
        
        # --- Phase 2: LLM 融合增强 (Fusion & Enhancement) ---
        
        # 4. Judger (LLM): 分析上述输入，尝试生成融合动作
        # 输入：Planner 建议, Solver 建议, 历史, 当前状态
        # 输出：经过 parser.py 验证的动作列表
        llm_candidates = self.judger.generate_fused_actions(
            observation, 
            planner_candidates, 
            solver_candidate, 
            history_context
        )
        # type: List[CandidateAction], source="LLM_Fusion"
        
        # --- Phase 3: 仿真竞技场 (The Arena) ---
        
        # 5. 构建混合动作空间 (去重)
        all_candidates = unique(planner_candidates + solver_candidate + llm_candidates)
        
        verified_results = []
        for candidate in all_candidates:
            # 对每个候选动作进行仿真
            sim_obs, _, sim_done, sim_info = observation.simulate(candidate.action_obj)
            
            # 评估仿真结果 (包含安全性检查、Rho计算、Reward预估)
            eval_result = self.simulator.evaluate(observation, sim_obs, sim_done, sim_info)
            
            # 记录结果
            candidate.simulation_result = eval_result
            if eval_result['is_safe']:
                verified_results.append(candidate)
        
        # 6. 择优 (Selection)
        # 依据：1. 是否消除过载 2. Max Rho 最小 3. Reward 最大
        if not verified_results:
             # 如果所有动作都导致游戏结束，返回空动作或 Solver 的降级方案
             final_action = self.action_space({}) 
             best_record = None
        else:
             verified_results.sort(key=lambda x: (x.simulation_result['rho_max'], -x.simulation_result['reward']))
             best_record = verified_results[0] # 取 Rho 最小的
             final_action = best_record.action_obj
        
        # --- Phase 4: 闭环学习 (Learning) ---
        
        if best_record:
            self.summarizer.summarize(observation, best_record)
        
        return final_action

```

## 5. 关键实现细节与坑点规避

1. **ExpertAgent 代码迁移**：
* 必须将 `ExpertAgent` 中依赖 `alphaDeesp` 的核心逻辑（如 `getRankedOverloads`, `compute_new_network_changes`）提取出来，重新实现到 analysis/expert 模块中，确保不依赖外部不可控环境。


2. **Parser 的重要性**：
* Judger (LLM) 输出的必须是能够被解析的指令（例如 JSON 格式描述："Action: combine, Topo: sub_id_1, Redispatch: gen_2 +10"）。`parser.py` 需要极其健壮，能够将这些文本描述精确转换为 Planner 和 Solver 生成的原始 Action 对象的组合。


3. **计算开销控制**：
* 全量仿真（Brute-force）非常消耗时间。
* **优化策略**：Planner 限制 Top-5，Solver 限制 1 个，LLM 限制 Top-3。总仿真次数控制在 10 次以内/步。可以使用多线程并行仿真 (`obs.simulate` 通常是 CPU 密集的)。


4. **Do Nothing 处理**：
* 如果当前电网状态安全（Rho < 0.9），可以直接跳过复杂的融合流程，或者仅运行 Solver 的经济调度模式，节省 Token 和计算时间。



## 6. 文件结构建议

```
ADA/
├── __init__.py              # 模块导出
├── agent.py                 # ADA_Agent 主类 (实现 grid2op.BaseAgent 接口)
├── evaluate.py              # 评估脚本 (参考 Template/evaluate.py)
├── main.py                  # 主入口脚本 (参考 Template/main.py)
├── README.md                # 使用文档和说明
├── core/                    # 核心模块目录
│   ├── __init__.py
│   ├── planner.py           # 复刻 ExpertAgent (物理规则拓扑专家)
│   ├── solver.py            # 复刻 OptimCVXPY (凸优化调度专家)
│   ├── judger.py            # LLM 融合策略生成
│   ├── simulator.py         # 仿真评价体系 (暴力搜索与排序逻辑)
│   └── summarizer.py        # 经验总结与入库逻辑
├── analysis/                # 分析模块目录 (参考 Planner/analysis/)
│   ├── __init__.py
│   ├── expert.py            # ExpertAgent分析、计算方法
│   └── optim.py             # OptimCVXPY优化问题计算方法
├── knowledgebase/           # 知识库模块
│   ├── storage/              # 知识库数据存储目录 (JSON/Vector DB 文件)
│   ├── __init__.py
│   ├── service.py           # 知识库服务接口
│   ├── VectorBase.py        # 向量数据库封装
│   └── utils.py             # 知识库工具函数
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── formatters.py            # Grid2Op Observation -> 文本描述的转换逻辑
│   ├── parser.py                # 关键：将 LLM 文本指令 -> Grid2Op Action 的解析逻辑
│   ├── prompts.py               # ADA 的 System Prompt
│   └── const.py       # CandidateAction 类定义等共享数据结构

```