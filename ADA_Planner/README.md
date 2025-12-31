# ADA_Planner Baseline Agent

基于 ADA_Planner (Reasoning + Acting) 范式的电网调度智能体，作为与 ADA 智能体的对比 Baseline。

## 文件结构

```
ADA_Planner/
├── __init__.py          # 模块导出
├── agent.py             # 核心类 ADA_Planner，实现 grid2op.BaseAgent 接口
├── formatters.py        # Grid2Op Observation -> 文本描述的转换逻辑
├── parser.py            # 文本指令 -> Grid2Op Action 的解析逻辑
├── prompts.py           # ADA_Planner 的 System Prompt 和 Few-shot Examples
├── analysis/            # 分析模块
│   ├── expert_insight.py    # 专家洞察服务（Expert-Augmented 核心）
└── README.md            # 本文件
```

## 使用方法

### 方法 1: 使用评估程序（推荐）

```bash
# 使用命令行参数
python ADA_Planner/evaluate.py \
    --data_dir l2rpn_case14_sandbox \
    --nb_episode 7 \
    --verbose \
    --llm_api_key your-api-key \
    --llm_base_url https://api.deepseek.com \
    --llm_model deepseek-chat \
    --max_ADA_Planner_steps 3
```

或者使用 `main.py`：

```bash
# 需要先配置环境变量
export CLOUD_API_KEY="your-api-key"
export CLOUD_BASE_URL="https://api.deepseek.com"
export CLOUD_MODEL="deepseek-chat"

# 运行
python ADA_Planner/main.py
```

### 方法 2: 在代码中使用

```python
from grid2op import make
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

from ADA_Planner import ADA_Planner
from ADA.utils.llm import OpenAIChat

# 1. 创建环境
env = make(
    "l2rpn_case14_sandbox",
    reward_class=RedispReward,
    backend=LightSimBackend(),
    other_rewards={
        "bridge": BridgeReward,
        "overflow": CloseToOverflowReward,
        "distance": DistanceReward
    }
)

# 2. 创建 LLM 客户端
llm_client = OpenAIChat(
    model="deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

# 3. 创建 ADA_Planner Agent
agent = ADA_Planner(
    action_space=env.action_space,
    observation_space=env.observation_space,
    llm_client=llm_client,
    max_ADA_Planner_steps=3,  # ADA_Planner 循环最大重试次数
    name="ADA_Planner"
)

# 4. 运行
obs = env.reset()
done = False
while not done:
    action = agent.act(obs, reward=0.0, done=False)
    obs, reward, done, info = env.step(action)
```

## 核心特性

1. **Expert-Augmented ADA_Planner (EARA)**: 集成 ExpertAgent 的数学求解能力，实现"符号引导，神经执行"（v3.0 新增）
2. **纯 LLM 推理**: 不依赖数学优化求解器，完全依赖 LLM 的推理能力
3. **ADA_Planner 循环**: 通过 Thought-Action-Observation 循环，让 LLM 根据环境反馈调整策略
4. **安全验证**: 使用 `observation.simulate()` 验证动作安全性，避免危险操作
5. **错误反馈**: 如果动作非法或不安全，将错误信息反馈给 LLM，让其重新思考
6. **缓解策略支持**: 允许渐进式改善，不要求一步到位完全消除过载（v2.0 新增）
7. **预防性调度**: 默认在负载率达到 92% 时开始介入，避免等到过载（v2.0 新增）
8. **拓扑感知**: 提供发电机和线路的拓扑信息（变电站连接），帮助 LLM 做出更合理的调度决策（v2.0 新增）

## 工作流程

1. **Observe**: 获取环境观测，转换为文本描述
2. **Think**: LLM 分析当前状态，生成思考过程
3. **Act**: LLM 生成文本动作指令（如 `redispatch(2, 10.5)`）
4. **Execute**: 
   - 解析动作为 Grid2Op Action
   - 在模拟环境中验证动作安全性
   - 如果安全: 输出该动作
   - 如果不安全/非法: 生成错误反馈，返回第 2 步

## 支持的动作

- `redispatch(gen_id, amount_mw)`: 调整发电机出力
- `set_line_status(line_id, status)`: 改变线路状态（+1 开启，-1 关闭）
- `execute_expert_solution(index)`: 执行专家系统推荐的拓扑动作（v3.0 新增，推荐使用）
- `do_nothing()`: 保持现状

## 配置参数

### Agent 参数
- `max_ADA_Planner_steps`: ADA_Planner 循环的最大重试次数（默认 3）
- `rho_danger`: 危险阈值，当最大负载率超过此值时才调用 LLM（默认 0.92，即负载率 > 92% 时介入）
- `llm_client`: LLM 客户端（必须提供 OpenAIChat 实例）
- `grid_name`: 电网名称，用于 ExpertInsight（默认 "IEEE14"，可选 "IEEE118", "IEEE118_R2"）
- `use_expert_insight`: 是否启用专家洞察服务（默认 True，需要安装 alphaDeesp）

### 评估程序参数
- `--data_dir`: 环境名称或路径（必需）
- `--nb_episode`: 评估的 episode 数量（默认 1）
- `--max_steps`: 每个 episode 的最大步数（-1 表示不限制，默认 -1）
- `--logs_dir`: 日志保存路径（默认 `./logs-eval/ADA_Planner-baseline`）
- `--verbose`: 显示详细输出
- `--gif`: 保存 GIF 可视化（需要 l2rpn_baselines）
- `--llm_model`: LLM 模型名称（默认从环境变量 `CLOUD_MODEL` 读取）
- `--llm_api_key`: LLM API Key（默认从环境变量 `CLOUD_API_KEY` 读取）
- `--llm_base_url`: LLM Base URL（默认从环境变量 `CLOUD_BASE_URL` 读取）
- `--llm_temperature`: LLM 温度参数（默认 0.7）
- `--max_ADA_Planner_steps`: ADA_Planner 循环最大重试次数（默认 3）
- `--rho_danger`: 危险阈值（默认 0.92，即负载率 > 92% 时调用 LLM）

## 重要改进

### v3.0: Expert-Augmented ADA_Planner (EARA)

**核心设计理念**: "Symbolic Guidance, Neural Execution"（符号引导，神经执行）

将 ExpertAgent 中经过验证的、基于规则和数学的求解能力完全剥离并封装为 `ExpertInsightService`。在 ADA_Planner 循环启动时，强制执行一次专家诊断，将诊断结果作为**强提示**注入给 LLM。

#### 主要特性：
1. **专家洞察服务（ExpertInsightService）**
   - 复用 ExpertAgent 的核心算法（影响图构建、候选拓扑生成、模拟打分）
   - 不直接执行动作，而是返回分析数据（Top-K 推荐方案）
   - 每个方案包含：评分（0-4分）、预期效果、动作对象

2. **快捷执行指令（execute_expert_solution）**
   - LLM 可以直接使用 `execute_expert_solution(index)` 执行专家推荐的拓扑动作
   - 零错误率：直接使用专家系统计算好的动作对象，无需手动构造复杂的拓扑向量
   - 完美复用 ExpertAgent 的计算结果

3. **降级策略**
   - 如果 ExpertInsight 不可用（未安装 alphaDeesp）或失败，系统返回空提示，让 LLM 基于物理知识自行决策
   - ExpertInsight 内部已有降级逻辑：如果找不到拓扑解，会自动提供基于灵敏度分析的再调度建议

#### 使用要求：
- 需要安装 `alphaDeesp` 包：`pip install alphaDeesp`
- 如果未安装，Agent 会自动降级到备用分析器，功能不受影响

### v2.0: 缓解策略、预防性调度、拓扑感知

#### 1. 缓解策略（Mitigation Strategy）
解决了"完美主义陷阱"问题：当电网已经过载时，系统会接受**缓解性动作**：
- ✅ **接受**：动作后过载线路数量减少
- ✅ **接受**：动作后最大负载率明显下降（即使仍有过载）
- ❌ **拒绝**：动作后情况恶化或没有改善
- ❌ **拒绝**：动作导致极度过载（>150%）或电网崩溃

**关键理解**：在过载情况下，不要求一步到位完全消除过载。只要动作能让情况变好，就是有效的。可以分多步逐步缓解。

#### 2. 预防性调度（Preventive Action）
- 默认阈值从 `1.0` 降低到 `0.92`
- 在负载率达到 92% 时就开始介入，避免等到过载
- 给 Agent 更多时间窗口来预防和缓解问题

#### 3. 拓扑感知（Topology Awareness）
- 观测中显示发电机和线路连接的变电站信息
- LLM 可以利用拓扑关系做出更合理的调度决策
- 例如：调整与过载线路连接的变电站附近的发电机出力

## 注意事项

- 需要配置 LLM API Key（通过环境变量或命令行参数）
- 动作解析使用正则表达式，要求 LLM 严格按照格式输出
- 如果多次重试后仍无法找到安全动作，会返回 `do_nothing()`
- 评估程序会自动统计 Agent 的性能指标（成功率、平均 ADA_Planner 循环次数等）
- **v2.0 行为变化**：Agent 可能会连续多步进行再调度，逐步缓解过载，这是正常且期望的行为
- **v3.0 Expert-Augmented**：
  - 推荐安装 `alphaDeesp` 以获得最佳性能
  - 专家系统会在检测到过载时自动运行，提供 Top-K 推荐方案
  - LLM 应优先考虑评分 3-4 分的专家方案，使用 `execute_expert_solution(index)` 执行
  - 如果 ExpertInsight 不可用或失败，系统会返回空提示，让 LLM 基于物理知识自行决策

