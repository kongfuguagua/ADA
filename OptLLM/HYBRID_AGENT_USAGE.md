# OptAgent (OptLLM) 使用指南

## 概述

`OptAgent` (OptLLM) 是一个混合智能优化代理，将 `OptimCVXPY`（凸优化）作为基础求解器，将 LLM 作为增强器，实现以下两个核心工作流：

1. **动作增强与优选 (Action Ensemble & Selection)**：并行生成"纯优化动作"和"LLM增强动作"，通过仿真模拟择优录取。
2. **动态参数配置 (Dynamic Meta-Optimization)**：LLM 根据当前电网场景，动态调整优化器的超参数。

## 架构设计

```
Observation → Check Rho
    ↓
Safe (rho < 0.95) → OptimCVXPY Only (Fast Path)
    ↓
Danger (rho >= 0.95) → Hybrid Intelligence
    ├─ Workflow 1: Parallel Generation
    │   ├─ Base Optimizer Action
    │   ├─ Tuned Optimizer Action (LLM 参数调优)
    │   └─ LLM Topology Action (拓扑增强)
    └─ Workflow 2: Simulation & Selection
        └─ Evaluate all candidates → Select Best
```

## 快速开始

### 基本使用

```python
import grid2op
from grid2op.Reward import RedispReward
from lightsim2grid import LightSimBackend
from OptLLM import OptAgent
from utils import OpenAIChat

# 创建环境
env = grid2op.make(
    "l2rpn_case14_sandbox",
    reward_class=RedispReward,
    backend=LightSimBackend()
)

# 初始化 LLM 客户端
llm_client = OpenAIChat(
    api_key=os.getenv("CLOUD_API_KEY"),
    base_url=os.getenv("CLOUD_BASE_URL"),
    model=os.getenv("CLOUD_MODEL", "gpt-4")
)

# 创建 OptAgent
agent = OptAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    env=env,
    llm_client=llm_client,
    llm_trigger_rho=0.95,  # 超过此阈值才激活 LLM
    rho_safe=0.85,
    rho_danger=0.95
)

# 运行一个 episode
obs = env.reset()
done = False
while not done:
    action = agent.act(obs, reward=0.0, done=done)
    obs, reward, done, info = env.step(action)
```

### 参数说明

- `llm_trigger_rho` (默认 0.95): LLM 激活阈值。当 `max(rho) < llm_trigger_rho` 时，直接使用优化器，不消耗 LLM token。
- `rho_safe` (默认 0.85): 安全阈值，传递给 `OptimCVXPY`。
- `rho_danger` (默认 0.95): 危险阈值，传递给 `OptimCVXPY`。
- `**optimizer_kwargs`: 其他传递给 `OptimCVXPY` 的参数（如 `margin_th_limit`, `penalty_curtailment_unsafe` 等）。

## 工作流详解

### Workflow 1: LLM 拓扑增强

当优化器只能处理连续变量（调度、削减、储能）时，LLM 可以建议拓扑调整（如开关线路、母线分裂）来进一步降低负载率。

**流程**：
1. 优化器生成基础动作（调度、削减等）
2. LLM 分析当前状态和优化器动作
3. LLM 判断是否需要拓扑调整
4. 如果需要，生成拓扑动作并与优化器动作组合

**示例输出**：
```python
# LLM 可能输出：
set_line_status(4, -1)  # 关闭过载线路 4
set_line_status(6, +1)  # 开启备用线路 6
```

### Workflow 2: LLM 参数调优

LLM 根据当前电网场景（过载严重程度、拥塞位置），动态调整优化器的超参数。

**可调参数**：
- `margin_th_limit`: 热极限安全裕度 (0.5 ~ 1.0)
- `penalty_curtailment`: 切负荷惩罚 (0.001 ~ 10.0)
- `penalty_redispatch`: 再调度惩罚 (0.01 ~ 1.0)
- `penalty_storage`: 储能惩罚 (0.1 ~ 1.0)

**调优策略**：
- **紧急情况** (Max Rho > 110%): 提高 `margin_th_limit` 到 0.95~1.0，降低 `penalty_curtailment` 到 0.001~0.01
- **优化器无解**: 提高 `margin_th_limit` 0.05~0.1，降低 `penalty_curtailment` 10 倍
- **模拟失败** (AC/DC 误差): 降低 `margin_th_limit` 0.05~0.1

### 仿真评估与择优

所有候选动作（基准优化、参数调优、拓扑增强）都会通过 `observation.simulate()` 进行仿真评估。

**评分指标**：
1. **安全性** (权重最高): `rho < 1.0` 且未导致 Game Over
2. **有效性**: `rho` 下降了多少
3. **成本**: 动作的经济成本（调度 > 拓扑）

**评分公式**：
```python
score = safety_score * 2.0 + recovery_score - cost
```

## 统计信息

`OptAgent` 会记录以下统计信息：

```python
agent.stats = {
    "total_steps": 0,
    "optimizer_only": 0,      # 仅使用优化器的次数
    "llm_activated": 0,       # LLM 被激活的次数
    "workflow1_count": 0,     # Workflow 1 使用次数
    "workflow2_count": 0,      # Workflow 2 使用次数
    "simulation_count": 0,    # 仿真评估次数
    "best_action_selected": {} # 每种策略被选中的次数
}
```

## 与 OptAgent 的区别

| 特性 | 旧版 OptAgent | 新版 OptAgent (OptLLM) |
|------|----------|----------------|
| 基础求解器 | OptimizationService | OptimCVXPY (直接集成) |
| 参数调优 | ✅ (通过模式选择) | ✅ (动态参数调优) |
| 拓扑增强 | ❌ | ✅ (LLM 拓扑建议) |
| 动作选择 | 单一路径 | 多候选路径 + 仿真择优 |
| 性能 | 较快 | 较慢（但更智能） |
| 适用场景 | 一般过载 | 严重过载、复杂场景 |

## 注意事项

1. **依赖**: `HybridOptAgent` 需要 `OptimCVXPY` 可用。如果导入失败，会抛出 `RuntimeError`。
2. **性能**: 在危险状态下，`HybridOptAgent` 会生成多个候选动作并进行仿真，因此比 `OptAgent` 慢，但通常能产生更好的结果。
3. **LLM Token 消耗**: 只有在 `rho >= llm_trigger_rho` 时才会调用 LLM，安全状态下直接使用优化器，不消耗 token。
4. **仿真开销**: 每个候选动作都需要进行一次仿真，如果候选动作较多，会增加计算时间。

## 故障排除

### 问题 1: 无法导入 OptimCVXPY

**错误**: `RuntimeError: 无法导入 OptimCVXPY，OptAgent 需要 OptimCVXPY 作为基础求解器`

**解决**: 确保 `example/OptimCVXPY/optimCVXPY.py` 存在且可导入。

### 问题 2: LLM 响应解析失败

**现象**: 日志中出现 "无法解析 LLM 参数调优响应" 或 "无法解析 LLM 拓扑动作"

**解决**: 
- 检查 LLM 响应格式是否符合预期
- 查看 `parser.py` 中的解析逻辑
- 考虑调整 prompt 以提高 LLM 输出格式的一致性

### 问题 3: 仿真失败

**现象**: 某些候选动作的仿真失败

**解决**: 
- 这是正常的，`OptAgent` 会自动跳过失败的候选动作
- 如果所有候选动作都失败，会返回空动作
- 检查动作是否合法（如线路编号、母线编号等）

## 示例：完整评估流程

```python
import grid2op
from grid2op.Runner import Runner
from OptLLM import OptAgent
from utils import OpenAIChat
import os

# 创建环境
env = grid2op.make("l2rpn_case14_sandbox")

# 初始化 LLM
llm_client = OpenAIChat(
    api_key=os.getenv("CLOUD_API_KEY"),
    base_url=os.getenv("CLOUD_BASE_URL"),
    model=os.getenv("CLOUD_MODEL")
)

# 创建 agent
agent = OptAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    env=env,
    llm_client=llm_client,
    llm_trigger_rho=0.95
)

# 运行评估
runner = Runner(**env.get_params_for_runner(), agentClass=None)
runner.run(
    env=env,
    agent=agent,
    nb_episode=1,
    max_iter=1000
)

# 查看统计信息
print(agent.stats)
```

## 参考

- [OptLLM README](README.md) - OptLLM 项目总体说明
- [OptimCVXPY 工作原理](../example/OptimCVXPY/OptimCVXPY工作原理.md) - OptimCVXPY 详细说明
- [HybridAgent](../HybridAgent/README.md) - 另一个混合智能体实现

