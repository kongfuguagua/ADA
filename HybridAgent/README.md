# Hybrid Agent (神经符号混合架构) 使用指南

## 概述

`HybridAgent` 是一个分层混合控制架构，结合了：
- **Layer 1 (Muscle/肌肉)**: `OptimCVXPY` - 负责数值优化（再调度、切负荷）
- **Layer 2 (Brain/大脑)**: `LLM-Topology` - 负责拓扑调整（母线分裂）

## 设计哲学

### 核心思想

传统的 `OptimCVXPY` 擅长处理连续变量优化（再调度、切负荷），但在非凸问题（如拓扑改变）上无能为力。`HybridAgent` 通过以下策略解决这个问题：

1. **正常情况下**：使用 `OptimCVXPY` 进行快速、精确的数值优化
2. **优化失败时**：激活 LLM 进行拓扑调整（母线分裂），改变电网结构以缓解过载

### 控制流程

```
Observation
    ↓
[1] 快速检查: rho < 0.85? → 直接使用 OptimCVXPY
    ↓
[2] 尝试优化: OptimCVXPY.act()
    ↓
[3] 模拟验证: 检查优化后的状态
    ↓
[4] 判断: 优化后 rho > 1.05? → 激活 LLM 拓扑调整
    ↓
[5] LLM 介入: 生成母线分裂方案
    ↓
[6] 模拟验证: 比较 LLM 动作 vs 优化器动作
    ↓
[7] 选择更好的动作执行
```

## 使用方法

### 基本用法

```python
import grid2op
from lightsim2grid import LightSimBackend
from HybridAgent.hybrid_agent import HybridAgent
from utils import OpenAIChat

# 1. 创建环境
env = grid2op.make(
    "l2rpn_case14_sandbox",
    backend=LightSimBackend()
)

# 2. 初始化 LLM 客户端
llm_client = OpenAIChat(
    api_key=os.getenv("CLOUD_API_KEY"),
    base_url=os.getenv("CLOUD_BASE_URL"),
    model=os.getenv("CLOUD_MODEL", "gpt-4"),
    temperature=0.7
)

# 3. 创建 HybridAgent
agent = HybridAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    env=env,
    llm_client=llm_client,
    rho_safe=0.85,          # 安全阈值
    rho_danger=0.95,        # 危险阈值
    rho_llm_threshold=1.05  # LLM 激活阈值（优化后仍超过此值才激活）
)

# 4. 运行评估
from grid2op.Runner import Runner
runner = Runner(**env.get_params_for_runner(), agentClass=None)
runner.init_agent(agent)

res = runner.run(
    nb_episode=7,
    path_save="./logs-eval/hybrid-agent"
)

# 5. 查看统计信息
stats = agent.get_stats()
print(stats)
```

### 通过 main.py 使用

直接运行 HybridAgent 的 main.py：

```bash
export CLOUD_API_KEY=your_api_key
export CLOUD_BASE_URL=your_base_url
export CLOUD_MODEL=gpt-4

python HybridAgent/main.py
```

## 核心模块

### 1. HybridAgent (`hybrid_agent.py`)

主控制器，负责协调 OptimCVXPY 和 LLM 拓扑调整。

**关键参数**：
- `rho_safe`: 安全阈值（低于此值认为安全，直接使用 OptimCVXPY）
- `rho_danger`: 危险阈值（用于 OptimCVXPY 内部决策）
- `rho_llm_threshold`: LLM 激活阈值（优化后仍超过此值才激活 LLM）

### 2. TopologyPrompter (`topology_prompter.py`)

为 LLM 生成关于母线分裂的提示词。

**功能**：
- 提取过载线路连接的变电站信息
- 生成简洁的提示词，只包含局部信息（RAG 思路）
- 指导 LLM 建议母线分裂配置

### 3. TopologyParser (`topology_parser.py`)

将 LLM 返回的 JSON（母线分裂配置）转换为 Grid2Op Action。

**功能**：
- 解析 LLM 返回的 JSON
- 将元素名称映射到 Grid2Op 内部 ID
- 构建 Grid2Op Action（set_bus）

## LLM 输出格式

LLM 需要输出以下 JSON 格式：

```json
{
    "substation_id": 5,
    "bus_1": ["line_3", "load_2"],
    "bus_2": ["line_5", "gen_1"],
    "reasoning": "将过载线路与重载发电机分离，减少线路 3 的流量"
}
```

### 元素命名规则

- 发电机: `"gen_<id>"` (例如: `"gen_0"`, `"gen_5"`)
- 负荷: `"load_<id>"` (例如: `"load_2"`, `"load_10"`)
- 线路: `"line_<id>"` (例如: `"line_3"`, `"line_15"`)

### 约束条件

1. **必须包含过载线路**：过载线路必须放在 bus_1 或 bus_2 中
2. **完整性**：变电站内的所有设备必须被分配到 bus_1 或 bus_2（不能遗漏）
3. **非空性**：bus_1 和 bus_2 都不能为空
4. **互斥性**：每个设备只能出现在 bus_1 或 bus_2 中（不能重复）

## 统计信息

`HybridAgent` 提供以下统计信息：

```python
stats = agent.get_stats()
# {
#     "total_steps": 1000,
#     "optimizer_only": 850,      # 只使用优化器的步数
#     "llm_activated": 150,        # LLM 激活的次数
#     "llm_success": 120,          # LLM 成功的次数
#     "llm_failed": 30,            # LLM 失败的次数
#     "simulation_failures": 5,    # 模拟失败的次数
#     "optimizer_only_rate": 0.85, # 优化器使用率
#     "llm_activation_rate": 0.15, # LLM 激活率
#     "llm_success_rate": 0.80     # LLM 成功率
# }
```

## 与 OptAgent 的区别

| 特性 | OptAgent | HybridAgent |
|------|----------|-------------|
| **优化器** | OptimizationService (基于 OptimCVXPY) | OptimCVXPY (直接使用) |
| **LLM 作用** | 配置优化器参数 | 拓扑调整（母线分裂） |
| **动作类型** | 再调度、切负荷 | 拓扑调整 + 再调度、切负荷 |
| **适用场景** | 连续变量优化 | 连续 + 离散变量优化 |
| **性能** | 较慢（需要 LLM 配置参数） | 较快（大部分时间使用 OptimCVXPY） |

## 故障排除

### 1. OptimCVXPY 导入失败

确保 `example/OptimCVXPY/optimCVXPY.py` 存在，或安装 `l2rpn_baselines`：

```bash
pip install l2rpn-baselines
```

### 2. LLM 无法生成有效动作

检查：
- LLM API Key 是否正确
- 提示词是否包含足够的上下文信息
- LLM 响应格式是否符合 JSON 规范

### 3. 拓扑动作无效

可能原因：
- 变电站正在冷却中（无法操作）
- 元素名称映射错误
- Grid2Op Action 格式不正确

查看日志以获取详细错误信息。

## 未来改进

1. **多变电站拓扑调整**：支持同时调整多个变电站
2. **拓扑动作缓存**：缓存有效的拓扑动作，避免重复计算
3. **自适应阈值**：根据历史性能动态调整 `rho_llm_threshold`
4. **拓扑动作评估**：使用更复杂的评估函数选择最佳拓扑动作

