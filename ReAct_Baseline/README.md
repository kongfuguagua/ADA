# ReAct Baseline Agent

基于 ReAct (Reasoning + Acting) 范式的电网调度智能体，作为与 ADA 智能体的对比 Baseline。

## 文件结构

```
ReAct_Baseline/
├── __init__.py          # 模块导出
├── agent.py             # 核心类 ReActAgent，实现 grid2op.BaseAgent 接口
├── formatters.py        # Grid2Op Observation -> 文本描述的转换逻辑
├── parser.py            # 文本指令 -> Grid2Op Action 的解析逻辑
├── prompts.py           # ReAct 的 System Prompt 和 Few-shot Examples
└── README.md            # 本文件
```

## 使用方法

### 方法 1: 使用评估程序（推荐）

```bash
# 使用命令行参数
python ReAct_Baseline/evaluate.py \
    --data_dir l2rpn_case14_sandbox \
    --nb_episode 7 \
    --verbose \
    --llm_api_key your-api-key \
    --llm_base_url https://api.deepseek.com \
    --llm_model deepseek-chat \
    --max_react_steps 3
```

或者使用 `main.py`：

```bash
# 需要先配置环境变量
export CLOUD_API_KEY="your-api-key"
export CLOUD_BASE_URL="https://api.deepseek.com"
export CLOUD_MODEL="deepseek-chat"

# 运行
python ReAct_Baseline/main.py
```

### 方法 2: 在代码中使用

```python
from grid2op import make
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend

from ReAct_Baseline import ReActAgent
from ADA.utils.llm import OpenAIChat

# 1. 创建环境
env = make(
    "l2rpn_case14_sandbox",
    reward_class=L2RPNReward,
    backend=LightSimBackend()
)

# 2. 创建 LLM 客户端
llm_client = OpenAIChat(
    model="deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

# 3. 创建 ReAct Agent
agent = ReActAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    llm_client=llm_client,
    max_react_steps=3,  # ReAct 循环最大重试次数
    name="ReActAgent",
    enable_rag=True,  # 启用 RAG 功能（默认 True）
    knowledge_path=None  # 知识库路径，None 则使用 ADA/knowledgebase/storage
)

# 4. 运行
obs = env.reset()
done = False
while not done:
    action = agent.act(obs, reward=0.0, done=False)
    obs, reward, done, info = env.step(action)
```

## 核心特性

1. **纯 LLM 推理**: 不依赖数学优化求解器，完全依赖 LLM 的推理能力
2. **ReAct 循环**: 通过 Thought-Action-Observation 循环，让 LLM 根据环境反馈调整策略
3. **安全验证**: 使用 `observation.simulate()` 验证动作安全性，避免危险操作
4. **错误反馈**: 如果动作非法或不安全，将错误信息反馈给 LLM，让其重新思考
5. **缓解策略支持**: 允许渐进式改善，不要求一步到位完全消除过载（v2.0 新增）
6. **预防性调度**: 默认在负载率达到 92% 时开始介入，避免等到过载（v2.0 新增）
7. **拓扑感知**: 提供发电机和线路的拓扑信息（变电站连接），帮助 LLM 做出更合理的调度决策（v2.0 新增）
8. **RAG 增强**: 集成 ADA 知识库，在决策前检索相似历史场景，提供参考策略（v2.1 新增）

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
- `do_nothing()`: 保持现状

## 配置参数

### Agent 参数
- `max_react_steps`: ReAct 循环的最大重试次数（默认 3）
- `rho_danger`: 危险阈值，当最大负载率超过此值时才调用 LLM（默认 0.92，即负载率 > 92% 时介入）
- `rho_safe`: 安全阈值，当最大负载率低于此值时直接返回 do_nothing（默认 0.80，即负载率 < 80% 时提前终止）
- `min_redispatch_threshold`: 动作剪枝阈值，小于此值的 redispatch 将被过滤（默认 0.5 MW）
- `llm_client`: LLM 客户端（必须提供 OpenAIChat 实例）
- `enable_rag`: 是否启用 RAG 功能（默认 True）
- `knowledge_path`: 知识库存储路径（默认使用 `ADA/knowledgebase/storage`）
- `config`: 配置字典，可以一次性设置多个参数（可选）

### 配置示例

```python
# 使用默认配置
agent = ReActAgent(..., enable_rag=True)

# 自定义配置
agent = ReActAgent(
    ...,
    rho_danger=0.95,  # 更保守的阈值
    rho_safe=0.75,   # 更激进的提前终止
    min_redispatch_threshold=1.0,  # 更大的剪枝阈值
    enable_rag=True
)

# 使用配置字典
agent = ReActAgent(
    ...,
    config={
        "rho_danger": 0.95,
        "rho_safe": 0.75,
        "min_redispatch_threshold": 1.0,
        "rag_top_k": 3  # RAG 检索返回 3 条
    }
)
```

### 评估程序参数
- `--data_dir`: 环境名称或路径（必需）
- `--nb_episode`: 评估的 episode 数量（默认 1）
- `--max_steps`: 每个 episode 的最大步数（-1 表示不限制，默认 -1）
- `--logs_dir`: 日志保存路径（默认 `./logs-eval/react-baseline`）
- `--verbose`: 显示详细输出
- `--gif`: 保存 GIF 可视化（需要 l2rpn_baselines）
- `--llm_model`: LLM 模型名称（默认从环境变量 `CLOUD_MODEL` 读取）
- `--llm_api_key`: LLM API Key（默认从环境变量 `CLOUD_API_KEY` 读取）
- `--llm_base_url`: LLM Base URL（默认从环境变量 `CLOUD_BASE_URL` 读取）
- `--llm_temperature`: LLM 温度参数（默认 0.7）
- `--max_react_steps`: ReAct 循环最大重试次数（默认 3）
- `--rho_danger`: 危险阈值（默认 0.92，即负载率 > 92% 时调用 LLM）

## 重要改进

### v2.1 深度优化（最新）

#### 1. RAG 预验证机制
- **历史动作预仿真**：在将 RAG 检索到的历史经验发给 LLM 之前，先进行预仿真验证
- **智能标注**：如果历史动作在当前环境下安全，标注为"强烈推荐"；如果不可行，标注为"仅参考思路"
- **上下文压缩**：只在第一轮 ReAct 注入 RAG 上下文，后续步骤不再重复，大幅节省 Token

#### 2. 量化仿真反馈
- **详细反馈**：`_simulate_action` 现在返回详细的量化反馈，包括：
  - 负载率变化量（delta_rho）
  - 过载线路数量变化
  - 方向性指导（"方向正确但力度不够" vs "方向错误"）
- **智能提示**：LLM 可以知道"虽然失败了，但离成功还有多远"

#### 3. 语义增强的 RAG 检索
- **过载严重程度描述**：根据负载率自动分类为 "Severe/Moderate/Slight overload"
- **过载线路数量描述**：区分单线路、少量线路、多线路过载
- **混合特征**：结合语义描述和具体 Line ID，提高检索命中率

#### 4. 动作剪枝（Action Pruning）
- **过滤微小调整**：自动过滤掉小于阈值（默认 0.5 MW）的 redispatch 操作
- **NaN/None 处理**：增强了对异常值的处理
- **提升效率**：避免浪费步骤在无效的微小调整上

#### 5. 奖励感知的 Prompt
- **最小干预原则**：明确要求优先选择调整量最小、操作步数最少的方案
- **Token 节约**：引导 LLM 在自信时直接输出 Action，无需冗长描述

#### 6. 配置抽象
- **统一配置管理**：所有硬编码参数提取到 `DEFAULT_CONFIG`
- **灵活配置**：支持通过参数或配置字典自定义所有阈值
- **便捷访问**：常用配置可通过属性直接访问

#### 7. 提前终止优化
- **双重阈值**：`rho_safe`（0.80）用于提前终止，`rho_danger`（0.92）用于预防性调度
- **节省资源**：系统非常安全时直接返回 do_nothing，跳过 LLM 调用

#### 8. 历史压缩
- **自动压缩**：当 ReAct 步数超过 2 步时，自动压缩历史对话
- **保留关键信息**：只保留最近的 2 轮对话，之前的摘要化处理
- **Token 优化**：大幅减少长对话场景下的 Token 消耗

### v2.0 改进

### 1. 缓解策略（Mitigation Strategy）
解决了"完美主义陷阱"问题：当电网已经过载时，系统会接受**缓解性动作**：
- ✅ **接受**：动作后过载线路数量减少
- ✅ **接受**：动作后最大负载率明显下降（即使仍有过载）
- ❌ **拒绝**：动作后情况恶化或没有改善
- ❌ **拒绝**：动作导致极度过载（>150%）或电网崩溃

**关键理解**：在过载情况下，不要求一步到位完全消除过载。只要动作能让情况变好，就是有效的。可以分多步逐步缓解。

### 2. 预防性调度（Preventive Action）
- 默认阈值从 `1.0` 降低到 `0.92`
- 在负载率达到 92% 时就开始介入，避免等到过载
- 给 Agent 更多时间窗口来预防和缓解问题

### 3. 拓扑感知（Topology Awareness）
- 观测中显示发电机和线路连接的变电站信息
- LLM 可以利用拓扑关系做出更合理的调度决策
- 例如：调整与过载线路连接的变电站附近的发电机出力

## RAG 功能说明（v2.1）

ReAct Agent 现在集成了 ADA 的知识库模块，能够在决策前检索相似的历史场景：

### 工作原理

1. **检索阶段**: 在每次 `act()` 调用时，根据当前观测（过载线路、负载率等）构建查询，从知识库中检索相似历史场景
2. **增强阶段**: 将检索到的历史经验注入到 System Prompt 中，作为 LLM 的参考上下文
3. **生成阶段**: LLM 参考历史经验生成指令，经由现有的仿真模块验证后执行

### 使用要求

- 需要配置 Embedding API（通过环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`）
- 知识库路径默认指向 `ADA/knowledgebase/storage`，确保该路径存在且包含知识库数据
- 如果 ADA 知识库不存在或 RAG 初始化失败，Agent 会自动降级为纯 LLM 模式（不启用 RAG）

### 配置示例

```python
# 启用 RAG（默认）
agent = ReActAgent(..., enable_rag=True)

# 禁用 RAG
agent = ReActAgent(..., enable_rag=False)

# 使用自定义知识库路径
agent = ReActAgent(..., knowledge_path="/path/to/custom/knowledge/storage")
```

## 注意事项

- 需要配置 LLM API Key（通过环境变量或命令行参数）
- 如果启用 RAG，还需要配置 Embedding API（`OPENAI_API_KEY` 和 `OPENAI_BASE_URL`）
- 动作解析使用正则表达式，要求 LLM 严格按照格式输出
- 如果多次重试后仍无法找到安全动作，会返回 `do_nothing()`
- 评估程序会自动统计 Agent 的性能指标（成功率、平均 ReAct 循环次数等）
- **新版本行为变化**：Agent 可能会连续多步进行再调度，逐步缓解过载，这是正常且期望的行为
- **RAG 注意事项**：LLM 会批判性地参考历史经验，但仍需通过仿真验证，避免盲目照抄历史动作

