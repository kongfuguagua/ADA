# ADA: Agile Dispatch Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ADA (Agile Dispatch Agent)** 是一个基于双时间尺度演化的端到端智能体架构，用于解决复杂物理系统（如电力系统）中的自动化优化问题。ADA 通过解耦任务知识和动作知识的演化过程，有效解决了大语言模型在操作研究中的**归因模糊性（Attribution Ambiguity）**问题。

## 📋 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
- [实验结果](#实验结果)
- [引用](#引用)

## 🎯 核心特性

### 1. 双时间尺度演化（Dual-Timescale Evolution）

ADA 将操作自动化解耦为两个独立的演化过程：

- **慢时间尺度（Slow Timescale）**：更新建模可行性区域，通过知识库持续学习历史经验
- **快时间尺度（Fast Timescale）**：采用主动状态增广和动态流形特征匹配，将非结构化需求精确转化为最优解策略

### 2. 主动状态增广（Active State Augmentation）

利用蒙特卡洛树搜索（MCTS）机制，将模糊观测精确映射到严格的数学公式，确保在动态环境中的建模准确性和物理可行性。

### 3. 动态流形特征匹配（Dynamic Manifold Feature Matching）

基于问题流形特征自动检索和适配最优求解器，根据优化问题的数学特性（凸性、约束刚度、求解复杂度等）动态选择算法，显著增强系统在处理非结构化需求时的鲁棒性。

### 4. 混合智能架构（Hybrid Intelligence）

- **Planner（规划器）**：基于物理规则和专家知识的拓扑动作生成（复刻 ExpertAgent）
- **Solver（求解器）**：基于凸优化的数学调度方案生成（复刻 OptimCVXPY）
- **Judger（判断器）**：LLM 驱动的策略融合和增强
- **Simulator（仿真器）**：暴力搜索与实证排序，确保最优动作选择

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Grid2Op Environment                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ Observation
                        ▼
        ┌───────────────────────────────────────┐
        │         ADA Agent (Main Loop)           │
        └───────────────┬─────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌──────────┐
   │ Planner │    │ Solver  │    │Knowledge │
   │(Expert) │    │(Optim)  │    │  Base    │
   └────┬────┘    └────┬────┘    └────┬─────┘
        │              │               │
        └──────────────┼───────────────┘
                       │
                       ▼
                ┌──────────────┐
                │   Judger     │
                │   (LLM)      │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Simulator   │
                │  (Arena)     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Best Action │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Summarizer   │
                │ (Learning)   │
                └──────────────┘
```

### 工作流程

1. **并行候选生成**：Planner 和 Solver 并行生成拓扑和调度候选动作
2. **知识检索**：从知识库中检索相似历史场景
3. **LLM 融合**：Judger 分析各方输入，生成融合策略
4. **仿真竞技场**：对所有候选动作进行暴力搜索和排序
5. **最优选择**：基于安全性、过载消除、稳定性和奖励等多维指标选择最优动作
6. **闭环学习**：Summarizer 将成功经验存入知识库

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Grid2Op 1.12.2+
- CUDA（可选，用于加速计算）

### 安装

1. **克隆仓库**

```bash
git clone <repository-url>
cd ADA
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

复制 `env.example` 为 `.env` 并配置 LLM API 信息：

```bash
cp env.example .env
```

编辑 `.env` 文件，配置以下变量：

```env
# Embedding 模型（用于知识库向量检索）
OPENAI_API_MODEL=Pro/BAAI/bge-m3
OPENAI_API_KEY=your-embedding-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

# Chat 模型（用于 Judger 和 Summarizer）
CLOUD_MODEL=Qwen3-32B
CLOUD_API_KEY=your-chat-api-key
CLOUD_BASE_URL=https://ai.api.coregpu.cn/v1/
```

### 基本使用

#### 方法 1：使用评估脚本（推荐）

```bash
cd ADA
python main.py
```

#### 方法 2：在代码中使用

```python
import grid2op
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend
from ADA.agent import ADA_Agent
from utils.llm import OpenAIChat

# 1. 创建环境
env = grid2op.make(
    "l2rpn_wcci_2022",
    reward_class=RedispReward,
    backend=LightSimBackend(),
    other_rewards={
        "bridge": BridgeReward,
        "overflow": CloseToOverflowReward,
        "distance": DistanceReward
    }
)

# 2. 创建 LLM 客户端
llm_client = OpenAIChat()

# 3. 创建 ADA Agent
agent = ADA_Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    llm_client=llm_client,
    rho_danger=0.95,  # 危险阈值
    rho_safe=0.85,    # 安全阈值
    max_planner_candidates=5,
    max_llm_candidates=3,
    enable_knowledge_base=True
)

# 4. 运行评估
from ADA.evaluate import evaluate

results = evaluate(
    env=env,
    agent=agent,
    nb_episode=7,
    verbose=True,
    logs_path="./logs-eval/ada"
)
```

### 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `rho_danger` | 危险阈值，超过此值进入危险模式 | 0.95 |
| `rho_safe` | 安全阈值，低于此值进入安全模式 | 0.85 |
| `max_planner_candidates` | Planner 返回的最大候选数 | 5 |
| `max_llm_candidates` | LLM 返回的最大候选数 | 3 |
| `enable_knowledge_base` | 是否启用知识库 | True |
| `llm_temperature` | LLM 温度参数 | 0.7 |

## 📁 项目结构

```
ADA/
├── ADA/                          # ADA 核心实现
│   ├── __init__.py
│   ├── agent.py                  # ADA Agent 主类
│   ├── main.py                   # 主入口脚本
│   ├── evaluate.py               # 评估脚本
│   ├── readme.md                 # ADA 设计文档
│   ├── core/                     # 核心模块
│   │   ├── planner.py            # 规划器（拓扑专家）
│   │   ├── solver.py             # 求解器（优化专家）
│   │   ├── judger.py             # 判断器（LLM 融合）
│   │   ├── simulator.py          # 仿真器（竞技场）
│   │   └── summarizer.py         # 总结器（学习模块）
│   ├── analysis/                 # 分析模块
│   │   └── expert_insight.py     # 专家洞察服务
│   ├── knowledgebase/            # 知识库模块
│   │   ├── service.py            # 知识库服务
│   │   ├── VectorBase.py         # 向量数据库封装
│   │   └── utils.py              # 工具函数
│   └── utils/                    # 工具模块
│       ├── definitions.py        # 数据结构定义
│       ├── formatters.py         # 格式化工具
│       ├── parser.py             # 动作解析器
│       └── prompts.py            # Prompt 模板
│
├── ADA_Planner/                    # ADA_Planner Baseline（对比基线）
│   ├── agent.py                   # ReAct 风格智能体
│   └── ...
│
├── example/                        # 示例实现
│   ├── DoNothing/                 # 空动作基线
│   ├── ExpertAgent/               # 专家智能体
│   ├── OptimCVXPY/                # 优化求解器智能体
│   ├── PPO_SB3/                   # PPO 强化学习基线
│   └── Template/                   # 模板代码
│
├── HybridAgent/                    # 混合智能体
├── OptLLM/                         # 优化 LLM 智能体
├── ReAct_Baseline/                 # ReAct 基线
├── utils/                          # 共享工具
│   ├── llm.py                     # LLM 客户端
│   ├── logger.py                  # 日志工具
│   ├── embeddings.py              # 嵌入向量工具
│   └── json_utils.py              # JSON 工具
│
├── logs-eval/                      # 评估日志
├── result/                         # 实验结果
├── rl_saved_model/                 # RL 模型保存
├── requirements.txt                # 依赖列表
├── env.example                     # 环境变量示例
└── README.md                       # 本文件
```

## 🔧 核心模块

### Planner（规划器）

基于物理规则和专家知识的拓扑动作生成模块，复刻了 ExpertAgent 的核心算法：

- **敏感度分析**：计算过载线路的敏感度
- **影响图构建**：构建电网影响图
- **拓扑搜索**：执行节点分裂和线路开关搜索
- **状态增广**：生成 Top-N 拓扑变更方案

**特点**：纯 Python 实现，不依赖 LLM，提供基于物理规则的离散拓扑动作。

### Solver（求解器）

基于凸优化的数学调度方案生成模块，复刻了 OptimCVXPY 的核心逻辑：

- **DC 潮流模型**：基于直流潮流模型构建优化问题
- **目标函数**：最小化过载（危险模式）或最小化成本/最大化余量（安全模式）
- **约束条件**：线路热极限、发电机爬坡、储能限制

**特点**：纯 Python + CVXPY 实现，不依赖 LLM，提供基于数学优化的连续变量动作。

### Judger（判断器）

LLM 驱动的策略融合和增强模块：

- **多源信息融合**：分析 Planner 和 Solver 的输出
- **历史经验利用**：结合知识库中的历史案例
- **策略生成**：生成融合策略（如拓扑+重调度的组合）
- **规范化输出**：通过 parser 将文本指令转换为 Grid2Op Action

**特点**：利用 LLM 的推理能力发现单一方法无法覆盖的融合策略。

### Simulator（仿真器）

暴力搜索与实证排序模块，系统的核心过滤器：

- **动作池构建**：收集所有候选动作（Planner + Solver + LLM）
- **全量仿真**：对每个候选动作执行 `observation.simulate()`
- **多维排序**：
  1. **安全性**：必须无异常、无发散、无解列
  2. **过载消除**：优先选择最大负载率最低的
  3. **稳定性**：选择安全裕度最大的
  4. **奖励**：Grid2Op 返回的奖励最大化
  5. **成本**：操作代价最小

**特点**：纯粹依据仿真结果择优执行，确保最优动作选择。

### Summarizer（总结器）

经验总结与知识库更新模块：

- **场景分析**：分析当前场景和最终选中的动作
- **经验总结**：生成结构化经验描述
- **知识入库**：将成功经验存入知识库，实现持续学习

**特点**：实现闭环学习，使系统能够从历史经验中持续改进。

## 📊 实验结果

### L2RPN 基准测试

在 L2RPN（Learning to Run a Power Network）基准测试中，ADA 取得了以下成果：

#### 1. 运行安全性

- **接近 100% 的操作安全性**：在所有测试场景中，ADA 均能维持电网稳定运行
- **平均负载率**：$\bar{\rho} \approx 0.715$，显著优于基线方法（ReAct: $\bar{\rho} \approx 0.782$）
- **峰值抑制**：最大负载率严格控制在 $0.980$，而基线方法经常超过 $1.0$ 导致游戏结束

#### 2. 长期稳定性

在长时域场景（8000 步）中：

- **生存能力**：ADA 成功导航整个 8000 步时域，而大多数基线方法在前 4000 步内出现"突然死亡"现象
- **稳定性**：ADA 的轨迹保持平滑且持续低位，而基线方法（如 ReAct）表现出高频振荡
- **安全裕度**：ADA 主动维持更大的安全缓冲，应对随机注入

#### 3. Pareto 最优权衡

在高度约束的搜索空间中，ADA 实现了**优越的 Pareto 最优权衡**，显著优于最先进的 LLM 智能体基线。

### 与基线方法对比

| 方法 | 平均负载率 | 峰值负载率 | 生存步数 | 安全性 |
|------|-----------|-----------|---------|--------|
| **ADA** | **0.715** | **0.980** | **8000+** | **✓** |
| ReAct | 0.782 | >1.0 | 4000-7000 | ✗ |
| DoNothing | - | 1.356 | <4000 | ✗ |
| SAC | - | - | <4000 | ✗ |
| HRL | - | 1.593 | <4000 | ✗ |

### 理论保证

论文提供了理论分析，证明**异步更新机制在非平稳环境中保证渐近收敛到局部最优策略**。

## 📚 相关项目

- **ADA_Planner**：基于 ReAct 范式的对比基线
- **ExpertAgent**：基于专家规则的拓扑优化智能体
- **OptimCVXPY**：基于凸优化的调度智能体
- **PPO_SB3**：基于强化学习的基线方法

## 🔬 理论背景

### 归因模糊性问题

在端到端的操作研究自动化中，当解决方案失败时，很难确定应该改进：
- LLM 的反思能力？
- 求解算法？
- 建模精度？

这种**归因模糊性**导致无效的搜索轨迹和逻辑幻觉。

### ADA 的解决方案

通过**双时间尺度演化**解耦建模和求解过程：

- **慢时间尺度**：更新建模可行性区域（Task Knowledge $\mathcal{T}\mathcal{K}$）
- **快时间尺度**：适配非结构化需求和解决方案策略（Action Knowledge $\mathcal{A}\mathcal{K}$）

这种解耦机制消除了归因模糊性，使系统能够高效地端到端自动化操作研究。

## 📖 引用

如果您在研究中使用了 ADA，请引用：

```bibtex
@article{ada2025,
  title={ADA: Resolving Attribution Ambiguity in Automated Optimization via Two-Time-Scale Stochastic Approximation},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请通过 Issue 或 Pull Request 提交您的建议和改进。

## 📧 联系方式

如有问题或建议，请通过 Issue 联系我们。

---

**注意**：本项目是研究代码，用于学术研究目的。在生产环境中使用前，请进行充分测试和验证。

