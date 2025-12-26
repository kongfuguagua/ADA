# ADA - Agile Dispatch Agent

知识驱动的复杂系统敏捷调度智能体框架

## 论文需求响应表

基于 ACL 论文 "Agile Dispatch Agent" 的设计需求：

| 论文核心概念 | 实现模块 | 状态 |
|------------|---------|------|
| 知识驱动的动态演化过程 | 整体架构 | ✅ |
| 主动状态增广 (Active State Augmentation) | `Planner/core.py` | ✅ |
| 问题-算法对齐 (Problem-Algorithm Alignment) | `Solver/matcher.py` | ✅ |
| 混合评分 (Physical + LLM-as-a-Judge) | `Judger/` | ✅ |
| MCTS 驱动的知识更新 | `Summarizer/core.py` | ✅ |
| 任务知识 (TK) 检索与更新 | `knowledgebase/` | ✅ |
| 动作知识 (AK) 检索与更新 | `knowledgebase/` | ✅ |
| 闭环反馈与 Self-Correction | `main.py` | ✅ |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        ADA 系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Planner  │───▶│  Solver  │───▶│  Judger  │───▶│Summarizer│  │
│  └────┬─────┘    └──────────┘    └────┬─────┘    └────┬─────┘  │
│       │                               │               │        │
│       │    ┌─────────────────────────┐│               │        │
│       └───▶│    KnowledgeBase        │◀───────────────┘        │
│            │   (TK + AK 向量库)       │                         │
│            └─────────────────────────┘                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Grid2Op 环境                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
ADA/
├── main.py              # 系统入口，编排器
├── readme.md            # 本文档
├── requirements.txt     # 依赖
│
├── utils/               # 通用工具层
│   ├── const.py         # 数据契约定义
│   ├── interact.py      # Agent 接口定义
│   ├── llm.py           # LLM 服务封装
│   ├── embeddings.py    # Embedding 服务封装
│   └── logger.py        # 日志系统
│
├── config/              # 配置层
│   ├── llm_config.py    # LLM API 配置
│   └── system_config.py # 系统参数配置
│
├── knowledgebase/       # 知识库
│   ├── service.py       # 知识服务层
│   └── VectorBase.py    # 向量存储
│
├── Planner/             # 规划智能体
│   ├── core.py          # 主逻辑
│   ├── prompt.py        # Prompt 模板
│   └── tools/           # 分析工具（为 Planner 决策服务）
│
├── Solver/              # 求解智能体
│   ├── solver.py        # 主入口
│   ├── feature.py       # 特征提取
│   ├── matcher.py       # 算法匹配
│   ├── prompt.py        # Prompt 模板
│   └── Template/        # 算法库
│
├── Judger/              # 评估智能体
│   ├── core.py          # 主逻辑
│   ├── prompt.py        # Prompt 模板
│   ├── Reward/          # 评分模块
│   └── Debug/           # 诊断模块
│
├── Summarizer/          # 总结智能体
│   ├── core.py          # 主逻辑 + MCTS
│   ├── prompt.py        # Prompt 模板
│   └── knowledge_updater.py  # 知识更新
│
└── env/                 # 环境交互层
    ├── config.py        # Grid2Op 配置
    ├── grid2op_env.py   # 环境封装
    └── tools.py         # 环境交互工具（发送命令、获取数据）
```

## 工具职责划分

### Planner 分析工具 (`Planner/tools/`)

为 **Planner 决策** 服务，提供分析结论：
- `grid_status_analysis`: 电网状态分析、风险评估
- `overflow_risk_analysis`: 过载风险识别、处理建议
- `generator_capacity_analysis`: 发电机容量分析、调度灵活性评估
- `load_trend_analysis`: 负荷趋势分析、峰谷预测

### 环境交互工具 (`env/tools.py`)

与 **Grid2Op 环境** 直接交互：
- `get_observation`: 获取原始观测数据
- `simulate_action`: 模拟动作效果
- `execute_action`: 执行动作
- `get_grid_info`: 获取电网拓扑信息
- `get_forecast`: 获取预测数据

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# LLM 配置
CLOUD_API_KEY=your-api-key-here
CLOUD_BASE_URL=https://api.deepseek.com
CLOUD_MODEL=deepseek-chat

# Embedding 配置（可选，默认使用 LLM 配置）
CLOUD_EMBEDDING_API_KEY=your-embedding-api-key
CLOUD_EMBEDDING_BASE_URL=https://api.openai.com/v1
CLOUD_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. 运行示例

```python
from main import ADAOrchestrator
from utils.const import EnvironmentState

# 创建编排器
orchestrator = ADAOrchestrator()

# 定义任务
state = EnvironmentState(
    user_instruction="优化电网调度，最小化发电成本",
    real_data={"total_load": 100.0}
)

# 运行
result = orchestrator.run(state)

print(f"成功: {result['success']}")
print(f"算法: {result['solution'].algorithm_used}")
print(f"目标值: {result['solution'].objective_value}")
```

### 4. 与 Grid2Op 环境集成

```python
from main import ADAOrchestrator
from env.grid2op_env import Grid2OpEnvironment
from env.config import SANDBOX_CASE14

# 创建环境
env = Grid2OpEnvironment(SANDBOX_CASE14)
env.reset()

# 创建编排器并绑定环境
orchestrator = ADAOrchestrator(env=env)

# 运行一个回合
result = orchestrator.run_episode(max_steps=100)
print(f"总奖励: {result['total_reward']}")
```

## 注意事项

1. **必须配置有效的 LLM API Key**，系统不支持 Mock 模式
2. 如果 API 额度用尽，系统会抛出异常而非静默失败
3. 建议使用 DeepSeek 或 OpenAI 兼容的 API

## 扩展指南

### 添加新的分析工具

```python
from utils.interact import BaseTool

class MyAnalysisTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_analysis"
    
    @property
    def description(self) -> str:
        return "我的分析工具"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # 实现分析逻辑
        return {"result": "分析结论"}
```

### 添加新的求解算法

```python
from Solver.Template.base import BaseAlgorithm

class MyOptimizer(BaseAlgorithm):
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="MyOptimizer",
            capabilities={
                "convex_handling": 0.8,
                "non_convex_handling": 0.6,
                "constraint_handling": 0.7,
                "speed": 0.9,
                "global_optimality": 0.5,
            }
        )
    
    def _solve_impl(self, problem) -> Dict[str, Any]:
        # 实现求解逻辑
        pass
```
