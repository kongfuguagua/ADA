# 环境模块 (env)

Grid2Op 环境封装模块，提供统一的接口用于创建和交互 L2RPN 比赛环境。

## 模块结构

```
env/
├── __init__.py          # 模块导出接口
├── config.py            # 环境配置定义
├── grid2op_env.py      # 环境封装和仿真器
├── tools.py             # 环境交互工具
└── README.md            # 本文档
```

## 核心约定

1. **约定大于配置**：使用预定义配置，减少手动设置
2. **统一接口**：所有环境通过 `create_grid2op_env()` 创建
3. **职责分离**：
   - `Grid2OpEnvironment`: 环境管理器，提供高级接口
   - `env/tools.py`: 环境交互工具（发送命令、获取数据）
   - `Planner/tools/`: 分析工具（统计数据、规则检测）

## 快速开始

### 1. 创建环境

```python
from env import create_grid2op_env

# 方式1: 使用配置名称（推荐）
env = create_grid2op_env("wcci_2022", seed=42)
obs = env.reset()

# 方式2: 使用预定义配置对象
from env import WCCI_2022
env = create_grid2op_env(WCCI_2022, seed=42)
obs = env.reset()

# 方式3: 直接使用 Grid2Op 环境名（自动创建临时配置）
env = create_grid2op_env("l2rpn_wcci_2022", seed=42)
```

### 2. 查看可用配置

```python
from env import list_env_configs, get_env_config, print_config_info

# 列出所有配置
configs = list_env_configs()
print(configs)
# ['neurips_2020_track1', 'neurips_2020_track2', 'icaps_2021', 'wcci_2022', 'sandbox_case14']

# 获取配置
config = get_env_config("wcci_2022")
print(config.name)  # WCCI 2022 - 未来能源与碳中和
print(config.has_storage)  # True

# 打印详细信息
print_config_info(config)
```

### 3. 环境交互

```python
# 重置环境
obs = env.reset()

# 执行动作
action = env.get_do_nothing_action()
obs, reward, done, info = env.step(action)

# 模拟动作（不实际执行）
sim_obs, sim_reward, sim_done, sim_info = env.simulate(action)

# 获取状态信息
state = env.get_state_for_planner()  # 用于 Planner
obs_info = env.get_observation_info()  # 详细观测信息
grid_info = env.get_grid_info()  # 电网拓扑信息

# 关闭环境
env.close()
```

## 支持的 L2RPN 环境

### NeurIPS 2020

- **Track 1 (鲁棒性)**: `neurips_2020_track1`
  - 应对对抗性线路攻击
  - 动作：拓扑变更、再调度
  
- **Track 2 (适应性)**: `neurips_2020_track2`
  - 处理可变的可再生能源发电
  - 动作：拓扑变更、再调度

### ICAPS 2021

- **信任赛道**: `icaps_2021`
  - 引入告警机制，实现人机协作
  - 动作：拓扑变更、再调度、告警

### WCCI 2022

- **碳中和赛道**: `wcci_2022`
  - 包含储能单元和弃风功能
  - 动作：拓扑变更、再调度、弃风、储能控制

### 开发测试

- **沙盒环境**: `sandbox_case14`
  - 基于 IEEE 14 节点系统
  - 用于快速开发和测试

## 环境工具 (env/tools.py)

环境工具用于与 Grid2Op 环境进行直接交互：

```python
from env.tools import create_env_tools

# 创建工具集
tools = create_env_tools(env=env)

# 使用工具
for tool in tools:
    if tool.name == "get_observation":
        result = tool.execute()
    elif tool.name == "simulate_action":
        result = tool.execute(redispatch={0: 10.0})
```

可用工具：
- `GetObservationTool`: 获取当前观测数据
- `SimulateActionTool`: 模拟动作效果
- `ExecuteActionTool`: 执行动作
- `GetGridInfoTool`: 获取电网拓扑信息
- `GetForecastTool`: 获取负荷/发电预测

## 与 ADA 系统集成

```python
from main import ADAOrchestrator
from env import create_grid2op_env

# 创建环境
env = create_grid2op_env("wcci_2022", seed=42)

# 创建编排器并绑定环境
orchestrator = ADAOrchestrator(
    system_config=system_config,
    llm_config=llm_config,
    env=env
)

# 运行回合
result = orchestrator.run_episode(max_steps=100)
```

## 配置说明

### EnvConfig 属性

- `name`: 配置名称
- `env_name`: Grid2Op 环境标识符
- `competition`: 所属比赛
- `use_lightsim`: 是否使用 LightSim2Grid 后端（更快）
- `action_class`: 动作空间类
- `has_storage`: 是否有储能单元
- `has_renewable`: 是否有可再生能源
- `has_curtailment`: 是否支持弃风
- `has_redispatch`: 是否支持再调度
- `has_alarm`: 是否有告警系统
- `max_episode_steps`: 最大回合步数

### 自定义配置

```python
from env import EnvConfig, Competition

custom_config = EnvConfig(
    name="自定义环境",
    env_name="l2rpn_custom",
    competition=Competition.SANDBOX,
    description="自定义环境描述",
    use_lightsim=True,
    has_storage=True,
    has_renewable=True,
)

env = create_grid2op_env(custom_config)
```

## 注意事项

1. **安装依赖**：
   ```bash
   pip install grid2op lightsim2grid
   ```

2. **环境数据下载**：首次使用某个环境时，Grid2Op 会自动下载数据

3. **后端选择**：
   - LightSim2Grid: 更快，推荐使用
   - PandaPower: 默认后端，兼容性更好

4. **环境关闭**：使用完毕后记得调用 `env.close()` 释放资源

## 与 GRIDenv 目录的关系

`env/GRIDenv/` 目录包含演示代码和旧版本实现，保留用于参考。
主模块 `env/` 已整合所有功能，推荐使用主模块接口。

## 故障排除

### 导入错误

如果遇到 `Grid2Op 环境模块未找到` 警告：
1. 检查是否安装了 `grid2op`: `pip install grid2op`
2. 检查 Python 环境是否正确

### 环境创建失败

1. 检查环境名称是否正确
2. 确认网络连接（首次使用需要下载数据）
3. 查看日志获取详细错误信息

### 性能问题

1. 使用 LightSim2Grid 后端：`use_lightsim=True`
2. 对于大型环境，考虑使用 `_small` 版本进行测试

