# L2RPN 演示框架

用于初始化和交互 L2RPN（学习运行电网）比赛环境的模块化框架。

## 概述

本框架为不同的 L2RPN 比赛提供统一的接口：

| 比赛 | 年份 | 关键特性 | 环境名称 |
|------|------|----------|----------|
| NeurIPS 赛道1 | 2020 | 鲁棒性（对抗性攻击） | `l2rpn_neurips_2020_track1` |
| NeurIPS 赛道2 | 2020 | 适应性（可再生能源整合） | `l2rpn_neurips_2020_track2` |
| ICAPS | 2021 | 信任（告警机制） | `l2rpn_icaps_2021` |
| WCCI | 2022 | 碳中和（储能、弃风） | `l2rpn_wcci_2022` |

## 快速开始

### 安装依赖

```bash
# 安装必需的包
pip install grid2op lightsim2grid numpy

# 对于 WCCI 2022（基于优化的智能体）
pip install cvxpy
```

### 基本使用

```python
from env_factory import create_env, EnvManager
from config import WCCI_2022

# 方式1：简单创建环境
env = create_env("wcci_2022", seed=42)
obs = env.reset()
action = env.action_space({})  # "什么都不做"动作
obs, reward, done, info = env.step(action)

# 方式2：使用 EnvManager（推荐）
manager = EnvManager("wcci_2022", seed=42)
obs = manager.reset()
manager.print_status()  # 打印当前电网状态

# 执行一步
action = manager.get_do_nothing_action()
obs, reward, done, info = manager.step(action)

# 获取电网信息
grid_info = manager.get_grid_info()
print(f"线路数: {grid_info['n_line']}, 变电站数: {grid_info['n_sub']}")

# 清理
manager.close()
```

### 运行演示

```bash
# 列出可用环境
python main.py --list-envs

# 运行特定环境演示
python main.py --env wcci_2022
python main.py --env neurips_2020_track1 --max-steps 200

# 显示环境详情
python main.py --info wcci_2022

# 比较所有环境
python main.py --compare
```

## 项目结构

```
demo/
├── config.py           # 环境配置定义
├── env_factory.py      # 环境创建工具
├── main.py             # 统一入口点
├── demo_neurips_2020.py    # NeurIPS 2020 演示
├── demo_icaps_2021.py      # ICAPS 2021 演示
├── demo_wcci_2022.py       # WCCI 2022 演示
└── README.md           # 本文件
```

## 配置系统

### 使用预定义配置

```python
from config import (
    NEURIPS_2020_TRACK1,
    NEURIPS_2020_TRACK2,
    ICAPS_2021,
    WCCI_2022,
    get_config,
    list_configs
)

# 根据名称获取配置
config = get_config("wcci_2022")

# 列出所有可用配置
print(list_configs())
# 输出: ['neurips_2020_track1', 'neurips_2020_track2', 'icaps_2021', 'wcci_2022', 'sandbox_case14']

# 打印配置详情
from config import print_config_info
print_config_info(WCCI_2022)
```

### 配置属性

每个配置包含以下属性：

| 属性 | 描述 |
|------|------|
| `name` | 可读名称 |
| `env_name` | Grid2Op 环境标识符 |
| `competition` | 比赛枚举值 |
| `use_lightsim` | 使用 LightSim2Grid 后端（更快） |
| `has_storage` | 是否有储能单元 |
| `has_renewable` | 是否有可再生能源发电机 |
| `has_curtailment` | 是否支持弃风动作 |
| `has_redispatch` | 是否支持再调度动作 |
| `has_alarm` | 是否有告警机制 |

## 各比赛环境特性

### NeurIPS 2020 - 赛道1（鲁棒性）

**挑战**：处理对抗性线路攻击

```python
from env_factory import EnvManager

manager = EnvManager("neurips_2020_track1", seed=42)
obs = manager.reset()

# 关键特性：
# - 对抗性对手攻击随机线路
# - 专注于拓扑变更和再调度
# - 目标：尽可能长时间存活

# 示例：重连断开的线路
action = manager.env.action_space({"set_line_status": [(line_id, +1)]})
```

### NeurIPS 2020 - 赛道2（适应性）

**挑战**：处理可变的可再生能源发电

```python
manager = EnvManager("neurips_2020_track2", seed=42)
obs = manager.reset()

# 关键特性：
# - 可变的可再生能源
# - 通过再调度平衡发电
# - 目标：适应变化的发电模式

# 检查可再生能源发电机
gen_renewable = manager.env.gen_renewable
renewable_gen = obs.gen_p[gen_renewable].sum()
```

### ICAPS 2021（信任）

**挑战**：带告警的人机协作

```python
manager = EnvManager("icaps_2021", seed=42)
obs = manager.reset()

# 关键特性：
# - 用于通知操作员的告警机制
# - 注意力预算限制告警频率
# - 目标：通过准确预测建立信任

# 发出告警
action = manager.get_do_nothing_action()
if obs.attention_budget[0] >= 1 and not obs.is_alarm_illegal:
    action.raise_alarm = [zone_id]  # 告警特定区域
```

### WCCI 2022（未来能源）

**挑战**：利用储能和弃风实现碳中和

```python
manager = EnvManager("wcci_2022", seed=42)
obs = manager.reset()

# 关键特性：
# - 用于能源管理的储能单元
# - 可再生能源发电机的弃风
# - 组合离散+连续动作

# 储能动作
action = manager.get_do_nothing_action()
storage_power = np.zeros(manager.env.n_storage)
storage_power[0] = 5.0  # 吸收 5 MW
action.storage_p = storage_power

# 弃风动作
curtail = np.ones(manager.env.n_gen) * (-1)  # -1 = 不改变
curtail[gen_id] = 0.5  # 限制到 50% 容量
action.curtail = curtail

# 再调度
action.redispatch = [(gen_id, 10.0)]  # +10 MW
```

## 开发自己的智能体

### 智能体模板

```python
from grid2op.Agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, action_space, **kwargs):
        super().__init__(action_space)
        # 初始化你的智能体
        
    def act(self, observation, reward, done):
        """
        根据当前观测选择动作。
        
        参数:
            observation: 当前电网状态
            reward: 上一个动作的奖励
            done: 回合是否已结束
            
        返回:
            action: Grid2Op 动作
        """
        # 你的决策逻辑
        action = self.action_space({})  # 默认"什么都不做"
        
        # 示例：对过载做出反应
        if observation.rho.max() > 0.95:
            # 实现你的策略
            pass
            
        return action
```

### 测试你的智能体

```python
from env_factory import EnvManager

# 创建环境
manager = EnvManager("wcci_2022", seed=42)
env = manager.env

# 初始化你的智能体
agent = MyAgent(env.action_space)

# 运行回合
obs = manager.reset()
total_reward = 0
done = False

while not done:
    action = agent.act(obs, total_reward, done)
    obs, reward, done, info = manager.step(action)
    total_reward += reward

print(f"回合结束，奖励: {total_reward}")
```

### 使用模拟进行规划

```python
# 在执行前模拟动作
obs = manager.reset()
action = manager.env.action_space({"set_line_status": [(0, +1)]})

# 模拟而不影响真实环境
sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)

if not sim_done and sim_obs.rho.max() < 1.0:
    # 动作安全，执行它
    obs, reward, done, info = manager.step(action)
```

## 常用模式

### 1. 线路重连

```python
def reconnect_lines(obs, action_space):
    """重连任何可以重连的断开线路。"""
    disconnected = np.where(obs.line_status == False)[0]
    
    for line_id in disconnected:
        if obs.time_before_cooldown_line[line_id] == 0:
            action = action_space({"set_line_status": [(line_id, +1)]})
            return action
    
    return action_space({})  # 什么都不做
```

### 2. 拓扑重置

```python
def reset_topology(obs, action_space):
    """将拓扑重置到参考状态。"""
    ref_actions = action_space.get_back_to_ref_state(obs)
    if 'substation' in ref_actions:
        for action in ref_actions['substation']:
            sub_id = action.as_dict()['set_bus_vect']['modif_subs_id'][0]
            if obs.time_before_cooldown_sub[int(sub_id)] == 0:
                return action
    return action_space({})
```

### 3. 过载处理

```python
def handle_overflow(obs, action_space, threshold=0.95):
    """处理线路过载情况。"""
    if obs.rho.max() < threshold:
        return action_space({})
    
    # 找到过载线路
    overloaded = np.where(obs.rho > threshold)[0]
    
    # 尝试拓扑动作来降低负载
    # ... 实现你的策略 ...
```

## 性能优化建议

1. **使用 LightSim2Grid**：比 PandaPower 后端快得多
   ```python
   manager = EnvManager("wcci_2022")  # 默认使用 LightSim
   ```

2. **批量模拟**：模拟多个动作找到最佳方案
   ```python
   best_action = None
   best_rho = float('inf')
   
   for action in candidate_actions:
       sim_obs, _, done, _ = obs.simulate(action)
       if not done and sim_obs.rho.max() < best_rho:
           best_rho = sim_obs.rho.max()
           best_action = action
   ```

3. **预计算动作空间**：一次性生成单一动作
   ```python
   line_actions = action_space.get_all_unitary_line_set(action_space)
   topo_actions = action_space.get_all_unitary_topologies_set(action_space)
   ```

## 故障排除

### 环境未找到

```
RuntimeError: Cannot find environment 'l2rpn_wcci_2022'
```

**解决方案**：下载环境数据
```python
import grid2op
env = grid2op.make("l2rpn_wcci_2022")  # 会提示下载
```

### LightSim2Grid 不可用

```
Warning: LightSim2Grid not available, falling back to PandaPower
```

**解决方案**：安装 LightSim2Grid
```bash
pip install lightsim2grid
```

### 动作非法

```
Info: {'is_illegal': True, 'is_ambiguous': False}
```

**解决方案**：检查冷却时间和动作有效性
```python
# 检查变电站冷却时间
print(obs.time_before_cooldown_sub)

# 检查线路冷却时间
print(obs.time_before_cooldown_line)
```

## 参考资料

- [Grid2Op 文档](https://grid2op.readthedocs.io/)
- [L2RPN 比赛主页](https://l2rpn.chalearn.org/)
- [L2RPN 基线代码库](https://github.com/rte-france/l2rpn-baselines)

## 许可证

本框架仅供研究和教育目的使用。
具体条款请参考原始比赛的许可证。
