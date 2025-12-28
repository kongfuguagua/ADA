# DoNothing Baseline 代码详解

## 概述

`DoNothing` 是 L2RPN Baselines 中最简单的基准实现。它不执行任何控制动作，作为性能对比的下界（lower bound baseline）。任何有效的控制策略都应该比 DoNothing 表现更好。

---

## 文件结构

```
DoNothing/
├── __init__.py          # 模块导出
├── doNothing.py         # DoNothing Agent 核心实现
├── eval_donothing.py    # 评估脚本
└── main.py             # 简单使用示例
```

---

## 1. doNothing.py - Agent 核心实现

### 完整代码

```python
from grid2op.Agent import BaseAgent

class DoNothing(BaseAgent):
    """
    Do nothing agent of grid2op, as a lowerbond baseline for l2rpn competition.
    """
    def __init__(self, action_space, observation_space, name, **kwargs):
        super().__init__(action_space)
        self.name = name

    def act(self, observation, reward, done):
        # Just return an empty action aka. "do nothing"
        return self.action_space({})

    def reset(self, observation):
        # No internal states to reset
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass
```

### 详细解析

#### 类定义

```python
class DoNothing(BaseAgent):
```

- **继承关系**：继承自 `grid2op.Agent.BaseAgent`
- **作用**：实现 grid2op 的标准 Agent 接口

#### `__init__` 方法

```python
def __init__(self, action_space, observation_space, name, **kwargs):
    super().__init__(action_space)
    self.name = name
```

**参数说明**：
- `action_space`：动作空间，定义所有可能的动作
- `observation_space`：观测空间（虽然不使用，但为保持接口一致性）
- `name`：Agent 的名称
- `**kwargs`：其他可选参数（未使用）

**实现细节**：
- 调用父类构造函数初始化动作空间
- 保存名称（用于日志和识别）

#### `act` 方法 - 核心决策函数

```python
def act(self, observation, reward, done):
    # Just return an empty action aka. "do nothing"
    return self.action_space({})
```

**功能**：这是 Agent 的核心方法，在每个时间步被调用。

**参数说明**：
- `observation`：当前电网状态观测（包含线路负载、发电量、负荷等信息）
- `reward`：上一步获得的奖励
- `done`：是否结束标志

**返回值**：
- `self.action_space({})`：创建一个空动作对象
  - `{}` 表示空字典，即不执行任何操作
  - 等价于：不改变发电机出力、不改变拓扑、不操作储能等

**关键理解**：
- 虽然接收了 `observation`、`reward`、`done` 等参数，但完全忽略它们
- 无论电网状态如何，都返回空动作
- 这是"什么都不做"策略的直接体现

#### `reset` 方法

```python
def reset(self, observation):
    # No internal states to reset
    pass
```

**功能**：在每个新场景（episode）开始时调用。

**参数**：
- `observation`：新场景的初始观测

**实现**：
- `pass` 表示什么都不做
- DoNothing Agent 没有内部状态需要重置

#### `load` 和 `save` 方法

```python
def load(self, path):
    # Nothing to load
    pass

def save(self, path):
    # Nothing to save
    pass
```

**功能**：实现 Baseline 接口的保存/加载方法。

**说明**：
- DoNothing Agent 不需要保存任何模型或参数
- 这些方法存在是为了保持接口一致性
- 其他需要训练的 Agent（如 PPO、DQN）会在这里保存/加载神经网络权重

---

## 2. eval_donothing.py - 评估脚本

### 完整代码结构

这个文件提供了完整的评估框架，可以：
- 在多个场景上评估 DoNothing Agent
- 支持多进程并行
- 生成日志和可视化（GIF）

### 详细解析

#### 命令行参数解析

```python
def cli():
    parser = argparse.ArgumentParser(description="Eval baseline DDDQN")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOGS_DIR, type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=DEFAULT_NB_PROCESS, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=DEFAULT_MAX_STEPS, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    return parser.parse_args()
```

**参数说明**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_dir` | str | ✅ | - | 数据集根目录路径 |
| `--logs_dir` | str | ❌ | `./logs-eval/do-nothing-baseline` | 日志输出目录 |
| `--nb_episode` | int | ❌ | 1 | 评估的场景数量 |
| `--nb_process` | int | ❌ | 1 | 并行进程数 |
| `--max_steps` | int | ❌ | -1 | 每个场景的最大步数（-1表示无限制） |
| `--gif` | flag | ❌ | False | 是否生成 GIF 可视化 |
| `--verbose` | flag | ❌ | False | 是否显示详细输出 |

#### `evaluate` 函数 - 核心评估逻辑

```python
def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             verbose=False,
             save_gif=False):
```

**参数说明**：
- `env`：grid2op 环境对象
- `load_path`：模型加载路径（DoNothing 不需要，保留为接口一致性）
- `logs_path`：日志保存路径
- `nb_episode`：评估的场景数量
- `nb_process`：并行进程数
- `max_steps`：每个场景的最大步数
- `verbose`：是否显示详细信息
- `save_gif`：是否保存 GIF 可视化

**执行流程**：

##### 步骤 1：创建 Runner

```python
runner_params = env.get_params_for_runner()
runner_params["verbose"] = verbose

# Build runner
runner = Runner(**runner_params,
                agentClass=DoNothingAgent)
```

**说明**：
- `Runner` 是 grid2op 提供的评估工具
- `get_params_for_runner()` 获取环境配置参数
- `agentClass=DoNothingAgent` 指定使用 DoNothing Agent
  - 注意：这里使用的是 `grid2op.Agent.DoNothingAgent`（grid2op 内置版本）
  - 而不是 `l2rpn_baselines.DoNothing.DoNothing`（虽然功能相同）

##### 步骤 2：运行评估

```python
os.makedirs(logs_path, exist_ok=True)
res = runner.run(path_save=logs_path,
                 nb_episode=nb_episode,
                 nb_process=nb_process,
                 max_iter=max_steps,
                 pbar=True)
```

**参数说明**：
- `path_save`：日志保存路径
- `nb_episode`：场景数量
- `nb_process`：并行进程数（可以加速评估）
- `max_iter`：每个场景的最大步数
- `pbar=True`：显示进度条

**返回值 `res`**：
- 是一个列表，每个元素是一个元组：`(agent, chron_name, cum_reward, nb_time_step, max_ts)`
  - `agent`：Agent 实例
  - `chron_name`：场景名称
  - `cum_reward`：累积奖励
  - `nb_time_step`：实际完成的步数
  - `max_ts`：场景的最大步数

##### 步骤 3：打印结果摘要

```python
print("Evaluation summary:")
for _, chron_name, cum_reward, nb_time_step, max_ts in res:
    msg_tmp = "chronics at: {}".format(chron_name)
    msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
    msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
    print(msg_tmp)
```

**输出示例**：
```
Evaluation summary:
chronics at: scenario_0    total reward: -1234.567890    time steps: 150/288
chronics at: scenario_1    total reward: -2345.678901    time steps: 288/288
```

**结果解读**：
- `time steps: 150/288` 表示在 150 步时电网崩溃（game over），未能完成全部 288 步
- `time steps: 288/288` 表示成功完成整个场景
- `total reward` 是累积奖励（通常为负值，因为 DoNothing 策略表现很差）

##### 步骤 4：生成 GIF（可选）

```python
if save_gif:
    save_log_gif(logs_path, res)
```

**功能**：如果启用了 `--gif` 参数，会生成可视化 GIF 文件，展示电网状态随时间的变化。

#### 主程序入口

```python
if __name__ == "__main__":
    # Parse command line
    args = cli()
    # Create dataset env
    env = make(args.data_dir,
               reward_class=RedispReward,
               action_class=TopologyChangeAction,
               other_rewards={
                   "bridge": BridgeReward,
                   "overflow": CloseToOverflowReward,
                   "distance": DistanceReward
               })
    # Call evaluation interface
    evaluate(env,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif)
```

**环境配置说明**：
- `reward_class=RedispReward`：主要奖励函数（基于重新调度）
- `action_class=TopologyChangeAction`：允许拓扑操作的动作类
- `other_rewards`：额外的奖励函数（用于详细分析）
  - `BridgeReward`：评估拓扑操作能力
  - `CloseToOverflowReward`：评估过载预防能力
  - `DistanceReward`：评估状态恢复能力
  
  > **注意**：额外奖励函数的值可以通过 `info["rewards"]` 字典访问，详见 `docs/额外奖励函数详解.md`

---

## 3. main.py - 简单使用示例

```python
import grid2op
from l2rpn_baselines.DoNothing import evaluate

env = grid2op.make("D:\\Projets\\RTE\\ExpertOp4Grid\\alphaDeesp\\ressources\\parameters\\l2rpn_2019_ltc_9")
res = evaluate(env)
```

**说明**：
- 这是一个最简单的使用示例
- 直接创建环境并调用 `evaluate` 函数
- 注意：路径是硬编码的，实际使用时应该使用命令行参数

**改进版本**：
```python
import grid2op
from l2rpn_baselines.DoNothing import evaluate

# 使用标准环境名称
env = grid2op.make("l2rpn_case14_sandbox")
res = evaluate(env, nb_episode=7, verbose=True)
```

---

## 4. __init__.py - 模块导出

```python
__all__ = [
    "DoNothing",
    "evaluate"
]

from l2rpn_baselines.DoNothing.doNothing import DoNothing
from l2rpn_baselines.DoNothing.eval_donothing import evaluate
```

**功能**：
- 定义模块的公共接口
- 导出 `DoNothing` 类和 `evaluate` 函数
- 允许通过 `from l2rpn_baselines.DoNothing import DoNothing, evaluate` 导入

---

## 使用示例

### 示例 1：直接使用 DoNothing Agent

```python
import grid2op
from l2rpn_baselines.DoNothing import DoNothing

# 创建环境
env = grid2op.make("l2rpn_case14_sandbox")

# 创建 Agent
agent = DoNothing(env.action_space, 
                  env.observation_space, 
                  name="DoNothing")

# 运行一个场景
obs = env.reset()
agent.reset(obs)
done = False
step = 0

while not done:
    action = agent.act(obs, None, done)
    obs, reward, done, info = env.step(action)
    step += 1
    print(f"Step {step}: reward = {reward:.2f}")

print(f"Completed {step} steps")
```

### 示例 2：使用评估函数

```python
import grid2op
from l2rpn_baselines.DoNothing import evaluate

# 创建环境
env = grid2op.make("l2rpn_case14_sandbox")

# 评估多个场景
res = evaluate(env, 
               nb_episode=10,
               logs_path="./do_nothing_logs",
               verbose=True)

# 分析结果
total_steps = sum(nb_time_step for _, _, _, nb_time_step, _ in res)
print(f"Total steps completed: {total_steps}")
```

### 示例 3：命令行使用

```bash
python -m l2rpn_baselines.DoNothing.eval_donothing \
    --data_dir /path/to/dataset \
    --nb_episode 10 \
    --nb_process 4 \
    --verbose \
    --gif
```

---

## 为什么需要 DoNothing Baseline？

### 1. 性能对比基准
- 任何有效的控制策略都应该比 DoNothing 表现更好
- 如果某个 Agent 的性能低于 DoNothing，说明策略有问题

### 2. 问题难度评估
- 如果 DoNothing 能在大部分场景中完成，说明环境相对简单
- 如果 DoNothing 经常失败，说明控制挑战较大

### 3. 奖励函数验证
- 验证奖励函数设计是否合理
- DoNothing 应该获得较低的奖励

### 4. 调试工具
- 作为最简单的 Agent，用于测试环境是否正常工作
- 验证评估流程是否正确

---

## 与其他 Agent 的对比

| 特性 | DoNothing | OptimCVXPY | PPO_SB3 |
|------|-----------|------------|---------|
| 训练需求 | ❌ 无需训练 | ❌ 无需训练 | ✅ 需要训练 |
| 计算复杂度 | O(1) | O(n³) | O(n) |
| 性能 | 最差（基准） | 中等 | 较好（需训练） |
| 可解释性 | 极高 | 高 | 低 |
| 适用场景 | 基准对比 | 快速原型 | 生产使用 |

---

## 总结

`DoNothing` Baseline 虽然简单，但在 L2RPN 项目中扮演重要角色：

1. **最简单的实现**：展示了如何实现一个 Baseline
2. **性能基准**：作为其他方法的对比标准
3. **测试工具**：用于验证环境和评估流程
4. **学习起点**：理解 grid2op Agent 接口的最佳起点

**关键要点**：
- `act()` 方法返回空动作 `self.action_space({})`
- 不需要任何内部状态或模型
- 实现简单但功能完整，符合 Baseline 接口规范

---

## 相关资源

- **Grid2Op 文档**：https://grid2op.readthedocs.io/
- **BaseAgent 接口**：查看 `grid2op.Agent.BaseAgent`
- **Runner 使用**：查看 `grid2op.Runner.Runner`

