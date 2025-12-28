# PPO_SB3 - 基于 Stable Baselines3 的 PPO 智能体

## 📖 项目简介

PPO_SB3 是一个基于 **Stable Baselines3** 框架实现的 **PPO (Proximal Policy Optimization)** 强化学习智能体，专门用于在 **Grid2Op** 电力系统环境中进行连续动作控制。

本项目是 L2RPN (Learning to Run a Power Network) 竞赛的基线实现之一，主要用于研究连续动作对电力系统的影响，包括：

- **储能单元 (Storage Units)** 的充放电控制
- **可调度发电机 (Dispatchable Generators)** 的重新调度 (Redispatch)
- **可再生能源发电机** 的削减控制 (Curtailment)

## ✨ 核心特性

- 🎯 **连续动作空间**：专注于连续变量控制，适合电力系统的精细调节
- 🔄 **Gym 兼容**：使用 Grid2Op 的 `gym_compat` 模块，将 Grid2Op 环境转换为标准 Gym 环境
- 📊 **TensorBoard 支持**：完整的训练日志记录，便于监控和调试
- 💾 **模型检查点**：支持训练过程中的自动保存和恢复
- 🔍 **评估回调**：训练过程中自动评估模型性能
- 🎛️ **灵活配置**：可自定义观察空间、动作空间和网络架构
- 📈 **归一化支持**：支持观察和动作空间的自动归一化

## 📦 依赖要求

### 必需依赖

```bash
# 核心依赖
grid2op>=1.7.0
stable-baselines3>=1.5.0
lightsim2grid>=0.7.0  # 强烈推荐，用于加速训练

# 可选但推荐
tensorboard  # 用于可视化训练过程
```

### 安装方法

```bash
# 安装 l2rpn-baselines（包含 PPO_SB3）
pip install l2rpn-baselines

# 或从源码安装
git clone https://github.com/rte-france/l2rpn-baselines.git
cd l2rpn-baselines
pip install -e .
```

## 🚀 快速开始

### 1. 基本训练示例

```python
import re
import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from l2rpn_baselines.PPO_SB3 import train

# 创建环境
env_name = "l2rpn_case14_sandbox"
env = grid2op.make(
    env_name,
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),  # 使用 LightSimBackend 加速
    chronics_class=MultifolderWithCache  # 缓存数据以加速训练
)

# 过滤训练数据（可选）
env.chronics_handler.real_data.set_filter(
    lambda x: re.match(".*00$", x) is not None
)
env.chronics_handler.real_data.reset()

try:
    # 训练智能体
    trained_agent = train(
        env,
        iterations=10_000,  # 训练步数
        logs_dir="./logs",  # TensorBoard 日志目录
        save_path="./saved_model",  # 模型保存路径
        name="my_ppo_agent",  # 模型名称
        net_arch=[200, 200, 200],  # 神经网络架构 [隐藏层1, 隐藏层2, 隐藏层3]
        save_every_xxx_steps=2000,  # 每 2000 步保存一次检查点
        eval_every_xxx_steps=1000,  # 每 1000 步评估一次（需要提供 eval_env）
    )
finally:
    env.close()
```

### 2. 带评估环境的训练

```python
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.PPO_SB3 import train

# 创建训练环境
env = grid2op.make(
    "l2rpn_case14_sandbox",
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache
)

# 创建评估环境（使用测试集）
eval_env = grid2op.make(
    "l2rpn_case14_sandbox",
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache,
    test=True  # 使用测试集
)

try:
    trained_agent = train(
        env,
        iterations=10_000,
        logs_dir="./logs",
        save_path="./saved_model",
        name="my_ppo_agent",
        net_arch=[200, 200, 200],
        save_every_xxx_steps=2000,
        eval_every_xxx_steps=1000,  # 启用评估回调
        eval_env=eval_env,  # 提供评估环境
    )
finally:
    env.close()
    eval_env.close()
```

### 3. 评估训练好的模型

```python
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from l2rpn_baselines.PPO_SB3 import evaluate

# 创建评估环境
env = grid2op.make(
    "l2rpn_case14_sandbox",
    reward_class=LinesCapacityReward,
    backend=LightSimBackend()
)

try:
    # 评估智能体
    trained_agent, results = evaluate(
        env,
        load_path="./saved_model",  # 模型加载路径（需与训练时一致）
        name="my_ppo_agent",  # 模型名称（需与训练时一致）
        nb_episode=7,  # 评估的回合数
        nb_process=1,  # 并行进程数
        logs_path="./logs-eval",  # 评估日志保存路径
        verbose=True,  # 显示详细信息
        save_gif=False,  # 是否保存 GIF（可能占用大量内存）
    )
    
    # 打印评估结果
    print("评估完成！")
    for _, chron_name, cum_reward, nb_time_step, max_ts in results:
        print(f"场景: {chron_name}")
        print(f"  总奖励: {cum_reward:.6f}")
        print(f"  步数: {nb_time_step}/{max_ts}")
finally:
    env.close()
```

## 📁 代码结构

```
PPO_SB3/
├── __init__.py          # 模块导出
├── train.py             # 训练函数
├── evaluate.py          # 评估函数
└── utils.py             # 工具函数和 SB3Agent 类
```

### 主要模块说明

#### `train.py`
- **`train()`**: 主训练函数，负责创建 Gym 环境、初始化 PPO 模型、训练和保存
- **`build_gym_env()`**: 将 Grid2Op 环境转换为 Gym 环境

#### `evaluate.py`
- **`evaluate()`**: 评估函数，加载训练好的模型并在环境中运行评估

#### `utils.py`
- **`SB3Agent`**: 核心智能体类，封装了 Stable Baselines3 的 PPO 模型
- **`default_obs_attr_to_keep`**: 默认观察属性列表
- **`default_act_attr_to_keep`**: 默认动作属性列表
- **`remove_non_usable_attr()`**: 移除不可用的动作属性
- **`save_used_attribute()`**: 保存使用的观察和动作属性

## 🔧 主要参数详解

### 训练参数 (`train()`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | `grid2op.Environment` | - | Grid2Op 环境（必需） |
| `name` | `str` | `"PPO_SB3"` | 模型名称 |
| `iterations` | `int` | `1` | 训练步数（不是回合数） |
| `save_path` | `str` | `None` | 模型保存路径 |
| `load_path` | `str` | `None` | 模型加载路径（用于继续训练） |
| `net_arch` | `list` | `None` | 神经网络架构，如 `[200, 200, 200]` |
| `logs_dir` | `str` | `None` | TensorBoard 日志目录 |
| `learning_rate` | `float` | `3e-4` | 学习率 |
| `save_every_xxx_steps` | `int` | `None` | 每 N 步保存检查点 |
| `eval_every_xxx_steps` | `int` | `None` | 每 N 步评估一次（需提供 `eval_env`） |
| `obs_attr_to_keep` | `list` | 默认列表 | 保留的观察属性 |
| `act_attr_to_keep` | `list` | 默认列表 | 保留的动作属性 |
| `normalize_obs` | `bool` | `False` | 是否归一化观察空间 |
| `normalize_act` | `bool` | `False` | 是否归一化动作空间 |
| `eval_env` | `grid2op.Environment` | `None` | 评估环境（用于训练时评估） |

### 评估参数 (`evaluate()`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | `grid2op.Environment` | - | Grid2Op 环境（必需） |
| `load_path` | `str` | `"."` | 模型加载路径 |
| `name` | `str` | `"PPO_SB3"` | 模型名称（需与训练时一致） |
| `nb_episode` | `int` | `1` | 评估回合数 |
| `nb_process` | `int` | `1` | 并行进程数 |
| `max_steps` | `int` | `-1` | 最大步数（-1 表示无限制） |
| `logs_path` | `str` | `None` | 评估日志保存路径 |
| `save_gif` | `bool` | `False` | 是否保存 GIF 动画 |
| `iter_num` | `int` | `None` | 加载特定训练迭代的模型 |

### 默认观察属性

```python
default_obs_attr_to_keep = [
    "day_of_week", "hour_of_day", "minute_of_hour",  # 时间信息
    "prod_p", "prod_v",  # 发电功率和电压
    "load_p", "load_q",  # 负荷功率
    "actual_dispatch", "target_dispatch",  # 调度信息
    "topo_vect",  # 拓扑向量
    "time_before_cooldown_line", "time_before_cooldown_sub",  # 冷却时间
    "rho",  # 线路负载率
    "timestep_overflow",  # 过载时间步
    "line_status",  # 线路状态
    "storage_power", "storage_charge",  # 储能信息
]
```

### 默认动作属性

```python
default_act_attr_to_keep = [
    "redispatch",    # 重新调度（可调度发电机）
    "curtail",       # 削减（可再生能源）
    "set_storage",   # 储能控制
]
```

## 🎯 高级用法

### 自定义观察和动作空间

```python
from l2rpn_baselines.PPO_SB3 import train

# 自定义观察属性
custom_obs_attr = [
    "rho",  # 只保留线路负载率
    "prod_p",  # 发电功率
    "load_p",  # 负荷功率
]

# 自定义动作属性
custom_act_attr = [
    "redispatch",  # 只使用重新调度
    # "curtail",  # 不使用削减
    # "set_storage",  # 不使用储能
]

trained_agent = train(
    env,
    iterations=10_000,
    save_path="./saved_model",
    name="custom_agent",
    obs_attr_to_keep=custom_obs_attr,
    act_attr_to_keep=custom_act_attr,
    net_arch=[256, 256, 128],  # 自定义网络架构
)
```

### 启用归一化

```python
trained_agent = train(
    env,
    iterations=10_000,
    save_path="./saved_model",
    name="normalized_agent",
    normalize_obs=True,  # 归一化观察空间
    normalize_act=True,  # 归一化动作空间
)
```

### 继续训练（从检查点恢复）

```python
trained_agent = train(
    env,
    iterations=20_000,  # 总训练步数
    load_path="./saved_model",  # 从该路径加载
    save_path="./saved_model",  # 保存到该路径
    name="my_ppo_agent",
    # 其他参数会自动从保存的配置中恢复
)
```

### 使用 TensorBoard 监控训练

```python
# 训练时指定 logs_dir
trained_agent = train(
    env,
    iterations=10_000,
    logs_dir="./logs",  # TensorBoard 日志目录
    save_path="./saved_model",
    name="my_ppo_agent",
)

# 启动 TensorBoard
# 在终端运行: tensorboard --logdir=./logs
```

### 加载特定迭代的模型

```python
# 评估时指定 iter_num
trained_agent, results = evaluate(
    env,
    load_path="./saved_model",
    name="my_ppo_agent",
    iter_num=8000,  # 加载第 8000 步的模型
    nb_episode=7,
)
```

## 📊 使用示例脚本

项目根目录的 `train.py` 和 `evaluate.py` 文件包含可直接运行的示例：

### 运行训练示例

```bash
# 直接运行 train.py
python -m l2rpn_baselines.PPO_SB3.train

# 或修改 train.py 中的参数后运行
python l2rpn_baselines/PPO_SB3/train.py
```

### 运行评估示例

```bash
# 直接运行 evaluate.py
python -m l2rpn_baselines.PPO_SB3.evaluate

# 或修改 evaluate.py 中的参数后运行
python l2rpn_baselines/PPO_SB3/evaluate.py
```

## ⚠️ 注意事项

### 1. Gym/Gymnasium 兼容性

目前 Grid2Op 使用的是旧版 Gym API，而 Stable Baselines3 可能使用新版 Gymnasium API。这可能导致兼容性问题。建议：

- 使用兼容版本的 `stable-baselines3`
- 关注 Grid2Op 的更新，未来可能会迁移到 Gymnasium

### 2. 训练时间

- 训练可能需要较长时间，特别是使用大量数据时
- 建议使用 `LightSimBackend` 加速计算
- 使用 `MultifolderWithCache` 缓存数据以减少 I/O 开销

### 3. 内存使用

- 保存 GIF 动画会占用大量内存，评估时谨慎使用 `save_gif=True`
- 大网络架构和长时间训练可能需要更多内存

### 4. 模型保存

- 训练时务必指定 `save_path`，否则模型不会被保存
- 模型保存时会同时保存观察和动作空间的配置，评估时必须使用相同的配置

### 5. 动作空间限制

- 默认只支持连续动作（`redispatch`, `curtail`, `set_storage`）
- 不支持拓扑操作（开关线路、改变变电站配置等）
- 如需拓扑操作，请考虑使用其他基线（如 `ExpertAgent`）

## 🔍 常见问题

### Q: 如何选择合适的网络架构？

A: 建议从 `[200, 200, 200]` 开始，根据环境复杂度调整：
- 简单环境：`[100, 100]` 或 `[128, 128]`
- 复杂环境：`[256, 256, 256]` 或 `[512, 512, 256]`

### Q: 训练时如何选择合适的观察属性？

A: 建议从默认列表开始，然后根据任务需求调整：
- 如果只关注功率平衡，可以只保留 `prod_p`, `load_p`, `rho`
- 如果需要时间信息，保留 `day_of_week`, `hour_of_day` 等

### Q: 如何提高训练效率？

A: 
1. 使用 `LightSimBackend` 替代默认后端
2. 使用 `MultifolderWithCache` 缓存数据
3. 过滤训练数据，只使用部分场景
4. 调整 `save_every_xxx_steps` 和 `eval_every_xxx_steps` 以减少 I/O

### Q: 评估结果不稳定怎么办？

A: 
1. 增加评估回合数 `nb_episode`
2. 检查模型是否充分训练
3. 尝试不同的随机种子
4. 检查环境配置是否一致

### Q: 如何与其他基线对比？

A: 可以参考项目中的其他基线实现，如：
- `DoNothing`: 不执行任何动作的基准
- `OptimCVXPY`: 基于优化的基线
- `ExpertAgent`: 基于规则的专家系统

## 📚 参考资源

- **项目仓库**: https://github.com/rte-france/l2rpn-baselines
- **Grid2Op 文档**: https://grid2op.readthedocs.io/
- **Stable Baselines3 文档**: https://stable-baselines3.readthedocs.io/
- **L2RPN 竞赛**: https://l2rpn.chalearn.org/
- **官方文档**: `docs/ppo_stable_baselines.rst`

## 📝 示例项目

项目根目录的 `examples/` 文件夹包含完整的使用示例：

- `examples/ppo_stable_baselines/`: 基础使用示例
- `examples/ppo_stable_baselines_idf_2023/`: IDF 2023 竞赛示例

建议参考这些示例了解完整的工作流程。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循 Mozilla Public License 2.0 (MPL-2.0)。

---

**Happy Training! 🚀**

