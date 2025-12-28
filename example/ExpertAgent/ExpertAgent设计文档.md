# ExpertAgent 设计文档

## 1. 概述

ExpertAgent 是一个基于专家系统的智能体，用于解决电力系统运行中的线路过载问题。该智能体不需要训练，通过实时分析电网状态、计算影响图、评估拓扑变化来解决过载问题。

### 1.1 核心特性

- **无需训练**：基于规则和专家系统，无需机器学习模型训练
- **实时决策**：针对每个过载情况动态计算最优拓扑操作
- **多目标优化**：平衡解决过载、避免恶化其他线路、恢复参考拓扑等多个目标
- **网格特定优化**：针对不同电网（IEEE14、IEEE118等）提供特定优化策略

## 2. 架构设计

### 2.1 模块结构

```
ExpertAgent/
├── expertAgent.py      # 核心智能体类
├── evaluate.py          # 评估接口
├── make_agent.py       # 智能体创建工厂函数
├── main.py             # 简单运行脚本
└── __init__.py         # 模块导出
```

### 2.2 类层次结构

```
BaseAgent (grid2op)
    └── ExpertAgent
        ├── 状态管理
        ├── 过载检测与排序
        ├── 拓扑搜索与评估
        └── 动作选择策略
```

## 3. 核心组件

### 3.1 ExpertAgent 类

#### 3.1.1 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `action_space` | ActionSpace | - | Grid2Op动作空间 |
| `observation_space` | ObservationSpace | - | Grid2Op观测空间 |
| `name` | str | - | 智能体名称 |
| `gridName` | str | "IEEE118" | 电网标识符（IEEE14/IEEE118/IEEE118_R2） |

#### 3.1.2 关键属性

- **状态跟踪**：
  - `curr_iter`: 当前迭代次数
  - `sub_2nodes`: 被分割为两个节点的变电站集合
  - `lines_disconnected`: 已断开的线路集合

- **配置参数**：
  - `threshold_powerFlow_safe`: 安全功率流阈值（0.95）
  - `maxOverloadsAtATime`: 同时处理的最大过载数量（3）
  - `config`: 专家系统配置字典

- **专家系统配置**：
  ```python
  {
      "totalnumberofsimulatedtopos": 25,      # 总模拟拓扑数
      "numberofsimulatedtopospernode": 5,     # 每个节点模拟拓扑数
      "maxUnusedLines": 2,                    # 最大未使用线路数
      "ratioToReconsiderFlowDirection": 0.75, # 重新考虑流向的比例
      "ratioToKeepLoop": 0.25,                # 保持环路比例
      "ThersholdMinPowerOfLoop": 0.1,         # 环路最小功率阈值
      "ThresholdReportOfLine": 0.2            # 线路报告阈值
  }
  ```

### 3.2 主要方法

#### 3.2.1 `act(observation, reward, done)`

**功能**：根据当前观测选择动作

**决策流程**：

1. **过载检测与排序**
   - 调用 `getRankedOverloads()` 获取排序后的过载线路列表
   - 优先处理关键过载（即将断开的线路）

2. **无过载情况处理**
   - 尝试恢复参考拓扑（`recover_reference_topology`）
   - 尝试重连断开的线路（`reco_line`）
   - 如果都不可行，执行空动作

3. **过载处理流程**
   ```
   For each 过载线路 (按优先级):
       ├─ 创建 Grid2opSimulation 模拟器
       ├─ 调用 expert_operator 获取候选动作
       ├─ 评估动作得分（0-4分）
       └─ 选择最佳动作
   
   如果得分 <= 1:
       ├─ 尝试网格特定动作（如IEEE14的线路14断开）
       ├─ 尝试恢复参考拓扑
       └─ 选择最小恶化动作
   ```

4. **动作评分系统**
   - **4分**：解决所有过载
   - **3分**：解决目标过载
   - **2分**：部分解决目标过载
   - **1分**：解决目标过载但恶化其他线路
   - **0分**：失败

#### 3.2.2 `getRankedOverloads(observation)`

**功能**：对过载线路进行排序

**排序策略**：
1. 按功率流使用率（rho）降序排序
2. 优先处理关键过载（`timestep_overflow == NB_TIMESTEP_OVERFLOW_ALLOWED`）
3. 返回：`[关键过载列表] + [非关键过载列表]`

#### 3.2.3 `reco_line(observation)`

**功能**：尝试重连断开的线路

**逻辑**：
- 查找断开且冷却时间已过的线路
- 模拟重连动作
- 如果安全（所有线路 rho < 0.95），返回重连动作

#### 3.2.4 `recover_reference_topology(observation, sub_id)`

**功能**：尝试恢复变电站到参考拓扑

**逻辑**：
- 检查变电站是否被分割（存在节点2）
- 创建恢复参考拓扑的动作
- 模拟验证安全性
- 如果安全，返回动作并更新状态

#### 3.2.5 `try_out_reference_topologies(...)`

**功能**：尝试将多个变电站恢复为参考拓扑

**使用场景**：当专家系统得分较低时，尝试恢复拓扑作为备选方案

#### 3.2.6 `bonus_action_IEEE14(...)`

**功能**：IEEE14电网的特殊优化动作

**特殊处理**：对于IEEE14电网，尝试断开线路14，这是一个已知的有效操作

#### 3.2.7 `additionalLinesToCut(lineToCut)`

**功能**：识别并行过载线路

**用途**：对于IEEE118等大型电网，识别并行线路并一起处理

**支持的电网**：
- IEEE118_R2: 线路对 (22,23), (33,35), (34,32)
- IEEE118: 线路对 (135,136), (149,147), (148,146)

#### 3.2.8 `get_action_with_least_worsened_lines(...)`

**功能**：从得分1的动作中选择恶化最少的动作

**策略**：
- 优先选择恶化现有过载线路最少的动作
- 其次选择恶化新线路最少的动作

### 3.3 MinMargin_reward 类

**功能**：自定义奖励函数，用于评估动作效果

**计算公式**：
```python
reward = min(1.0 - rho)  # 最小安全裕度
```

**范围**：[-1.0, 1.0]

## 4. 工作流程

### 4.1 完整决策流程

```
开始
  │
  ├─> 检测过载
  │     │
  │     ├─> 无过载？
  │     │     ├─> 尝试恢复参考拓扑
  │     │     ├─> 尝试重连线路
  │     │     └─> 返回空动作
  │     │
  │     └─> 有过载
  │           │
  │           ├─> 排序过载（关键优先）
  │           │
  │           ├─> For each 过载（最多3个）:
  │           │     │
  │           │     ├─> 创建模拟器
  │           │     ├─> 调用专家系统
  │           │     ├─> 获取候选动作
  │           │     ├─> 评估得分
  │           │     │
  │           │     ├─> 得分 == 4?
  │           │     │     └─> 选择动作，退出循环
  │           │     │
  │           │     ├─> 得分 == 3 且 关键?
  │           │     │     └─> 选择动作，退出循环
  │           │     │
  │           │     └─> 记录最佳动作
  │           │
  │           ├─> 得分 <= 1?
  │           │     │
  │           │     ├─> 尝试网格特定动作（如IEEE14）
  │           │     ├─> 尝试恢复参考拓扑
  │           │     └─> 选择最小恶化动作
  │           │
  │           └─> 更新状态（记录分割的变电站）
  │
  └─> 返回动作
```

### 4.2 专家系统集成

ExpertAgent 依赖于 `alphaDeesp` 库的专家系统：

1. **Grid2opSimulation**：创建模拟环境
2. **expert_operator**：执行专家系统算法
   - 计算影响图
   - 排序变电站和拓扑
   - 模拟候选拓扑
   - 返回评分结果

## 5. 使用方式

### 5.1 简单运行

```python
import grid2op
from l2rpn_baselines.ExpertAgent import evaluate

env = grid2op.make("l2rpn_case14_sandbox")
res = evaluate(env, nb_episode=7, verbose=True, save_gif=True, grid="IEEE14")
```

### 5.2 使用 make_agent

```python
from l2rpn_baselines.ExpertAgent import make_agent

agent = make_agent(env, dir_path="./", gridName="IEEE14")
```

### 5.3 直接使用 ExpertAgent

```python
from l2rpn_baselines.ExpertAgent import ExpertAgent

agent = ExpertAgent(
    env.action_space,
    env.observation_space,
    name="MyExpertAgent",
    gridName="IEEE14"
)
```

## 6. 配置与调优

### 6.1 可调参数

#### 6.1.1 专家系统配置

在 `ExpertAgent.__init__` 中的 `self.config` 字典：

- `totalnumberofsimulatedtopos`: 增加可探索更多拓扑，但计算时间增加
- `numberofsimulatedtopospernode`: 每个节点的探索深度
- `maxUnusedLines`: 允许的最大未使用线路数

#### 6.1.2 行为参数

- `maxOverloadsAtATime`: 同时处理的最大过载数（默认3）
- `threshold_powerFlow_safe`: 安全功率流阈值（默认0.95）

### 6.2 网格特定优化

#### IEEE14
- 特殊处理：线路14断开策略
- 适用于小型测试电网

#### IEEE118 / IEEE118_R2
- 并行线路识别与处理
- 适用于大型电网

## 7. 依赖项

### 7.1 必需依赖

- `grid2op`: Grid2Op框架
- `numpy`: 数值计算
- `pandas`: 数据处理
- `alphaDeesp`: 专家系统核心库

### 7.2 可选依赖

- `lightsim2grid`: 快速功率流计算后端（可选）

## 8. 性能考虑

### 8.1 计算复杂度

- **时间复杂度**：O(n × m × k)
  - n: 过载数量
  - m: 每个过载的候选拓扑数（约25个）
  - k: 每个拓扑的模拟时间

- **空间复杂度**：O(n × m)（存储候选动作和结果）

### 8.2 优化建议

1. **限制过载数量**：`maxOverloadsAtATime = 3`
2. **限制模拟拓扑数**：`totalnumberofsimulatedtopos = 25`
3. **提前退出**：找到得分4的动作立即退出
4. **并行处理**：使用 `nb_process > 1` 进行多进程评估

## 9. 限制与注意事项

### 9.1 已知限制

1. **依赖 alphaDeesp**：如果库不可用，智能体无法运行
2. **计算时间**：每个决策需要多次模拟，可能较慢
3. **网格特定**：某些优化仅适用于特定电网

### 9.2 使用注意事项

1. **网格名称匹配**：确保 `gridName` 参数与实际电网匹配
2. **冷却时间**：智能体会考虑变电站和线路的冷却时间
3. **奖励函数**：默认使用 `MinMargin_reward`，可通过 `other_rewards` 配置

## 10. 扩展与改进

### 10.1 可能的改进方向

1. **自适应参数**：根据电网状态动态调整配置参数
2. **缓存机制**：缓存已评估的拓扑，避免重复计算
3. **并行模拟**：并行执行多个拓扑模拟
4. **学习机制**：结合历史经验优化决策

### 10.2 扩展点

- 添加新的网格特定优化
- 实现新的奖励函数
- 集成其他专家系统算法
- 添加动作历史分析

## 11. 测试与验证

### 11.1 测试场景

- IEEE14 电网过载处理
- IEEE118 电网多过载场景
- 无过载情况下的拓扑恢复
- 关键过载的紧急处理

### 11.2 评估指标

- 过载解决率
- 平均动作得分
- 计算时间
- 累积奖励

## 12. 参考文献

- Grid2Op 文档：https://grid2op.readthedocs.io/
- alphaDeesp 专家系统
- L2RPN 竞赛基准

---

**版本**：1.0  
**最后更新**：2024  
**维护者**：L2RPN Baselines Team

