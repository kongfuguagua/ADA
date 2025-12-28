# OptimCVXPY 工作原理详解

## 概述

OptimCVXPY 是一个基于凸优化（Convex Optimization）的智能体，用于电力系统运行控制。它使用 CVXPY 库求解优化问题，通过优化重调度（redispatching）、削减（curtailment）和储能（storage）操作来维持电网的安全稳定运行。

## 核心思想

OptimCVXPY 采用**基于优化的控制策略**，而不是基于强化学习。它在每个时间步求解一个凸优化问题，以确定最优的控制动作。

## 三种运行模式

### 1. 危险模式（Unsafe Mode）
**触发条件**：`obs.rho.max() > rho_danger`（默认 0.95）

**目标**：快速降低线路负载率，防止过载

**优化目标函数**：
- 最小化热极限违反：`cp.sum_squares(cp.pos(cp.abs(f_or_corr) - margin_th_limit * th_lim_mw))`
- 惩罚削减：`penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw)`
- 惩罚储能：`penalty_storage_unsafe * cp.sum_squares(storage)`
- 惩罚重调度：`penalty_redispatching_unsafe * cp.sum_squares(redispatching)`

**约束条件**：
- 基尔霍夫电流定律（KCL）：每个节点的功率平衡
- 重调度限制：`-redisp_down <= redispatching <= redisp_up`
- 削减限制：`-curtail_down <= curtailment_mw <= curtail_up`
- 储能限制：`-storage_down <= storage <= storage_up`
- 能量平衡：`sum(curtailment) + sum(storage) - sum(redispatching) = 0`

### 2. 安全模式（Safe Mode）
**触发条件**：`obs.rho.max() < rho_safe`（默认 0.85）

**目标**：恢复电网到参考状态

**优化目标函数**：
- 最小化重调度偏差：`weight_redisp_target * cp.sum_squares(dispatch_after_this)`
- 最小化储能偏差：`weight_storage_target * cp.sum_squares(state_of_charge_after - storage_target_bus)`
- 最小化削减：`weight_curtail_target * cp.sum_squares(curtailment_mw + curtail_down)`
- 惩罚项（通常为0）：`penalty_curtailment_safe`, `penalty_storage_safe`, `penalty_redispatching_safe`

**约束条件**：
- 所有危险模式的约束
- **额外约束**：线路潮流必须在热极限内：`-margin_th_limit * th_lim_mw <= f_or_corr <= margin_th_limit * th_lim_mw`

**特殊功能**：
- 可以尝试重新连接断开的线路（如果冷却时间已过）

### 3. 中间模式（Do Nothing Mode）
**触发条件**：`rho_safe <= obs.rho.max() <= rho_danger`

**行为**：不采取任何动作，避免在安全模式和危险模式之间振荡

## 技术细节

### 1. DC 潮流模型

OptimCVXPY 使用**直流（DC）潮流近似**来建模电网：

```
f_or = (1 / x) * (theta[bus_or] - theta[bus_ex])
```

其中：
- `f_or`：线路有功潮流
- `x`：线路电抗（per unit）
- `theta`：节点电压相角
- `bus_or`：线路起始节点
- `bus_ex`：线路终止节点

### 2. 拓扑建模

电网拓扑通过以下参数建模：
- `bus_or`：每条线路的起始节点
- `bus_ex`：每条线路的终止节点
- `bus_load`：每个负荷连接的节点
- `bus_gen`：每个发电机连接的节点
- `bus_storage`：每个储能单元连接的节点

**关键处理**：
- 当线路断开时，将其两端都连接到参考节点（slack bus，节点0）
- 当设备连接到母线2时，节点ID需要加上 `n_sub`（变电站数量）

### 3. 参数更新流程

每次 `act()` 调用时，会更新以下参数：

1. **拓扑参数**（`_update_topo_param`）：
   - 根据观察更新所有设备的连接节点
   - 处理断开的线路

2. **热极限参数**（`_update_th_lim_param`）：
   - 考虑无功功率和电压的影响
   - 计算有效的有功功率热极限

3. **注入参数**（`_update_inj_param`）：
   - 计算每个节点的负荷和发电功率
   - 调整负荷以匹配发电（功率平衡）

4. **约束参数**（`_update_constraints_param_unsafe` 或 `_update_constraints_param_safe`）：
   - 更新重调度、削减、储能的上下限
   - 考虑运行约束（如爬坡率、储能容量等）

### 4. 误差修正

为了改善 DC 模型的准确性，OptimCVXPY 使用误差修正：

```
f_or_corr = f_or - alpha_por_error * prev_por_error
```

其中：
- `prev_por_error`：上一次 DC 计算与实际观测的误差
- `alpha_por_error`：误差修正系数（默认 0.5）
- 只保留负误差（低估的潮流），因为高估是安全的

### 5. 求解器

OptimCVXPY 尝试多个求解器，按顺序：
1. **OSQP**：通常最快且准确
2. **SCS**：几乎总是收敛，但精度较低
3. **SCIPY**：很少收敛，作为最后备选

如果所有求解器都失败，智能体会记录错误并返回零动作。

### 6. 动作转换

优化结果（每个节点的 MW 值）需要转换为 grid2op 动作：

1. **重调度**：按比例分配给同一节点的所有可调度发电机
2. **削减**：转换为最大允许功率（MW）
3. **储能**：直接设置储能功率

**稀疏化处理**：小于 `margin_sparse`（默认 5e-3 MW）的值被设为 0，避免数值噪声。

## 关键参数

### 阈值参数
- `rho_danger`（默认 0.95）：危险模式阈值
- `rho_safe`（默认 0.85）：安全模式阈值
- `margin_th_limit`（默认 0.9）：热极限裕度

### 惩罚参数（危险模式）
- `penalty_curtailment_unsafe`（默认 0.1）：削减惩罚
- `penalty_redispatching_unsafe`（默认 0.03）：重调度惩罚
- `penalty_storage_unsafe`（默认 0.3）：储能惩罚

### 权重参数（安全模式）
- `weight_redisp_target`（默认 1.0）：重调度目标权重
- `weight_storage_target`（默认 1.0）：储能目标权重
- `weight_curtail_target`（默认 1.0）：削减目标权重

### 其他参数
- `alpha_por_error`（默认 0.5）：误差修正系数
- `margin_rounding`（默认 0.01）：舍入裕度
- `margin_sparse`（默认 5e-3）：稀疏化阈值

## 工作流程

```
1. 环境重置 → agent.reset(obs)
   ├─ 初始化误差为 0
   └─ 运行 DC 潮流计算，初始化误差

2. 每个时间步 → agent.act(obs)
   ├─ 更新误差（只保留负误差）
   ├─ 判断模式（危险/安全/中间）
   │
   ├─ 危险模式：
   │  ├─ update_parameters(obs, unsafe=True)
   │  ├─ compute_optimum_unsafe()
   │  └─ to_grid2op(obs, curtailment, storage, redispatching)
   │
   ├─ 安全模式：
   │  ├─ 尝试重新连接线路（如果可能）
   │  ├─ update_parameters(obs, unsafe=False)
   │  ├─ compute_optimum_safe(obs, l_id)
   │  └─ to_grid2op(obs, curtailment, storage, redispatching, safe=True)
   │
   └─ 中间模式：
      └─ 返回空动作
```

## 优势与局限

### 优势
1. **可解释性强**：基于物理模型和优化理论
2. **稳定性好**：不依赖随机性，结果可重复
3. **快速响应**：直接优化，无需训练
4. **物理约束**：自动满足功率平衡和运行约束

### 局限
1. **DC 近似误差**：DC 模型忽略无功功率，可能不够准确
2. **计算开销**：每个时间步都需要求解优化问题
3. **拓扑动作有限**：主要依赖重调度、削减和储能
4. **参数敏感**：需要仔细调整阈值和惩罚参数

## 使用建议

1. **参数调优**：根据具体环境调整 `rho_danger`、`rho_safe` 和惩罚参数
2. **求解器选择**：如果 OSQP 不可用，确保 SCS 已安装
3. **性能监控**：关注优化问题的收敛性和求解时间
4. **误差分析**：检查 `flow_computed` 与实际观测的差异

## 总结

OptimCVXPY 是一个基于优化的控制智能体，通过求解凸优化问题来维持电网安全。它根据电网状态（通过 `rho` 指标判断）在三种模式间切换，使用 DC 潮流模型和误差修正来平衡计算效率和准确性。虽然不如强化学习方法灵活，但在稳定性和可解释性方面具有优势。

