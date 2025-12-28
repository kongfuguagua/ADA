# OptimAgent 的数学原理

## 目录
1. [概述](#概述)
2. [DC 潮流模型](#dc-潮流模型)
3. [优化问题建模](#优化问题建模)
4. [危险模式（Unsafe Mode）](#危险模式unsafe-mode)
5. [安全模式（Safe Mode）](#安全模式safe-mode)
6. [模式切换机制](#模式切换机制)
7. [数据获取与参数更新](#数据获取与参数更新)
8. [优化求解](#优化求解)
9. [解的下发与环境交互](#解的下发与环境交互)
10. [数学公式总结](#数学公式总结)

---

## 概述

OptimCVXPY 是一个基于凸优化的电力系统运行控制智能体。它通过求解凸优化问题来确定最优的控制动作（重调度、削减、储能），以维持电网的安全稳定运行。

核心思想：
- 使用 **DC 潮流模型**近似电网物理特性
- 将控制问题转化为**凸优化问题**
- 根据电网状态（通过 `rho` 指标）在三种模式间切换
- 通过 **CVXPY** 库求解优化问题

---

## DC 潮流模型

### 2.1 基本方程

DC 潮流模型基于以下假设：
1. 忽略线路电阻（只考虑电抗）
2. 电压幅值恒定（标幺值 ≈ 1.0）
3. 相角差很小，sin(θ) ≈ θ

**线路有功潮流方程**：

\[
f_{or}^l = \frac{1}{x^l} (\theta_{or}^l - \theta_{ex}^l)
\]

其中：
- \( f_{or}^l \)：线路 \( l \) 的有功潮流（MW）
- \( x^l \)：线路 \( l \) 的电抗（per unit）
- \( \theta_{or}^l \)：线路起始节点的电压相角
- \( \theta_{ex}^l \)：线路终止节点的电压相角

**代码实现**（第 767 行）：
```python
f_or = cp.multiply(1. / self._powerlines_x , (theta[bus_or_idx] - theta[bus_ex_idx]))
```

### 2.2 误差修正

为了改善 DC 模型的准确性，引入误差修正项：

\[
f_{or,corr}^l = f_{or}^l - \alpha_{por\_error} \cdot \epsilon_{prev}^l
\]

其中：
- \( \alpha_{por\_error} \)：误差修正系数（默认 0.5）
- \( \epsilon_{prev}^l \)：上一次 DC 计算与实际观测的误差
- 只保留负误差（低估的潮流），因为高估是安全的

**代码实现**（第 768 行）：
```python
f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
```

### 2.3 基尔霍夫电流定律（KCL）

对于每个节点 \( i \)：

\[
\sum_{j \in \text{connected to } i} f_{or}^j - \sum_{j \in \text{leaving from } i} f_{ex}^j + P_{inj}^i = 0
\]

其中：
- \( P_{inj}^i = P_{load}^i - P_{gen}^i \)：节点 \( i \) 的净注入功率
- 考虑重调度、削减、储能后的净注入为：
  \[
  P_{inj}^i = (P_{load}^i + P_{storage}^i) - (P_{gen}^i + \Delta P_{redisp}^i - \Delta P_{curtail}^i)
  \]

**代码实现**（第 653-665 行）：
```python
def _aux_compute_kcl(self, inj_bus, f_or):
    KCL_eq = []
    bus_or_int = self.bus_or.value.astype(int)
    bus_ex_int = self.bus_ex.value.astype(int)
    for bus_id in range(self.nb_max_bus):
        tmp = inj_bus[bus_id]
        if np.any(bus_or_int == bus_id):
            tmp += cp.sum(f_or[bus_or_int == bus_id])
        if np.any(bus_ex_int == bus_id):
            tmp -= cp.sum(f_or[bus_ex_int == bus_id])
        KCL_eq.append(tmp)
    return KCL_eq
```

---

## 优化问题建模

### 3.1 决策变量

在每个时间步，优化问题求解以下决策变量：

1. **\( \theta_i \)**：每个节点 \( i \) 的电压相角（rad）
2. **\( \Delta P_{curtail}^i \)**：每个节点 \( i \) 的削减功率（MW，正值表示削减）
3. **\( \Delta P_{storage}^i \)**：每个节点 \( i \) 的储能功率（MW，正值表示充电）
4. **\( \Delta P_{redisp}^i \)**：每个节点 \( i \) 的重调度功率（MW，正值表示增加发电）

**代码实现**（第 758-761 行，危险模式）：
```python
theta = cp.Variable(shape=self.nb_max_bus)
curtailment_mw = cp.Variable(shape=self.nb_max_bus)
storage = cp.Variable(shape=self.nb_max_bus)
redispatching = cp.Variable(shape=self.nb_max_bus)
```

### 3.2 约束条件

#### 3.2.1 参考节点约束

设置参考节点（slack bus）的相角为 0：

\[
\theta_i = 0, \quad \forall i \in \text{isolated nodes or slack bus}
\]

**代码实现**（第 777 行）：
```python
[theta[theta_is_zero] == 0]
```

#### 3.2.2 功率平衡约束

每个节点的 KCL 必须满足：

\[
\sum_{l: or(l)=i} f_{or}^l - \sum_{l: ex(l)=i} f_{ex}^l + P_{inj}^i = 0, \quad \forall i
\]

**代码实现**（第 780 行）：
```python
[el == 0 for el in KCL_eq]
```

#### 3.2.3 运行限制约束

**重调度限制**：
\[
-\Delta P_{redisp,down}^i \leq \Delta P_{redisp}^i \leq \Delta P_{redisp,up}^i, \quad \forall i
\]

**削减限制**：
\[
-\Delta P_{curtail,down}^i \leq \Delta P_{curtail}^i \leq \Delta P_{curtail,up}^i, \quad \forall i
\]

**储能限制**：
\[
-\Delta P_{storage,down}^i \leq \Delta P_{storage}^i \leq \Delta P_{storage,up}^i, \quad \forall i
\]

**代码实现**（第 783-787 行）：
```python
[redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
[curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
[storage <= self.storage_up, storage >= -self.storage_down]
```

#### 3.2.4 能量平衡约束

总能量变化必须平衡：

\[
\sum_i \Delta P_{curtail}^i + \sum_i \Delta P_{storage}^i - \sum_i \Delta P_{redisp}^i - P_{storage,obs} = 0
\]

其中 \( P_{storage,obs} \) 是当前观测到的储能功率总和。

**代码实现**（第 790 行）：
```python
[energy_added == 0]
```

#### 3.2.5 热极限约束（仅安全模式）

在安全模式下，线路潮流必须在热极限内：

\[
-m_{th} \cdot P_{th,lim}^l \leq f_{or,corr}^l \leq m_{th} \cdot P_{th,lim}^l, \quad \forall l
\]

其中 \( m_{th} \) 是热极限裕度（默认 0.9）。

**代码实现**（第 1049-1050 行）：
```python
[f_or_corr <= self._margin_th_limit * self._th_lim_mw] +
[f_or_corr >= -self._margin_th_limit * self._th_lim_mw]
```

---

## 危险模式（Unsafe Mode）

### 4.1 触发条件

当电网最大负载率超过危险阈值时：

\[
\max(\rho) > \rho_{danger}
\]

其中 \( \rho_{danger} \) 默认值为 0.95。

**代码实现**（第 1128 行）：
```python
if obs.rho.max() > self.rho_danger:
```

### 4.2 优化目标函数

危险模式的目标是**最小化热极限违反**，同时惩罚控制动作：

\[
\min \quad J_{unsafe} = w_{curtail} \sum_i (\Delta P_{curtail}^i)^2 + w_{storage} \sum_i (\Delta P_{storage}^i)^2 + w_{redisp} \sum_i (\Delta P_{redisp}^i)^2 + \sum_l \left[ \max(0, |f_{or,corr}^l| - m_{th} \cdot P_{th,lim}^l) \right]^2
\]

其中：
- \( w_{curtail} = 0.1 \)：削减惩罚系数
- \( w_{storage} = 0.3 \)：储能惩罚系数
- \( w_{redisp} = 0.03 \)：重调度惩罚系数
- \( m_{th} = 0.9 \)：热极限裕度

**数学解释**：
- 第一项：惩罚削减操作（避免不必要的削减）
- 第二项：惩罚储能操作（避免频繁充放电）
- 第三项：惩罚重调度操作（避免频繁调整）
- 第四项：**核心项**，惩罚超过热极限的潮流（使用 `max(0, ...)` 确保只惩罚违反）

**代码实现**（第 795-798 行）：
```python
cost = ( self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw) + 
         self._penalty_storage_unsafe * cp.sum_squares(storage) +
         self._penalty_redispatching_unsafe * cp.sum_squares(redispatching) +
         cp.sum_squares(cp.pos(cp.abs(f_or_corr) - self._margin_th_limit * self._th_lim_mw))
)
```

其中 `cp.pos(x) = max(0, x)`。

### 4.3 约束条件

危险模式使用所有基本约束（3.2 节），但**不包含热极限硬约束**（因为目标是减少违反，而不是完全避免）。

---

## 安全模式（Safe Mode）

### 5.1 触发条件

当电网最大负载率低于安全阈值时：

\[
\max(\rho) < \rho_{safe}
\]

其中 \( \rho_{safe} \) 默认值为 0.85。

**代码实现**（第 1137 行）：
```python
elif obs.rho.max() < self.rho_safe:
```

### 5.2 优化目标函数

安全模式的目标是**恢复电网到参考状态**，最小化与目标值的偏差：

\[
\min \quad J_{safe} = w_{curtail,safe} \sum_i (\Delta P_{curtail}^i)^2 + w_{storage,safe} \sum_i (\Delta P_{storage}^i)^2 + w_{redisp,safe} \sum_i (\Delta P_{redisp}^i)^2 + w_{redisp,target} \sum_i (P_{dispatch,after}^i)^2 + w_{storage,target} \sum_i (SoC_{after}^i - SoC_{target}^i)^2 + w_{curtail,target} \sum_i (\Delta P_{curtail}^i + \Delta P_{curtail,down}^i)^2
\]

其中：
- \( P_{dispatch,after}^i = P_{dispatch,past}^i + \Delta P_{redisp}^i \)：重调度后的总调度量
- \( SoC_{after}^i = SoC_{past}^i + \frac{\Delta P_{storage}^i \cdot \Delta t}{60} \)：储能后的荷电状态
- \( SoC_{target}^i = 0.5 \cdot E_{max}^i \)：目标荷电状态（50% 容量）
- 默认权重：\( w_{curtail,safe} = 0.0 \), \( w_{storage,safe} = 0.0 \), \( w_{redisp,safe} = 0.0 \), \( w_{redisp,target} = 1.0 \), \( w_{storage,target} = 1.0 \), \( w_{curtail,target} = 1.0 \)

**数学解释**：
- 前 three 项：控制动作的惩罚（通常为 0，允许自由调整）
- 第四项：最小化重调度总量（目标是回到 0）
- 第五项：最小化储能与目标值的偏差（目标是 50% 容量）
- 第六项：最小化削减（目标是取消所有削减）

**代码实现**（第 1064-1069 行）：
```python
cost = ( self._penalty_curtailment_safe * cp.sum_squares(curtailment_mw) +  
         self._penalty_storage_safe * cp.sum_squares(storage) +
         self._penalty_redispatching_safe * cp.sum_squares(redispatching) +
         self._weight_redisp_target * cp.sum_squares(dispatch_after_this)  +
         self._weight_storage_target * cp.sum_squares(state_of_charge_after - self._storage_target_bus) +
         self._weight_curtail_target * cp.sum_squares(curtailment_mw + self.curtail_down)
)
```

其中：
- `dispatch_after_this = self._past_dispatch + redispatching`（第 1038 行）
- `state_of_charge_after = self._past_state_of_charge + storage / (60. / obs.delta_time)`（第 1039 行）

### 5.3 约束条件

安全模式使用所有基本约束，**并包含热极限硬约束**（确保潮流在安全范围内）。

### 5.4 线路重连

在安全模式下，如果线路冷却时间已过，可以尝试重新连接断开的线路。

**代码实现**（第 1145-1152 行）：
```python
can_be_reco = (obs.time_before_cooldown_line == 0) & (~obs.line_status)
l_id = None
if np.any(can_be_reco):
    l_id = np.where(can_be_reco)[0][0]
    act.line_set_status = [(l_id, +1)]
```

---

## 模式切换机制

### 6.1 三种模式

1. **危险模式**：\( \max(\rho) > \rho_{danger} \)（默认 0.95）
2. **安全模式**：\( \max(\rho) < \rho_{safe} \)（默认 0.85）
3. **中间模式**：\( \rho_{safe} \leq \max(\rho) \leq \rho_{danger} \)

### 6.2 切换逻辑

**代码实现**（第 1128-1164 行）：
```python
if obs.rho.max() > self.rho_danger:
    # 危险模式：求解 minimize thermal violation
    self.update_parameters(obs, unsafe=True)
    curtailment, storage, redispatching = self.compute_optimum_unsafe()
    act = self.to_grid2op(obs, curtailment, storage, redispatching, safe=False)
elif obs.rho.max() < self.rho_safe:
    # 安全模式：求解 minimize deviation from target
    self.update_parameters(obs, unsafe=False)
    curtailment, storage, redispatching = self.compute_optimum_safe(obs, l_id)
    act = self.to_grid2op(obs, curtailment, storage, redispatching, act, safe=True)
else:
    # 中间模式：不采取任何动作
    act = self.action_space()
```

### 6.3 防振荡机制

中间模式的存在是为了**避免在危险模式和安全模式之间频繁振荡**。当电网状态处于中间区域时，智能体不采取动作，等待状态自然变化。

---

## 数据获取与参数更新

### 7.1 参数更新流程

每次 `act()` 调用时，通过 `update_parameters()` 更新所有优化参数：

**代码实现**（第 630-651 行）：
```python
def update_parameters(self, obs: BaseObservation, unsafe: bool = True):
    self._update_topo_param(obs)      # 更新拓扑
    self._update_th_lim_param(obs)     # 更新热极限
    self._update_inj_param(obs)        # 更新注入功率
    if unsafe:
        self._update_constraints_param_unsafe(obs)  # 更新约束（危险模式）
    else:
        self._update_constraints_param_safe(obs)   # 更新约束（安全模式）
    self._validate_param_values()      # 验证参数值
```

### 7.2 拓扑参数更新

**目的**：根据当前观察更新所有设备的连接节点。

**关键处理**：
1. 处理母线切换：如果设备连接到母线 2，节点 ID 需要加上 \( n_{sub} \)（变电站数量）
2. 处理断开线路：将断开线路的两端都连接到参考节点（节点 0）

**数学表示**：
\[
\text{bus}_i = \begin{cases}
\text{sub}_i & \text{if bus} = 1 \\
\text{sub}_i + n_{sub} & \text{if bus} = 2 \\
0 & \text{if disconnected}
\end{cases}
\]

**代码实现**（第 509-536 行）：
```python
def _update_topo_param(self, obs: BaseObservation):
    tmp_ = 1 * obs.line_or_to_subid
    tmp_[obs.line_or_bus == 2] += obs.n_sub
    self.bus_or.value[:] = tmp_.astype(int)
    # ... 类似处理 bus_ex, bus_load, bus_gen, bus_storage
```

### 7.3 热极限参数更新

**目的**：考虑无功功率和电压的影响，计算有效的有功功率热极限。

**数学公式**：
\[
P_{th,lim,eff}^l = \sqrt{\max\left(1, \left(\frac{I_{th,lim}^l \cdot V_{or}^l \cdot \sqrt{3}}{1000}\right)^2 \cdot 3 - Q_{or}^2\right)}
\]

其中：
- \( I_{th,lim}^l \)：线路热极限电流（A）
- \( V_{or}^l \)：线路起始端电压（kV）
- \( Q_{or}^l \)：线路起始端无功功率（MVar）

**代码实现**（第 538-549 行）：
```python
def _update_th_lim_param(self, obs: BaseObservation):
    self._th_lim_mw.value[:] = (0.001 * obs.thermal_limit)**2 * obs.v_or **2 * 3. - obs.q_or**2
    mask_ok = self._th_lim_mw.value >= threshold_
    self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
    self._th_lim_mw.value[~mask_ok] = threshold_
```

### 7.4 注入功率参数更新

**目的**：计算每个节点的负荷和发电功率。

**关键处理**：
1. 调整负荷以匹配发电（确保功率平衡）
2. 考虑储能功率的影响

**数学公式**：
\[
P_{load,scaled}^i = P_{load}^i \cdot \frac{\sum P_{gen} - P_{storage,obs}}{\sum P_{load}}
\]

\[
P_{load,per\_bus}^i = \sum_{j: \text{load } j \text{ at bus } i} P_{load,scaled}^j
\]

\[
P_{gen,per\_bus}^i = \sum_{j: \text{gen } j \text{ at bus } i} P_{gen}^j
\]

**代码实现**（第 554-565 行）：
```python
def _update_inj_param(self, obs: BaseObservation):
    self.load_per_bus.value[:] = 0.
    self.gen_per_bus.value[:] = 0.
    load_p = 1.0 * obs.load_p
    load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
    for bus_id in range(self.nb_max_bus):
        self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
        self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()
```

### 7.5 约束参数更新

#### 7.5.1 危险模式约束

**重调度限制**：
\[
\Delta P_{redisp,up}^i = \sum_{j: \text{gen } j \text{ at bus } i} \Delta P_{gen,up}^j
\]

\[
\Delta P_{redisp,down}^i = \sum_{j: \text{gen } j \text{ at bus } i} \Delta P_{gen,down}^j
\]

**削减限制**：
\[
\Delta P_{curtail,up}^i = \sum_{j: \text{renewable gen } j \text{ at bus } i} P_{gen}^j
\]

\[
\Delta P_{curtail,down}^i = 0
\]

**储能限制**：
\[
\Delta P_{storage,up}^i = \min\left(\sum_{j: \text{storage } j \text{ at bus } i} P_{max,absorb}^j, \frac{E_{max}^j - E_{charge}^j}{\Delta t / 60}\right)
\]

\[
\Delta P_{storage,down}^i = \min\left(\sum_{j: \text{storage } j \text{ at bus } i} P_{max,prod}^j, \frac{E_{charge}^j}{\Delta t / 60}\right)
\]

**代码实现**（第 592-608 行）：
```python
def _update_constraints_param_unsafe(self, obs: BaseObservation):
    for bus_id in range(self.nb_max_bus):
        self._add_redisp_const(obs, bus_id)
        mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
        self.curtail_up.value[bus_id] = tmp_[mask_].sum()
        self._add_storage_const(obs, bus_id)
    self._remove_margin_rounding()
```

#### 7.5.2 安全模式约束

类似危险模式，但：
- \( \Delta P_{curtail,down}^i = \sum_{j} (P_{gen,before\_curtail}^j - P_{gen}^j) \)（允许取消削减）
- \( \Delta P_{curtail,up}^i = 0 \)（不允许增加削减）

**代码实现**（第 987-1012 行）：
```python
def _update_constraints_param_safe(self, obs):
    for bus_id in range(self.nb_max_bus):
        # ... 类似危险模式
        self.curtail_down.value[bus_id] = obs.gen_p_before_curtail[mask_].sum() - tmp_[mask_].sum()
        self.curtail_up.value[:] = 0.  # 不允许增加削减
```

### 7.6 舍入裕度处理

为了避免数值误差导致约束违反，所有约束上限都会减去一个小的裕度：

\[
\Delta P_{limit,adjusted} = \max(0, \Delta P_{limit} - \epsilon_{rounding})
\]

其中 \( \epsilon_{rounding} = 0.01 \) MW。

**代码实现**（第 610-616 行）：
```python
def _remove_margin_rounding(self):
    self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
    # ... 类似处理其他约束
```

---

## 优化求解

### 8.1 求解器选择

OptimCVXPY 按顺序尝试多个求解器：

1. **OSQP**：通常最快且准确
2. **SCS**：几乎总是收敛，但精度较低
3. **SCIPY**：很少收敛，作为最后备选

**代码实现**（第 817-843 行）：
```python
def _solve_problem(self, prob, solver_type=None):
    if solver_type is None:
        for solver_type in type(self).SOLVER_TYPES:
            res = self._solve_problem(prob, solver_type=solver_type)
            if res:
                return True
        return False
    # ... 尝试求解
```

### 8.2 求解流程

1. 构建优化问题：`cp.Problem(cp.Minimize(cost), constraints)`
2. 尝试求解：`prob.solve(solver=solver_type)`
3. 检查收敛性：`np.isfinite(tmp_)`
4. 如果失败，尝试下一个求解器

### 8.3 求解失败处理

如果所有求解器都失败：
- 记录错误日志
- 返回零动作（所有控制变量为 0）

**代码实现**（第 809-813 行）：
```python
if has_converged:
    res = (curtailment_mw.value, storage.value, redispatching.value)
else:
    tmp_ = np.zeros(shape=self.nb_max_bus)
    res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
```

---

## 解的下发与环境交互

### 9.1 稀疏化处理

首先，将小于阈值的值设为 0（避免数值噪声）：

\[
x_{cleaned} = \begin{cases}
0 & \text{if } |x| < \epsilon_{sparse} \\
x & \text{otherwise}
\end{cases}
\]

其中 \( \epsilon_{sparse} = 5 \times 10^{-3} \) MW。

**代码实现**（第 845-849 行）：
```python
def _clean_vect(self, curtailment, storage, redispatching):
    curtailment[np.abs(curtailment) < self.margin_sparse] = 0.
    storage[np.abs(storage) < self.margin_sparse] = 0.
    redispatching[np.abs(redispatching) < self.margin_sparse] = 0.
```

### 9.2 储能动作转换

优化结果给出每个节点的储能功率，需要转换为每个储能单元的动作：

\[
P_{storage}^j = P_{storage,node}^i, \quad \text{where storage } j \text{ is at node } i
\]

**代码实现**（第 893-897 行）：
```python
if act.n_storage and np.any(np.abs(storage) > 0.):
    storage_ = np.zeros(shape=act.n_storage)
    storage_[:] = storage[self.bus_storage.value.astype(int)]
    act.storage_p = storage_
```

### 9.3 削减动作转换

**关键点**：优化器给出的是**削减的 MW 量**，但 grid2op 需要的是**最大允许功率**。

**转换公式**：
\[
P_{curtail,max}^j = P_{gen}^j - \Delta P_{curtail,node}^i \cdot \frac{P_{gen}^j}{P_{gen,node}^i}
\]

其中：
- \( P_{gen}^j \)：发电机 \( j \) 的当前功率
- \( P_{gen,node}^i \)：节点 \( i \) 的总发电功率
- \( \Delta P_{curtail,node}^i \)：节点 \( i \) 的削减量

**代码实现**（第 903-918 行）：
```python
if np.any(np.abs(curtailment) > 0.):
    gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
    idx_gen = self.bus_gen.value[gen_curt].astype(int)
    tmp_ = curtailment[idx_gen]
    aux_[modif_gen_optim] = (gen_p[gen_curt][modif_gen_optim] - 
                             tmp_[modif_gen_optim] * 
                             gen_p[gen_curt][modif_gen_optim] / 
                             self.gen_per_bus.value[idx_gen][modif_gen_optim])
    act.curtail_mw = curtailment_mw
```

**安全模式特殊处理**：如果削减量达到最大值，设置为 `gen_pmax`（取消所有削减）。

### 9.4 重调度动作转换

**关键点**：优化器给出每个节点的总重调度量，需要按比例分配给该节点的所有可调度发电机。

**转换公式**：
\[
\Delta P_{redisp}^j = \Delta P_{redisp,node}^i \cdot \frac{\Delta P_{gen,available}^j}{\sum_{k: \text{gen } k \text{ at node } i} \Delta P_{gen,available}^k}
\]

其中：
- 如果 \( \Delta P_{redisp,node}^i > 0 \)：使用 \( \Delta P_{gen,up}^j \)
- 如果 \( \Delta P_{redisp,node}^i < 0 \)：使用 \( \Delta P_{gen,down}^j \)

**代码实现**（第 936-984 行）：
```python
if np.any(np.abs(redispatching) > 0.):
    gen_redi = obs.gen_redispatchable
    idx_gen = self.bus_gen.value[gen_redi].astype(int)
    tmp_ = redispatching[idx_gen]
    
    # 计算每个节点的可用重调度量
    for bus_id in range(self.nb_max_bus):
        if redispatching[bus_id] > 0.:
            redisp_avail[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
        elif redispatching[bus_id] < 0.:
            redisp_avail[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
    
    # 按比例分配
    prop_to_gen[redisp_up] = obs.gen_margin_up[redisp_up]
    prop_to_gen[redisp_down] = obs.gen_margin_down[redisp_down]
    redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail[idx_gen]
    act.redispatch = redisp_
```

### 9.5 动作下发

最终，所有动作通过 `BaseAction` 对象下发到环境：

```python
act = self.action_space()
act.storage_p = storage_          # 储能动作
act.curtail_mw = curtailment_mw   # 削减动作
act.redispatch = redisp_          # 重调度动作
act.line_set_status = [(l_id, +1)]  # 线路重连（仅安全模式）
```

---

## 数学公式总结

### 10.1 DC 潮流模型

\[
f_{or}^l = \frac{1}{x^l} (\theta_{or}^l - \theta_{ex}^l)
\]

\[
f_{or,corr}^l = f_{or}^l - \alpha_{por\_error} \cdot \epsilon_{prev}^l
\]

### 10.2 危险模式优化问题

**目标函数**：
\[
\min \quad w_{curtail} \sum_i (\Delta P_{curtail}^i)^2 + w_{storage} \sum_i (\Delta P_{storage}^i)^2 + w_{redisp} \sum_i (\Delta P_{redisp}^i)^2 + \sum_l \left[ \max(0, |f_{or,corr}^l| - m_{th} \cdot P_{th,lim}^l) \right]^2
\]

**约束条件**：
- KCL：\( \sum_{l: or(l)=i} f_{or}^l - \sum_{l: ex(l)=i} f_{ex}^l + P_{inj}^i = 0 \)
- 运行限制：\( -\Delta P_{limit,down}^i \leq \Delta P^i \leq \Delta P_{limit,up}^i \)
- 能量平衡：\( \sum_i \Delta P_{curtail}^i + \sum_i \Delta P_{storage}^i - \sum_i \Delta P_{redisp}^i = P_{storage,obs} \)

### 10.3 安全模式优化问题

**目标函数**：
\[
\min \quad w_{redisp,target} \sum_i (P_{dispatch,after}^i)^2 + w_{storage,target} \sum_i (SoC_{after}^i - SoC_{target}^i)^2 + w_{curtail,target} \sum_i (\Delta P_{curtail}^i + \Delta P_{curtail,down}^i)^2
\]

**约束条件**：
- 所有危险模式的约束
- 热极限硬约束：\( -m_{th} \cdot P_{th,lim}^l \leq f_{or,corr}^l \leq m_{th} \cdot P_{th,lim}^l \)

### 10.4 模式切换

\[
\text{mode} = \begin{cases}
\text{unsafe} & \text{if } \max(\rho) > \rho_{danger} \\
\text{safe} & \text{if } \max(\rho) < \rho_{safe} \\
\text{intermediate} & \text{otherwise}
\end{cases}
\]

---

## 总结

OptimCVXPY 通过以下步骤实现智能控制：

1. **建模**：使用 DC 潮流模型近似电网物理特性
2. **优化**：将控制问题转化为凸优化问题
3. **求解**：使用 CVXPY 求解器求解优化问题
4. **转换**：将优化结果转换为 grid2op 动作
5. **下发**：通过 `BaseAction` 对象下发到环境

关键优势：
- **可解释性强**：基于物理模型和优化理论
- **稳定性好**：不依赖随机性，结果可重复
- **快速响应**：直接优化，无需训练

主要局限：
- **DC 近似误差**：忽略无功功率，可能不够准确
- **计算开销**：每个时间步都需要求解优化问题
- **参数敏感**：需要仔细调整阈值和惩罚参数

