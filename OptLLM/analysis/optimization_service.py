# -*- coding: utf-8 -*-
"""
Optimization Service for OARA
基于 OptimCVXPY 关键逻辑的可配置优化服务（独立实现，便于修改）
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import logging
import warnings
import cvxpy as cp

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Action import ActionSpace
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend

logger = logging.getLogger("OptimizationService")


class OptimizationService:
    """
    可配置的优化服务
    
    独立实现 OptimCVXPY 的关键逻辑，提供动态参数配置接口。
    核心方法：solve_with_config(observation, config_dict)
    """
    
    SOLVER_TYPES = [cp.OSQP, cp.SCS, cp.SCIPY]
    
    def __init__(
        self,
        action_space: ActionSpace,
        env: Environment,
        lines_x_pu: Optional[np.array] = None,
        margin_th_limit: float = 0.9,
        alpha_por_error: float = 0.5,
        penalty_curtailment_unsafe: float = 0.1,
        penalty_redispatching_unsafe: float = 0.03,
        penalty_storage_unsafe: float = 0.3,
        penalty_overflow_unsafe: float = 1.0,
        margin_rounding: float = 0.01,
        margin_sparse: float = 5e-3,
        **kwargs
    ):
        """
        初始化优化服务
        
        Args:
            action_space: Grid2Op 动作空间
            env: Grid2Op 环境
            lines_x_pu: 线路电抗（可选，如果不提供则从环境读取）
            margin_th_limit: 热极限安全裕度（默认 0.9）
            alpha_por_error: 误差修正系数（默认 0.5）
            penalty_curtailment_unsafe: 切负荷惩罚（默认 0.1）
            penalty_redispatching_unsafe: 再调度惩罚（默认 0.03）
            penalty_storage_unsafe: 储能惩罚（默认 0.3）
            penalty_overflow_unsafe: 过载惩罚权重（默认 1.0，生存模式可设为 1000.0）
            margin_rounding: 舍入裕度（默认 0.01）
            margin_sparse: 稀疏化阈值（默认 5e-3）
        """
        # 环境检查
        if env.n_storage > 0 and not env.action_space.supports_type("set_storage"):
            raise RuntimeError("环境不支持储能操作，但电网中有储能单元")
        
        if np.any(env.gen_renewable) and not env.action_space.supports_type("curtail"):
            raise RuntimeError("环境不支持切负荷操作，但电网中有可再生能源")
        
        if not env.action_space.supports_type("redispatch"):
            raise RuntimeError("环境必须支持再调度操作")
        
        # 手动保存 action_space（不再继承 BaseAgent）
        self.action_space = action_space
        
        # 初始化 CVXPY 参数
        self._margin_th_limit: cp.Parameter = cp.Parameter(value=margin_th_limit, nonneg=True)
        self._penalty_curtailment_unsafe: cp.Parameter = cp.Parameter(
            value=penalty_curtailment_unsafe, nonneg=True
        )
        self._penalty_redispatching_unsafe: cp.Parameter = cp.Parameter(
            value=penalty_redispatching_unsafe, nonneg=True
        )
        self._penalty_storage_unsafe: cp.Parameter = cp.Parameter(
            value=penalty_storage_unsafe, nonneg=True
        )
        self._penalty_overflow_unsafe: cp.Parameter = cp.Parameter(
            value=penalty_overflow_unsafe, nonneg=True
        )
        self._alpha_por_error: cp.Parameter = cp.Parameter(value=alpha_por_error, nonneg=True)
        
        # 电网参数
        self.nb_max_bus: int = 2 * env.n_sub
        self._storage_power_obs: cp.Parameter = cp.Parameter(value=0.)
        
        # 储能目标
        SoC = np.zeros(shape=self.nb_max_bus)
        self._storage_setpoint: np.ndarray = 0.5 * env.storage_Emax
        for bus_id in range(self.nb_max_bus):
            SoC[bus_id] = 0.5 * self._storage_setpoint[env.storage_to_subid == bus_id].sum()
        self._storage_target_bus = cp.Parameter(shape=self.nb_max_bus, value=1.0 * SoC, nonneg=True)
        
        # 其他参数
        self.margin_rounding: float = float(margin_rounding)
        self.margin_sparse: float = float(margin_sparse)
        
        # 读取线路电抗
        if lines_x_pu is not None:
            powerlines_x = 1.0 * np.array(lines_x_pu).astype(float)
        elif isinstance(env.backend, LightSimBackend):
            powerlines_x = np.array(
                [float(el.x_pu) for el in env.backend._grid.get_lines()] +
                [float(el.x_pu) for el in env.backend._grid.get_trafos()]
            )
        elif isinstance(env.backend, PandaPowerBackend):
            # PandaPowerBackend 需要从后端直接读取
            # 注意：如果无法读取，请提供 lines_x_pu 参数
            raise RuntimeError(
                f"PandaPowerBackend 暂不支持自动读取线路电抗。"
                "请提供 lines_x_pu 参数。"
            )
        else:
            raise RuntimeError(
                f"无法从后端类型 {type(env.backend)} 读取线路电抗。"
                "请提供 lines_x_pu 参数。"
            )
        
        if powerlines_x.shape[0] != env.n_line:
            raise ValueError("线路电抗数量与电网线路数量不匹配")
        if np.any(powerlines_x <= 0.):
            raise ValueError("所有线路电抗必须为正数")
        
        self._powerlines_x: cp.Parameter = cp.Parameter(
            shape=powerlines_x.shape, value=1.0 * powerlines_x, pos=True
        )
        self._prev_por_error: cp.Parameter = cp.Parameter(
            shape=powerlines_x.shape, value=np.zeros(env.n_line)
        )
        
        # 拓扑参数
        self.bus_or: cp.Parameter = cp.Parameter(
            shape=env.n_line, value=env.line_or_to_subid, integer=True
        )
        self.bus_ex: cp.Parameter = cp.Parameter(
            shape=env.n_line, value=env.line_ex_to_subid, integer=True
        )
        self.bus_load: cp.Parameter = cp.Parameter(
            shape=env.n_load, value=env.load_to_subid, integer=True
        )
        self.bus_gen: cp.Parameter = cp.Parameter(
            shape=env.n_gen, value=env.gen_to_subid, integer=True
        )
        
        if env.n_storage:
            self.bus_storage: cp.Parameter = cp.Parameter(
                shape=env.n_storage, value=env.storage_to_subid, integer=True
            )
        else:
            self.bus_storage = None
        
        # 注入和约束参数
        this_zeros_ = np.zeros(self.nb_max_bus)
        self.load_per_bus: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.gen_per_bus: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.redisp_up: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.redisp_down: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.curtail_down: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.curtail_up: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.storage_down: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.storage_up: cp.Parameter = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self._th_lim_mw: cp.Parameter = cp.Parameter(
            shape=env.n_line, value=env.get_thermal_limit(), nonneg=True
        )
        
        self._v_ref: np.ndarray = 1.0 * env.get_obs().v_or
        
        # 日志
        self.logger = logger
        
        # 潮流计算结果
        self.flow_computed = np.zeros(env.n_line, dtype=float)
        self.flow_computed[:] = np.nan
        
        logger.info("OptimizationService 初始化完成")
    
    def reset(self, obs: BaseObservation):
        """重置优化服务状态"""
        self._prev_por_error.value[:] = 0.
        conv_ = self.run_dc(obs)
        if conv_:
            self._prev_por_error.value[:] = self.flow_computed - obs.p_or
        else:
            self.logger.warning("DC 潮流计算未收敛，无法初始化误差")
    
    def solve_with_config(
        self,
        observation: BaseObservation,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        核心接口：接收观测 + 配置，返回动作
        
        Args:
            observation: 当前电网观测
            config: 配置字典，包含以下键：
                - penalty_curtailment: 切负荷惩罚权重（默认 0.1）
                - penalty_redispatch: 再调度惩罚权重（默认 0.03）
                - penalty_storage: 储能惩罚权重（默认 0.3，可选）
                - penalty_overflow: 过载惩罚权重（默认 1.0，生存模式可设为 1000.0）
                - margin_th_limit: 热极限安全裕度（默认 0.9，范围 0.8-1.0）
                - strategy_description: 策略描述（可选，用于日志）
        
        Returns:
            结果字典，包含：
                - status: "SUCCESS" | "INFEASIBLE" | "ERROR"
                - action: Grid2Op Action（如果成功）
                - reason: 错误原因（如果失败）
                - metrics: 指标字典（如果成功）
        """
        try:
            # Step 1: 注入动态参数（核心创新点）
            penalty_curtailment = config.get("penalty_curtailment", 0.1)
            penalty_redispatch = config.get("penalty_redispatch", 0.03)
            penalty_storage = config.get("penalty_storage", 0.3)
            penalty_overflow = config.get("penalty_overflow", 1.0)
            margin_th_limit = config.get("margin_th_limit", 0.9)
            
            # 验证参数范围
            if not (0.0 <= penalty_curtailment <= 100.0):
                logger.warning(f"penalty_curtailment 超出合理范围: {penalty_curtailment}，将使用默认值 0.1")
                penalty_curtailment = 0.1
            if not (0.0 <= penalty_redispatch <= 100.0):
                logger.warning(f"penalty_redispatch 超出合理范围: {penalty_redispatch}，将使用默认值 0.03")
                penalty_redispatch = 0.03
            if not (0.0 <= penalty_storage <= 100.0):
                logger.warning(f"penalty_storage 超出合理范围: {penalty_storage}，将使用默认值 0.3")
                penalty_storage = 0.3
            if not (0.0 <= penalty_overflow <= 10000.0):
                logger.warning(f"penalty_overflow 超出合理范围: {penalty_overflow}，将使用默认值 1.0")
                penalty_overflow = 1.0
            if not (0.5 <= margin_th_limit <= 1.0):
                logger.warning(f"margin_th_limit 超出合理范围: {margin_th_limit}，将使用默认值 0.9")
                margin_th_limit = 0.9
            
            # 更新 CVXPY Parameter 的值
            self._penalty_curtailment_unsafe.value = float(penalty_curtailment)
            self._penalty_redispatching_unsafe.value = float(penalty_redispatch)
            self._penalty_storage_unsafe.value = float(penalty_storage)
            self._penalty_overflow_unsafe.value = float(penalty_overflow)
            self._margin_th_limit.value = float(margin_th_limit)
            
            strategy_desc = config.get("strategy_description", "未指定")
            logger.info(f"优化配置: penalty_curtailment={penalty_curtailment:.4f}, "
                       f"penalty_redispatch={penalty_redispatch:.4f}, "
                       f"penalty_overflow={penalty_overflow:.4f}, "
                       f"margin_th_limit={margin_th_limit:.4f}, "
                       f"策略: {strategy_desc}")
            
            # Step 2: 更新电网状态参数
            self.update_parameters(observation, unsafe=True)
            
            # Step 3: 求解
            curtailment, storage, redispatching = self.compute_optimum_unsafe()
            
            # Step 4: 结果验证
            if np.all(np.isnan(curtailment)) or np.all(np.isnan(redispatching)):
                # 诊断无解原因
                diagnosis = self._diagnose_infeasibility(observation, config)
                return {
                    "status": "INFEASIBLE",
                    "reason": diagnosis
                }
            
            if np.any(np.isnan(curtailment)) or np.any(np.isnan(redispatching)):
                logger.warning("优化结果包含 NaN 值，可能存在问题")
            
            # Step 5: 转换为 Grid2Op 动作
            action = self.to_grid2op(
                observation, curtailment, storage, redispatching, safe=False
            )
            
            # 计算指标
            curtailment_sum = float(np.sum(np.abs(curtailment[~np.isnan(curtailment)])))
            redispatch_sum = float(np.sum(np.abs(redispatching[~np.isnan(redispatching)])))
            storage_sum = float(np.sum(np.abs(storage[~np.isnan(storage)])))
            
            logger.info(f"优化成功: 切负荷={curtailment_sum:.2f} MW, "
                       f"再调度={redispatch_sum:.2f} MW, "
                       f"储能={storage_sum:.2f} MW")
            
            return {
                "status": "SUCCESS",
                "action": action,
                "metrics": {
                    "curtailment_mw": curtailment_sum,
                    "redispatch_mw": redispatch_sum,
                    "storage_mw": storage_sum
                }
            }
            
        except Exception as e:
            error_msg = f"优化求解过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "ERROR",
                "reason": error_msg
            }
    
    # ========== 以下是从 OptimCVXPY 提取的关键方法 ==========
    
    def update_parameters(self, obs: BaseObservation, unsafe: bool = True):
        """更新优化问题的参数"""
        self._update_topo_param(obs)
        self._update_th_lim_param(obs)
        self._update_inj_param(obs)
        if unsafe:
            self._update_constraints_param_unsafe(obs)
        self._validate_param_values()
    
    def _update_topo_param(self, obs: BaseObservation):
        """更新拓扑参数"""
        tmp_ = 1 * obs.line_or_to_subid
        tmp_[obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_.astype(int)
        tmp_ = 1 * obs.line_ex_to_subid
        tmp_[obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_.astype(int)
        
        # 断开线路的处理
        self.bus_ex.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value[:] = self.bus_or.value.astype(int)
        self.bus_ex.value[:] = self.bus_ex.value.astype(int)
        
        tmp_ = obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_.astype(int)
        
        tmp_ = obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_.astype(int)
        
        if self.bus_storage is not None:
            tmp_ = obs.storage_to_subid
            tmp_[obs.storage_bus == 2] += obs.n_sub
            self.bus_storage.value[:] = tmp_.astype(int)
    
    def _update_th_lim_param(self, obs: BaseObservation):
        """更新热极限参数"""
        threshold_ = 1.
        self._th_lim_mw.value[:] = (0.001 * obs.thermal_limit)**2 * obs.v_or**2 * 3. - obs.q_or**2
        mask_ok = self._th_lim_mw.value >= threshold_
        self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
        self._th_lim_mw.value[~mask_ok] = threshold_
        
        index_disc = obs.v_or == 0.
        self._th_lim_mw.value[index_disc] = 0.001 * (obs.thermal_limit * self._v_ref)[index_disc] * np.sqrt(3.)
    
    def _update_storage_power_obs(self, obs: BaseObservation):
        """更新储能功率观测"""
        self._storage_power_obs.value += obs.storage_power.sum()
    
    def _update_inj_param(self, obs: BaseObservation):
        """更新注入参数"""
        self._update_storage_power_obs(obs)
        self.load_per_bus.value[:] = 0.
        self.gen_per_bus.value[:] = 0.
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()
    
    def _add_redisp_const(self, obs: BaseObservation, bus_id: int):
        """添加再调度约束"""
        self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
        self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
    
    def _add_storage_const(self, obs: BaseObservation, bus_id: int):
        """添加储能约束"""
        if self.bus_storage is None:
            return
        stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
        stor_down = min(stor_down, obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
        self.storage_down.value[bus_id] = stor_down
        stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
        stor_up = min(stor_up, (obs.storage_Emax - obs.storage_charge)[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
        self.storage_up.value[bus_id] = stor_up
    
    def _update_constraints_param_unsafe(self, obs: BaseObservation):
        """更新不安全模式的约束参数"""
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        for bus_id in range(self.nb_max_bus):
            self._add_redisp_const(obs, bus_id)
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = 0.
            self.curtail_up.value[bus_id] = tmp_[mask_].sum()
            self._add_storage_const(obs, bus_id)
        self._remove_margin_rounding()
    
    def _remove_margin_rounding(self):
        """移除舍入裕度"""
        self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
        self.storage_up.value[self.storage_up.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_down.value[self.curtail_down.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_up.value[self.curtail_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_up.value[self.redisp_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_down.value[self.redisp_down.value > self.margin_rounding] -= self.margin_rounding
    
    def _validate_param_values(self):
        """验证参数值"""
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)
        self.curtail_down._validate_value(self.curtail_down.value)
        self.curtail_up._validate_value(self.curtail_up.value)
        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)
        self._th_lim_mw._validate_value(self._th_lim_mw.value)
    
    def _aux_compute_kcl(self, inj_bus, f_or):
        """计算基尔霍夫电流定律约束"""
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
    
    def _mask_theta_zero(self):
        """标记哪些节点的相角应该为0（slack bus）"""
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value.astype(int)] = False
        theta_is_zero[self.bus_ex.value.astype(int)] = False
        theta_is_zero[self.bus_load.value.astype(int)] = False
        theta_is_zero[self.bus_gen.value.astype(int)] = False
        if self.bus_storage is not None:
            theta_is_zero[self.bus_storage.value.astype(int)] = False
        theta_is_zero[0] = True  # slack bus
        return theta_is_zero
    
    def compute_optimum_unsafe(self):
        """计算不安全模式下的最优解"""
        # 变量
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)
        
        # 计算潮流
        bus_or_idx = self.bus_or.value.astype(int)
        bus_ex_idx = self.bus_ex.value.astype(int)
        f_or = cp.multiply(1. / self._powerlines_x, (theta[bus_or_idx] - theta[bus_ex_idx]))
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching) - self._storage_power_obs
        
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        # 约束
        constraints = (
            [theta[theta_is_zero] == 0] +
            [el == 0 for el in KCL_eq] +
            [redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
            [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
            [storage <= self.storage_up, storage >= -self.storage_down] +
            [energy_added == 0]
        )
        
        # 目标函数
        cost = (
            self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw) +
            self._penalty_storage_unsafe * cp.sum_squares(storage) +
            self._penalty_redispatching_unsafe * cp.sum_squares(redispatching) +
            self._penalty_overflow_unsafe * cp.sum_squares(cp.pos(cp.abs(f_or_corr) - self._margin_th_limit * self._th_lim_mw))
        )
        
        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.
        else:
            self.logger.error(f"优化求解失败，所有求解器都未收敛")
            self.flow_computed[:] = np.nan
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return res
    
    def _solve_problem(self, prob, solver_type=None):
        """尝试多个求解器直到找到一个收敛的"""
        if solver_type is None:
            for solver_type in self.SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver_type)
                if res:
                    self.logger.info(f"求解器 {solver_type} 收敛")
                    return True
            return False
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tmp_ = prob.solve(solver=solver_type, warm_start=False)
            if np.isfinite(tmp_):
                return True
            else:
                self.logger.warning(f"求解器 {solver_type} 返回无穷大")
                raise cp.error.SolverError("Infinite value")
        except cp.error.SolverError as exc_:
            self.logger.warning(f"求解器 {solver_type} 失败: {exc_}")
            return False
    
    def run_dc(self, obs: BaseObservation):
        """运行 DC 潮流计算"""
        self._update_topo_param(obs)
        self._update_inj_param(obs)
        
        theta = cp.Variable(shape=self.nb_max_bus)
        bus_or_idx = self.bus_or.value.astype(int)
        bus_ex_idx = self.bus_ex.value.astype(int)
        f_or = cp.multiply(1. / self._powerlines_x, (theta[bus_or_idx] - theta[bus_ex_idx]))
        inj_bus = self.load_per_bus - self.gen_per_bus
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        constraints = ([theta[theta_is_zero] == 0] + [el == 0 for el in KCL_eq])
        cost = 1.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
        else:
            self.logger.error("DC 潮流计算未收敛")
            self.flow_computed[:] = np.nan
        
        return has_converged
    
    def _clean_vect(self, curtailment, storage, redispatching):
        """清理向量，将小于阈值的值设为0"""
        curtailment[np.abs(curtailment) < self.margin_sparse] = 0.
        storage[np.abs(storage) < self.margin_sparse] = 0.
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.
    
    def to_grid2op(
        self,
        obs: BaseObservation,
        curtailment: np.ndarray,
        storage: np.ndarray,
        redispatching: np.ndarray,
        act: BaseAction = None,
        safe: bool = False
    ) -> BaseAction:
        """将优化结果转换为 Grid2Op 动作"""
        self._clean_vect(curtailment, storage, redispatching)
        
        if act is None:
            act = self.action_space({})
        
        # 储能
        if act.n_storage and np.any(np.abs(storage) > 0.):
            storage_ = np.zeros(shape=act.n_storage)
            storage_[:] = storage[self.bus_storage.value.astype(int)]
            act.storage_p = storage_
        
        # 切负荷
        if np.any(np.abs(curtailment) > 0.):
            curtailment_mw = np.zeros(shape=act.n_gen) - 1.
            gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
            idx_gen = self.bus_gen.value[gen_curt].astype(int)
            tmp_ = curtailment[idx_gen]
            modif_gen_optim = tmp_ != 0.
            gen_p = 1.0 * obs.gen_p
            aux_ = curtailment_mw[gen_curt]
            aux_[modif_gen_optim] = (
                gen_p[gen_curt][modif_gen_optim] -
                tmp_[modif_gen_optim] * gen_p[gen_curt][modif_gen_optim] /
                self.gen_per_bus.value[idx_gen][modif_gen_optim]
            )
            aux_[~modif_gen_optim] = -1.
            curtailment_mw[gen_curt] = aux_
            curtailment_mw[~gen_curt] = -1.
            act.curtail_mw = curtailment_mw
        
        # 再调度
        if np.any(np.abs(redispatching) > 0.):
            redisp_ = np.zeros(obs.n_gen)
            gen_redi = obs.gen_redispatchable
            idx_gen = self.bus_gen.value[gen_redi].astype(int)
            tmp_ = redispatching[idx_gen]
            gen_p = 1.0 * obs.gen_p
            
            redisp_avail = np.zeros(self.nb_max_bus)
            for bus_id in range(self.nb_max_bus):
                if redispatching[bus_id] > 0.:
                    redisp_avail[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
                elif redispatching[bus_id] < 0.:
                    redisp_avail[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
            
            prop_to_gen = np.zeros(obs.n_gen)
            redisp_up = np.zeros(obs.n_gen, dtype=bool)
            redisp_up[gen_redi] = tmp_ > 0.
            prop_to_gen[redisp_up] = obs.gen_margin_up[redisp_up]
            redisp_down = np.zeros(obs.n_gen, dtype=bool)
            redisp_down[gen_redi] = tmp_ < 0.
            prop_to_gen[redisp_down] = obs.gen_margin_down[redisp_down]
            
            nothing_happens = (redisp_avail[idx_gen] == 0.) & (prop_to_gen[gen_redi] == 0.)
            set_to_one_nothing = 1.0 * redisp_avail[idx_gen]
            set_to_one_nothing[nothing_happens] = 1.0
            redisp_avail[idx_gen] = set_to_one_nothing
            
            if np.any(np.abs(redisp_avail[idx_gen]) <= self.margin_sparse):
                self.logger.warning("某些发电机的再调度量被取消（无可用裕度）")
                this_fix_ = 1.0 * redisp_avail[idx_gen]
                too_small_here = np.abs(this_fix_) <= self.margin_sparse
                tmp_[too_small_here] = 0.
                this_fix_[too_small_here] = 1.
                redisp_avail[idx_gen] = this_fix_
            
            redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail[idx_gen]
            redisp_[~gen_redi] = 0.
            act.redispatch = redisp_
        
        return act
    
    def _diagnose_infeasibility(
        self,
        observation: BaseObservation,
        config: Dict[str, Any]
    ) -> str:
        """
        诊断无解原因
        
        分析为什么优化器无法找到可行解，生成详细的反馈信息
        """
        margin_th_limit = config.get("margin_th_limit", 0.9)
        
        # 1. 检查过载情况
        max_rho = float(observation.rho.max())
        overflow_mask = observation.rho > 1.0
        overflow_count = int(overflow_mask.sum())
        
        if overflow_count > 0:
            # 计算过载量
            overflow_lines = observation.rho[overflow_mask]
            max_overflow_rho = float(overflow_lines.max())
            max_overflow_mw = 0.0
            
            # 估算需要减少的功率流
            for line_id in np.where(overflow_mask)[0]:
                thermal_limit = float(observation.thermal_limit[line_id])
                current_flow = float(observation.p_or[line_id])
                target_flow = thermal_limit * margin_th_limit
                overflow_mw = current_flow - target_flow
                max_overflow_mw = max(max_overflow_mw, overflow_mw)
            
            # 2. 检查可用的调节能力
            if hasattr(observation, 'gen_redispatchable'):
                redispatchable_mask = observation.gen_redispatchable
                total_margin_up = float(observation.gen_margin_up[redispatchable_mask].sum())
                total_margin_down = float(observation.gen_margin_down[redispatchable_mask].sum())
            else:
                total_margin_up = 0.0
                total_margin_down = 0.0
            
            # 检查切负荷能力
            if hasattr(observation, 'gen_renewable'):
                renewable_mask = observation.gen_renewable
                total_curtailment_cap = float(observation.gen_p[renewable_mask].sum())
            else:
                total_curtailment_cap = 0.0
            
            total_relief_capability = total_margin_down + total_curtailment_cap
            
            # 3. 生成诊断信息
            if max_overflow_mw > total_relief_capability * 1.5:
                return (
                    f"物理极限：过载量（约 {max_overflow_mw:.1f}MW）远超可用调节能力 "
                    f"（{total_relief_capability:.1f}MW）。即使全切负荷也无法满足约束。"
                    f"建议：将 margin_th_limit 提高到 0.98 或 1.0"
                )
            elif max_overflow_mw > total_relief_capability:
                return (
                    f"调节能力不足：过载量（约 {max_overflow_mw:.1f}MW）超过可用调节能力 "
                    f"（{total_relief_capability:.1f}MW）。"
                    f"建议：将 margin_th_limit 提高到 0.95 或 0.98，"
                    f"或将 penalty_curtailment 降低到 0.001"
                )
            else:
                return (
                    f"约束过紧：当前 margin_th_limit={margin_th_limit:.2f} 可能过于严格。"
                    f"过载线路 {overflow_count} 条，最大负载率 {max_rho:.1%}。"
                    f"建议：将 margin_th_limit 提高到 {min(0.95, margin_th_limit + 0.05):.2f}"
                )
        else:
            # 没有过载但无解，可能是其他约束问题
            return (
                f"约束冲突：即使当前无过载，优化器仍无法找到可行解。"
                f"可能是 penalty 权重设置导致。"
                f"建议：检查 penalty_curtailment 和 penalty_redispatch 的设置"
            )
    
    def _diagnose_infeasibility(
        self,
        observation: BaseObservation,
        config: Dict[str, Any]
    ) -> str:
        """
        诊断优化无解的原因
        
        通过分析约束和调节能力，给出详细的失败原因（字符串格式，用于反馈给 LLM）
        
        Args:
            observation: 当前观测
            config: 优化配置
            
        Returns:
            诊断信息字符串
        """
        
        # 计算最大过载量
        overflow_mask = observation.rho > 1.0
        if not np.any(overflow_mask):
            return "无过载但求解失败，可能是数值问题。建议检查 penalty 权重设置"
        
        max_rho = float(observation.rho.max())
        max_rho_line = int(np.argmax(observation.rho))
        thermal_limit = float(observation.thermal_limit[max_rho_line])
        margin_th_limit = config.get("margin_th_limit", 0.9)
        target_limit = margin_th_limit * thermal_limit
        
        # 计算实际过载量（相对于目标限制）
        actual_flow = observation.p_or[max_rho_line]
        overflow_mw = actual_flow - target_limit
        
        # 计算可用调节能力
        if hasattr(observation, 'gen_redispatchable'):
            redispatchable_mask = observation.gen_redispatchable
            total_margin_up = float(observation.gen_margin_up[redispatchable_mask].sum())
            total_margin_down = float(observation.gen_margin_down[redispatchable_mask].sum())
        else:
            total_margin_up = 0.0
            total_margin_down = 0.0
        
        # 可切负荷容量
        if hasattr(observation, 'gen_renewable'):
            renewable_mask = observation.gen_renewable
            renewable_capacity = float(observation.gen_p[renewable_mask].sum())
        else:
            renewable_capacity = 0.0
        
        # 储能能力
        storage_capacity = 0.0
        if hasattr(observation, 'storage_Emax') and observation.n_storage > 0:
            storage_capacity = float(observation.storage_Emax.sum())
        
        total_relief_capability = total_margin_down + renewable_capacity + storage_capacity
        
        # 诊断逻辑
        if overflow_mw > total_relief_capability * 1.5:
            return (
                f"物理极限：线路 {max_rho_line} 过载约 {overflow_mw:.1f} MW，"
                f"远超可用调节能力 {total_relief_capability:.1f} MW。"
                f"即使全切负荷也无法满足约束。建议：将 margin_th_limit 提高到 0.98 或 1.0"
            )
        elif overflow_mw > total_relief_capability:
            return (
                f"调节能力不足：线路 {max_rho_line} 过载约 {overflow_mw:.1f} MW，"
                f"超过可用调节能力 {total_relief_capability:.1f} MW。"
                f"建议：将 margin_th_limit 提高到 0.95 或 0.98，"
                f"或将 penalty_curtailment 降低到 0.001"
            )
        elif margin_th_limit >= 0.98:
            return (
                f"约束过紧：margin_th_limit={margin_th_limit:.2f} 已接近极限。"
                f"线路 {max_rho_line} 需要降至 {target_limit:.1f} MW，"
                f"但当前流量为 {actual_flow:.1f} MW。"
                f"建议：考虑其他策略（如拓扑调整）"
            )
        else:
            return (
                f"约束冲突：当前配置 (penalty_curtailment={config.get('penalty_curtailment', 0.1):.3f}, "
                f"margin_th_limit={margin_th_limit:.2f}) 无法找到可行解。"
                f"线路 {max_rho_line} 过载约 {overflow_mw:.1f} MW。"
                f"建议：将 margin_th_limit 提高到 {min(0.95, margin_th_limit + 0.05):.2f}，"
                f"或将 penalty_curtailment 降低到 {max(0.001, config.get('penalty_curtailment', 0.1) * 0.1):.3f}"
            )
