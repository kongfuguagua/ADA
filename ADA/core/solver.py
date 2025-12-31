# -*- coding: utf-8 -*-
"""
Solver 模块：数学优化调度专家
复刻 OptimCVXPY 的核心优化逻辑，基于凸优化解决功率流平衡问题
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import numpy as np
import cvxpy as cp
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend

from ADA.utils.definitions import CandidateAction
from utils import get_logger

# 尝试导入 lightsim2grid 的 init 函数
try:
    from importlib.metadata import version as version_str
    from packaging import version
    lightsim2grid_version = version.parse(version_str("lightsim2grid"))
    if lightsim2grid_version >= version.parse("0.9.1"):
        from lightsim2grid.gridmodel import init_from_pandapower as init
    else:
        from lightsim2grid.gridmodel import init
except ImportError:
    init = None

logger = get_logger("ADA.Solver")


class Solver:
    """
    数学优化调度专家
    
    基于 DC 潮流模型和凸优化，生成重调度、削减和储能动作。
    完全复刻 OptimCVXPY 的核心逻辑，不依赖 LLM。
    """
    
    SOLVER_TYPES = [cp.OSQP, cp.SCS, cp.SCIPY]
    
    def __init__(
        self,
        action_space,
        observation_space,
        env=None,
        lines_x_pu: Optional[np.ndarray] = None,
        margin_th_limit: float = 0.9,
        alpha_por_error: float = 0.5,
        rho_danger: float = 0.95,
        rho_safe: float = 0.85,
        penalty_curtailment_unsafe: float = 0.1,
        penalty_redispatching_unsafe: float = 0.03,
        penalty_storage_unsafe: float = 0.3,
        penalty_curtailment_safe: float = 0.0,
        penalty_redispatching_safe: float = 0.0,
        penalty_storage_safe: float = 0.0,
        weight_redisp_target: float = 1.0,
        weight_storage_target: float = 1.0,
        weight_curtail_target: float = 1.0,
        margin_rounding: float = 0.01,
        margin_sparse: float = 5e-3,
        **kwargs
    ):
        """
        初始化 Solver
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            env: Grid2Op 环境（用于读取线路电抗）
            lines_x_pu: 线路电抗（per unit），如果为 None 则从环境读取
            margin_th_limit: 热极限裕度
            alpha_por_error: 误差修正系数
            rho_danger: 危险阈值
            rho_safe: 安全阈值
            penalty_*: 各种惩罚参数
            weight_*: 各种权重参数
            margin_rounding: 舍入裕度
            margin_sparse: 稀疏化阈值
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.env = env
        
        # 参数
        self.margin_th_limit = cp.Parameter(value=margin_th_limit, nonneg=True)
        self.alpha_por_error = cp.Parameter(value=alpha_por_error, nonneg=True)
        self.rho_danger = float(rho_danger)
        self.rho_safe = float(rho_safe)
        self.margin_rounding = float(margin_rounding)
        self.margin_sparse = float(margin_sparse)
        
        # 惩罚参数
        self._penalty_curtailment_unsafe = cp.Parameter(value=penalty_curtailment_unsafe, nonneg=True)
        self._penalty_redispatching_unsafe = cp.Parameter(value=penalty_redispatching_unsafe, nonneg=True)
        self._penalty_storage_unsafe = cp.Parameter(value=penalty_storage_unsafe, nonneg=True)
        self._penalty_curtailment_safe = cp.Parameter(value=penalty_curtailment_safe, nonneg=True)
        self._penalty_redispatching_safe = cp.Parameter(value=penalty_redispatching_safe, nonneg=True)
        self._penalty_storage_safe = cp.Parameter(value=penalty_storage_safe, nonneg=True)
        
        # 权重参数
        self._weight_redisp_target = cp.Parameter(value=weight_redisp_target, nonneg=True)
        self._weight_storage_target = cp.Parameter(value=weight_storage_target, nonneg=True)
        self._weight_curtail_target = cp.Parameter(value=weight_curtail_target, nonneg=True)
        
        # 获取线路电抗
        if lines_x_pu is not None:
            powerlines_x = np.array(lines_x_pu).astype(float)
        elif env is not None:
            if isinstance(env.backend, LightSimBackend):
                powerlines_x = np.array(
                    [float(el.x_pu) for el in env.backend._grid.get_lines()] +
                    [float(el.x_pu) for el in env.backend._grid.get_trafos()]
                )
            elif isinstance(env.backend, PandaPowerBackend) and init is not None:
                pp_net = env.backend._grid
                grid_model = init(pp_net)
                powerlines_x = np.array(
                    [float(el.x_pu) for el in grid_model.get_lines()] +
                    [float(el.x_pu) for el in grid_model.get_trafos()]
                )
            else:
                raise RuntimeError(
                    f"无法从环境读取线路电抗。请提供 lines_x_pu 参数或使用 LightSimBackend/PandaPowerBackend。"
                )
        else:
            raise RuntimeError("必须提供 lines_x_pu 或 env 参数")
        
        if powerlines_x.shape[0] != observation_space.n_line:
            raise ValueError(f"线路数量不匹配：{powerlines_x.shape[0]} != {observation_space.n_line}")
        if np.any(powerlines_x <= 0.):
            raise ValueError("所有线路电抗必须严格为正")
        
        self._powerlines_x = cp.Parameter(shape=powerlines_x.shape, value=powerlines_x, pos=True)
        self._prev_por_error = cp.Parameter(shape=powerlines_x.shape, value=np.zeros(observation_space.n_line))
        
        # 拓扑参数
        self.nb_max_bus = 2 * observation_space.n_sub
        self.bus_or = cp.Parameter(shape=observation_space.n_line, value=observation_space.line_or_to_subid, integer=True)
        self.bus_ex = cp.Parameter(shape=observation_space.n_line, value=observation_space.line_ex_to_subid, integer=True)
        self.bus_load = cp.Parameter(shape=observation_space.n_load, value=observation_space.load_to_subid, integer=True)
        self.bus_gen = cp.Parameter(shape=observation_space.n_gen, value=observation_space.gen_to_subid, integer=True)
        
        if observation_space.n_storage > 0:
            self.bus_storage = cp.Parameter(shape=observation_space.n_storage, value=observation_space.storage_to_subid, integer=True)
        else:
            self.bus_storage = None
        
        # 注入参数
        self.load_per_bus = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.gen_per_bus = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        
        # 约束参数
        self.redisp_up = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.redisp_down = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.curtail_down = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.curtail_up = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.storage_down = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self.storage_up = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        # 从 env 获取热极限（与 OptimCVXPY 一致）
        if env is not None:
            thermal_limit = env.get_thermal_limit()
        else:
            # 降级方案：从 observation_space 获取（如果可用）
            if hasattr(observation_space, 'get_thermal_limit'):
                thermal_limit = observation_space.get_thermal_limit()
            else:
                # 最后降级：使用 observation_space 的属性（如果存在）
                thermal_limit = getattr(observation_space, 'thermal_limit', np.ones(observation_space.n_line) * 1000.0)
        self._th_lim_mw = cp.Parameter(shape=observation_space.n_line, value=thermal_limit, nonneg=True)
        
        # 历史状态
        self._past_dispatch = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus))
        self._past_state_of_charge = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        self._storage_power_obs = cp.Parameter(value=0.)
        
        # 储能目标
        if observation_space.n_storage > 0:
            storage_setpoint = 0.5 * observation_space.storage_Emax
            SoC = np.zeros(self.nb_max_bus)
            for bus_id in range(self.nb_max_bus):
                SoC[bus_id] = 0.5 * storage_setpoint[observation_space.storage_to_subid == bus_id].sum()
            self._storage_target_bus = cp.Parameter(shape=self.nb_max_bus, value=SoC, nonneg=True)
        else:
            self._storage_target_bus = None
        
        # 初始化电压参考值（与 OptimCVXPY 一致）
        if env is not None:
            try:
                obs_init = env.get_obs()
                self._v_ref = 1.0 * obs_init.v_or
            except Exception as e:
                logger.warning(f"无法从环境获取初始观测，将在首次更新时使用观测值: {e}")
                self._v_ref = None
        else:
            self._v_ref = None
        
        # 计算状态
        self.flow_computed = np.zeros(observation_space.n_line, dtype=float)
        self.flow_computed[:] = np.nan
        
        logger.info("Solver 初始化完成")
    
    def solve_dispatch(self, observation: BaseObservation) -> List[CandidateAction]:
        """
        求解调度问题，返回候选动作列表
        
        Args:
            observation: 当前观测
            
        Returns:
            CandidateAction 列表（通常为 1 个）
        """
        candidates = []
        
        try:
            # 更新误差（只保留负误差）
            prev_ok = np.isfinite(self.flow_computed)
            self._prev_por_error.value[prev_ok] = np.minimum(
                self.flow_computed[prev_ok] - observation.p_or[prev_ok], 0.
            )
            self._prev_por_error.value[~prev_ok] = 0.
            
            max_rho = float(observation.rho.max())
            
            # 判断模式（消除 0.85-0.95 死区，引入预防模式）
            if max_rho >= self.rho_danger:
                # 危险模式：最小化过载
                logger.debug("Solver: 危险模式")
                self.update_parameters(observation, unsafe=True)
                curtailment, storage, redispatching = self.compute_optimum_unsafe()
                action = self.to_grid2op(
                    observation, curtailment, storage, redispatching, safe=False
                )
                
                description = f"危险模式优化：最小化过载 (max_rho={max_rho:.3f})"
                
            elif max_rho >= self.rho_safe:
                # 预防模式：在接近危险阈值时，提前做温和优化，避免迟滞
                logger.debug("Solver: 预防模式（接近危险阈值，提前优化）")
                # 这里仍复用“unsafe”参数集，但可以在未来根据需要单独调参
                self.update_parameters(observation, unsafe=True)
                curtailment, storage, redispatching = self.compute_optimum_unsafe()
                action = self.to_grid2op(
                    observation, curtailment, storage, redispatching, safe=False
                )
                
                description = f"预防模式优化：缓解高负载风险 (max_rho={max_rho:.3f})"
                
            else:
                # 安全模式：恢复/维持参考状态
                logger.debug("Solver: 安全模式")
                self.update_parameters(observation, unsafe=False)
                curtailment, storage, redispatching = self.compute_optimum_safe(observation)
                action = self.to_grid2op(
                    observation, curtailment, storage, redispatching, safe=True
                )
                
                description = f"安全模式优化：恢复/维持参考状态 (max_rho={max_rho:.3f})"
            
            # 包装为 CandidateAction
            candidate = CandidateAction(
                source="Math_Dispatch",
                action_obj=action,
                description=description,
                priority=1 if max_rho > self.rho_danger else 0
            )
            candidates.append(candidate)
            
        except Exception as e:
            logger.error(f"Solver 求解失败: {e}", exc_info=True)
            # 降级：返回空动作
            fallback_action = self.action_space({})
            candidates.append(CandidateAction(
                source="Math_Dispatch",
                action_obj=fallback_action,
                description=f"Solver 求解失败，返回 Do Nothing: {str(e)}",
                priority=-1
            ))
        
        return candidates
    
    def update_parameters(self, obs: BaseObservation, unsafe: bool = True) -> None:
        """更新优化参数"""
        # 更新拓扑参数
        self._update_topo_param(obs)
        
        # 更新热极限参数
        self._update_th_lim_param(obs)
        
        # 更新注入参数
        self._update_inj_param(obs)
        
        # 更新约束参数
        if unsafe:
            self._update_constraints_param_unsafe(obs)
        else:
            self._update_constraints_param_safe(obs)
    
    def _update_topo_param(self, obs: BaseObservation) -> None:
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
    
    def _update_th_lim_param(self, obs: BaseObservation) -> None:
        """更新热极限参数"""
        threshold_ = 1.
        self._th_lim_mw.value[:] = (0.001 * obs.thermal_limit) ** 2 * obs.v_or ** 2 * 3. - obs.q_or ** 2
        mask_ok = self._th_lim_mw.value >= threshold_
        self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
        self._th_lim_mw.value[~mask_ok] = threshold_
        
        index_disc = obs.v_or == 0.
        if hasattr(obs, 'v_or') and len(obs.v_or) > 0:
            # 使用保存的 _v_ref，如果不存在则使用当前观测值
            v_ref = self._v_ref if self._v_ref is not None else obs.v_or
            # 更新 _v_ref 为当前观测值（与 OptimCVXPY 一致）
            self._v_ref = 1.0 * obs.v_or
            self._th_lim_mw.value[index_disc] = 0.001 * (obs.thermal_limit[index_disc] * v_ref[index_disc]) * np.sqrt(3.)
    
    def _update_inj_param(self, obs: BaseObservation) -> None:
        """更新注入参数"""
        self._storage_power_obs.value += obs.storage_power.sum() if hasattr(obs, 'storage_power') else 0.
        
        self.load_per_bus.value[:] = 0.
        self.gen_per_bus.value[:] = 0.
        
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
        
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()
    
    def _update_constraints_param_unsafe(self, obs: BaseObservation) -> None:
        """更新危险模式约束参数"""
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        
        for bus_id in range(self.nb_max_bus):
            # 重调度约束
            self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
            self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
            
            # 削减约束
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = 0.
            self.curtail_up.value[bus_id] = tmp_[mask_].sum()
            
            # 储能约束
            if self.bus_storage is not None:
                stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
                stor_down = min(stor_down, obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
                self.storage_down.value[bus_id] = stor_down
                
                stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
                stor_up = min(stor_up, (obs.storage_Emax - obs.storage_charge)[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
                self.storage_up.value[bus_id] = stor_up
        
        self._remove_margin_rounding()
    
    def _update_constraints_param_safe(self, obs: BaseObservation) -> None:
        """更新安全模式约束参数"""
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        
        for bus_id in range(self.nb_max_bus):
            # 重调度约束
            self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
            self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
            
            # 储能约束
            if self.bus_storage is not None:
                stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
                stor_down = min(stor_down, obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
                self.storage_down.value[bus_id] = stor_down
                
                stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
                stor_up = min(stor_up, (obs.storage_Emax - obs.storage_charge)[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
                self.storage_up.value[bus_id] = stor_up
                
                # 储能目标
                self._storage_target_bus.value[bus_id] = (0.5 * obs.storage_Emax)[self.bus_storage.value == bus_id].sum()
            
            # 削减约束（安全模式下尽量取消削减）
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = obs.gen_p_before_curtail[mask_].sum() - tmp_[mask_].sum() if hasattr(obs, 'gen_p_before_curtail') else 0.
            self.curtail_up.value[bus_id] = 0.
            
            # 历史状态
            if self.bus_storage is not None:
                self._past_state_of_charge.value[bus_id] = obs.storage_charge[self.bus_storage.value == bus_id].sum()
            self._past_dispatch.value[bus_id] = obs.target_dispatch[self.bus_gen.value == bus_id].sum()
        
        self._remove_margin_rounding()
    
    def _remove_margin_rounding(self) -> None:
        """移除舍入裕度"""
        self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
        self.storage_up.value[self.storage_up.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_down.value[self.curtail_down.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_up.value[self.curtail_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_up.value[self.redisp_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_down.value[self.redisp_down.value > self.margin_rounding] -= self.margin_rounding
    
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
        """标记参考节点（相角为0）"""
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value.astype(int)] = False
        theta_is_zero[self.bus_ex.value.astype(int)] = False
        theta_is_zero[self.bus_load.value.astype(int)] = False
        theta_is_zero[self.bus_gen.value.astype(int)] = False
        if self.bus_storage is not None:
            theta_is_zero[self.bus_storage.value.astype(int)] = False
        theta_is_zero[0] = True  # slack bus
        return theta_is_zero
    
    def compute_optimum_unsafe(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算危险模式下的最优解"""
        # 决策变量
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)
        
        # 潮流计算
        bus_or_idx = self.bus_or.value.astype(int)
        bus_ex_idx = self.bus_ex.value.astype(int)
        f_or = cp.multiply(1. / self._powerlines_x, (theta[bus_or_idx] - theta[bus_ex_idx]))
        f_or_corr = f_or - self.alpha_por_error * self._prev_por_error
        
        # 注入功率
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching) - self._storage_power_obs
        
        # KCL 约束
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        # 约束条件
        constraints = (
            [theta[theta_is_zero] == 0] +  # slack bus
            [el == 0 for el in KCL_eq] +  # KCL
            [redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
            [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
            [storage <= self.storage_up, storage >= -self.storage_down] +
            [energy_added == 0]
        )
        
        # 目标函数：最小化热极限违反
        cost = (
            self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw) +
            self._penalty_storage_unsafe * cp.sum_squares(storage) +
            self._penalty_redispatching_unsafe * cp.sum_squares(redispatching) +
            cp.sum_squares(cp.pos(cp.abs(f_or_corr) - self.margin_th_limit * self._th_lim_mw))
        )
        
        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.
        else:
            logger.warning("危险模式优化未收敛")
            self.flow_computed[:] = np.nan
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return res
    
    def compute_optimum_safe(self, obs: BaseObservation) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算安全模式下的最优解"""
        # 决策变量
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)
        
        # 潮流计算
        bus_or_idx = self.bus_or.value.astype(int)
        bus_ex_idx = self.bus_ex.value.astype(int)
        f_or = cp.multiply(1. / self._powerlines_x, (theta[bus_or_idx] - theta[bus_ex_idx]))
        f_or_corr = f_or - self.alpha_por_error * self._prev_por_error
        
        # 注入功率
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching) - self._storage_power_obs
        
        # KCL 约束
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        # 状态变量
        dispatch_after_this = self._past_dispatch + redispatching
        if self.bus_storage is not None:
            state_of_charge_after = self._past_state_of_charge + storage / (60. / obs.delta_time)
        else:
            state_of_charge_after = None
        
        # 约束条件
        constraints = (
            [theta[theta_is_zero] == 0] +  # slack bus
            [el == 0 for el in KCL_eq] +  # KCL
            [f_or_corr <= self.margin_th_limit * self._th_lim_mw] +  # 热极限硬约束
            [f_or_corr >= -self.margin_th_limit * self._th_lim_mw] +
            [redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
            [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
            [storage <= self.storage_up, storage >= -self.storage_down] +
            [energy_added == 0]
        )
        
        # 目标函数：恢复参考状态
        cost = (
            self._penalty_curtailment_safe * cp.sum_squares(curtailment_mw) +
            self._penalty_storage_safe * cp.sum_squares(storage) +
            self._penalty_redispatching_safe * cp.sum_squares(redispatching) +
            self._weight_redisp_target * cp.sum_squares(dispatch_after_this) +
            self._weight_curtail_target * cp.sum_squares(curtailment_mw + self.curtail_down)
        )
        
        if state_of_charge_after is not None and self._storage_target_bus is not None:
            cost += self._weight_storage_target * cp.sum_squares(state_of_charge_after - self._storage_target_bus)
        
        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.
        else:
            logger.warning("安全模式优化未收敛")
            self.flow_computed[:] = np.nan
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return res
    
    def _solve_problem(self, prob: cp.Problem, solver_type=None) -> bool:
        """求解优化问题"""
        if solver_type is None:
            for solver_type in self.SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver_type)
                if res:
                    logger.debug(f"Solver {solver_type} 收敛")
                    return True
            return False
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tmp_ = prob.solve(solver=solver_type, warm_start=False)
            
            if np.isfinite(tmp_):
                return True
            else:
                logger.warning(f"Solver {solver_type} 返回无穷值")
                return False
        except cp.error.SolverError as e:
            logger.warning(f"Solver {solver_type} 失败: {e}")
            return False
    
    def _clean_vect(self, curtailment: np.ndarray, storage: np.ndarray, redispatching: np.ndarray) -> None:
        """清理向量（移除过小的值）"""
        curtailment[np.abs(curtailment) < self.margin_sparse] = 0.
        storage[np.abs(storage) < self.margin_sparse] = 0.
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.
    
    def to_grid2op(
        self,
        obs: BaseObservation,
        curtailment: np.ndarray,
        storage: np.ndarray,
        redispatching: np.ndarray,
        act: Optional[BaseAction] = None,
        safe: bool = False
    ) -> BaseAction:
        """将优化结果转换为 grid2op 动作"""
        self._clean_vect(curtailment, storage, redispatching)
        
        if act is None:
            act = self.action_space({})
        
        # 储能
        if hasattr(act, 'n_storage') and act.n_storage > 0 and np.any(np.abs(storage) > 0.):
            storage_ = np.zeros(shape=act.n_storage)
            if self.bus_storage is not None:
                storage_[:] = storage[self.bus_storage.value.astype(int)]
            act.storage_p = storage_
        
        # 削减（简化实现）
        if np.any(np.abs(curtailment) > 0.) and hasattr(obs, 'gen_renewable'):
            curtailment_mw = np.zeros(shape=obs.n_gen) - 1.
            gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
            if np.any(gen_curt) and self.bus_gen is not None:
                idx_gen = self.bus_gen.value[gen_curt].astype(int)
                tmp_ = curtailment[idx_gen]
                modif_gen_optim = tmp_ != 0.
                gen_p = 1.0 * obs.gen_p
                aux_ = curtailment_mw[gen_curt]
                if np.any(modif_gen_optim):
                    gen_per_bus_val = self.gen_per_bus.value[idx_gen]
                    aux_[modif_gen_optim] = (
                        gen_p[gen_curt][modif_gen_optim] -
                        tmp_[modif_gen_optim] * gen_p[gen_curt][modif_gen_optim] /
                        np.maximum(gen_per_bus_val[modif_gen_optim], 1e-6)
                    )
                aux_[~modif_gen_optim] = -1.
                curtailment_mw[gen_curt] = aux_
                curtailment_mw[~gen_curt] = -1.
                act.curtail_mw = curtailment_mw
        
        # 重调度（简化实现）
        if np.any(np.abs(redispatching) > 0.) and hasattr(obs, 'gen_redispatchable'):
            redisp_ = np.zeros(obs.n_gen)
            gen_redi = obs.gen_redispatchable
            if np.any(gen_redi) and self.bus_gen is not None:
                idx_gen = self.bus_gen.value[gen_redi].astype(int)
                tmp_ = redispatching[idx_gen]
                
                # 计算比例分配
                prop_to_gen = np.zeros(obs.n_gen)
                redisp_up_mask = np.zeros(obs.n_gen, dtype=bool)
                redisp_up_mask[gen_redi] = tmp_ > 0.
                prop_to_gen[redisp_up_mask] = obs.gen_margin_up[redisp_up_mask]
                
                redisp_down_mask = np.zeros(obs.n_gen, dtype=bool)
                redisp_down_mask[gen_redi] = tmp_ < 0.
                prop_to_gen[redisp_down_mask] = obs.gen_margin_down[redisp_down_mask]
                
                # 计算可用容量
                redisp_avail = np.zeros(self.nb_max_bus)
                for bus_id in range(self.nb_max_bus):
                    if redispatching[bus_id] > 0.:
                        redisp_avail[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
                    elif redispatching[bus_id] < 0.:
                        redisp_avail[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
                
                # 避免除零
                redisp_avail_idx = redisp_avail[idx_gen]
                redisp_avail_idx[redisp_avail_idx == 0.] = 1.0
                prop_to_gen[gen_redi] = np.maximum(prop_to_gen[gen_redi], 1e-6)
                
                redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail_idx
                redisp_[~gen_redi] = 0.
                act.redispatch = redisp_
        
        return act
    
    def reset(self, observation: BaseObservation) -> None:
        """重置 Solver 状态"""
        self._prev_por_error.value[:] = 0.
        
        # === [修复开始] ===
        # 显式重置储能注入功率，防止跨 Episode 累积
        self._storage_power_obs.value = 0.
        # 重置电压参考值（或者设为 None 以便在 update_parameters 中重新初始化）
        self._v_ref = None
        # 重置历史状态，防止跨 Episode 污染
        self._past_dispatch.value[:] = 0.
        self._past_state_of_charge.value[:] = 0.
        # === [修复结束] ===
        
        # 运行一次 DC 潮流计算初始化误差
        try:
            self._update_topo_param(observation)
            self._update_inj_param(observation)
            # 简化：直接使用观测的潮流
            self.flow_computed[:] = observation.p_or if hasattr(observation, 'p_or') else np.zeros(self.observation_space.n_line)
        except Exception as e:
            logger.warning(f"重置时 DC 潮流计算失败: {e}")

