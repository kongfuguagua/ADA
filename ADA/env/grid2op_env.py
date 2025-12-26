# -*- coding: utf-8 -*-
"""
Grid2Op 环境封装
提供统一的环境交互接口
"""

import logging
from typing import Union, Optional, Dict, Any, Tuple, List
import numpy as np

from .config import EnvConfig, get_env_config

logger = logging.getLogger(__name__)


def _get_backend(use_lightsim: bool = True):
    """获取仿真后端"""
    if use_lightsim:
        try:
            from lightsim2grid import LightSimBackend
            logger.info("使用 LightSim2Grid 后端")
            return LightSimBackend()
        except ImportError:
            logger.warning("LightSim2Grid 不可用，回退到 PandaPower")
    
    from grid2op.Backend import PandaPowerBackend
    return PandaPowerBackend()


def _get_action_class(action_class_name: Optional[str]):
    """根据名称获取动作类"""
    if action_class_name is None:
        return None
    
    from grid2op.Action import (
        TopologyAction,
        TopologyAndDispatchAction,
        PlayableAction,
        PowerlineSetAction,
        DontAct,
    )
    
    action_classes = {
        "TopologyAction": TopologyAction,
        "TopologyAndDispatchAction": TopologyAndDispatchAction,
        "PlayableAction": PlayableAction,
        "PowerlineSetAction": PowerlineSetAction,
        "DontAct": DontAct,
    }
    
    return action_classes.get(action_class_name)


class Grid2OpEnvironment:
    """
    Grid2Op 环境管理器
    
    提供便捷的环境交互方法、观测处理和动作空间探索功能。
    """
    
    def __init__(
        self,
        config: Union[str, EnvConfig],
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        初始化环境管理器
        
        Args:
            config: 配置名称或 EnvConfig 对象
            seed: 随机种子
            **kwargs: 环境创建的额外参数
        """
        if isinstance(config, str):
            self.config = get_env_config(config)
        else:
            self.config = config
        
        self.env = self._create_env(seed, **kwargs)
        self.current_obs = None
        self._seed = seed
    
    def _create_env(self, seed: Optional[int] = None, **kwargs):
        """创建 Grid2Op 环境"""
        import grid2op
        
        make_params = self.config.to_dict()
        make_params["backend"] = _get_backend(self.config.use_lightsim)
        
        action_class = _get_action_class(self.config.action_class)
        if action_class is not None:
            make_params["action_class"] = action_class
        
        make_params.update(kwargs)
        
        env = grid2op.make(**make_params)
        
        if seed is not None:
            env.seed(seed)
        
        logger.info(f"环境创建成功: {self.config.name}")
        logger.info(f"  线路数: {env.n_line}, 变电站数: {env.n_sub}")
        
        return env
    
    def reset(self, **kwargs):
        """重置环境"""
        self.current_obs = self.env.reset(**kwargs)
        return self.current_obs
    
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """执行一步"""
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        return obs, reward, done, info
    
    def get_do_nothing_action(self):
        """获取空动作"""
        return self.env.action_space({})
    
    def simulate(self, action, time_step: int = 1) -> Tuple[Any, float, bool, Dict]:
        """模拟动作（不实际执行）"""
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        return self.current_obs.simulate(action, time_step=time_step)
    
    def get_grid_info(self) -> Dict[str, Any]:
        """获取电网信息"""
        env = self.env
        return {
            "n_line": env.n_line,
            "n_sub": env.n_sub,
            "n_gen": env.n_gen,
            "n_load": env.n_load,
            "n_storage": getattr(env, 'n_storage', 0),
            "thermal_limits": env.get_thermal_limit().tolist(),
            "line_names": env.name_line.tolist(),
            "gen_names": env.name_gen.tolist(),
            "load_names": env.name_load.tolist(),
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """获取动作空间信息"""
        action_space = self.env.action_space
        return {
            "action_size": action_space.size(),
            "can_change_bus": action_space.supports_type("change_bus"),
            "can_set_bus": action_space.supports_type("set_bus"),
            "can_change_line": action_space.supports_type("change_line_status"),
            "can_set_line": action_space.supports_type("set_line_status"),
            "can_redispatch": action_space.supports_type("redispatch"),
            "can_curtail": action_space.supports_type("curtail") if hasattr(action_space, 'supports_type') else False,
            "can_storage": action_space.supports_type("set_storage") if hasattr(action_space, 'supports_type') else False,
        }
    
    def get_observation_info(self) -> Dict[str, Any]:
        """获取当前观测信息"""
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        
        obs = self.current_obs
        info = {
            "timestep": obs.current_step,
            "max_rho": float(obs.rho.max()),
            "total_load": float(obs.load_p.sum()),
            "total_gen": float(obs.gen_p.sum()),
            "lines_in_overflow": int((obs.rho > 1.0).sum()),
            "lines_disconnected": int((obs.line_status == False).sum()),
            "gen_p": obs.gen_p.tolist(),
            "load_p": obs.load_p.tolist(),
            "rho": obs.rho.tolist(),
            "line_status": obs.line_status.tolist(),
        }
        
        if hasattr(obs, 'storage_charge') and len(obs.storage_charge) > 0:
            info["storage_charge"] = obs.storage_charge.tolist()
            info["storage_power"] = obs.storage_power.tolist()
        
        return info
    
    def get_state_for_planner(self) -> Dict[str, Any]:
        """
        获取用于 Planner 的状态信息
        将 Grid2Op 观测转换为 ADA 可用的格式
        """
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        
        obs = self.current_obs
        
        return {
            # 时间信息
            "timestep": obs.current_step,
            "hour_of_day": obs.hour_of_day,
            "day_of_week": obs.day_of_week,
            
            # 负荷信息
            "total_load_mw": float(obs.load_p.sum()),
            "load_p": obs.load_p.tolist(),
            "load_q": obs.load_q.tolist(),
            
            # 发电信息
            "total_gen_mw": float(obs.gen_p.sum()),
            "gen_p": obs.gen_p.tolist(),
            "gen_p_max": obs.gen_pmax.tolist() if hasattr(obs, 'gen_pmax') else None,
            "gen_p_min": obs.gen_pmin.tolist() if hasattr(obs, 'gen_pmin') else None,
            
            # 线路信息
            "max_rho": float(obs.rho.max()),
            "rho": obs.rho.tolist(),
            "line_status": obs.line_status.tolist(),
            "thermal_limits": self.env.get_thermal_limit().tolist(),
            
            # 告警信息
            "lines_in_overflow": int((obs.rho > 1.0).sum()),
            "lines_near_overflow": int((obs.rho > 0.9).sum()),
            "lines_disconnected": int((obs.line_status == False).sum()),
            
            # 储能信息（如果有）
            "storage_charge": obs.storage_charge.tolist() if hasattr(obs, 'storage_charge') and len(obs.storage_charge) > 0 else [],
            "storage_power": obs.storage_power.tolist() if hasattr(obs, 'storage_power') and len(obs.storage_power) > 0 else [],
        }
    
    def close(self) -> None:
        """关闭环境"""
        if self.env is not None:
            self.env.close()


class Grid2OpSimulator:
    """
    Grid2Op 仿真器
    用于 Judger 评估解的物理可行性
    """
    
    def __init__(self, env: Grid2OpEnvironment):
        """
        初始化仿真器
        
        Args:
            env: Grid2Op 环境实例
        """
        self.env = env
    
    def run(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行仿真
        
        Args:
            action_dict: 动作字典，包含：
                - redispatch: 再调度量 {gen_name: delta_p}
                - set_bus: 母线设置 {element: bus_id}
                - set_line_status: 线路状态 {line_id: status}
                - curtailment: 弃风量 {gen_id: ratio}
                - storage: 储能功率 {storage_id: power}
        
        Returns:
            仿真结果，包含物理指标
        """
        if self.env.current_obs is None:
            return {"success": False, "error": "环境未初始化"}
        
        try:
            # 构建 Grid2Op 动作
            action = self._build_action(action_dict)
            
            # 模拟动作
            sim_obs, sim_reward, sim_done, sim_info = self.env.simulate(action)
            
            # 提取物理指标
            return {
                "success": True,
                "is_safe": not sim_done,
                "reward": float(sim_reward),
                "max_rho": float(sim_obs.rho.max()),
                "lines_in_overflow": int((sim_obs.rho > 1.0).sum()),
                "total_loss": float(sim_obs.gen_p.sum() - sim_obs.load_p.sum()),
                "violation_details": self._check_violations(sim_obs),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _build_action(self, action_dict: Dict[str, Any]):
        """将动作字典转换为 Grid2Op 动作"""
        action_space = self.env.env.action_space
        action = action_space({})
        
        # 再调度
        if "redispatch" in action_dict and action_dict["redispatch"]:
            redispatch = action_dict["redispatch"]
            if isinstance(redispatch, dict):
                for gen_name, delta_p in redispatch.items():
                    gen_idx = list(self.env.env.name_gen).index(gen_name)
                    action.redispatch = [(gen_idx, delta_p)]
        
        # 线路状态
        if "set_line_status" in action_dict and action_dict["set_line_status"]:
            for line_id, status in action_dict["set_line_status"].items():
                action.set_line_status = [(int(line_id), int(status))]
        
        return action
    
    def _check_violations(self, obs) -> Dict[str, Any]:
        """检查物理约束违规"""
        violations = {}
        
        # 线路过载
        overflow_lines = np.where(obs.rho > 1.0)[0]
        if len(overflow_lines) > 0:
            violations["overflow_lines"] = {
                int(i): float(obs.rho[i]) for i in overflow_lines
            }
        
        # 电压越限（如果有）
        if hasattr(obs, 'v_or') and obs.v_or is not None:
            low_voltage = np.where(obs.v_or < 0.95)[0]
            high_voltage = np.where(obs.v_or > 1.05)[0]
            if len(low_voltage) > 0:
                violations["low_voltage"] = low_voltage.tolist()
            if len(high_voltage) > 0:
                violations["high_voltage"] = high_voltage.tolist()
        
        return violations
    
    def reset(self) -> None:
        """重置仿真器"""
        self.env.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.env.get_observation_info()


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 Grid2Op 环境封装")
    print("=" * 50)
    
    try:
        # 创建沙盒环境
        env = Grid2OpEnvironment("sandbox_case14", seed=42)
        obs = env.reset()
        
        print("\n电网信息:")
        grid_info = env.get_grid_info()
        print(f"  线路数: {grid_info['n_line']}")
        print(f"  变电站数: {grid_info['n_sub']}")
        print(f"  发电机数: {grid_info['n_gen']}")
        
        print("\n当前状态:")
        obs_info = env.get_observation_info()
        print(f"  最大负载率: {obs_info['max_rho']:.2%}")
        print(f"  总负荷: {obs_info['total_load']:.2f} MW")
        print(f"  总发电: {obs_info['total_gen']:.2f} MW")
        
        print("\nPlanner 状态:")
        planner_state = env.get_state_for_planner()
        print(f"  时间步: {planner_state['timestep']}")
        print(f"  过载线路: {planner_state['lines_in_overflow']}")
        
        # 测试仿真器
        print("\n测试仿真器:")
        simulator = Grid2OpSimulator(env)
        result = simulator.run({})
        print(f"  仿真成功: {result['success']}")
        print(f"  安全: {result.get('is_safe', 'N/A')}")
        
        env.close()
        print("\n测试完成!")
        
    except ImportError as e:
        print(f"Grid2Op 未安装: {e}")
        print("请运行: pip install grid2op lightsim2grid")

