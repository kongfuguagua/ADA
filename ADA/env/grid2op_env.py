# -*- coding: utf-8 -*-
"""
Grid2Op 环境封装
提供统一的环境交互接口

约定：
- Grid2OpEnvironment: 环境管理器，提供高级接口
- Grid2OpSimulator: 仿真器，用于 Judger 评估
- create_grid2op_env: 便捷函数，快速创建环境
"""

import logging
from typing import Union, Optional, Dict, Any, Tuple
import numpy as np

from .config import EnvConfig, get_env_config, list_env_configs, Competition

logger = logging.getLogger(__name__)


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
        """
        创建 Grid2Op 环境
        
        Args:
            seed: 随机种子
            **kwargs: 传递给 grid2op.make() 的额外参数
            
        Returns:
            Grid2Op 环境实例
        """
        try:
            import grid2op
        except ImportError:
            raise ImportError(
                "需要安装 grid2op。使用: pip install grid2op"
            )
        
        # 自动尝试使用 LightSim2Grid（更快），失败则使用默认后端
        make_params = self.config.to_dict()
        if "backend" not in make_params:
            try:
                from lightsim2grid import LightSimBackend
                make_params["backend"] = LightSimBackend()
                logger.info("使用 LightSim2Grid 后端（快速仿真）")
            except ImportError:
                logger.info("使用默认后端（PandaPower）")
        
        make_params.update(kwargs)
        
        try:
            env = grid2op.make(**make_params)
        except Exception as e:
            logger.error(f"创建环境失败: {e}")
            raise
        
        if seed is not None:
            env.seed(seed)
            logger.info(f"环境随机种子设置为: {seed}")
        
        logger.info(f"环境创建成功: {self.config.name}")
        logger.info(f"  - 线路数: {env.n_line}")
        logger.info(f"  - 变电站数: {env.n_sub}")
        logger.info(f"  - 发电机数: {env.n_gen}")
        logger.info(f"  - 负荷数: {env.n_load}")
        if hasattr(env, 'n_storage') and env.n_storage > 0:
            logger.info(f"  - 储能单元数: {env.n_storage}")
        
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
    
    
    def get_observation_info(self) -> Dict[str, Any]:
        """获取当前观测信息（简化版，用于快速检查）"""
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        
        obs = self.current_obs
        max_rho = float(obs.rho.max())
        
        return {
            "timestep": obs.current_step,
            "max_rho": max_rho,
            "total_load": float(obs.load_p.sum()),
            "total_gen": float(obs.gen_p.sum()),
            "overflow_count": int((obs.rho > 1.0).sum()),
            "lines_disconnected": int((obs.line_status == False).sum()),
            # 状态判断（参考 OptimCVXPY 的启发式）
            "is_safe": max_rho < 0.85,
            "is_danger": max_rho > 0.95,
        }
    
    def get_state_for_planner(self) -> Dict[str, Any]:
        """
        获取用于 Planner 的状态信息
        将 Grid2Op 观测转换为 ADA 可用的格式
        """
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        
        obs = self.current_obs
        
        # 核心指标（用于启发式判断）
        max_rho = float(obs.rho.max())
        overflow_count = int((obs.rho > 1.0).sum())
        near_overflow_count = int((obs.rho > 0.9).sum())
        
        return {
            # 核心指标（用于启发式判断）
            "max_rho": max_rho,
            "overflow_count": overflow_count,
            "near_overflow_count": near_overflow_count,
            "is_safe": max_rho < 0.85,
            "is_danger": max_rho > 0.95,
            "is_critical": overflow_count > 0,
            
            # 时间信息
            "timestep": obs.current_step,
            "hour_of_day": obs.hour_of_day,
            
            # 负荷信息
            "total_load_mw": float(obs.load_p.sum()),
            
            # 发电信息
            "total_gen_mw": float(obs.gen_p.sum()),
            
            # 线路信息（简化）
            "lines_disconnected": int((obs.line_status == False).sum()),
            
            # 完整数据（仅在需要时使用）
            "full_data": {
                "load_p": obs.load_p.tolist(),
                "gen_p": obs.gen_p.tolist(),
                "rho": obs.rho.tolist(),
                "line_status": obs.line_status.tolist(),
            }
        }
    
    def close(self) -> None:
        """关闭环境"""
        if self.env is not None:
            self.env.close()
            logger.info("环境已关闭")


class Grid2OpSimulator:
    """
    Grid2Op 仿真器
    用于 Judger 评估解的物理可行性
    简化版本，专注于核心功能
    """
    
    def __init__(self, env: Grid2OpEnvironment):
        """初始化仿真器"""
        self.env = env
    
    def simulate_action(self, action) -> Dict[str, Any]:
        """
        模拟动作效果（简化接口）
        
        Args:
            action: Grid2Op Action 对象或动作字典
        
        Returns:
            仿真结果
        """
        if self.env.current_obs is None:
            return {"success": False, "error": "环境未初始化"}
        
        try:
            # 如果是字典，转换为 Action
            if isinstance(action, dict):
                action = self._build_action(action)
            
            # 模拟动作
            sim_obs, sim_reward, sim_done, sim_info = self.env.simulate(action)
            
            max_rho = float(sim_obs.rho.max())
            
            return {
                "success": True,
                "is_safe": max_rho <= 1.0 and not sim_done,
                "reward": float(sim_reward),
                "max_rho": max_rho,
                "overflow_count": int((sim_obs.rho > 1.0).sum()),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _build_action(self, action_dict: Dict[str, Any]):
        """将动作字典转换为 Grid2Op 动作（简化版）"""
        action_space = self.env.env.action_space
        action = action_space({})
        
        # 再调度
        if "redispatch" in action_dict:
            redispatch = action_dict["redispatch"]
            if isinstance(redispatch, dict):
                action.redispatch = [(int(k), float(v)) for k, v in redispatch.items()]
        
        # 线路状态
        if "set_line_status" in action_dict:
            action.set_line_status = [
                (int(k), int(v)) for k, v in action_dict["set_line_status"].items()
            ]
        
        return action


def create_grid2op_env(
    config: Union[str, EnvConfig],
    seed: Optional[int] = None,
    **kwargs
) -> Grid2OpEnvironment:
    """
    便捷函数：创建 Grid2Op 环境管理器
    
    这是创建环境的推荐方式，返回 Grid2OpEnvironment 实例。
    
    Args:
        config: 配置名称（字符串）或 EnvConfig 对象
        seed: 随机种子，用于可重复性
        **kwargs: 传递给环境创建的额外参数
        
    Returns:
        Grid2OpEnvironment 实例
        
    Examples:
        >>> # 使用配置名称
        >>> env = create_grid2op_env("wcci_2022", seed=42)
        >>> obs = env.reset()
        
        >>> # 使用 EnvConfig 对象
        >>> from env.config import WCCI_2022
        >>> env = create_grid2op_env(WCCI_2022, seed=42)
        
        >>> # 使用环境名称（直接使用 Grid2Op 环境名）
        >>> env = create_grid2op_env("l2rpn_wcci_2022", seed=42)
    """
    # 如果传入的是字符串，尝试作为配置名称查找
    if isinstance(config, str):
        # 先尝试作为配置名称
        try:
            config = get_env_config(config)
        except KeyError:
            # 如果不是已知配置，创建一个临时配置
            logger.warning(
                f"'{config}' 不是已知配置，将作为 Grid2Op 环境名使用。"
                f"可用配置: {list_env_configs()}"
            )
            config = EnvConfig(
                name=f"Custom: {config}",
                env_name=config,
                competition=Competition.SANDBOX,
                description=f"自定义环境: {config}"
            )
    
    return Grid2OpEnvironment(config, seed=seed, **kwargs)


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 Grid2Op 环境封装")
    print("=" * 50)
    
    try:
        # 测试便捷函数
        print("\n1. 测试 create_grid2op_env 函数:")
        env = create_grid2op_env("sandbox_case14", seed=42)
        obs = env.reset()
        print("   ✓ 环境创建成功")
        
        print("\n2. 当前状态:")
        obs_info = env.get_observation_info()
        print(f"  最大负载率: {obs_info['max_rho']:.2%}")
        print(f"  总负荷: {obs_info['total_load']:.2f} MW")
        print(f"  总发电: {obs_info['total_gen']:.2f} MW")
        
        print("\n4. Planner 状态:")
        planner_state = env.get_state_for_planner()
        print(f"  时间步: {planner_state['timestep']}")
        print(f"  过载线路: {planner_state['lines_in_overflow']}")
        
        print("\n5. 测试仿真器:")
        simulator = Grid2OpSimulator(env)
        result = simulator.simulate_action(env.get_do_nothing_action())
        print(f"  仿真成功: {result['success']}")
        print(f"  安全: {result.get('is_safe', 'N/A')}")
        
        env.close()
        print("\n✓ 测试完成!")
        
    except ImportError as e:
        print(f"✗ Grid2Op 未安装: {e}")
        print("请运行: pip install grid2op lightsim2grid")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

