# -*- coding: utf-8 -*-
"""
L2RPN 环境工厂模块

本模块提供了一个统一的接口，用于创建不同 L2RPN 比赛的 Grid2Op 环境。
它处理后端选择、动作空间配置和环境初始化。

使用方法:
    from env_factory import create_env, EnvManager
    from config import WCCI_2022
    
    # 简单创建
    env = create_env("wcci_2022")
    
    # 使用自定义配置
    env = create_env(WCCI_2022, seed=42)
    
    # 使用 EnvManager 获取高级功能
    manager = EnvManager("wcci_2022")
    env = manager.get_env()
    obs = manager.reset()
"""

import logging
from typing import Union, Optional, Dict, Any, Tuple
import numpy as np

try:
    import grid2op
    from grid2op.Environment import Environment
    from grid2op.Observation import BaseObservation
    from grid2op.Action import BaseAction
except ImportError:
    raise ImportError("需要安装 grid2op。使用: pip install grid2op")

from config import EnvConfig, get_config, ENV_CONFIGS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_backend(use_lightsim: bool = True):
    """
    获取适当的仿真后端
    
    参数:
        use_lightsim: 如果为 True，尝试使用 LightSim2Grid（更快）。
                     如果不可用，则回退到 PandaPower。
    
    返回:
        后端实例
    """
    if use_lightsim:
        try:
            from lightsim2grid import LightSimBackend
            logger.info("使用 LightSim2Grid 后端（快速仿真）")
            return LightSimBackend()
        except ImportError:
            logger.warning(
                "LightSim2Grid 不可用，回退到 PandaPower 后端。"
                "安装方法: pip install lightsim2grid"
            )
    
    from grid2op.Backend import PandaPowerBackend
    logger.info("使用 PandaPower 后端")
    return PandaPowerBackend()


def _get_action_class(action_class_name: Optional[str]):
    """
    根据名称获取动作类
    
    参数:
        action_class_name: 动作类名称
        
    返回:
        动作类或 None
    """
    if action_class_name is None:
        return None
    
    from grid2op.Action import (
        TopologyAction,
        TopologyAndDispatchAction,
        PlayableAction,
        PowerlineSetAction,
        DontAct,
    )
    
    # 动作类映射表
    action_classes = {
        "TopologyAction": TopologyAction,              # 仅拓扑动作
        "TopologyAndDispatchAction": TopologyAndDispatchAction,  # 拓扑+再调度
        "PlayableAction": PlayableAction,              # 所有可用动作
        "PowerlineSetAction": PowerlineSetAction,      # 仅线路动作
        "DontAct": DontAct,                           # 无动作
    }
    
    if action_class_name not in action_classes:
        logger.warning(f"未知动作类: {action_class_name}，使用默认值")
        return None
    
    return action_classes[action_class_name]


def create_env(
    config: Union[str, EnvConfig],
    seed: Optional[int] = None,
    **kwargs
) -> Environment:
    """
    根据配置创建 Grid2Op 环境
    
    参数:
        config: 配置名称（字符串）或 EnvConfig 对象
        seed: 随机种子，用于可重复性
        **kwargs: 传递给 grid2op.make() 的额外参数
        
    返回:
        初始化的 Grid2Op 环境
        
    示例:
        >>> env = create_env("wcci_2022", seed=42)
        >>> obs = env.reset()
        >>> print(f"电网有 {env.n_line} 条线路和 {env.n_sub} 个变电站")
    """
    # 获取配置
    if isinstance(config, str):
        config = get_config(config)
    
    logger.info(f"创建环境: {config.name}")
    logger.info(f"Grid2Op 环境: {config.env_name}")
    
    # 准备参数
    make_params = config.to_dict()
    
    # 添加后端
    backend = _get_backend(config.use_lightsim)
    make_params["backend"] = backend
    
    # 如果指定了动作类，则添加
    action_class = _get_action_class(config.action_class)
    if action_class is not None:
        make_params["action_class"] = action_class
    
    # 合并额外的 kwargs
    make_params.update(kwargs)
    
    # 创建环境
    try:
        env = grid2op.make(**make_params)
        logger.info(f"环境创建成功")
        logger.info(f"  - 线路数: {env.n_line}")
        logger.info(f"  - 变电站数: {env.n_sub}")
        logger.info(f"  - 发电机数: {env.n_gen}")
        logger.info(f"  - 负荷数: {env.n_load}")
        if hasattr(env, 'n_storage') and env.n_storage > 0:
            logger.info(f"  - 储能单元数: {env.n_storage}")
    except Exception as e:
        logger.error(f"创建环境失败: {e}")
        raise
    
    # 如果提供了种子，则设置
    if seed is not None:
        env.seed(seed)
        logger.info(f"环境随机种子设置为: {seed}")
    
    return env


class EnvManager:
    """
    高级环境管理器
    
    提供便捷的环境交互方法、观测处理和动作空间探索功能。
    
    属性:
        config: 环境配置
        env: Grid2Op 环境实例
        current_obs: 当前观测
    """
    
    def __init__(
        self,
        config: Union[str, EnvConfig],
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        初始化环境管理器
        
        参数:
            config: 配置名称或 EnvConfig 对象
            seed: 随机种子
            **kwargs: 环境创建的额外参数
        """
        if isinstance(config, str):
            self.config = get_config(config)
        else:
            self.config = config
        
        self.env = create_env(self.config, seed=seed, **kwargs)
        self.current_obs: Optional[BaseObservation] = None
        self._seed = seed
    
    def reset(self, **kwargs) -> BaseObservation:
        """
        重置环境
        
        参数:
            **kwargs: 传递给 env.reset() 的参数
            
        返回:
            初始观测
        """
        self.current_obs = self.env.reset(**kwargs)
        return self.current_obs
    
    def step(self, action: BaseAction) -> Tuple[BaseObservation, float, bool, Dict]:
        """
        在环境中执行一步
        
        参数:
            action: 要执行的动作
            
        返回:
            元组 (观测, 奖励, 是否结束, 信息)
        """
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        return obs, reward, done, info
    
    def get_env(self) -> Environment:
        """获取底层的 Grid2Op 环境"""
        return self.env
    
    def get_do_nothing_action(self) -> BaseAction:
        """获取一个"什么都不做"的动作"""
        return self.env.action_space({})
    
    def simulate(
        self,
        action: BaseAction,
        time_step: int = 1
    ) -> Tuple[BaseObservation, float, bool, Dict]:
        """
        模拟一个动作而不实际执行它
        
        参数:
            action: 要模拟的动作
            time_step: 向前模拟的步数
            
        返回:
            元组 (模拟观测, 奖励, 是否结束, 信息)
        """
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        return self.current_obs.simulate(action, time_step=time_step)
    
    def get_grid_info(self) -> Dict[str, Any]:
        """
        获取电网信息
        
        返回:
            包含电网信息的字典
        """
        env = self.env
        info = {
            "n_line": env.n_line,           # 线路数
            "n_sub": env.n_sub,             # 变电站数
            "n_gen": env.n_gen,             # 发电机数
            "n_load": env.n_load,           # 负荷数
            "n_storage": getattr(env, 'n_storage', 0),  # 储能数
            "thermal_limits": env.get_thermal_limit().tolist(),  # 热限制
            "line_names": env.name_line.tolist(),        # 线路名称
            "sub_names": env.name_sub.tolist() if hasattr(env, 'name_sub') else None,  # 变电站名称
            "gen_names": env.name_gen.tolist(),          # 发电机名称
            "load_names": env.name_load.tolist(),        # 负荷名称
        }
        
        # 添加可再生能源发电机信息（如果可用）
        if hasattr(env, 'gen_renewable'):
            info["gen_renewable"] = env.gen_renewable.tolist()
        
        # 添加可再调度发电机信息
        if hasattr(env, 'gen_redispatchable'):
            info["gen_redispatchable"] = env.gen_redispatchable.tolist()
        
        return info
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        获取动作空间信息
        
        返回:
            包含动作空间信息的字典
        """
        action_space = self.env.action_space
        info = {
            "action_size": action_space.size(),                          # 动作空间大小
            "can_change_bus": action_space.supports_type("change_bus"),  # 是否支持改变母线
            "can_set_bus": action_space.supports_type("set_bus"),        # 是否支持设置母线
            "can_change_line": action_space.supports_type("change_line_status"),  # 是否支持改变线路状态
            "can_set_line": action_space.supports_type("set_line_status"),        # 是否支持设置线路状态
            "can_redispatch": action_space.supports_type("redispatch"),  # 是否支持再调度
        }
        
        # 添加弃风和储能信息
        if hasattr(action_space, 'supports_type'):
            info["can_curtail"] = action_space.supports_type("curtail")      # 是否支持弃风
            info["can_storage"] = action_space.supports_type("set_storage")  # 是否支持储能控制
        
        return info
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        获取当前观测信息
        
        返回:
            包含观测信息的字典
        """
        if self.current_obs is None:
            raise ValueError("没有当前观测。请先调用 reset()")
        
        obs = self.current_obs
        info = {
            "timestep": obs.current_step,                              # 当前时间步
            "max_rho": float(obs.rho.max()),                          # 最大线路负载率
            "total_load": float(obs.load_p.sum()),                    # 总负荷功率
            "total_gen": float(obs.gen_p.sum()),                      # 总发电功率
            "lines_in_overflow": int((obs.rho > 1.0).sum()),          # 过载线路数
            "lines_disconnected": int((obs.line_status == False).sum()),  # 断开线路数
        }
        
        # 添加储能信息（如果可用）
        if hasattr(obs, 'storage_charge') and len(obs.storage_charge) > 0:
            info["storage_charge"] = obs.storage_charge.tolist()   # 储能电量
            info["storage_power"] = obs.storage_power.tolist()     # 储能功率
        
        return info
    
    def print_status(self) -> None:
        """打印当前环境状态"""
        if self.current_obs is None:
            print("环境尚未重置。请先调用 reset()")
            return
        
        info = self.get_observation_info()
        print(f"\n{'='*50}")
        print(f"环境状态 - 第 {info['timestep']} 步")
        print(f"{'='*50}")
        print(f"最大线路负载率 (rho): {info['max_rho']:.2%}")
        print(f"总负荷: {info['total_load']:.2f} MW")
        print(f"总发电: {info['total_gen']:.2f} MW")
        print(f"过载线路数: {info['lines_in_overflow']}")
        print(f"断开线路数: {info['lines_disconnected']}")
        
        if 'storage_charge' in info:
            print(f"储能电量: {info['storage_charge']}")
        print(f"{'='*50}\n")
    
    def close(self) -> None:
        """关闭环境"""
        if self.env is not None:
            self.env.close()
            logger.info("环境已关闭")


def run_episode(
    config: Union[str, EnvConfig],
    agent=None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    运行单个回合（可选择使用智能体）
    
    参数:
        config: 配置名称或 EnvConfig 对象
        agent: 具有 act(obs, reward, done) 方法的智能体。如果为 None，使用"什么都不做"动作。
        max_steps: 每回合最大步数
        seed: 随机种子
        verbose: 是否打印进度信息
        
    返回:
        包含回合统计信息的字典
    """
    manager = EnvManager(config, seed=seed)
    
    if isinstance(config, str):
        config = get_config(config)
    
    if max_steps is None:
        max_steps = config.max_episode_steps or 8064  # 默认：4周
    
    obs = manager.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    if verbose:
        print(f"开始回合，配置: {config.name}")
        print(f"最大步数: {max_steps}")
    
    while not done and step_count < max_steps:
        # 获取动作
        if agent is not None:
            action = agent.act(obs, total_reward, done)
        else:
            action = manager.get_do_nothing_action()
        
        # 执行一步
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        step_count += 1
        
        # 打印进度
        if verbose and step_count % 100 == 0:
            print(f"第 {step_count} 步: 奖励={reward:.2f}, 最大rho={obs.rho.max():.2%}")
    
    # 回合结束
    result = {
        "steps": step_count,                          # 总步数
        "total_reward": total_reward,                 # 总奖励
        "done": done,                                 # 是否结束
        "max_steps_reached": step_count >= max_steps, # 是否达到最大步数
    }
    
    if verbose:
        print(f"\n回合结束:")
        print(f"  步数: {step_count}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  是否结束: {done}")
    
    manager.close()
    return result


if __name__ == "__main__":
    # 演示：为每个比赛创建环境
    print("="*60)
    print("L2RPN 环境工厂演示")
    print("="*60)
    
    # 列出可用配置
    print("\n可用配置:")
    for name in ENV_CONFIGS.keys():
        print(f"  - {name}")
    
    # 使用沙盒环境进行演示（更小、更快）
    print("\n" + "="*60)
    print("创建沙盒环境进行演示...")
    print("="*60)
    
    try:
        manager = EnvManager("sandbox_case14", seed=42)
        obs = manager.reset()
        
        print("\n电网信息:")
        grid_info = manager.get_grid_info()
        for key, value in grid_info.items():
            if not isinstance(value, list) or len(value) < 10:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: [{len(value)} 个元素]")
        
        print("\n动作空间信息:")
        action_info = manager.get_action_space_info()
        for key, value in action_info.items():
            print(f"  {key}: {value}")
        
        manager.print_status()
        
        # 运行几步
        print("使用'什么都不做'智能体运行 10 步...")
        for i in range(10):
            action = manager.get_do_nothing_action()
            obs, reward, done, info = manager.step(action)
            if done:
                print(f"回合在第 {i+1} 步结束")
                break
        
        manager.print_status()
        manager.close()
        
    except Exception as e:
        print(f"演示失败: {e}")
        print("如果 grid2op 环境未安装，这是预期的。")
        print("安装方法: pip install grid2op[optional]")
