#!/usr/bin/env python3
# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import warnings
from typing import Optional, List
import numpy as np

from grid2op.Runner import Runner
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv

# --- 兼容本地运行和包运行的导入 ---
try:
    from l2rpn_baselines.PPO_SB3.utils import SB3Agent
except ImportError:
    try:
        from .utils import SB3Agent
    except ImportError:
        from utils import SB3Agent

try:
    from l2rpn_baselines.utils.save_log_gif import save_log_gif
except ImportError:
    try:
        from l2rpn_baselines.utils import save_log_gif
    except ImportError:
        def save_log_gif(logs_path, res):
            warnings.warn("save_log_gif not available. GIF saving skipped.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    _CAN_USE_STABLE_BASELINE = True
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    )


# ============================================================================
# 约定：标准路径结构
# ============================================================================
# 模型保存路径约定: {save_path}/{name}/
#   - {name}.zip                    # 模型文件
#   - vec_normalize.pkl             # 归一化统计（如果使用）
#   - obs_attr_to_keep.json         # 观察属性配置
#   - act_attr_to_keep.json         # 动作属性配置
# ============================================================================


def get_available_obs_attributes(env) -> List[str]:
    """自动检测可用的观察属性"""
    priority_attrs = [
        "day_of_week", "hour_of_day", "minute_of_hour",
        "prod_p", "prod_v", "load_p", "load_q",
        "actual_dispatch", "target_dispatch",
        "topo_vect", "time_before_cooldown_line", "time_before_cooldown_sub",
        "rho", "timestep_overflow", "line_status",
        "storage_power", "storage_charge",
    ]
    
    available_attrs = []
    try:
        obs = env.reset()
        for attr_name in priority_attrs:
            if hasattr(obs, attr_name):
                try:
                    if getattr(obs, attr_name) is not None:
                        available_attrs.append(attr_name)
                except (AttributeError, TypeError, ValueError):
                    pass
    except Exception:
        # 降级到基本属性
        available_attrs = ["prod_p", "load_p", "rho", "line_status"]
    
    if not available_attrs:
        raise ValueError("No valid observation attributes found!")
    return available_attrs


def get_available_act_attributes(env) -> List[str]:
    """自动检测可用的动作属性"""
    priority_attrs = ["redispatch", "curtail", "set_storage"]
    available_attrs = []
    act_space = env.action_space
    
    for attr_name in priority_attrs:
        try:
            if act_space.supports_type(attr_name):
                available_attrs.append(attr_name)
        except Exception:
            pass
    
    if not available_attrs:
        raise ValueError("No valid action attributes found!")
    return available_attrs


def make_grid2op_gym_env(env, obs_attr_to_keep: List[str], act_attr_to_keep: List[str]):
    """创建 Gym 兼容的 Grid2Op 环境"""
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(
        env.observation_space,
        attr_to_keep=obs_attr_to_keep
    )
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(
        env.action_space,
        attr_to_keep=act_attr_to_keep
    )
    return env_gym


def _find_model_dir(load_path: str, name: str) -> str:
    """
    查找模型目录（约定：load_path/name 或 load_path 本身）
    
    Returns:
        模型所在的基础目录路径
    """
    candidates = [
        os.path.join(load_path, name),  # 标准结构
        load_path,  # load_path 本身就是模型目录
    ]
    
    for base_dir in candidates:
        # 检查模型文件是否存在
        model_file = os.path.join(base_dir, f"{name}.zip")
        if os.path.exists(model_file):
            return base_dir
        # 检查 best_model
        best_model = os.path.join(base_dir, "best_model.zip")
        if os.path.exists(best_model):
            return base_dir
    
    # 找不到模型文件
    searched = [os.path.join(c, f"{name}.zip") for c in candidates]
    raise FileNotFoundError(
        f"无法找到模型文件。搜索路径:\n" + "\n".join(f"  - {p}" for p in searched)
    )


def _load_attributes(base_dir: str, env, verbose: bool):
    """加载训练时保存的属性配置"""
    obs_attr_path = os.path.join(base_dir, "obs_attr_to_keep.json")
    act_attr_path = os.path.join(base_dir, "act_attr_to_keep.json")
    
    # 加载观察属性
    if os.path.exists(obs_attr_path):
        with open(obs_attr_path, "r", encoding="utf-8") as f:
            obs_attr_to_keep = json.load(f)
        if verbose:
            print(f"✓ 加载了 {len(obs_attr_to_keep)} 个观察属性")
    else:
        warnings.warn(f"未找到 obs_attr_to_keep.json，将自动检测")
        obs_attr_to_keep = get_available_obs_attributes(env)
    
    # 加载动作属性
    if os.path.exists(act_attr_path):
        with open(act_attr_path, "r", encoding="utf-8") as f:
            act_attr_to_keep = json.load(f)
        if verbose:
            print(f"✓ 加载了 {len(act_attr_to_keep)} 个动作属性")
    else:
        warnings.warn(f"未找到 act_attr_to_keep.json，将自动检测")
        act_attr_to_keep = get_available_act_attributes(env)
    
    return obs_attr_to_keep, act_attr_to_keep


def evaluate(
    env,
    load_path: str = ".",
    name: str = "PPO_SB3",
    logs_path: Optional[str] = None,
    nb_episode: int = 1,
    nb_process: int = 1,
    max_steps: int = -1,
    verbose: bool = False,
    save_gif: bool = False,
    iter_num: Optional[int] = None,
    **kwargs
):
    """
    评估训练好的 PPO 智能体
    
    约定：
    - load_path: 模型保存的根目录（默认: "."）
    - name: 模型名称（默认: "PPO_SB3"）
    - 模型文件路径: {load_path}/{name}/{name}.zip
    """
    if not _CAN_USE_STABLE_BASELINE:
        raise ImportError("stable_baselines3 is required")
    
    # 1. 查找模型目录
    base_dir = _find_model_dir(load_path, name)
    if verbose:
        print(f"✓ 模型目录: {base_dir}")
    
    # 2. 加载属性配置
    obs_attr_to_keep, act_attr_to_keep = _load_attributes(base_dir, env, verbose)
    
    # 3. 创建 Gym 环境
    env_gym = make_grid2op_gym_env(env, obs_attr_to_keep, act_attr_to_keep)
    
    # 4. 检查 VecNormalize
    vecnorm_path = os.path.join(base_dir, "vec_normalize.pkl")
    use_vec_normalize = os.path.exists(vecnorm_path)
    
    eval_vec_env = None
    if use_vec_normalize:
        if verbose:
            print("✓ 检测到 VecNormalize 统计，正在加载...")
        eval_vec_env = DummyVecEnv([lambda: env_gym])
        eval_vec_env = VecNormalize.load(vecnorm_path, eval_vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
    
    # 5. 确定模型文件路径
    if iter_num is not None:
        model_file = os.path.join(base_dir, f"{name}_{iter_num}_steps.zip")
    else:
        model_file = os.path.join(base_dir, f"{name}.zip")
        if not os.path.exists(model_file):
            best_model = os.path.join(base_dir, "best_model.zip")
            if os.path.exists(best_model):
                model_file = best_model
                if verbose:
                    print("✓ 使用 best_model.zip")
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    # 6. 加载 PPO 模型
    if verbose:
        print(f"✓ 加载模型: {model_file}")
    
    custom_objects = {
        "action_space": env_gym.action_space,
        "observation_space": env_gym.observation_space
    }
    
    try:
        if use_vec_normalize:
            model = PPO.load(model_file, env=eval_vec_env, custom_objects=custom_objects)
        else:
            model = PPO.load(model_file, env=env_gym, custom_objects=custom_objects)
    except Exception as e:
        raise RuntimeError(
            f"模型加载失败: {e}\n"
            "可能原因: 观察空间/动作空间与训练时不匹配"
        ) from e
    
    # 7. 创建 SB3Agent（提供 nn_kwargs 以满足基类要求）
    grid2op_agent = SB3Agent(
        g2op_action_space=env.action_space,
        gym_act_space=env_gym.action_space,
        gym_obs_space=env_gym.observation_space,
        nn_type=PPO,
        nn_kwargs={"policy": "MlpPolicy", "env": env_gym},  # 占位符
        gymenv=env_gym,
        iter_num=iter_num,
    )
    # 手动设置已加载的模型
    grid2op_agent.nn_model = model
    
    # 8. 如果使用 VecNormalize，重写 get_act 方法
    if use_vec_normalize:
        def normalized_get_act(gym_obs, reward, done):
            if not isinstance(gym_obs, np.ndarray):
                gym_obs = np.array(gym_obs)
            obs_batch = eval_vec_env.normalize_obs(gym_obs.reshape(1, -1))
            action, _ = grid2op_agent.nn_model.predict(obs_batch[0], deterministic=True)
            return action
        grid2op_agent.get_act = normalized_get_act
    
    # 9. 运行评估
    if nb_episode == 0:
        return grid2op_agent, []
    
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    runner = Runner(**runner_params, agentClass=None, agentInstance=grid2op_agent)
    
    if logs_path:
        os.makedirs(logs_path, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始评估: {nb_episode} 个回合")
        print(f"{'='*60}\n")
    
    res = runner.run(
        path_save=logs_path,
        nb_episode=nb_episode,
        nb_process=nb_process,
        max_iter=max_steps,
        pbar=verbose,
        **kwargs
    )
    
    # 10. 打印评估摘要
    if verbose and res:
        print("\n" + "="*60)
        print("评估摘要")
        print("="*60)
        total_reward = sum(r[2] for r in res)
        total_steps = sum(r[3] for r in res)
        completed = sum(1 for r in res if r[3] >= r[4])
        
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            print(f"场景: {chron_name}")
            print(f"  总奖励: {cum_reward:.6f}")
            print(f"  步数: {nb_time_step}/{max_ts}")
            print(f"  完成: {'是' if nb_time_step >= max_ts else '否'}")
        
        print("\n" + "-"*60)
        print(f"平均奖励: {total_reward / len(res):.6f}")
        print(f"平均步数: {total_steps / len(res):.1f}")
        print(f"完成率: {completed}/{len(res)} ({100*completed/len(res):.1f}%)")
        print("="*60 + "\n")
    
    # 11. 保存 GIF（如果请求）
    if save_gif and logs_path:
        try:
            save_log_gif(logs_path, res)
        except Exception as e:
            warnings.warn(f"保存 GIF 失败: {e}")
    
    # 12. 清理
    env_gym.close()
    if eval_vec_env:
        eval_vec_env.close()
    
    return grid2op_agent, res


if __name__ == "__main__":
    """命令行入口"""
    import grid2op
    from l2rpn_baselines.utils import cli_eval
    
    args_cli = cli_eval().parse_args()
    
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except ImportError:
        warnings.warn("lightsim2grid not available, using default backend")
        backend = None
    
    try:
        data_dir = getattr(args_cli, 'data_dir', None) or getattr(args_cli, 'load_path', None)
        if data_dir:
            env = grid2op.make(data_dir, backend=backend) if backend else grid2op.make(data_dir)
        else:
            env = grid2op.make(backend=backend) if backend else grid2op.make()
    except Exception as e:
        warnings.warn(f"环境创建失败: {e}，使用默认环境")
        env = grid2op.make(backend=backend) if backend else grid2op.make()
    
    try:
        evaluate(
            env,
            load_path=args_cli.load_path,
            logs_path=args_cli.logs_path,
            nb_episode=args_cli.nb_episode,
            nb_process=args_cli.nb_process,
            max_steps=args_cli.max_steps,
            verbose=args_cli.verbose,
            save_gif=args_cli.save_gif,
        )
    finally:
        env.close()
