#!/usr/bin/env python3
# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import warnings
import os
import json
from typing import List, Optional
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward

import numpy as np

from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv

from l2rpn_baselines.PPO_SB3.utils import SB3Agent, save_used_attribute, remove_non_usable_attr

try:
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
    _CAN_USE_STABLE_BASELINE = True
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    )

try:
    import swanlab
    from swanlab.integration.sb3 import SwanLabCallback
    _CAN_USE_SWANLAB = True
except ImportError:
    _CAN_USE_SWANLAB = False
    SwanLabCallback = None


# Priority list of observation attributes (in order of importance)
PRIORITY_OBS_ATTRS = [
    "day_of_week", "hour_of_day", "minute_of_hour",
    "prod_p", "prod_v", "load_p", "load_q",
    "actual_dispatch", "target_dispatch",
    "topo_vect", "time_before_cooldown_line", "time_before_cooldown_sub",
    "rho", "timestep_overflow", "line_status",
    "storage_power", "storage_charge",
]

# Priority list of action attributes
PRIORITY_ACT_ATTRS = [
    "redispatch",
    "curtail",
    "set_storage",
]


# --- 新增：自定义回调函数，用于打印场景信息 ---
class Grid2OpLogCallback(BaseCallback):
    """
    自定义回调函数：
    1. 在控制台打印当前完成的 Episode 信息（场景名、奖励）。
    2. 将场景名记录到日志（如果支持文本日志）。
    """
    def __init__(self, verbose=0):
        super(Grid2OpLogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 检查是否有环境完成了 Episode (dones=True)
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                # 尝试从 info 中获取 episode 信息 (SB3 Monitor wrapper)
                ep_info = info.get("episode")
                
                # 尝试获取 Grid2Op 原始 info (通常 GymEnv 会保留部分信息，但不一定包含 chronics_name)
                # 注意：在 VecEnv 中获取具体的 chronic name 比较困难，
                # 但我们可以打印 step 结束时的 reward 等信息。
                
                if ep_info:
                    ep_rew = ep_info.get("r", 0)
                    ep_len = ep_info.get("l", 0)
                    if self.verbose > 0:
                        print(f"[Episode Finished] Env {idx}: Reward={ep_rew:.2f}, Length={ep_len}")
                    
                    # 如果使用了 SwanLab，可以手动记录更多指标
                    if _CAN_USE_SWANLAB and swanlab.get_run():
                        swanlab.log({"grid2op/episode_reward": ep_rew})

        return True


def get_available_obs_attributes(env) -> List[str]:
    """Dynamically detect available observation attributes from the environment."""
    available_attrs = []
    try:
        obs = env.reset()
        for attr_name in PRIORITY_OBS_ATTRS:
            if hasattr(obs, attr_name):
                try:
                    attr_value = getattr(obs, attr_name)
                    if attr_value is not None:
                        available_attrs.append(attr_name)
                except (AttributeError, TypeError, ValueError) as e:
                    warnings.warn(
                        f"Attribute '{attr_name}' exists but cannot be accessed: {e}. Skipping.",
                        UserWarning
                    )
    except Exception as e:
        warnings.warn(
            f"Could not inspect observation space dynamically: {e}. Using basic attributes.",
            UserWarning
        )
        basic_attrs = ["prod_p", "load_p", "rho", "line_status"]
        try:
            obs = env.reset()
            for attr in basic_attrs:
                if hasattr(obs, attr):
                    available_attrs.append(attr)
        except:
            available_attrs = PRIORITY_OBS_ATTRS[:4]
    
    if not available_attrs:
        raise ValueError("No valid observation attributes found! The environment may be incompatible.")
    
    return available_attrs


def get_available_act_attributes(env) -> List[str]:
    """Dynamically detect available action attributes from the environment."""
    available_attrs = []
    act_space = env.action_space
    
    for attr_name in PRIORITY_ACT_ATTRS:
        try:
            if act_space.supports_type(attr_name):
                available_attrs.append(attr_name)
        except Exception as e:
            warnings.warn(
                f"Could not check support for action attribute '{attr_name}': {e}",
                UserWarning
            )
    
    if not available_attrs:
        raise ValueError("No valid action attributes found! The environment action space may be incompatible.")
    
    return available_attrs


def make_grid2op_gym_env(
    env,
    obs_attr_to_keep: Optional[List[str]] = None,
    act_attr_to_keep: Optional[List[str]] = None,
):
    """Create a Gymnasium-compatible Grid2Op environment with dynamic attribute detection."""
    # Auto-detect attributes if not provided
    if obs_attr_to_keep is None:
        obs_attr_to_keep = get_available_obs_attributes(env)
    
    if act_attr_to_keep is None:
        act_attr_to_keep = get_available_act_attributes(env)
    
    # Create GymEnv wrapper
    env_gym = GymEnv(env)
    
    # Replace observation space with BoxGymObsSpace
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(
        env.observation_space,
        attr_to_keep=obs_attr_to_keep
    )
    
    # Replace action space with BoxGymActSpace
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(
        env.action_space,
        attr_to_keep=act_attr_to_keep
    )
    
    return env_gym


def make_env_fn(
    env,
    rank: int,
    seed: int = 0,
    obs_attr_to_keep: Optional[List[str]] = None,
    act_attr_to_keep: Optional[List[str]] = None,
):
    """Create a function that returns a single environment instance for DummyVecEnv."""
    def _init():
        # For DummyVecEnv, we create a new gym env wrapper each time
        # Note: For true parallelism with SubprocVecEnv, we'd need to recreate from env_name
        # but that requires additional parameters. For now, we use DummyVecEnv.
        env_gym = make_grid2op_gym_env(
            env,
            obs_attr_to_keep=obs_attr_to_keep,
            act_attr_to_keep=act_attr_to_keep,
        )
        # Gymnasium uses reset(seed=...) instead of seed()
        # Set seed by resetting with seed parameter
        try:
            env_gym.reset(seed=seed + rank)
        except (TypeError, AttributeError):
            # Fallback for older gym versions or if reset doesn't accept seed
            try:
                env_gym.seed(seed + rank)
            except AttributeError:
                # If neither works, just reset without seed
                env_gym.reset()
        return env_gym
    return _init


def train(env,
          name="PPO_SB3",
          iterations=100000,
          save_path=None,
          load_path=None,
          net_arch=None,
          logs_dir=None,
          learning_rate=3e-4,
          save_every_xxx_steps=None,
          eval_every_xxx_steps=None,
          n_envs=8,
          use_vec_normalize=True,
          norm_obs=True,
          norm_reward=True,
          clip_obs=10.0,
          clip_reward=10.0,
          gamma=0.99,
          seed=None,
          use_swanlab=True,
          swanlab_project="PPO_SB3",
          swanlab_experiment_name=None,
          verbose=True,
          **kwargs):
    """
    Train a PPO agent with parallel environments and VecNormalize.
    
    This function will use stable baselines 3 to train a PPO agent on
    a grid2op environment "env".

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        The environment on which you need to train your agent.

    name: ``str```
        The name of your agent.

    iterations: ``int```
        Total timesteps to train (total_timesteps in SB3). 
        NOTE: This is NOT the number of update steps. It is the total number of environment interactions.
        Recommendation: > 100,000 for meaningful training.

    save_path: ``str```
        Where do you want to save your baseline.

    load_path: ``str```
        If you want to reload your baseline, specify the path where it is located.

    net_arch: ``list```
        The neural network architecture (e.g., [256, 256, 256])

    logs_dir: ``str```
        Where to store the logs during the training.

    learning_rate: ``float```
        The learning rate.

    save_every_xxx_steps: ``int```
        Save checkpoint every N steps.

    eval_every_xxx_steps: ``int```
        Evaluate every N steps (requires eval_env in kwargs).

    n_envs: ``int```
        Number of parallel environments (recommended: CPU cores).

    use_vec_normalize: ``bool```
        Whether to use VecNormalize (highly recommended).

    norm_obs: ``bool```
        Normalize observations.

    norm_reward: ``bool```
        Normalize rewards.

    clip_obs: ``float```
        Clip observations to [-clip_obs, clip_obs].

    clip_reward: ``float```
        Clip rewards to [-clip_reward, clip_reward].

    gamma: ``float```
        Discount factor.

    seed: ``int```
        Random seed.

    use_swanlab: ``bool```
        Use SwanLab for logging (if available).

    swanlab_project: ``str```
        SwanLab project name.

    swanlab_experiment_name: ``str```
        SwanLab experiment name.

    verbose: ``bool```
        Verbosity level.

    kwargs:
        Extra parameters passed to PPO (e.g., eval_env for evaluation).

    Returns
    -------
    baseline: 
        The trained baseline as a SB3Agent.
    """
    if not _CAN_USE_STABLE_BASELINE:
        raise ImportError("Cannot use this function as stable baselines3 is not installed")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
    
    # Detect available attributes
    obs_attr_to_keep = get_available_obs_attributes(env)
    act_attr_to_keep = get_available_act_attributes(env)
    act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)
    
    if verbose:
        print(f"Using {len(obs_attr_to_keep)} observation attributes: {obs_attr_to_keep[:5]}...")
        print(f"Using {len(act_attr_to_keep)} action attributes: {act_attr_to_keep}")
    
    # Save the attributes used (for evaluation)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_used_attribute(save_path, name, obs_attr_to_keep, act_attr_to_keep)
    
    # Create vectorized environments
    if verbose:
        print(f"Creating {n_envs} parallel environments...")
    
    if n_envs == 1:
        # Single environment
        env_gym = make_grid2op_gym_env(env, obs_attr_to_keep, act_attr_to_keep)
        vec_env = DummyVecEnv([lambda: env_gym])
    else:
        # Multiple environments - use DummyVecEnv
        # Note: For true parallelism with SubprocVecEnv, we'd need to recreate environments
        # from env_name in each process, but that requires additional setup.
        # DummyVecEnv with multiple copies still provides some benefits for data collection.
        if verbose:
            print(f"Using DummyVecEnv with {n_envs} environment copies")
        env_fns = [
            make_env_fn(env, rank=i, seed=seed or 0, 
                       obs_attr_to_keep=obs_attr_to_keep,
                       act_attr_to_keep=act_attr_to_keep)
            for i in range(n_envs)
        ]
        vec_env = DummyVecEnv(env_fns)
    
    # Wrap with VecNormalize for robust normalization
    if use_vec_normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            training=True
        )
        if verbose:
            print("Using VecNormalize for observation and reward normalization")
    
    # Create evaluation environment if requested
    eval_vec_env = None
    eval_env = kwargs.get('eval_env', None)
    if eval_every_xxx_steps is not None and eval_env is not None:
        if verbose:
            print("Creating evaluation environment...")
        
        eval_env_gym = make_grid2op_gym_env(
            eval_env,
            obs_attr_to_keep=obs_attr_to_keep,
            act_attr_to_keep=act_attr_to_keep,
        )
        
        eval_vec_env = DummyVecEnv([lambda: eval_env_gym])
        
        if use_vec_normalize:
            eval_vec_env = VecNormalize(
                eval_vec_env,
                norm_obs=norm_obs,
                norm_reward=False,
                clip_obs=clip_obs,
                training=False
            )
    
    # Configure network architecture
    if net_arch is None:
        sample_obs = vec_env.reset()
        obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 1 else len(sample_obs)
        
        if obs_dim < 100:
            net_arch = [200, 200, 200]
        elif obs_dim < 500:
            net_arch = [256, 256, 256]
        else:
            net_arch = [512, 512, 512]
        
        if verbose:
            print(f"Auto-selected network architecture: {net_arch} (obs_dim={obs_dim})")
    
    # Configure policy kwargs
    try:
        import torch
        activation_fn = torch.nn.Tanh
    except ImportError:
        activation_fn = None
    
    policy_kwargs = {"net_arch": net_arch}
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    
    # Configure logging
    tensorboard_log = None
    if not use_swanlab and logs_dir is not None:
        os.makedirs(logs_dir, exist_ok=True)
        tensorboard_log = os.path.join(logs_dir, name)
    
    # Create or load PPO model
    my_path = os.path.join(save_path, name) if save_path else None
    
    if load_path is None:
        # Create new model
        if verbose:
            print("Creating new PPO model...")
        
        model = PPO(
            policy=MlpPolicy,
            env=vec_env,
            learning_rate=learning_rate,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1 if verbose else 0,
            **{k: v for k, v in kwargs.items() if k != 'eval_env'}
        )
    else:
        # Load existing model
        if verbose:
            print(f"Loading model from {load_path}...")
        model_path = os.path.join(load_path, name, name)
        model = PPO.load(model_path, env=vec_env)
        
        # Load VecNormalize stats if they exist
        if use_vec_normalize and save_path is not None:
            vecnorm_path = os.path.join(load_path, name, "vec_normalize.pkl")
            if os.path.exists(vecnorm_path):
                vec_env.load(vecnorm_path)
                if verbose:
                    print(f"Loaded VecNormalize statistics from {vecnorm_path}")
    
    # Setup callbacks
    callbacks = []
    
    # 添加我们自定义的 Grid2Op 日志回调
    callbacks.append(Grid2OpLogCallback(verbose=1 if verbose else 0))
    
    # Checkpoint callback
    if save_every_xxx_steps is not None and save_path is not None:
        callbacks.append(
            CheckpointCallback(
                save_freq=save_every_xxx_steps,
                save_path=my_path,
                name_prefix=name
            )
        )
    
    # Evaluation callback
    if eval_every_xxx_steps is not None and eval_vec_env is not None:
        callbacks.append(
            EvalCallback(
                eval_vec_env,
                best_model_save_path=my_path,
                log_path=my_path,
                eval_freq=eval_every_xxx_steps,
                deterministic=True,
                render=False,
                verbose=1 if verbose else 0,
                n_eval_episodes=8,
            )
        )
    
    # SwanLab callback
    if use_swanlab and _CAN_USE_SWANLAB:
        swanlab_exp_name = swanlab_experiment_name or name
        if verbose:
            print(f"Initializing SwanLab: Project={swanlab_project}, Exp={swanlab_exp_name}")
        callbacks.append(
            SwanLabCallback(
                project=swanlab_project,
                experiment_name=swanlab_exp_name,
                description=f"PPO_SB3 training: {name}",
                verbose=1 if verbose else 0,
            )
        )
    elif use_swanlab:
        warnings.warn("SwanLab requested but not installed. Falling back to TensorBoard.")
    
    # Train the model
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting training for {iterations} timesteps")
        print(f"Progress bar: Enabled")
        print(f"{'='*60}\n")
    
    # --- 修改重点：开启 progress_bar ---
    model.learn(
        total_timesteps=iterations,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=True,  # 这里的 progress_bar 需要 sb3 >= 1.6.0
    )
    
    # Save final model
    if save_path is not None:
        os.makedirs(my_path, exist_ok=True)
        model.save(os.path.join(my_path, name))
        if verbose:
            print(f"\nModel saved to {os.path.join(my_path, name)}")
        
        # Save VecNormalize statistics (CRITICAL for evaluation!)
        if use_vec_normalize:
            vecnorm_path = os.path.join(my_path, "vec_normalize.pkl")
            vec_env.save(vecnorm_path)
            if verbose:
                print(f"VecNormalize statistics saved to {vecnorm_path}")
    
    # Finish SwanLab session
    if use_swanlab and _CAN_USE_SWANLAB:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass
    
    # Cleanup
    vec_env.close()
    if eval_vec_env is not None:
        eval_vec_env.close()
    
    # Create SB3Agent wrapper for Grid2Op compatibility
    env_gym = make_grid2op_gym_env(env, obs_attr_to_keep, act_attr_to_keep)
    grid2op_env = env
    
    agent = SB3Agent(
        g2op_action_space=grid2op_env.action_space,
        gym_act_space=env_gym.action_space,
        gym_obs_space=env_gym.observation_space,
        nn_type=PPO,
        nn_path=os.path.join(my_path, name) if save_path else None,
        nn_kwargs=None,
    )
    
    env_gym.close()
    
    return agent


if __name__ == "__main__":
    """
    PPO_SB3 训练脚本
    
    约定：
    - 模型保存路径: ./rl_saved_model/{name}/
    - 日志路径: ./rl_logs/{name}/
    """
    import grid2op
    from l2rpn_baselines.utils import cli_train
    from lightsim2grid import LightSimBackend

    # 参数解析（支持命令行）
    try:
        args_cli = cli_train().parse_args()
    except SystemExit:
        # 如果命令行解析失败，使用默认值
        class Args:
            pass
        args_cli = Args()

    # 创建环境
    print("="*60)
    print("PPO_SB3 模型训练")
    print("="*60)
    print("\n[1/3] 创建环境...")
    
    env = grid2op.make(
        "l2rpn_wcci_2020",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    print("✓ 环境创建成功")
    
    # 训练智能体
    print("\n[2/3] 开始训练...")
    try:
        baseline = train(
            env=env,
            name="PPO_SB3",
            iterations=5000000,  # 总环境步数（推荐 >= 100,000）
            save_path="./rl_saved_model",  # 约定：模型保存根目录
            logs_dir="./rl_logs",  # 约定：日志保存目录
            save_every_xxx_steps=10000,  # 每 10k 步保存检查点
            eval_every_xxx_steps=10000,  # 每 10k 步评估（需要 eval_env）
            n_envs=32,  # 并行环境数
            seed=42,
            use_swanlab=True,
            swanlab_project="Grid2Op_PPO",
            verbose=True,
        )
        print("\n[3/3] ✓ 训练完成！")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise
    finally:
        env.close()
        print("\n环境已关闭")
