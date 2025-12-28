# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import warnings
import copy
import os
import grid2op
import json

from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv

from l2rpn_baselines.PPO_SB3.utils import SB3Agent

try:
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    _CAN_USE_STABLE_BASELINE = True
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    class MlpPolicy(object):
        """
        Do not use, this class is a template when stable baselines3 is not installed.
        
        It represents `from stable_baselines3.ppo import MlpPolicy`
        """

try:
    from swanlab.integration.sb3 import SwanLabCallback
    _CAN_USE_SWANLAB = True
except ImportError:
    _CAN_USE_SWANLAB = False
    SwanLabCallback = None
    
from l2rpn_baselines.PPO_SB3.utils import (default_obs_attr_to_keep, 
                                           default_act_attr_to_keep,
                                           remove_non_usable_attr,
                                           save_used_attribute)

def build_gym_env(env, 
         gymenv_class, 
         gymenv_kwargs, 
         obs_attr_to_keep, 
         obs_space_kwargs, 
         act_attr_to_keep, 
         act_space_kwargs
         ):
    """
    Build the gym environment from the grid2op environment.
    Args:
        env: grid2op environment
        gymenv_class: gym environment class
        gymenv_kwargs: gym environment kwargs
        obs_attr_to_keep: list of string
            Grid2op attribute to use to build the BoxObservationSpace. It is passed
            as the "attr_to_keep" value of the
            BoxObservation space (see
            https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymObsSpace)

        obs_space_kwargs:
            Extra kwargs to build the BoxGymObsSpace (**NOT** saved then NOT restored)

        act_attr_to_keep: list of string
            Grid2op attribute to use to build the BoxGymActSpace. It is passed
            as the "attr_to_keep" value of the
            BoxAction space (see
            https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymActSpace)

        act_space_kwargs:
            Extra kwargs to build the BoxGymActSpace (**NOT** saved then NOT restored)
    Returns:
        gym environment
    """
    # define the gym environment from the grid2op env
    if gymenv_kwargs is None:
        gymenv_kwargs = {}
    env_gym = gymenv_class(env, **gymenv_kwargs)

    env_gym.observation_space.close()
    if obs_space_kwargs is None:
        obs_space_kwargs = {}
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep,
                                               **obs_space_kwargs)
    
    env_gym.action_space.close()
    if act_space_kwargs is None:
        act_space_kwargs = {}
    env_gym.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep,
                                          **act_space_kwargs)
    return env_gym

def train(env,
          name="PPO_SB3",
          iterations=1,
          save_path=None,
          load_path=None,
          net_arch=None,
          logs_dir=None,
          learning_rate=3e-4,
          checkpoint_callback=None,
          save_every_xxx_steps=None,
          eval_every_xxx_steps=None,
          model_policy=MlpPolicy,
          obs_attr_to_keep=copy.deepcopy(default_obs_attr_to_keep),
          obs_space_kwargs=None,
          act_attr_to_keep=copy.deepcopy(default_act_attr_to_keep),
          act_space_kwargs=None,
          policy_kwargs=None,
          normalize_obs=False,
          normalize_act=False,
          gymenv_class=GymEnv,
          gymenv_kwargs=None,
          verbose=True,
          seed=None,  # TODO
          eval_env=None,
          use_swanlab=True,
          swanlab_project="PPO_SB3",
          swanlab_experiment_name=None,
          swanlab_description=None,
          **kwargs):
    """
    This function will use stable baselines 3 to train a PPO agent on
    a grid2op environment "env".

    It will use the grid2op "gym_compat" module to convert the action space
    to a BoxActionSpace and the observation to a BoxObservationSpace.

    It is suited for the studying the impact of continuous actions:

    - on storage units
    - on dispatchable generators
    - on generators with renewable energy sources

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        The environment on which you need to train your agent.

    name: ``str```
        The name of your agent.

    iterations: ``int``
        For how many iterations (steps) do you want to train your agent. NB these are not episode, these are steps.

    save_path: ``str``
        Where do you want to save your baseline.

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    net_arch:
        The neural network architecture, used to create the neural network
        of the PPO (see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

    logs_dir: ``str``
        Where to store the logs during the training. ``None`` if you don't want to log them.
        Note: If use_swanlab=True, this parameter is ignored and SwanLab is used for logging.

    learning_rate: ``float``
        The learning rate, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    save_every_xxx_steps: ``int``
        If set (by default it's None) the stable baselines3 model will be saved
        to the hard drive each `save_every_xxx_steps` steps performed in the
        environment.

    eval_every_xxx_steps: ``int``
        If set (by default it's None) the stable baselines3 model will be evaluated
        each `eval_every_xxx_steps` steps performed in the environment.

    model_policy: 
        Type of neural network model trained in stable baseline. By default
        it's `MlpPolicy`

    obs_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxObservationSpace. It is passed
        as the "attr_to_keep" value of the
        BoxObservation space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymObsSpace)
        
    obs_space_kwargs:
        Extra kwargs to build the BoxGymObsSpace (**NOT** saved then NOT restored)

    act_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxGymActSpace. It is passed
        as the "attr_to_keep" value of the
        BoxAction space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymActSpace)
        
    act_space_kwargs:
        Extra kwargs to build the BoxGymActSpace (**NOT** saved then NOT restored)

    verbose: ``bool``
        If you want something to be printed on the terminal (a better logging strategy will be put at some point)

    normalize_obs: ``bool``
        Attempt to normalize the observation space (so that gym-based stuff will only
        see numbers between 0 and 1)
    
    normalize_act: ``bool``
        Attempt to normalize the action space (so that gym-based stuff will only
        manipulate numbers between 0 and 1)
    
    gymenv_class: 
        The class to use as a gym environment. By default `GymEnv` (from module grid2op.gym_compat)
    
    gymenv_kwargs: ``dict``
        Extra key words arguments to build the gym environment., **NOT** saved / restored by this class
        
    policy_kwargs: ``dict``
        extra parameters passed to the PPO "policy_kwargs" key word arguments
        (defaults to ``None``)
    
    eval_env: :class:`grid2op.Environment`
        Evaluation environment for periodic evaluation during training. If provided and
        `eval_every_xxx_steps` is set, the model will be evaluated periodically.
    
    use_swanlab: ``bool``
        Whether to use SwanLab for logging instead of TensorBoard. Default: True
        
    swanlab_project: ``str``
        Project name for SwanLab. Default: "PPO_SB3"
        
    swanlab_experiment_name: ``str``
        Experiment name for SwanLab. If None, uses the `name` parameter. Default: None
        
    swanlab_description: ``str``
        Description for the SwanLab experiment. Default: None
        
    kwargs:
        extra parameters passed to the PPO from stable baselines 3

    Returns
    -------

    baseline: 
        The trained baseline as a stable baselines PPO element.


    .. _Example-ppo_stable_baseline:

    Examples
    ---------

    Here is an example on how to train a ppo_stablebaseline .

    First define a python script, for example

    .. code-block:: python

        import re
        import grid2op
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from grid2op.Chronics import MultifolderWithCache  # highly recommended
        from lightsim2grid import LightSimBackend  # highly recommended for training !
        from l2rpn_baselines.PPO_SB3 import train

        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name,
                           reward_class=LinesCapacityReward,
                           backend=LightSimBackend(),
                           chronics_class=MultifolderWithCache)

        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
        env.chronics_handler.real_data.reset()
        # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
        # for more information !

        try:
            trained_agent = train(
                  env,
                  iterations=10_000,  # any number of iterations you want
                  logs_dir="./logs",  # where the tensorboard logs will be put
                  save_path="./saved_model",  # where the NN weights will be saved
                  name="test",  # name of the baseline
                  net_arch=[100, 100, 100],  # architecture of the NN
                  save_every_xxx_steps=2000,  # save the NN every 2k steps
                  )
        finally:
            env.close()

    """
    if not _CAN_USE_STABLE_BASELINE:
        raise ImportError("Cannot use this function as stable baselines3 is not installed")
    
    # keep only usable attributes (if default is used)
    act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)
    
    # save the attributes kept
    if save_path is not None:
        my_path = os.path.join(save_path, name)
    save_used_attribute(save_path, name, obs_attr_to_keep, act_attr_to_keep)

    # define the gym environment from the grid2op env
    env_gym = build_gym_env(env,
                            gymenv_class,
                            gymenv_kwargs,
                            obs_attr_to_keep,
                            obs_space_kwargs,
                            act_attr_to_keep,
                            act_space_kwargs)
    if eval_env is not None:
        env_gym_eval = build_gym_env(eval_env,
                                    gymenv_class,
                                    gymenv_kwargs,
                                    obs_attr_to_keep,
                                    obs_space_kwargs,
                                    act_attr_to_keep,
                                    act_space_kwargs)

    if normalize_act:
        if save_path is not None:
            with open(os.path.join(my_path, ".normalize_act"), encoding="utf-8", 
                      mode="w") as f:
                f.write("I have encoded the action space !\n DO NOT MODIFY !")
        for attr_nm in act_attr_to_keep:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                # attribute is scaled elsewhere
                continue
            env_gym.action_space.normalize_attr(attr_nm)

    if normalize_obs:
        if save_path is not None:
            with open(os.path.join(my_path, ".normalize_obs"), encoding="utf-8", 
                      mode="w") as f:
                f.write("I have encoded the observation space !\n DO NOT MODIFY !")
        for attr_nm in obs_attr_to_keep:
            if (("divide" in obs_space_kwargs and attr_nm in obs_space_kwargs["divide"]) or 
                ("subtract" in obs_space_kwargs and attr_nm in obs_space_kwargs["subtract"]) 
               ):
                # attribute is scaled elsewhere
                continue
            env_gym.observation_space.normalize_attr(attr_nm)
            if eval_env is not None:
                env_gym_eval.observation_space.normalize_attr(attr_nm)
    
    # Save a checkpoint every "save_every_xxx_steps" steps
    callbacks = []
    if checkpoint_callback is None:
        if save_every_xxx_steps is not None:
            if save_path is None:
                warnings.warn("save_every_xxx_steps is set, but no path are "
                              "set to save the model (save_path is None). No model "
                              "will be saved.")
            else:
                callbacks.append(CheckpointCallback(save_freq=save_every_xxx_steps,
                                                        save_path=my_path,
                                                        name_prefix=name))

    # define the policy
    if load_path is None:
        if policy_kwargs is None:
            policy_kwargs = {}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch
        
        # Configure logging: use SwanLab if available and requested, otherwise TensorBoard
        tensorboard_log = None
        if not use_swanlab:
            if logs_dir is not None:
                if not os.path.exists(logs_dir):
                    os.mkdir(logs_dir)
                tensorboard_log = os.path.join(logs_dir, name)
        
        nn_kwargs = {
            "policy": model_policy,
            "env": env_gym,
            "verbose": verbose,
            "learning_rate": learning_rate,
            "tensorboard_log": tensorboard_log,
            "policy_kwargs": policy_kwargs,
            **kwargs
        }
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_kwargs=nn_kwargs,
        )
    else:        
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_path=os.path.join(load_path, name)
        )

    if eval_every_xxx_steps is not None:
        if eval_env is None:
            raise ValueError("eval_every_xxx_steps is set but eval_env is None. Please provide an evaluation environment.")
        callbacks.append(EvalCallback(eval_env=env_gym_eval,
                                    best_model_save_path=my_path,
                                    log_path=my_path,
                                    eval_freq=eval_every_xxx_steps,
                                    deterministic=True,
                                    render=False,
                                    verbose=verbose,
                                    n_eval_episodes=8,
                                    ))
    
    # Add SwanLab callback if requested and available
    if use_swanlab:
        if _CAN_USE_SWANLAB:
            swanlab_exp_name = swanlab_experiment_name if swanlab_experiment_name is not None else name
            swanlab_callback = SwanLabCallback(
                project=swanlab_project,
                experiment_name=swanlab_exp_name,
                description=swanlab_description or f"PPO_SB3 training: {name}",
                verbose=verbose,
            )
            callbacks.append(swanlab_callback)
        else:
            warnings.warn("SwanLab is requested but not installed. Falling back to TensorBoard (if logs_dir is set). "
                        "Install with: pip install swanlab")
    
    # train it
    agent.nn_model.learn(total_timesteps=iterations,
                         callback=CallbackList(callbacks),
                         )

    # save it
    if save_path is not None:
        agent.nn_model.save(os.path.join(my_path, name))

    # Finish SwanLab session if used
    if use_swanlab and _CAN_USE_SWANLAB:
        try:
            import swanlab
            swanlab.finish()
        except Exception as e:
            warnings.warn(f"Failed to finish SwanLab session: {e}")

    env_gym.close()
    return agent  # TODO

if __name__ == "__main__":

    import re
    import numpy as np
    import grid2op
    from grid2op.Reward import BaseReward
    from grid2op.Parameters import Parameters
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training

    # Improved reward function that focuses on grid stability
    class ImprovedReward(BaseReward):
        """
        Improved reward function that:
        - Strongly penalizes overloads (rho > 1.0)
        - Rewards low load states (rho < 0.9)
        - Penalizes disconnected lines that can be reconnected
        - Rewards completing episodes
        """
        def __init__(self, logger=None):
            BaseReward.__init__(self, logger=logger)
            self.reward_min = -10.0
            self.reward_max = 10.0
            self.rho_power = 4  # Power for rho penalty (higher = more sensitive to overloads)
            self.safe_rho_threshold = 0.9  # Grid is considered safe if rho < this
            self.penalty_disconnected_line = 2.0  # Penalty for disconnected lines that can be reconnected
            self.penalty_overflow_imminent = 3.0  # Penalty for lines about to overflow
            self.reward_safe_state = 0.1  # Small reward for maintaining safe state
            self.reward_per_step = 0.01  # Small reward for each step survived
            
        def initialize(self, env):
            # Calculate min/max rewards based on worst case scenarios
            max_rho_penalty = -np.sum((np.ones(env.n_line) * 2.0) ** self.rho_power) / env.n_line
            self.reward_min = max_rho_penalty - self.penalty_disconnected_line - self.penalty_overflow_imminent
            self.reward_min = min(self.reward_min, -5.0)  # Cap minimum reward
            
        def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
            if has_error or is_ambiguous:
                return self.reward_min
                
            if is_done:
                # Reward based on how long the episode lasted
                if env.nb_time_step >= env.max_episode_duration():
                    # Completed full episode - strong reward
                    return self.reward_max
                else:
                    # Early termination - reward proportional to steps survived
                    survival_ratio = env.nb_time_step / env.max_episode_duration()
                    return self.reward_max * survival_ratio
            
            if is_illegal:
                return self.reward_min * 0.5  # Less penalty for illegal than error
            
            obs = env.get_obs()
            if obs is None:
                return self.reward_min
            
            # Calculate reward based on grid state
            rho = obs.rho
            
            # Strong penalty for overloads (rho > 1.0)
            # Use exponential penalty for severe overloads
            overload_mask = rho > 1.0
            overload_penalty = 0.0
            if np.any(overload_mask):
                # Exponential penalty: more severe overloads get much higher penalty
                overload_rho = rho[overload_mask]
                overload_penalty = -np.sum((overload_rho ** self.rho_power)) / env.n_line
            
            # Reward for safe states (rho < safe_threshold)
            safe_mask = rho < self.safe_rho_threshold
            safe_reward = 0.0
            if np.any(safe_mask):
                safe_rho = rho[safe_mask]
                # Reward inversely proportional to rho (lower rho = higher reward)
                safe_reward = np.sum((1.0 - safe_rho / self.safe_rho_threshold)) * self.reward_safe_state / env.n_line
            
            # Penalty for moderate load (0.9 < rho < 1.0) - warning zone
            warning_mask = (rho >= self.safe_rho_threshold) & (rho < 1.0)
            warning_penalty = 0.0
            if np.any(warning_mask):
                warning_rho = rho[warning_mask]
                # Quadratic penalty in warning zone
                warning_penalty = -np.sum((warning_rho - self.safe_rho_threshold) ** 2) / env.n_line
            
            # Penalty for disconnected lines that can be reconnected
            disconnected_penalty = 0.0
            if hasattr(obs, 'time_before_cooldown_line') and hasattr(obs, 'line_status'):
                can_reconnect = (obs.time_before_cooldown_line == 0) & (~obs.line_status)
                if np.any(can_reconnect):
                    disconnected_penalty = -np.sum(can_reconnect) * self.penalty_disconnected_line / env.n_line
            
            # Penalty for lines about to overflow (close to disconnection)
            overflow_imminent_penalty = 0.0
            if hasattr(obs, 'timestep_overflow'):
                ts_overflow = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED - 1
                about_to_overflow = (obs.timestep_overflow >= ts_overflow) & (rho >= 1.0)
                if np.any(about_to_overflow):
                    overflow_imminent_penalty = -np.sum(about_to_overflow) * self.penalty_overflow_imminent / env.n_line
            
            # Base reward for surviving another step
            step_reward = self.reward_per_step
            
            # Combine all components
            total_reward = (overload_penalty + 
                          safe_reward + 
                          warning_penalty + 
                          disconnected_penalty + 
                          overflow_imminent_penalty + 
                          step_reward)
            
            # Ensure reward is finite and within bounds
            if not np.isfinite(total_reward):
                total_reward = self.reward_min
            else:
                total_reward = max(self.reward_min, min(self.reward_max, total_reward))
            
            return float(total_reward)

    env_name = "l2rpn_case14_sandbox"
    
    # Optimize parameters for better training
    game_param = Parameters()
    game_param.NB_TIMESTEP_COOLDOWN_SUB = 2  # Reduced cooldown for more flexibility
    game_param.NB_TIMESTEP_COOLDOWN_LINE = 2  # Reduced cooldown for more flexibility
    game_param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2  # Allow some overflow before disconnection
    game_param.HARD_OVERFLOW_THRESHOLD = 2.0  # Hard limit for overflow
    
    env = grid2op.make(env_name,
                       reward_class=ImprovedReward,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache,
                       param=game_param)
    eval_env = grid2op.make(env_name,
                            reward_class=ImprovedReward,
                            backend=LightSimBackend(),
                            chronics_class=MultifolderWithCache,
                            test=True,
                            param=game_param)

    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    eval_env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    eval_env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !
    train(env,
          iterations=100000,
          logs_dir="./logs",
          save_path="./saved_model", 
          name="PPO_SB3",
          net_arch=[200, 200, 200],
          save_every_xxx_steps=2000,
          eval_every_xxx_steps=1000,
          eval_env=eval_env,
          learning_rate=3e-4,  # Explicit learning rate
          gamma=0.99,  # Discount factor for long-term rewards
          )
