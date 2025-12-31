# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import argparse

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *
from optimCVXPY import OptimCVXPY

from l2rpn_baselines.utils.save_log_gif import save_log_gif

DEFAULT_LOGS_DIR = "./logs-eval/optimcvxpy-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def cli():
    parser = argparse.ArgumentParser(description="Evaluate OptimCVXPY baseline")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory or environment name (e.g., educ_case14_storage)")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOGS_DIR, type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=DEFAULT_NB_PROCESS, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=DEFAULT_MAX_STEPS, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    parser.add_argument("--test", action='store_true',
                        help="Use test mode (for educ_case14_storage)")
    parser.add_argument("--use_lightsim", action='store_true', default=True,
                        help="Use LightSimBackend (default: True)")
    # OptimCVXPY specific parameters
    parser.add_argument("--margin_th_limit", type=float, default=0.93,
                        help="Margin for thermal limit (default: 0.93)")
    parser.add_argument("--rho_safe", type=float, default=0.95,
                        help="Safe mode threshold (default: 0.95)")
    parser.add_argument("--rho_danger", type=float, default=0.97,
                        help="Danger mode threshold (default: 0.97)")
    parser.add_argument("--penalty_storage_unsafe", type=float, default=0.04,
                        help="Storage penalty in unsafe mode (default: 0.04)")
    parser.add_argument("--penalty_curtailment_unsafe", type=float, default=0.01,
                        help="Curtailment penalty in unsafe mode (default: 0.01)")
    parser.add_argument("--penalty_redispatching_unsafe", type=float, default=0.0,
                        help="Redispatching penalty in unsafe mode (default: 0.0)")
    return parser.parse_args()


def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             verbose=False,
             save_gif=False,
             # OptimCVXPY specific parameters
             margin_th_limit=0.93,
             rho_safe=0.9,
             rho_danger=0.95,
             penalty_storage_unsafe=0.04,
             penalty_curtailment_unsafe=0.01,
             penalty_redispatching_unsafe=0.0,
             alpha_por_error=0.5,
             weight_redisp_target=0.3,
             **kwargs):
    """
    Evaluate the OptimCVXPY agent.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The environment on which the agent will be evaluated.
        
    load_path : str, optional
        Not used for OptimCVXPY (no model to load), kept for interface consistency.
        
    logs_path : str, optional
        Path where the evaluation logs will be saved. Default: "./logs-eval/optimcvxpy-baseline"
        
    nb_episode : int, optional
        Number of episodes to evaluate. Default: 1
        
    nb_process : int, optional
        Number of processes to use for parallel evaluation. Default: 1
        
    max_steps : int, optional
        Maximum number of steps per episode. -1 means run until the end. Default: -1
        
    verbose : bool, optional
        Whether to print verbose output. Default: False
        
    save_gif : bool, optional
        Whether to save GIF visualizations. Default: False
        
    margin_th_limit : float, optional
        Margin for thermal limit. Default: 0.93
        
    rho_safe : float, optional
        Safe mode threshold. Default: 0.95
        
    rho_danger : float, optional
        Danger mode threshold. Default: 0.97
        
    penalty_storage_unsafe : float, optional
        Storage penalty in unsafe mode. Default: 0.04
        
    penalty_curtailment_unsafe : float, optional
        Curtailment penalty in unsafe mode. Default: 0.01
        
    penalty_redispatching_unsafe : float, optional
        Redispatching penalty in unsafe mode. Default: 0.0
        
    alpha_por_error : float, optional
        Power balance error weight. Default: 0.5
        
    weight_redisp_target : float, optional
        Redispatching target weight. Default: 0.3
        
    **kwargs : dict
        Additional keyword arguments passed to OptimCVXPY constructor.

    Returns
    -------
    list
        List of evaluation results, each element is a tuple:
        (agent, chronics_name, cumulative_reward, nb_time_step, max_ts)
    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create OptimCVXPY agent
    agent = OptimCVXPY(env.action_space,
                       env,
                       penalty_redispatching_unsafe=penalty_redispatching_unsafe,
                       penalty_storage_unsafe=penalty_storage_unsafe,
                       penalty_curtailment_unsafe=penalty_curtailment_unsafe,
                       rho_safe=rho_safe,
                       rho_danger=rho_danger,
                       margin_th_limit=margin_th_limit,
                       alpha_por_error=alpha_por_error,
                       weight_redisp_target=weight_redisp_target,
                       **kwargs)
    
    # Set storage setpoint if storage units exist
    if hasattr(env, 'storage_Emax') and env.storage_Emax is not None:
        agent.storage_setpoint = env.storage_Emax

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Run evaluation
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=True)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)
    
    return res


if __name__ == "__main__":
    # Parse command line
    args = cli()
    
    # Determine if data_dir is an environment name or path
    try:
        # Try to use it as environment name first
        if args.use_lightsim:
            try:
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
            except ImportError:
                print("Warning: lightsim2grid not available, using default backend")
                backend = None
        else:
            backend = None
        
        # Try to create environment with the name
        if backend:
            env = make(args.data_dir,
                       test=args.test,
                       backend=backend,
                       reward_class=RedispReward,
                       action_class=TopologyChangeAction,
                       other_rewards={
                           "bridge": BridgeReward,
                           "overflow": CloseToOverflowReward,
                           "distance": DistanceReward
                       })
        else:
            env = make(args.data_dir,
                       test=args.test,
                       reward_class=RedispReward,
                       action_class=TopologyChangeAction,
                       other_rewards={
                           "bridge": BridgeReward,
                           "overflow": CloseToOverflowReward,
                           "distance": DistanceReward
                       })
    except Exception as e:
        # If it fails, try as a path
        print(f"Warning: Could not create environment with name '{args.data_dir}': {e}")
        print("Trying as a path...")
        if args.use_lightsim:
            try:
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
            except ImportError:
                backend = None
        else:
            backend = None
        
        if backend:
            env = make(args.data_dir,
                       backend=backend,
                       reward_class=RedispReward,
                       action_class=TopologyChangeAction,
                       other_rewards={
                           "bridge": BridgeReward,
                           "overflow": CloseToOverflowReward,
                           "distance": DistanceReward
                       })
        else:
            env = make(args.data_dir,
                       reward_class=RedispReward,
                       action_class=TopologyChangeAction,
                       other_rewards={
                           "bridge": BridgeReward,
                           "overflow": CloseToOverflowReward,
                           "distance": DistanceReward
                       })
    
    # Call evaluation interface
    evaluate(env,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif,
             margin_th_limit=args.margin_th_limit,
             rho_safe=args.rho_safe,
             rho_danger=args.rho_danger,
             penalty_storage_unsafe=args.penalty_storage_unsafe,
             penalty_curtailment_unsafe=args.penalty_curtailment_unsafe,
             penalty_redispatching_unsafe=args.penalty_redispatching_unsafe)
