# -*- coding: utf-8 -*-
"""
OptLLM Agent 评估程序
参考 OptimCVXPY 和 ExpertAgent 的评估接口
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加当前目录到路径（以便导入同目录的模块）
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from grid2op.Action import TopologyChangeAction

# 导入 OptAgent（支持直接运行和作为模块导入）
try:
    # 如果作为模块导入，使用相对导入
    from .agent import OptAgent
except ImportError:
    # 如果直接运行，使用绝对导入
    try:
        from OptLLM.agent import OptAgent
    except ImportError:
        from agent import OptAgent

# 导入 LLM 和日志工具
from utils import OpenAIChat, get_logger

logger = get_logger("ReActEvaluate")

DEFAULT_LOGS_DIR = "./logs-eval/optllm"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def cli():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="Evaluate OptLLM Agent")
    
    # 环境参数
    parser.add_argument("--data_dir", required=True,
                       help="Path to the dataset root directory or environment name (e.g., l2rpn_case14_sandbox)")
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
                       help="Use test mode")
    parser.add_argument("--use_lightsim", action='store_true', default=True,
                       help="Use LightSimBackend (default: True)")
    
    # LLM 参数
    parser.add_argument("--llm_model", type=str, default=None,
                       help="LLM model name (default: from environment variable CLOUD_MODEL)")
    parser.add_argument("--llm_api_key", type=str, default=None,
                       help="LLM API key (default: from environment variable CLOUD_API_KEY)")
    parser.add_argument("--llm_base_url", type=str, default=None,
                       help="LLM base URL (default: from environment variable CLOUD_BASE_URL)")
    parser.add_argument("--llm_temperature", type=float, default=0.7,
                       help="LLM temperature (default: 0.7)")
    parser.add_argument("--llm_max_tokens", type=int, default=4096,
                       help="LLM max tokens (default: 4096)")
    
    # OptAgent 参数
    parser.add_argument("--max_react_steps", type=int, default=3,
                       help="Maximum retry steps for OptAgent (default: 3)")
    parser.add_argument("--rho_danger", type=float, default=0.92,
                       help="Rho safe threshold for OptAgent (default: 0.92, call LLM when rho >= 92% for preventive action)")
    
    # 场景选择参数
    parser.add_argument("--episode_id", type=str, default=None,
                       help="Comma-separated list of episode IDs to run (e.g., '0,1,2' or '0-6'). If not specified, runs episodes in default order.")
    parser.add_argument("--env_seeds", type=str, default=None,
                       help="Comma-separated list of environment seeds (e.g., '0,1,2')")
    
    return parser.parse_args()


def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             verbose=False,
             save_gif=False,
             # 场景选择参数
             episode_id=None,  # 指定场景编号列表，例如 [0, 1, 2] 或 None（使用默认顺序）
             env_seeds=None,   # 环境随机种子列表
             # LLM 参数
             llm_model=None,
             llm_api_key=None,
             llm_base_url=None,
             llm_temperature=0.7,
             llm_max_tokens=4096,
    # OptAgent 参数
    max_react_steps=3,
    rho_danger=0.92,  # 修改为 0.92：在过载前进行预防性调度（原值 1.0 太晚）
             **kwargs):
             
    # 创建 LLM 客户端
    try:
        llm_client = OpenAIChat(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        logger.info(f"LLM 客户端创建成功: model={llm_client.model}")
    except Exception as e:
        logger.error(f"LLM 客户端创建失败: {e}")
        raise
    
    # 创建 OptAgent
    agent = OptAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        llm_client=llm_client,
        env=env,
        max_retry_steps=max_react_steps,
        name="OptAgent",
        rho_safe=rho_danger,  # rho_safe: 当负载率 >= rho_safe 时调用LLM (与rho_danger语义一致)
        **kwargs
    )
    logger.info(f"OptAgent 创建成功: max_retry_steps={max_react_steps}, rho_safe={rho_danger}")
    
    # 构建 Runner
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)
    
    # 运行评估
    os.makedirs(logs_path, exist_ok=True)
    logger.info(f"开始评估: nb_episode={nb_episode}, max_steps={max_steps}")
    if episode_id is not None:
        logger.info(f"指定场景编号: {episode_id}")
    
    # 构建 run 参数
    run_kwargs = {
        "path_save": logs_path,
        "nb_episode": nb_episode,
        "nb_process": nb_process,
        "max_iter": max_steps,
        "pbar": True
    }
    
    # 如果指定了场景编号，添加到参数中
    if episode_id is not None:
        run_kwargs["episode_id"] = episode_id
    if env_seeds is not None:
        run_kwargs["env_seeds"] = env_seeds
    
    res = runner.run(**run_kwargs)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Evaluation summary:")
    print("=" * 60)
    total_reward = 0.0
    total_steps = 0
    completed_episodes = 0
    
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        total_reward += cum_reward
        total_steps += nb_time_step
        if nb_time_step >= max_ts:
            completed_episodes += 1
        
        msg_tmp = f"Chronic: {chron_name}"
        msg_tmp += f"\tTotal reward: {cum_reward:.6f}"
        msg_tmp += f"\tTime steps: {nb_time_step}/{max_ts}"
        if nb_time_step >= max_ts:
            msg_tmp += " ✓"
        else:
            msg_tmp += " ✗"
        print(msg_tmp)
    
    print("=" * 60)
    print(f"Overall: {completed_episodes}/{nb_episode} episodes completed")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Average reward: {total_reward/nb_episode:.6f}")
    print(f"Average steps: {total_steps/nb_episode:.1f}")
    
    # 打印 Agent 统计信息
    if hasattr(agent, 'get_stats'):
        stats = agent.get_stats()
        print("\nAgent Statistics:")
        print(f"  Total steps: {stats.get('total_steps', 0)}")
        print(f"  Successful actions: {stats.get('successful_actions', 0)}")
        print(f"  Failed actions: {stats.get('failed_actions', 0)}")
        print(f"  Do nothing count: {stats.get('do_nothing_count', 0)}")
        print(f"  Heuristic do nothing count: {stats.get('heuristic_do_nothing_count', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0.0):.2%}")
        print(f"  Heuristic do nothing rate: {stats.get('heuristic_do_nothing_rate', 0.0):.2%}")
        print(f"  Avg ReAct loops per step: {stats.get('avg_react_loops_per_step', 0.0):.2f}")
    
    print("=" * 60)
    
    # 保存 GIF（如果支持）
    if save_gif:
        try:
            from l2rpn_baselines.utils.save_log_gif import save_log_gif
            save_log_gif(logs_path, res)
            logger.info(f"GIF 已保存到 {logs_path}")
        except ImportError:
            logger.warning("l2rpn_baselines 未安装，无法保存 GIF")
        except Exception as e:
            logger.warning(f"保存 GIF 失败: {e}")
    
    return res


if __name__ == "__main__":
    # 解析命令行参数
    args = cli()
    
    # 确定 data_dir 是环境名称还是路径
    try:
        # 尝试作为环境名称使用
        if args.use_lightsim:
            try:
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
            except ImportError:
                print("Warning: lightsim2grid not available, using default backend")
                backend = None
        else:
            backend = None
        
        # 尝试创建环境
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
        # 如果失败，尝试作为路径
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
    
    # 解析 episode_id 和 env_seeds
    episode_id = None
    if args.episode_id:
        try:
            # 支持 "0,1,2" 或 "0-6" 格式
            if '-' in args.episode_id:
                start, end = map(int, args.episode_id.split('-'))
                episode_id = list(range(start, end + 1))
            else:
                episode_id = [int(x.strip()) for x in args.episode_id.split(',')]
        except ValueError:
            logger.warning(f"无法解析 episode_id: {args.episode_id}，将使用默认顺序")
            episode_id = None
    
    env_seeds = None
    if args.env_seeds:
        try:
            env_seeds = [int(x.strip()) for x in args.env_seeds.split(',')]
        except ValueError:
            logger.warning(f"无法解析 env_seeds: {args.env_seeds}，将使用默认种子")
            env_seeds = None
    
    # 调用评估接口
    evaluate(env,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif,
             episode_id=episode_id,
             env_seeds=env_seeds,
             llm_model=args.llm_model,
             llm_api_key=args.llm_api_key,
             llm_base_url=args.llm_base_url,
             llm_temperature=args.llm_temperature,
             llm_max_tokens=args.llm_max_tokens,
             max_react_steps=args.max_react_steps,
             rho_danger=args.rho_danger)

