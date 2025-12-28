# -*- coding: utf-8 -*-
"""
ADA 系统训练和监控工具
负责监控 agent 和环境交互的过程，记录日志、绘制图表、上传到 swanlab

参考：
- example/Template/train.py: 标准训练接口
- example/PPO_SB3/train.py: 完整的训练循环、SwanLab 集成
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from grid2op.Environment import Environment
from grid2op.Runner import Runner

# 导入配置
from config import SystemConfig, LLMConfig

# 导入智能体
from ADAgent import ADAgent
from make_agent import make_agent

# 导入工具
from utils.logger import get_logger

# 导入 SwanLab
import swanlab

logger = get_logger("Orchestrator")


class ADAOrchestrator:
    """
    ADA 系统训练和监控工具
    
    职责：
    1. 运行训练循环（多个 episode）
    2. 监控 agent 与环境交互
    3. 记录日志
    4. 绘制图表
    5. 上传到 SwanLab
    6. 保存检查点
    
    参考 example/PPO_SB3/train.py 的实现
    """
    
    def __init__(
        self,
        system_config: SystemConfig,
        llm_config: LLMConfig,
        kb_storage_path: str = None,
        use_swanlab: bool = True,
        swanlab_project: str = "ADA",
        swanlab_experiment_name: Optional[str] = None,
        swanlab_description: Optional[str] = None,
    ):
        """
        初始化训练和监控工具
        
        Args:
            system_config: 系统配置
            llm_config: LLM 配置
            kb_storage_path: 知识库存储路径
            use_swanlab: 是否使用 SwanLab
            swanlab_project: SwanLab 项目名称
            swanlab_experiment_name: SwanLab 实验名称
            swanlab_description: SwanLab 实验描述
        """
        self.system_config = system_config
        self.llm_config = llm_config
        self.kb_storage_path = kb_storage_path
        self.use_swanlab = use_swanlab
        
        # 初始化 SwanLab
        self.swanlab_run = None
        if use_swanlab:
            swanlab_exp_name = swanlab_experiment_name or f"ada_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.swanlab_run = swanlab.init(
                project=swanlab_project,
                experiment_name=swanlab_exp_name,
                description=swanlab_description or "ADA 系统训练实验",
                tags=["ada", "grid2op"],
            )
            logger.info(f"SwanLab 已初始化: {swanlab_exp_name}")
        
        # 训练统计
        self.training_stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "total_reward": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
        }
        
        logger.info("ADA 训练和监控工具初始化完成")
    
    def train(
        self,
        env: Environment,
        name: str = "ADAgent",
        iterations: Optional[int] = 1,
        save_path: Optional[str] = None,
        load_path: Optional[str] = None,
        save_every_xxx_steps: Optional[int] = None,
        eval_every_xxx_steps: Optional[int] = None,
        eval_env: Optional[Environment] = None,
        verbose: bool = True,
        **kwargs
    ) -> ADAgent:
        """
        训练 ADAgent（参考 example/Template/train.py 和 example/PPO_SB3/train.py）
        
        Parameters
        ----------
        env: Environment
            训练环境
        
        name: str
            智能体名称
        
        iterations: int, optional
            训练迭代次数（episode 数）。如果为 None，则运行所有可用场景（完全测试模式）
        
        save_path: str, optional
            保存路径
        
        load_path: str, optional
            加载路径
        
        save_every_xxx_steps: int, optional
            每 N 步保存一次检查点
        
        eval_every_xxx_steps: int, optional
            每 N 步评估一次
        
        eval_env: Environment, optional
            评估环境
        
        verbose: bool
            是否打印详细信息
        
        **kwargs
            其他参数（传递给 make_agent）
        
        Returns
        -------
        ADAgent
            训练后的智能体
        """
        # 创建智能体
        dir_path = save_path or "./ada_agent"
        os.makedirs(dir_path, exist_ok=True)
        
        agent = make_agent(
            env=env,
            dir_path=dir_path,
            name=name,
            system_config=self.system_config,
            llm_config=self.llm_config,
            **kwargs
        )
        
        # 加载已有模型（如果需要）
        if load_path is not None:
            agent.load(load_path)
            logger.info(f"从 {load_path} 加载智能体")
        
        # 使用 Runner 统一运行训练/测试
        return self._run_with_runner(
            env=env,
            agent=agent,
            name=name,
            iterations=iterations,
            save_path=save_path,
            save_every_xxx_steps=save_every_xxx_steps,
            eval_every_xxx_steps=eval_every_xxx_steps,
            eval_env=eval_env,
            verbose=verbose,
            **kwargs
        )
    
    def _run_with_runner(
        self,
        env: Environment,
        agent: ADAgent,
        name: str,
        iterations: Optional[int],
        save_path: Optional[str],
        save_every_xxx_steps: Optional[int],
        eval_every_xxx_steps: Optional[int],
        eval_env: Optional[Environment],
        verbose: bool,
        **kwargs
    ) -> ADAgent:
        """
        使用 Runner 统一运行训练/测试（合并了原来的 _run_episode 和 _run_full_test）
        
        Args:
            env: 环境
            agent: 智能体
            name: 智能体名称
            iterations: Episode 数量。如果为 None，则运行所有可用场景
            save_path: 保存路径
            save_every_xxx_steps: 每 N 个 episode 保存一次检查点
            eval_every_xxx_steps: 每 N 个 episode 评估一次
            eval_env: 评估环境
            verbose: 是否打印详细信息
            **kwargs: 其他参数（包括 max_steps）
        
        Returns:
            训练后的智能体
        """
        # 使用 Runner 运行场景
        runner_params = env.get_params_for_runner()
        runner_params["verbose"] = verbose
        
        runner = Runner(
            **runner_params,
            agentClass=None,
            agentInstance=agent
        )
        
        # 获取 max_steps（如果为 None，则使用 -1 表示不限制）
        max_steps = kwargs.get("max_steps", None)
        max_iter = -1 if max_steps is None else max_steps
        
        # 确定要运行的 episode 数量
        if iterations is None:
            # 完全测试模式：尝试获取总场景数
            try:
                if hasattr(env, 'chronics_handler') and hasattr(env.chronics_handler, 'n_chronics'):
                    nb_episode = env.chronics_handler.n_chronics
                    logger.info(f"完全测试模式：检测到 {nb_episode} 个场景")
                else:
                    nb_episode = 10000  # 使用大数字确保运行所有场景
                    logger.info(f"完全测试模式：无法获取总场景数，使用 {nb_episode} 作为上限")
            except Exception as e:
                nb_episode = 10000
                logger.warning(f"获取总场景数失败: {e}，使用 {nb_episode} 作为上限")
        else:
            # 正常训练模式：运行指定数量的 episode
            nb_episode = iterations
            logger.info(f"开始训练: {iterations} 个 episode")
        
        # 运行场景
        logger.info(f"运行配置: 场景数={nb_episode}, max_iter={max_iter}")
        results = runner.run(
            path_save=None,  # 不保存日志
            nb_episode=nb_episode,
            nb_process=1,
            max_iter=max_iter,
            pbar=verbose
        )
        
        # 处理结果
        for episode_idx, (_, chron_name, cum_reward, nb_time_step, max_ts) in enumerate(results, 1):
            # 更新统计
            self.training_stats["total_episodes"] += 1
            self.training_stats["total_steps"] += nb_time_step
            self.training_stats["total_reward"] += cum_reward
            self.training_stats["episode_rewards"].append(cum_reward)
            self.training_stats["episode_lengths"].append(nb_time_step)
            
            # 记录到 SwanLab
            if self.swanlab_run is not None:
                episode_result = {
                    "episode": episode_idx,
                    "steps": nb_time_step,
                    "total_reward": cum_reward,
                    "mean_reward": cum_reward / nb_time_step if nb_time_step > 0 else 0.0,
                    "max_rho_mean": 0.0,  # Runner 不提供这些详细信息
                    "overflow_count_mean": 0.0,
                    "agent_stats": agent.get_stats() if hasattr(agent, 'get_stats') else {},
                }
                self._log_episode_to_swanlab(episode_result, episode_idx)
            
            # 保存检查点
            if save_every_xxx_steps is not None and episode_idx % save_every_xxx_steps == 0:
                if save_path is not None:
                    agent.save(os.path.join(save_path, f"{name}_checkpoint_{episode_idx}"))
                    logger.info(f"保存检查点: episode {episode_idx}")
            
            # 评估（如果需要）
            if eval_every_xxx_steps is not None and episode_idx % eval_every_xxx_steps == 0:
                if eval_env is not None:
                    eval_result = self._evaluate_agent(eval_env, agent, verbose=verbose)
                    logger.info(f"评估结果 (episode {episode_idx}): {eval_result}")
                    if self.swanlab_run is not None:
                        self._log_eval_to_swanlab(eval_result, episode_idx)
            
            if verbose:
                completed = "✓" if nb_time_step >= max_ts else "✗"
                logger.info(f"场景 {episode_idx} ({chron_name}): {completed} "
                          f"奖励={cum_reward:.2f}, 步数={nb_time_step}/{max_ts}")
        
        # 保存最终模型
        if save_path is not None:
            agent.save(os.path.join(save_path, name))
            logger.info(f"保存最终模型到 {save_path}")
        
        # 完成 SwanLab 会话
        if self.swanlab_run is not None:
            swanlab.finish()
        
        return agent
    
    def _evaluate_agent(
        self,
        env: Environment,
        agent: ADAgent,
        nb_episode: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估智能体（参考 example/ExpertAgent/evaluate.py）
        
        Args:
            env: 评估环境
            agent: 智能体
            nb_episode: Episode 数量
            verbose: 是否打印详细信息
        
        Returns:
            评估结果
        """
        runner_params = env.get_params_for_runner()
        runner_params["verbose"] = verbose
        
        # 使用 Runner 运行评估
        runner = Runner(
            **runner_params,
            agentClass=None,
            agentInstance=agent
        )
        
        # 运行评估
        results = runner.run(
            path_save=None,  # 不保存日志
            nb_episode=nb_episode,
            nb_process=1,
            max_iter=-1,
            pbar=verbose
        )
        
        # 汇总结果
        total_reward = 0.0
        total_steps = 0
        completed_episodes = 0
        
        for _, chron_name, cum_reward, nb_time_step, max_ts in results:
            total_reward += cum_reward
            total_steps += nb_time_step
            if nb_time_step >= max_ts:
                completed_episodes += 1
        
        return {
            "nb_episode": nb_episode,
            "completed_episodes": completed_episodes,
            "total_reward": total_reward,
            "mean_reward": total_reward / nb_episode if nb_episode > 0 else 0.0,
            "total_steps": total_steps,
            "mean_steps": total_steps / nb_episode if nb_episode > 0 else 0.0,
        }
    
    def _log_episode_to_swanlab(self, episode_result: Dict[str, Any], episode_num: int):
        """记录 episode 结果到 SwanLab"""
        if self.swanlab_run is None:
            return
        
        try:
            metrics = {
                "episode/reward": episode_result["total_reward"],
                "episode/mean_reward": episode_result["mean_reward"],
                "episode/steps": episode_result["steps"],
                "episode/max_rho_mean": episode_result["max_rho_mean"],
                "episode/overflow_count_mean": episode_result["overflow_count_mean"],
            }
            
            # Agent 统计
            agent_stats = episode_result.get("agent_stats", {})
            if agent_stats:
                metrics.update({
                    "agent/safe_mode_count": agent_stats.get("safe_mode_count", 0),
                    "agent/full_ada_count": agent_stats.get("full_ada_count", 0),
                    "agent/success_count": agent_stats.get("success_count", 0),
                    "agent/failure_count": agent_stats.get("failure_count", 0),
                    "agent/success_rate": agent_stats.get("success_rate", 0.0),
                })
            
            swanlab.log(metrics, step=episode_num)
        except Exception as e:
            logger.warning(f"SwanLab 记录失败: {e}")
    
    def _log_eval_to_swanlab(self, eval_result: Dict[str, Any], episode_num: int):
        """记录评估结果到 SwanLab"""
        if self.swanlab_run is None:
            return
        
        try:
            metrics = {
                "eval/mean_reward": eval_result["mean_reward"],
                "eval/completed_episodes": eval_result["completed_episodes"],
                "eval/mean_steps": eval_result["mean_steps"],
            }
            swanlab.log(metrics, step=episode_num)
        except Exception as e:
            logger.warning(f"SwanLab 评估记录失败: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            **self.training_stats,
            "mean_episode_reward": (
                np.mean(self.training_stats["episode_rewards"])
                if self.training_stats["episode_rewards"] else 0.0
            ),
            "mean_episode_length": (
                np.mean(self.training_stats["episode_lengths"])
                if self.training_stats["episode_lengths"] else 0.0
            ),
        }

if __name__ == "__main__":
    """
    启动训练示例
    使用方式:
        python orchestrator.py
    """
    import grid2op
    from grid2op.Reward import L2RPNReward
    from lightsim2grid import LightSimBackend
    from pathlib import Path
    from config.yaml_loader import (
        load_config_from_yaml,
        create_system_config_from_yaml,
        create_llm_config_from_yaml,
    )
    
    # ============= 配置参数 =============
    # 配置文件路径
    config_path = "C:\\Users\\a2550\\Desktop\\ADA\\ADA\\yaml\\default.yaml"
    
    # 环境配置
    env_name = "l2rpn_case14_sandbox"  # 或 "l2rpn_wcci_2022" 等
    
    # 训练配置
    # 注意：如果 iterations=None 或 max_steps=None，将进入完全测试模式
    # - iterations=None: 运行所有可用场景（使用 Runner 运行所有 chronics）
    # - max_steps=None: 每个 episode 不限制步数，运行到 episode 自然结束
    iterations = 7  # 训练 episode 数（None 表示完全测试所有场景）
    max_steps = -1  # 每个 episode 的最大步数（None 表示不限制步数，运行到 episode 结束）
    save_path = "./saved_model"  # 模型保存路径
    load_path = None  # 模型加载路径（用于继续训练，None 表示从头训练）
    save_every_xxx_steps = None  # 每 N 个 episode 保存一次检查点（None 表示不保存检查点）
    eval_every_xxx_steps = None  # 每 N 个 episode 评估一次（None 表示不评估）
    
    # SwanLab 配置
    use_swanlab = True
    swanlab_project = "ADA"
    swanlab_experiment_name = None  # None 表示自动生成
    swanlab_description = "ADA 系统训练实验"
    
    verbose = True  # 是否打印详细信息
    
    # ============= 加载配置 =============
    if config_path:
        try:
            config_dict = load_config_from_yaml(str(config_path))
            system_config = create_system_config_from_yaml(config_dict)
            llm_config = create_llm_config_from_yaml(config_dict)
            logger.info(f"从 {config_path} 加载配置成功")
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            system_config = SystemConfig()
            llm_config = LLMConfig()
    else:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        system_config = SystemConfig()
        llm_config = LLMConfig()
    
    # ============= 创建环境 =============
    try:
        env = grid2op.make(
            env_name,
            reward_class=L2RPNReward,
            backend=LightSimBackend(),
        )
        logger.info(f"环境创建成功: {env_name}")
    except Exception as e:
        logger.error(f"环境创建失败: {e}")
        raise
    
    # ============= 创建评估环境（如果需要） =============
    eval_env = None
    if eval_every_xxx_steps is not None:
        try:
            eval_env = grid2op.make(
                env_name,
                reward_class=L2RPNReward,
                backend=LightSimBackend(),
                test=True
            )
            logger.info("评估环境已创建")
        except Exception as e:
            logger.warning(f"创建评估环境失败: {e}")
    
    # ============= 创建训练和监控工具 =============
    orchestrator = ADAOrchestrator(
        system_config=system_config,
        llm_config=llm_config,
        use_swanlab=use_swanlab,
        swanlab_project=swanlab_project,
        swanlab_experiment_name=swanlab_experiment_name,
        swanlab_description=swanlab_description,
    )
    
    # ============= 运行训练 =============
    try:
        agent = orchestrator.train(
            env=env,
            name="ADAgent",
            iterations=iterations,
            save_path=save_path,
            load_path=load_path,
            save_every_xxx_steps=save_every_xxx_steps,
            eval_every_xxx_steps=eval_every_xxx_steps,
            eval_env=eval_env,
            max_steps=max_steps,
            verbose=verbose,
        )
        
        # ============= 打印训练统计 =============
        stats = orchestrator.get_training_stats()
        logger.info("=" * 50)
        logger.info("训练完成")
        logger.info(f"  总 Episode 数: {stats['total_episodes']}")
        logger.info(f"  总步数: {stats['total_steps']}")
        logger.info(f"  总奖励: {stats['total_reward']:.2f}")
        logger.info(f"  平均 Episode 奖励: {stats['mean_episode_reward']:.2f}")
        logger.info(f"  平均 Episode 长度: {stats['mean_episode_length']:.1f}")
        logger.info("=" * 50)
        
    finally:
        # 关闭环境
        env.close()
        if eval_env is not None:
            eval_env.close()
        logger.info("环境已关闭")