# -*- coding: utf-8 -*-
"""
ADA 系统训练脚本
参考 example/Template/train.py 和 example/PPO_SB3/train.py

使用方式:
    python train.py --env_name l2rpn_wcci_2022 --iterations 100 --save_path ./saved_model
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import Optional
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import grid2op
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend

from orchestrator import ADAOrchestrator
from config import SystemConfig, LLMConfig
from config.yaml_loader import (
    load_config_from_yaml,
    create_system_config_from_yaml,
    create_llm_config_from_yaml,
)
from utils.logger import get_logger

logger = get_logger("Train")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 ADAgent")
    
    parser.add_argument(
        "--config",
        type=str,
        default="yaml/default.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--env_name",
        type=str,
        default="l2rpn_case14_sandbox",
        help="Grid2Op 环境名称"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="训练迭代次数（episode 数）"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        default="./saved_model",
        help="模型保存路径"
    )
    
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="模型加载路径（用于继续训练）"
    )
    
    parser.add_argument(
        "--save_every_xxx_steps",
        type=int,
        default=None,
        help="每 N 个 episode 保存一次检查点"
    )
    
    parser.add_argument(
        "--eval_every_xxx_steps",
        type=int,
        default=None,
        help="每 N 个 episode 评估一次"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="每个 episode 的最大步数"
    )
    
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
        default=True,
        help="使用 SwanLab 记录"
    )
    
    parser.add_argument(
        "--no_swanlab",
        action="store_true",
        help="禁用 SwanLab"
    )
    
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default="ADA",
        help="SwanLab 项目名称"
    )
    
    parser.add_argument(
        "--swanlab_experiment_name",
        type=str,
        default=None,
        help="SwanLab 实验名称"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细信息"
    )
    
    return parser.parse_args()


def train(
    env,
    name: str = "ADAgent",
    iterations: int = 1,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    save_every_xxx_steps: Optional[int] = None,
    eval_every_xxx_steps: Optional[int] = None,
    max_steps: int = 100,
    system_config: Optional[SystemConfig] = None,
    llm_config: Optional[LLMConfig] = None,
    use_swanlab: bool = True,
    swanlab_project: str = "ADA",
    swanlab_experiment_name: Optional[str] = None,
    swanlab_description: Optional[str] = None,
    verbose: bool = True,
    **kwargs
):
    """
    训练 ADAgent（参考 example/Template/train.py）
    
    Parameters
    ----------
    env: grid2op.Environment.Environment
        训练环境
    
    name: str
        智能体名称
    
    iterations: int
        训练迭代次数（episode 数）
    
    save_path: str, optional
        保存路径
    
    load_path: str, optional
        加载路径
    
    save_every_xxx_steps: int, optional
        每 N 个 episode 保存一次检查点
    
    eval_every_xxx_steps: int, optional
        每 N 个 episode 评估一次
    
    max_steps: int
        每个 episode 的最大步数
    
    system_config: SystemConfig, optional
        系统配置
    
    llm_config: LLMConfig, optional
        LLM 配置
    
    use_swanlab: bool
        是否使用 SwanLab
    
    swanlab_project: str
        SwanLab 项目名称
    
    swanlab_experiment_name: str, optional
        SwanLab 实验名称
    
    swanlab_description: str, optional
        SwanLab 实验描述
    
    verbose: bool
        是否打印详细信息
    
    **kwargs
        其他参数
    
    Returns
    -------
    ADAgent
        训练后的智能体
    """
    # 创建训练和监控工具
    orchestrator = ADAOrchestrator(
        system_config=system_config or SystemConfig(),
        llm_config=llm_config or LLMConfig(),
        use_swanlab=use_swanlab and not kwargs.get("no_swanlab", False),
        swanlab_project=swanlab_project,
        swanlab_experiment_name=swanlab_experiment_name,
        swanlab_description=swanlab_description,
    )
    
    # 运行训练
    agent = orchestrator.train(
        env=env,
        name=name,
        iterations=iterations,
        save_path=save_path,
        load_path=load_path,
        save_every_xxx_steps=save_every_xxx_steps,
        eval_every_xxx_steps=eval_every_xxx_steps,
        eval_env=eval_env,
        max_steps=max_steps,
        verbose=verbose,
        **kwargs
    )
    
    # 打印训练统计
    stats = orchestrator.get_training_stats()
    logger.info("训练完成")
    logger.info(f"  总 Episode 数: {stats['total_episodes']}")
    logger.info(f"  总步数: {stats['total_steps']}")
    logger.info(f"  总奖励: {stats['total_reward']:.2f}")
    logger.info(f"  平均 Episode 奖励: {stats['mean_episode_reward']:.2f}")
    logger.info(f"  平均 Episode 长度: {stats['mean_episode_length']:.1f}")
    
    return agent


if __name__ == "__main__":
    args = parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        try:
            config_dict = load_config_from_yaml(str(config_path))
            system_config = create_system_config_from_yaml(config_dict)
            llm_config = create_llm_config_from_yaml(config_dict)
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            system_config = SystemConfig()
            llm_config = LLMConfig()
    else:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        system_config = SystemConfig()
        llm_config = LLMConfig()
    
    # 创建环境
    try:
        env = grid2op.make(
            args.env_name,
            reward_class=L2RPNReward,
            backend=LightSimBackend(),
        )
        logger.info(f"环境创建成功: {args.env_name}")
    except Exception as e:
        logger.error(f"环境创建失败: {e}")
        raise
    
    # 创建评估环境（如果需要）
    eval_env = None
    if args.eval_every_xxx_steps is not None:
        try:
            eval_env = grid2op.make(
                args.env_name,
                reward_class=L2RPNReward,
                backend=LightSimBackend(),
                test=True
            )
            logger.info("评估环境已创建")
        except Exception as e:
            logger.warning(f"创建评估环境失败: {e}")
    
    # 运行训练
    try:
        agent = train(
            env=env,
            name="ADAgent",
            iterations=args.iterations,
            save_path=args.save_path,
            load_path=args.load_path,
            save_every_xxx_steps=args.save_every_xxx_steps,
            eval_every_xxx_steps=args.eval_every_xxx_steps,
            max_steps=args.max_steps,
            system_config=system_config,
            llm_config=llm_config,
            use_swanlab=args.use_swanlab and not args.no_swanlab,
            swanlab_project=args.swanlab_project,
            swanlab_experiment_name=args.swanlab_experiment_name,
            verbose=args.verbose,
        )
        logger.info("训练完成")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()

