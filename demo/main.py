# -*- coding: utf-8 -*-
"""
L2RPN 演示 - 主入口

所有 L2RPN 比赛演示的统一入口点。
支持 NeurIPS 2020（赛道1和2）、ICAPS 2021 和 WCCI 2022。

使用方法:
    python main.py --env wcci_2022
    python main.py --env neurips_2020_track1 --max-steps 200
    python main.py --list-envs
    python main.py --info wcci_2022
"""

import argparse
import sys
from typing import Optional

from config import (
    ENV_CONFIGS, 
    get_config, 
    list_configs, 
    print_config_info,
    Competition
)
from env_factory import EnvManager, create_env, run_episode


def run_demo(
    env_name: str,
    seed: Optional[int] = 42,
    max_steps: int = 100,
    verbose: bool = True
):
    """
    运行指定环境的演示
    
    参数:
        env_name: 配置名称
        seed: 随机种子
        max_steps: 最大运行步数
        verbose: 是否打印详细输出
    """
    config = get_config(env_name)
    
    print("\n" + "="*70)
    print(f"L2RPN 演示: {config.name}")
    print("="*70)
    
    if verbose:
        print_config_info(config)
    
    # 创建环境管理器
    print("正在初始化环境...")
    manager = EnvManager(config, seed=seed)
    
    # 重置环境
    obs = manager.reset()
    print("环境初始化成功!")
    
    # 打印电网信息
    if verbose:
        print("\n电网信息:")
        grid_info = manager.get_grid_info()
        print(f"  线路数: {grid_info['n_line']}")
        print(f"  变电站数: {grid_info['n_sub']}")
        print(f"  发电机数: {grid_info['n_gen']}")
        print(f"  负荷数: {grid_info['n_load']}")
        if grid_info['n_storage'] > 0:
            print(f"  储能单元数: {grid_info['n_storage']}")
    
    # 运行仿真
    print(f"\n运行仿真，最多 {max_steps} 步...")
    
    total_reward = 0.0
    step_count = 0
    
    for step in range(max_steps):
        # "什么都不做"动作
        action = manager.get_do_nothing_action()
        
        # 执行步骤
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        step_count += 1
        
        if done:
            print(f"\n[第 {step_count} 步] 回合结束!")
            break
        
        # 进度更新
        if verbose and (step + 1) % 20 == 0:
            print(f"  第 {step_count} 步: 最大rho={obs.rho.max():.2%}, 奖励={reward:.2f}")
    
    # 最终总结
    print("\n" + "-"*50)
    print("仿真总结:")
    print(f"  环境: {config.name}")
    print(f"  完成步数: {step_count}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  最终最大rho: {obs.rho.max():.2%}")
    print("-"*50)
    
    manager.close()
    return {"steps": step_count, "reward": total_reward}


def list_environments():
    """列出所有可用环境"""
    print("\n" + "="*60)
    print("可用的 L2RPN 环境")
    print("="*60)
    
    # 按比赛分组
    competitions = {}
    for name, config in ENV_CONFIGS.items():
        comp = config.competition.value
        if comp not in competitions:
            competitions[comp] = []
        competitions[comp].append((name, config))
    
    for comp, configs in competitions.items():
        print(f"\n{comp.upper().replace('_', ' ')}:")
        print("-" * 40)
        for name, config in configs:
            features = []
            if config.has_storage:
                features.append("储能")
            if config.has_renewable:
                features.append("可再生能源")
            if config.has_curtailment:
                features.append("弃风")
            if config.has_alarm:
                features.append("告警")
            
            feature_str = f" [{', '.join(features)}]" if features else ""
            print(f"  {name}: {config.name}{feature_str}")
    
    print("\n" + "="*60)
    print("使用方法: python main.py --env <环境名称>")
    print("示例: python main.py --env wcci_2022")
    print("="*60 + "\n")


def show_env_info(env_name: str):
    """显示环境的详细信息"""
    try:
        config = get_config(env_name)
        print_config_info(config)
    except KeyError as e:
        print(f"错误: {e}")
        print(f"\n可用环境: {list_configs()}")


def compare_environments():
    """比较所有环境的特性"""
    print("\n" + "="*80)
    print("L2RPN 环境特性对比")
    print("="*80)
    
    # 表头
    print(f"\n{'环境':<25} {'储能':<10} {'可再生能源':<12} {'弃风':<10} {'告警':<10}")
    print("-" * 80)
    
    for name, config in ENV_CONFIGS.items():
        storage = "是" if config.has_storage else "否"
        renewable = "是" if config.has_renewable else "否"
        curtail = "是" if config.has_curtailment else "否"
        alarm = "是" if config.has_alarm else "否"
        
        print(f"{name:<25} {storage:<10} {renewable:<12} {curtail:<10} {alarm:<10}")
    
    print("-" * 80)
    print("\n说明:")
    print("  储能: 是否有储能单元")
    print("  可再生能源: 是否有可变的可再生能源发电机")
    print("  弃风: 是否可以执行弃风动作")
    print("  告警: 是否有人机协作的告警机制")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="L2RPN 比赛演示 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --list-envs              # 列出所有环境
  python main.py --env wcci_2022          # 运行 WCCI 2022 演示
  python main.py --env neurips_2020_track1 --max-steps 200
  python main.py --info icaps_2021        # 显示环境信息
  python main.py --compare                # 比较所有环境
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=list(ENV_CONFIGS.keys()),
        help="要运行演示的环境"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可重复性"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="最大仿真步数"
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="列出所有可用环境"
    )
    parser.add_argument(
        "--info",
        type=str,
        metavar="ENV",
        help="显示指定环境的详细信息"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比较所有环境的特性"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少输出信息"
    )
    
    args = parser.parse_args()
    
    # 处理不同的命令
    if args.list_envs:
        list_environments()
    elif args.info:
        show_env_info(args.info)
    elif args.compare:
        compare_environments()
    elif args.env:
        try:
            run_demo(
                args.env,
                seed=args.seed,
                max_steps=args.max_steps,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"\n运行演示时出错: {e}")
            print("\n请确保已安装所需的包:")
            print("  pip install grid2op lightsim2grid")
            print("\n并确保环境数据可用。")
            sys.exit(1)
    else:
        # 没有参数 - 显示帮助
        parser.print_help()
        print("\n" + "-"*60)
        print("快速开始: python main.py --list-envs")
        print("-"*60)


if __name__ == "__main__":
    main()
