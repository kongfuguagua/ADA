# -*- coding: utf-8 -*-
"""
L2RPN NeurIPS 2020 比赛演示

本脚本演示如何初始化和交互 NeurIPS 2020 L2RPN 比赛环境（赛道1和赛道2）。

赛道1（鲁棒性）：处理对抗性线路攻击
赛道2（适应性）：处理可变的可再生能源发电

使用方法:
    python demo_neurips_2020.py --track 1
    python demo_neurips_2020.py --track 2
"""

import argparse
import numpy as np
from typing import Optional

from config import NEURIPS_2020_TRACK1, NEURIPS_2020_TRACK2, print_config_info
from env_factory import EnvManager, create_env


def demo_track1(seed: Optional[int] = 42, max_steps: int = 100):
    """
    NeurIPS 2020 赛道1（鲁棒性）演示
    
    该赛道专注于处理电力线路的对抗性攻击。
    智能体必须在随机线路断开的情况下维持电网稳定。
    """
    print("\n" + "="*70)
    print("NeurIPS 2020 - 赛道1（鲁棒性）演示")
    print("="*70)
    
    # 打印配置信息
    print_config_info(NEURIPS_2020_TRACK1)
    
    # 创建环境
    print("正在创建环境...")
    manager = EnvManager(NEURIPS_2020_TRACK1, seed=seed)
    
    # 重置并获取初始观测
    obs = manager.reset()
    print(f"\n已获取初始观测。")
    manager.print_status()
    
    # 显示电网信息
    print("\n电网信息:")
    grid_info = manager.get_grid_info()
    print(f"  线路数: {grid_info['n_line']}")
    print(f"  变电站数: {grid_info['n_sub']}")
    print(f"  发电机数: {grid_info['n_gen']}")
    print(f"  负荷数: {grid_info['n_load']}")
    
    # 显示动作空间能力
    print("\n动作空间能力:")
    action_info = manager.get_action_space_info()
    for key, value in action_info.items():
        print(f"  {key}: {value}")
    
    # 运行仿真
    print(f"\n运行仿真 {max_steps} 步...")
    total_reward = 0.0
    max_rho_history = []
    
    for step in range(max_steps):
        # 演示使用"什么都不做"动作
        action = manager.get_do_nothing_action()
        
        # 执行环境步骤
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        max_rho_history.append(obs.rho.max())
        
        # 检查问题
        if done:
            print(f"\n[第 {step+1} 步] 回合结束!")
            if 'exception' in info and info['exception']:
                print(f"  异常: {info['exception']}")
            break
        
        # 进度更新
        if (step + 1) % 20 == 0:
            print(f"  第 {step+1} 步: 最大rho={obs.rho.max():.2%}, 奖励={reward:.2f}")
    
    # 总结
    print("\n" + "-"*50)
    print("仿真总结:")
    print(f"  总步数: {step+1}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均最大rho: {np.mean(max_rho_history):.2%}")
    print(f"  峰值最大rho: {np.max(max_rho_history):.2%}")
    print("-"*50)
    
    manager.close()
    return total_reward


def demo_track2(seed: Optional[int] = 42, max_steps: int = 100):
    """
    NeurIPS 2020 赛道2（适应性）演示
    
    该赛道专注于处理可变的可再生能源发电。
    智能体必须适应不断变化的发电模式。
    """
    print("\n" + "="*70)
    print("NeurIPS 2020 - 赛道2（适应性）演示")
    print("="*70)
    
    # 打印配置信息
    print_config_info(NEURIPS_2020_TRACK2)
    
    # 创建环境
    print("正在创建环境...")
    manager = EnvManager(NEURIPS_2020_TRACK2, seed=seed)
    
    # 重置并获取初始观测
    obs = manager.reset()
    print(f"\n已获取初始观测。")
    manager.print_status()
    
    # 显示电网信息
    print("\n电网信息:")
    grid_info = manager.get_grid_info()
    print(f"  线路数: {grid_info['n_line']}")
    print(f"  变电站数: {grid_info['n_sub']}")
    print(f"  发电机数: {grid_info['n_gen']}")
    print(f"  负荷数: {grid_info['n_load']}")
    
    # 检查可再生能源发电机
    if 'gen_renewable' in grid_info:
        n_renewable = sum(grid_info['gen_renewable'])
        print(f"  可再生能源发电机数: {n_renewable}")
    
    # 检查可再调度发电机
    if 'gen_redispatchable' in grid_info:
        n_redispatch = sum(grid_info['gen_redispatchable'])
        print(f"  可再调度发电机数: {n_redispatch}")
    
    # 运行仿真
    print(f"\n运行仿真 {max_steps} 步...")
    total_reward = 0.0
    gen_history = []
    
    for step in range(max_steps):
        # 演示使用"什么都不做"动作
        action = manager.get_do_nothing_action()
        
        # 执行环境步骤
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        gen_history.append(obs.gen_p.sum())
        
        # 检查问题
        if done:
            print(f"\n[第 {step+1} 步] 回合结束!")
            break
        
        # 进度更新
        if (step + 1) % 20 == 0:
            print(f"  第 {step+1} 步: 总发电={obs.gen_p.sum():.1f}MW, 最大rho={obs.rho.max():.2%}")
    
    # 总结
    print("\n" + "-"*50)
    print("仿真总结:")
    print(f"  总步数: {step+1}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均发电量: {np.mean(gen_history):.1f} MW")
    print(f"  发电量波动（标准差）: {np.std(gen_history):.1f} MW")
    print("-"*50)
    
    manager.close()
    return total_reward


def demo_action_exploration(track: int = 1):
    """
    NeurIPS 2020 动作空间探索演示
    
    展示如何创建和应用不同类型的动作。
    """
    print("\n" + "="*70)
    print(f"动作空间探索 - 赛道 {track}")
    print("="*70)
    
    config = NEURIPS_2020_TRACK1 if track == 1 else NEURIPS_2020_TRACK2
    manager = EnvManager(config, seed=42)
    obs = manager.reset()
    
    action_space = manager.env.action_space
    
    # 1. 什么都不做动作
    print("\n1. 什么都不做动作:")
    do_nothing = action_space({})
    print(f"   {do_nothing}")
    
    # 2. 拓扑变更动作（改变母线）
    print("\n2. 拓扑变更动作（在变电站改变母线）:")
    try:
        # 获取一个有多个元件的变电站
        sub_id = 1  # 通常有多个元件
        topo_action = action_space({
            "change_bus": {"substations_id": [(sub_id, [True, False, False, False])]}
        })
        print(f"   在变电站 {sub_id} 改变母线")
        
        # 模拟动作
        sim_obs, sim_reward, sim_done, sim_info = manager.simulate(topo_action)
        print(f"   模拟后最大rho: {sim_obs.rho.max():.2%}")
    except Exception as e:
        print(f"   无法创建拓扑动作: {e}")
    
    # 3. 线路状态动作
    print("\n3. 线路状态动作（重连线路）:")
    try:
        # 找一条可以重连的线路
        line_id = 0
        line_action = action_space({"set_line_status": [(line_id, +1)]})
        print(f"   设置线路 {line_id} 状态为已连接")
    except Exception as e:
        print(f"   无法创建线路动作: {e}")
    
    # 4. 再调度动作
    print("\n4. 再调度动作:")
    try:
        # 找可再调度的发电机
        gen_redispatchable = manager.env.gen_redispatchable
        redisp_gen_ids = np.where(gen_redispatchable)[0]
        
        if len(redisp_gen_ids) > 0:
            gen_id = redisp_gen_ids[0]
            redisp_amount = 5.0  # MW
            redisp_action = action_space({"redispatch": [(gen_id, redisp_amount)]})
            print(f"   对发电机 {gen_id} 再调度 {redisp_amount} MW")
            
            # 模拟
            sim_obs, sim_reward, sim_done, sim_info = manager.simulate(redisp_action)
            print(f"   模拟后最大rho: {sim_obs.rho.max():.2%}")
        else:
            print("   没有可再调度的发电机")
    except Exception as e:
        print(f"   无法创建再调度动作: {e}")
    
    # 5. 获取所有单一动作
    print("\n5. 可用的单一动作:")
    try:
        line_actions = action_space.get_all_unitary_line_set(action_space)
        print(f"   线路设置动作数: {len(line_actions)}")
        
        topo_actions = action_space.get_all_unitary_topologies_set(action_space)
        print(f"   拓扑设置动作数: {len(topo_actions)}")
    except Exception as e:
        print(f"   无法枚举动作: {e}")
    
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="L2RPN NeurIPS 2020 比赛演示"
    )
    parser.add_argument(
        "--track", 
        type=int, 
        choices=[1, 2], 
        default=1,
        help="比赛赛道（1=鲁棒性, 2=适应性）"
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
        "--explore-actions",
        action="store_true",
        help="运行动作空间探索演示"
    )
    
    args = parser.parse_args()
    
    try:
        if args.explore_actions:
            demo_action_exploration(args.track)
        elif args.track == 1:
            demo_track1(seed=args.seed, max_steps=args.max_steps)
        else:
            demo_track2(seed=args.seed, max_steps=args.max_steps)
            
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已安装所需的包:")
        print("  pip install grid2op lightsim2grid")
        print("\n并下载了环境数据:")
        print("  赛道1: grid2op.make('l2rpn_neurips_2020_track1')")
        print("  赛道2: grid2op.make('l2rpn_neurips_2020_track2')")


if __name__ == "__main__":
    main()
