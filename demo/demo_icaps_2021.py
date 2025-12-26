# -*- coding: utf-8 -*-
"""
L2RPN ICAPS 2021 比赛演示

本脚本演示如何初始化和交互 ICAPS 2021 L2RPN 比赛环境（L2RPN with Trust）。

关键特性：用于人机协作的告警机制。
智能体可以发出告警通知人类操作员潜在问题。

使用方法:
    python demo_icaps_2021.py
    python demo_icaps_2021.py --max-steps 200
"""

import argparse
import numpy as np
from typing import Optional, List

from config import ICAPS_2021, print_config_info
from env_factory import EnvManager


def demo_icaps_2021(seed: Optional[int] = 42, max_steps: int = 100):
    """
    ICAPS 2021（L2RPN with Trust）演示
    
    该比赛引入了人机协作的告警机制。
    智能体可以在检测到潜在电网问题时发出告警。
    """
    print("\n" + "="*70)
    print("ICAPS 2021 - L2RPN 信任赛道演示")
    print("="*70)
    
    # 打印配置信息
    print_config_info(ICAPS_2021)
    
    # 创建环境
    print("正在创建环境...")
    manager = EnvManager(ICAPS_2021, seed=seed)
    env = manager.env
    
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
    
    # 显示告警相关信息
    print("\n告警系统信息:")
    if hasattr(env, 'alarms_lines_area'):
        print(f"  带告警区域的线路数: {len(env.alarms_lines_area)}")
    if hasattr(env, 'alarms_area_names'):
        print(f"  告警区域: {env.alarms_area_names}")
    
    # 检查注意力预算
    if hasattr(obs, 'attention_budget'):
        print(f"  初始注意力预算: {obs.attention_budget}")
    
    # 运行仿真
    print(f"\n运行仿真 {max_steps} 步...")
    total_reward = 0.0
    alarm_count = 0
    overflow_events = 0
    
    for step in range(max_steps):
        # 简单策略：当线路过载时发出告警
        action = manager.get_do_nothing_action()
        
        # 检查过载并考虑发出告警
        if obs.rho.max() > 1.0:
            overflow_events += 1
            
            # 检查是否可以发出告警
            if hasattr(obs, 'attention_budget') and obs.attention_budget[0] >= 1:
                if not obs.is_alarm_illegal:
                    # 找到过载线路及其区域
                    overloaded_lines = np.where(obs.rho > 1.0)[0]
                    zones_to_alert = get_zones_for_lines(env, obs, overloaded_lines)
                    
                    if zones_to_alert:
                        action.raise_alarm = zones_to_alert
                        alarm_count += 1
                        print(f"  [第 {step+1} 步] 为区域发出告警: {zones_to_alert}")
        
        # 执行环境步骤
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        
        # 检查问题
        if done:
            print(f"\n[第 {step+1} 步] 回合结束!")
            if 'exception' in info and info['exception']:
                print(f"  异常: {info['exception']}")
            break
        
        # 进度更新
        if (step + 1) % 20 == 0:
            budget_str = f", 预算={obs.attention_budget[0]}" if hasattr(obs, 'attention_budget') else ""
            print(f"  第 {step+1} 步: 最大rho={obs.rho.max():.2%}, 奖励={reward:.2f}{budget_str}")
    
    # 总结
    print("\n" + "-"*50)
    print("仿真总结:")
    print(f"  总步数: {step+1}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  过载事件数: {overflow_events}")
    print(f"  发出告警数: {alarm_count}")
    print("-"*50)
    
    manager.close()
    return total_reward


def get_zones_for_lines(env, obs, line_ids: List[int]) -> List[int]:
    """
    获取给定线路的告警区域
    
    参数:
        env: Grid2Op 环境
        obs: 当前观测
        line_ids: 线路 ID 列表
        
    返回:
        要告警的区域 ID 列表
    """
    if not hasattr(env, 'alarms_lines_area') or not hasattr(env, 'alarms_area_names'):
        return []
    
    zones_to_alert = set()
    zone_for_each_line = env.alarms_lines_area
    
    for line_id in line_ids:
        line_name = obs.name_line[line_id]
        if line_name in zone_for_each_line:
            for zone_name in zone_for_each_line[line_name]:
                zone_id = env.alarms_area_names.index(zone_name)
                zones_to_alert.add(zone_id)
    
    return list(zones_to_alert)


def demo_alarm_mechanism():
    """
    告警机制详细演示
    """
    print("\n" + "="*70)
    print("告警机制深入解析")
    print("="*70)
    
    manager = EnvManager(ICAPS_2021, seed=42)
    env = manager.env
    obs = manager.reset()
    
    # 解释告警系统
    print("\n告警系统概述:")
    print("-" * 40)
    print("""
ICAPS 2021 中的告警机制允许 AI 智能体与人类操作员沟通
潜在的电网问题。关键概念：

1. 注意力预算：发出告警的有限资源
2. 告警区域：可以告警的地理区域
3. 告警冷却：连续告警之间的时间间隔
4. 告警奖励：正确预测问题的奖励
""")
    
    # 显示告警区域
    if hasattr(env, 'alarms_area_names'):
        print("\n可用的告警区域:")
        for i, zone in enumerate(env.alarms_area_names):
            print(f"  区域 {i}: {zone}")
    
    # 显示线路到区域的映射
    if hasattr(env, 'alarms_lines_area'):
        print("\n线路到区域的映射（前10条线路）:")
        line_names = list(env.alarms_lines_area.keys())[:10]
        for line_name in line_names:
            zones = env.alarms_lines_area[line_name]
            print(f"  {line_name}: {zones}")
    
    # 显示注意力预算机制
    if hasattr(obs, 'attention_budget'):
        print(f"\n注意力预算机制:")
        print(f"  当前预算: {obs.attention_budget}")
        print(f"  预算会随时间恢复")
        print(f"  每次告警消耗 1 点预算")
    
    # 演示告警动作
    print("\n演示告警动作:")
    action = manager.get_do_nothing_action()
    
    if hasattr(env, 'alarms_area_names') and len(env.alarms_area_names) > 0:
        # 为第一个区域发出告警
        action.raise_alarm = [0]
        print(f"  为区域 0 发出告警: {env.alarms_area_names[0]}")
        
        # 检查动作是否有效
        if hasattr(obs, 'is_alarm_illegal'):
            print(f"  告警是否非法: {obs.is_alarm_illegal}")
        
        # 应用动作
        obs, reward, done, info = manager.step(action)
        print(f"  告警后奖励: {reward:.2f}")
        
        if hasattr(obs, 'attention_budget'):
            print(f"  剩余预算: {obs.attention_budget}")
    
    manager.close()


def demo_trust_based_agent_strategy():
    """
    基于信任的智能体策略演示
    """
    print("\n" + "="*70)
    print("基于信任的智能体策略演示")
    print("="*70)
    
    print("""
基于信任的智能体应该：
1. 持续监控电网状态
2. 在问题发生前预测潜在问题
3. 战略性地发出告警以最大化人类信任
4. 平衡自动化与人工监督

信任度关键指标：
- 告警准确性（真阳性 vs 假阳性）
- 响应时间（提前预警）
- 沟通清晰度（正确识别区域）
""")
    
    manager = EnvManager(ICAPS_2021, seed=42)
    obs = manager.reset()
    
    # 简单的基于信任的策略
    print("\n运行基于信任的策略 50 步...")
    
    alarm_threshold = 0.9  # 当 rho > 90% 时发出告警
    true_positives = 0
    false_positives = 0
    
    for step in range(50):
        action = manager.get_do_nothing_action()
        
        # 预测：是否即将发生过载？
        current_max_rho = obs.rho.max()
        
        if current_max_rho > alarm_threshold:
            # 预测到问题 - 发出告警
            if hasattr(obs, 'attention_budget') and obs.attention_budget[0] >= 1:
                if not obs.is_alarm_illegal:
                    overloaded = np.where(obs.rho > alarm_threshold)[0]
                    zones = get_zones_for_lines(manager.env, obs, overloaded)
                    if zones:
                        action.raise_alarm = zones
                        print(f"  [第 {step+1} 步] 警报! rho={current_max_rho:.2%}")
        
        obs, reward, done, info = manager.step(action)
        
        if done:
            break
    
    print(f"\n策略在 {step+1} 步后完成")
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="L2RPN ICAPS 2021 比赛演示"
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
        "--demo-alarm",
        action="store_true",
        help="运行详细的告警机制演示"
    )
    parser.add_argument(
        "--demo-strategy",
        action="store_true",
        help="运行基于信任的策略演示"
    )
    
    args = parser.parse_args()
    
    try:
        if args.demo_alarm:
            demo_alarm_mechanism()
        elif args.demo_strategy:
            demo_trust_based_agent_strategy()
        else:
            demo_icaps_2021(seed=args.seed, max_steps=args.max_steps)
            
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已安装所需的包:")
        print("  pip install grid2op lightsim2grid")
        print("\n并下载了环境数据:")
        print("  grid2op.make('l2rpn_icaps_2021')")


if __name__ == "__main__":
    main()
