# -*- coding: utf-8 -*-
"""
L2RPN WCCI 2022 比赛演示

本脚本演示如何初始化和交互 WCCI 2022 L2RPN 比赛环境（未来能源）。

关键特性：
- 储能单元用于能源管理
- 可再生能源发电机的弃风控制
- 连续动作空间（再调度、弃风、储能）
- 离散动作空间（拓扑变更）

使用方法:
    python demo_wcci_2022.py
    python demo_wcci_2022.py --demo-storage
    python demo_wcci_2022.py --demo-curtailment
"""

import argparse
import numpy as np
from typing import Optional

from config import WCCI_2022, print_config_info
from env_factory import EnvManager


def demo_wcci_2022(seed: Optional[int] = 42, max_steps: int = 100):
    """
    WCCI 2022（未来能源）演示
    
    该比赛引入储能单元和弃风功能，
    用于管理可再生能源的不确定性。
    """
    print("\n" + "="*70)
    print("WCCI 2022 - 未来能源与碳中和演示")
    print("="*70)
    
    # 打印配置信息
    print_config_info(WCCI_2022)
    
    # 创建环境
    print("正在创建环境...")
    manager = EnvManager(WCCI_2022, seed=seed)
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
    print(f"  储能单元数: {grid_info['n_storage']}")
    
    # 可再生能源发电机
    if 'gen_renewable' in grid_info:
        n_renewable = sum(grid_info['gen_renewable'])
        print(f"  可再生能源发电机数: {n_renewable}")
    
    # 储能信息
    print("\n储能信息:")
    if hasattr(env, 'storage_Emax'):
        print(f"  最大容量 (Emax): {env.storage_Emax} MWh")
    if hasattr(env, 'storage_max_p_prod'):
        print(f"  最大放电功率: {env.storage_max_p_prod} MW")
    if hasattr(env, 'storage_max_p_absorb'):
        print(f"  最大充电功率: {env.storage_max_p_absorb} MW")
    
    # 初始储能状态
    if hasattr(obs, 'storage_charge'):
        print(f"  当前电量: {obs.storage_charge} MWh")
    
    # 运行仿真
    print(f"\n运行仿真 {max_steps} 步...")
    total_reward = 0.0
    storage_history = []
    renewable_history = []
    
    for step in range(max_steps):
        # 演示使用"什么都不做"动作
        action = manager.get_do_nothing_action()
        
        # 执行环境步骤
        obs, reward, done, info = manager.step(action)
        total_reward += reward
        
        # 跟踪储能和可再生能源发电
        if hasattr(obs, 'storage_charge'):
            storage_history.append(obs.storage_charge.sum())
        
        # 跟踪可再生能源发电
        if hasattr(env, 'gen_renewable'):
            renewable_gen = obs.gen_p[env.gen_renewable].sum()
            renewable_history.append(renewable_gen)
        
        # 检查问题
        if done:
            print(f"\n[第 {step+1} 步] 回合结束!")
            if 'exception' in info and info['exception']:
                print(f"  异常: {info['exception']}")
            break
        
        # 进度更新
        if (step + 1) % 20 == 0:
            storage_str = f", 储能={obs.storage_charge.sum():.1f}MWh" if hasattr(obs, 'storage_charge') else ""
            print(f"  第 {step+1} 步: 最大rho={obs.rho.max():.2%}, 奖励={reward:.2f}{storage_str}")
    
    # 总结
    print("\n" + "-"*50)
    print("仿真总结:")
    print(f"  总步数: {step+1}")
    print(f"  总奖励: {total_reward:.2f}")
    
    if storage_history:
        print(f"  平均储能电量: {np.mean(storage_history):.1f} MWh")
        print(f"  储能电量范围: {np.min(storage_history):.1f} - {np.max(storage_history):.1f} MWh")
    
    if renewable_history:
        print(f"  平均可再生能源发电: {np.mean(renewable_history):.1f} MW")
        print(f"  可再生能源波动（标准差）: {np.std(renewable_history):.1f} MW")
    
    print("-"*50)
    
    manager.close()
    return total_reward


def demo_storage_actions():
    """
    储能单元控制动作演示
    """
    print("\n" + "="*70)
    print("储能控制演示")
    print("="*70)
    
    manager = EnvManager(WCCI_2022, seed=42)
    env = manager.env
    obs = manager.reset()
    
    print("\n储能单元概述:")
    print("-" * 40)
    
    n_storage = env.n_storage
    print(f"储能单元数量: {n_storage}")
    
    if n_storage == 0:
        print("该环境中没有储能单元。")
        manager.close()
        return
    
    # 储能属性
    print(f"\n储能属性:")
    print(f"  Emax（容量）: {env.storage_Emax}")
    print(f"  最大放电功率: {env.storage_max_p_prod}")
    print(f"  最大充电功率: {env.storage_max_p_absorb}")
    print(f"  效率: {getattr(env, 'storage_efficiency', '不可用')}")
    
    # 当前状态
    print(f"\n当前储能状态:")
    print(f"  电量: {obs.storage_charge}")
    print(f"  功率: {obs.storage_power}")
    
    # 演示储能动作
    print("\n演示储能动作:")
    
    # 1. 充电储能（吸收功率）
    print("\n1. 充电储能（吸收 5 MW）:")
    action = manager.get_do_nothing_action()
    storage_power = np.zeros(n_storage)
    storage_power[0] = 5.0  # 吸收 5 MW
    action.storage_p = storage_power
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   充电前: 电量={obs.storage_charge[0]:.2f} MWh")
        print(f"   充电后: 电量={sim_obs.storage_charge[0]:.2f} MWh")
        print(f"   模拟后最大rho: {sim_obs.rho.max():.2%}")
    
    # 2. 放电储能（产生功率）
    print("\n2. 放电储能（输出 5 MW）:")
    action = manager.get_do_nothing_action()
    storage_power = np.zeros(n_storage)
    storage_power[0] = -5.0  # 输出 5 MW
    action.storage_p = storage_power
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   放电前: 电量={obs.storage_charge[0]:.2f} MWh")
        print(f"   放电后: 电量={sim_obs.storage_charge[0]:.2f} MWh")
        print(f"   模拟后最大rho: {sim_obs.rho.max():.2%}")
    
    manager.close()


def demo_curtailment_actions():
    """
    可再生能源发电机弃风动作演示
    """
    print("\n" + "="*70)
    print("弃风控制演示")
    print("="*70)
    
    manager = EnvManager(WCCI_2022, seed=42)
    env = manager.env
    obs = manager.reset()
    
    print("\n可再生能源发电机概述:")
    print("-" * 40)
    
    # 找可再生能源发电机
    if not hasattr(env, 'gen_renewable'):
        print("没有可用的可再生能源发电机信息。")
        manager.close()
        return
    
    renewable_mask = env.gen_renewable
    renewable_ids = np.where(renewable_mask)[0]
    n_renewable = len(renewable_ids)
    
    print(f"总发电机数: {env.n_gen}")
    print(f"可再生能源发电机数: {n_renewable}")
    print(f"可再生能源发电机 ID: {renewable_ids.tolist()}")
    
    if n_renewable == 0:
        print("没有可弃风的可再生能源发电机。")
        manager.close()
        return
    
    # 当前可再生能源发电
    print(f"\n当前可再生能源发电:")
    for gen_id in renewable_ids[:5]:  # 显示前5个
        print(f"  发电机 {gen_id}: {obs.gen_p[gen_id]:.2f} MW")
    
    # 演示弃风
    print("\n演示弃风动作:")
    
    # 1. 将可再生能源发电机弃风到 50%
    gen_id = renewable_ids[0]
    current_gen = obs.gen_p[gen_id]
    
    print(f"\n1. 将发电机 {gen_id} 弃风到 50%:")
    action = manager.get_do_nothing_action()
    
    # 弃风指定为比率（0 到 1）
    curtail = np.ones(env.n_gen) * (-1)  # -1 表示不改变
    curtail[gen_id] = 0.5  # 限制到最大容量的 50%
    action.curtail = curtail
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   弃风前: {current_gen:.2f} MW")
        print(f"   弃风后: {sim_obs.gen_p[gen_id]:.2f} MW")
        print(f"   模拟后最大rho: {sim_obs.rho.max():.2%}")
    
    # 2. 完全弃风（0%）
    print(f"\n2. 完全弃风发电机 {gen_id}:")
    action = manager.get_do_nothing_action()
    curtail = np.ones(env.n_gen) * (-1)
    curtail[gen_id] = 0.0  # 完全弃风
    action.curtail = curtail
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   弃风前: {current_gen:.2f} MW")
        print(f"   弃风后: {sim_obs.gen_p[gen_id]:.2f} MW")
    
    # 3. 取消弃风（100%）
    print(f"\n3. 取消弃风（允许 100%）:")
    action = manager.get_do_nothing_action()
    curtail = np.ones(env.n_gen) * (-1)
    curtail[gen_id] = 1.0  # 不弃风
    action.curtail = curtail
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   取消后: {sim_obs.gen_p[gen_id]:.2f} MW")
    
    manager.close()


def demo_redispatching_actions():
    """
    再调度动作演示
    """
    print("\n" + "="*70)
    print("再调度控制演示")
    print("="*70)
    
    manager = EnvManager(WCCI_2022, seed=42)
    env = manager.env
    obs = manager.reset()
    
    print("\n可再调度发电机概述:")
    print("-" * 40)
    
    # 找可再调度发电机
    if not hasattr(env, 'gen_redispatchable'):
        print("没有可用的再调度信息。")
        manager.close()
        return
    
    redisp_mask = env.gen_redispatchable
    redisp_ids = np.where(redisp_mask)[0]
    n_redisp = len(redisp_ids)
    
    print(f"总发电机数: {env.n_gen}")
    print(f"可再调度发电机数: {n_redisp}")
    print(f"可再调度发电机 ID: {redisp_ids.tolist()}")
    
    if n_redisp == 0:
        print("没有可再调度的发电机。")
        manager.close()
        return
    
    # 显示爬坡限制
    print(f"\n爬坡限制:")
    for gen_id in redisp_ids[:5]:  # 显示前5个
        print(f"  发电机 {gen_id}: 上爬坡={env.gen_max_ramp_up[gen_id]:.1f} MW, "
              f"下爬坡={env.gen_max_ramp_down[gen_id]:.1f} MW")
    
    # 演示再调度
    print("\n演示再调度动作:")
    
    gen_id = redisp_ids[0]
    ramp_up = env.gen_max_ramp_up[gen_id]
    
    # 1. 增加发电
    print(f"\n1. 增加发电机 {gen_id} 的发电量 {ramp_up/2:.1f} MW:")
    action = manager.get_do_nothing_action()
    action.redispatch = [(gen_id, ramp_up / 2)]
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   调度前: {obs.gen_p[gen_id]:.2f} MW")
        print(f"   调度后: {sim_obs.gen_p[gen_id]:.2f} MW")
        print(f"   目标调度量: {sim_obs.target_dispatch[gen_id]:.2f} MW")
    
    # 2. 减少发电
    ramp_down = env.gen_max_ramp_down[gen_id]
    print(f"\n2. 减少发电机 {gen_id} 的发电量 {ramp_down/2:.1f} MW:")
    action = manager.get_do_nothing_action()
    action.redispatch = [(gen_id, -ramp_down / 2)]
    
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    if not sim_done:
        print(f"   调度前: {obs.gen_p[gen_id]:.2f} MW")
        print(f"   调度后: {sim_obs.gen_p[gen_id]:.2f} MW")
    
    manager.close()


def demo_combined_actions():
    """
    组合离散和连续动作演示
    """
    print("\n" + "="*70)
    print("组合动作演示")
    print("="*70)
    
    print("""
WCCI 2022 允许组合使用：
- 离散动作：拓扑变更、线路连接
- 连续动作：再调度、弃风、储能

典型策略：
1. 使用离散动作进行主要拓扑变更
2. 使用连续动作微调潮流
""")
    
    manager = EnvManager(WCCI_2022, seed=42)
    env = manager.env
    obs = manager.reset()
    
    print("\n创建组合动作:")
    action = manager.get_do_nothing_action()
    
    # 添加储能动作
    if env.n_storage > 0:
        storage_power = np.zeros(env.n_storage)
        storage_power[0] = 2.0  # 吸收 2 MW
        action.storage_p = storage_power
        print("  - 储能: 吸收 2 MW")
    
    # 添加再调度
    if hasattr(env, 'gen_redispatchable'):
        redisp_ids = np.where(env.gen_redispatchable)[0]
        if len(redisp_ids) > 0:
            action.redispatch = [(redisp_ids[0], 5.0)]
            print(f"  - 再调度: 发电机 {redisp_ids[0]} 增加 5 MW")
    
    # 模拟组合动作
    print("\n模拟组合动作:")
    sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
    
    if not sim_done:
        print(f"  模拟后最大rho: {sim_obs.rho.max():.2%}")
        print(f"  模拟奖励: {sim_reward:.2f}")
    else:
        print("  动作导致游戏结束!")
    
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="L2RPN WCCI 2022 比赛演示"
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
        "--demo-storage",
        action="store_true",
        help="运行储能控制演示"
    )
    parser.add_argument(
        "--demo-curtailment",
        action="store_true",
        help="运行弃风控制演示"
    )
    parser.add_argument(
        "--demo-redispatch",
        action="store_true",
        help="运行再调度控制演示"
    )
    parser.add_argument(
        "--demo-combined",
        action="store_true",
        help="运行组合动作演示"
    )
    
    args = parser.parse_args()
    
    try:
        if args.demo_storage:
            demo_storage_actions()
        elif args.demo_curtailment:
            demo_curtailment_actions()
        elif args.demo_redispatch:
            demo_redispatching_actions()
        elif args.demo_combined:
            demo_combined_actions()
        else:
            demo_wcci_2022(seed=args.seed, max_steps=args.max_steps)
            
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已安装所需的包:")
        print("  pip install grid2op lightsim2grid cvxpy")
        print("\n并下载了环境数据:")
        print("  grid2op.make('l2rpn_wcci_2022')")


if __name__ == "__main__":
    main()
