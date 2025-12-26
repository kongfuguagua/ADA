# -*- coding: utf-8 -*-
"""
L2RPN Demo 框架完整测试

测试所有环境配置和基本功能

使用方法:
    conda activate ada
    python test/test_demo.py
"""

import sys
import os

# 设置 UTF-8 输出（解决 Windows 编码问题）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加 demo 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

from config import (
    ENV_CONFIGS, 
    get_config, 
    list_configs, 
    print_config_info,
    NEURIPS_2020_TRACK1,
    NEURIPS_2020_TRACK2,
    ICAPS_2021,
    WCCI_2022,
    SANDBOX_CASE14,
)
from env_factory import create_env, EnvManager


def test_config_module():
    """测试配置模块"""
    print("\n" + "="*60)
    print("测试 1: 配置模块")
    print("="*60)
    
    # 测试 list_configs
    configs = list_configs()
    print(f"可用配置数量: {len(configs)}")
    print(f"配置列表: {configs}")
    assert len(configs) >= 5, "应该至少有5个配置"
    
    # 测试 get_config
    for name in configs:
        config = get_config(name)
        print(f"  [OK] {name}: {config.env_name}")
        assert config.env_name is not None
        assert config.name is not None
    
    # 测试预定义配置
    print("\n预定义配置:")
    print(f"  NEURIPS_2020_TRACK1: {NEURIPS_2020_TRACK1.env_name}")
    print(f"  NEURIPS_2020_TRACK2: {NEURIPS_2020_TRACK2.env_name}")
    print(f"  ICAPS_2021: {ICAPS_2021.env_name}")
    print(f"  WCCI_2022: {WCCI_2022.env_name}")
    print(f"  SANDBOX_CASE14: {SANDBOX_CASE14.env_name}")
    
    print("\n[PASS] 配置模块测试通过!")
    return True


def test_environment_creation(env_name: str):
    """测试单个环境创建"""
    print(f"\n  测试环境: {env_name}")
    
    try:
        # 创建环境
        manager = EnvManager(env_name, seed=42)
        env = manager.env
        
        print(f"    环境标识: {manager.config.env_name}")
        print(f"    线路数: {env.n_line}")
        print(f"    变电站数: {env.n_sub}")
        print(f"    发电机数: {env.n_gen}")
        print(f"    负荷数: {env.n_load}")
        
        # 检查储能
        if hasattr(env, 'n_storage') and env.n_storage > 0:
            print(f"    储能单元数: {env.n_storage}")
        
        # 重置环境
        obs = manager.reset()
        assert obs is not None, "观测不应为空"
        
        # 获取电网信息
        grid_info = manager.get_grid_info()
        assert 'n_line' in grid_info
        assert 'n_sub' in grid_info
        
        # 获取动作空间信息
        action_info = manager.get_action_space_info()
        assert 'action_size' in action_info
        
        # 执行几步
        total_reward = 0
        for step in range(5):
            action = manager.get_do_nothing_action()
            obs, reward, done, info = manager.step(action)
            total_reward += reward
            if done:
                print(f"    回合在第 {step+1} 步结束")
                break
        
        print(f"    5步总奖励: {total_reward:.2f}")
        print(f"    最终最大rho: {obs.rho.max():.2%}")
        
        # 测试模拟功能
        action = manager.get_do_nothing_action()
        sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
        print(f"    模拟测试: 成功")
        
        manager.close()
        print(f"    [OK] {env_name} 测试通过!")
        return True
        
    except Exception as e:
        print(f"    [FAIL] {env_name} 测试失败: {e}")
        return False


def test_all_environments():
    """测试所有环境"""
    print("\n" + "="*60)
    print("测试 2: 环境创建和基本操作")
    print("="*60)
    
    results = {}
    
    # 测试所有配置的环境
    for env_name in list_configs():
        results[env_name] = test_environment_creation(env_name)
    
    # 总结
    print("\n" + "-"*60)
    print("环境测试总结:")
    print("-"*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


def test_special_features():
    """测试特殊功能"""
    print("\n" + "="*60)
    print("测试 3: 特殊功能测试")
    print("="*60)
    
    all_passed = True
    
    # 测试 WCCI 2022 的储能功能
    print("\n测试 WCCI 2022 储能功能:")
    try:
        manager = EnvManager("wcci_2022", seed=42)
        env = manager.env
        obs = manager.reset()
        
        if env.n_storage > 0:
            print(f"  储能单元数: {env.n_storage}")
            print(f"  储能容量: {env.storage_Emax}")
            print(f"  当前电量: {obs.storage_charge}")
            
            # 测试储能动作
            import numpy as np
            action = manager.get_do_nothing_action()
            storage_power = np.zeros(env.n_storage)
            storage_power[0] = 2.0  # 吸收 2 MW
            action.storage_p = storage_power
            
            sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
            if not sim_done:
                print(f"  储能动作模拟: 成功")
            
            print("  [PASS] 储能功能测试通过!")
        else:
            print("  该环境无储能单元")
        
        manager.close()
        
    except Exception as e:
        print(f"  [FAIL] 储能功能测试失败: {e}")
        all_passed = False
    
    # 测试 ICAPS 2021 的告警功能
    print("\n测试 ICAPS 2021 告警功能:")
    try:
        manager = EnvManager("icaps_2021", seed=42)
        env = manager.env
        obs = manager.reset()
        
        if hasattr(env, 'alarms_area_names'):
            print(f"  告警区域数: {len(env.alarms_area_names)}")
            if hasattr(obs, 'attention_budget'):
                print(f"  注意力预算: {obs.attention_budget}")
            print("  [PASS] 告警功能测试通过!")
        else:
            print("  该环境无告警功能")
        
        manager.close()
        
    except Exception as e:
        print(f"  [FAIL] 告警功能测试失败: {e}")
        all_passed = False
    
    # 测试再调度功能
    print("\n测试再调度功能:")
    try:
        manager = EnvManager("neurips_2020_track1", seed=42)
        env = manager.env
        obs = manager.reset()
        
        import numpy as np
        if hasattr(env, 'gen_redispatchable'):
            redisp_ids = np.where(env.gen_redispatchable)[0]
            print(f"  可再调度发电机数: {len(redisp_ids)}")
            
            if len(redisp_ids) > 0:
                # 测试再调度动作
                action = manager.get_do_nothing_action()
                gen_id = redisp_ids[0]
                ramp_up = env.gen_max_ramp_up[gen_id]
                action.redispatch = [(gen_id, ramp_up / 4)]
                
                sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action)
                if not sim_done:
                    print(f"  再调度动作模拟: 成功")
            
            print("  [PASS] 再调度功能测试通过!")
        
        manager.close()
        
    except Exception as e:
        print(f"  [FAIL] 再调度功能测试失败: {e}")
        all_passed = False
    
    return all_passed


def test_main_entry():
    """测试主入口"""
    print("\n" + "="*60)
    print("测试 4: 主入口测试")
    print("="*60)
    
    from main import list_environments, compare_environments
    
    print("\n列出环境:")
    list_environments()
    
    print("\n比较环境:")
    compare_environments()
    
    print("\n[PASS] 主入口测试通过!")
    return True


def test_simulation_feature():
    """测试模拟功能"""
    print("\n" + "="*60)
    print("测试 5: 模拟功能测试")
    print("="*60)
    
    import numpy as np
    
    try:
        manager = EnvManager("sandbox_case14", seed=42)
        obs = manager.reset()
        
        print(f"  初始最大rho: {obs.rho.max():.2%}")
        
        # 测试多步模拟
        action = manager.get_do_nothing_action()
        
        for time_step in [1, 2, 3]:
            sim_obs, sim_reward, sim_done, sim_info = manager.simulate(action, time_step=time_step)
            if not sim_done:
                print(f"  模拟 {time_step} 步后最大rho: {sim_obs.rho.max():.2%}")
        
        # 测试拓扑动作模拟
        print("\n  测试拓扑动作模拟:")
        topo_action = manager.env.action_space({
            "set_bus": {"substations_id": [(1, [1, 1, 1, 1, 1, 1])]}
        })
        sim_obs, sim_reward, sim_done, sim_info = manager.simulate(topo_action)
        if not sim_done:
            print(f"    拓扑动作后最大rho: {sim_obs.rho.max():.2%}")
        
        manager.close()
        print("\n  [PASS] 模拟功能测试通过!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 模拟功能测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("L2RPN Demo 框架完整测试")
    print("="*60)
    print("测试环境: conda activate ada")
    print("="*60)
    
    results = []
    
    # 测试 1: 配置模块
    try:
        results.append(("配置模块", test_config_module()))
    except Exception as e:
        print(f"配置模块测试异常: {e}")
        results.append(("配置模块", False))
    
    # 测试 2: 环境创建
    try:
        results.append(("环境创建", test_all_environments()))
    except Exception as e:
        print(f"环境创建测试异常: {e}")
        results.append(("环境创建", False))
    
    # 测试 3: 特殊功能
    try:
        results.append(("特殊功能", test_special_features()))
    except Exception as e:
        print(f"特殊功能测试异常: {e}")
        results.append(("特殊功能", False))
    
    # 测试 4: 主入口
    try:
        results.append(("主入口", test_main_entry()))
    except Exception as e:
        print(f"主入口测试异常: {e}")
        results.append(("主入口", False))
    
    # 测试 5: 模拟功能
    try:
        results.append(("模拟功能", test_simulation_feature()))
    except Exception as e:
        print(f"模拟功能测试异常: {e}")
        results.append(("模拟功能", False))
    
    # 最终总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, v in results if v)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n所有测试通过!")
    else:
        print(f"\n警告: {total - passed} 个测试失败")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
