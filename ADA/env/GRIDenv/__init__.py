# -*- coding: utf-8 -*-
"""
L2RPN 演示框架

用于初始化和交互 L2RPN（学习运行电网）比赛环境的模块化框架。

支持的比赛:
- NeurIPS 2020 赛道1（鲁棒性）
- NeurIPS 2020 赛道2（适应性）
- ICAPS 2021（信任）
- WCCI 2022（未来能源与碳中和）

快速开始:
    from demo import create_env, EnvManager
    
    # 创建环境
    env = create_env("wcci_2022", seed=42)
    
    # 或使用 EnvManager 获取更多功能
    manager = EnvManager("wcci_2022", seed=42)
    obs = manager.reset()
    manager.print_status()
"""

from config import (
    EnvConfig,          # 环境配置数据类
    Competition,        # 比赛枚举类
    NEURIPS_2020_TRACK1,  # NeurIPS 2020 赛道1 配置
    NEURIPS_2020_TRACK2,  # NeurIPS 2020 赛道2 配置
    ICAPS_2021,         # ICAPS 2021 配置
    WCCI_2022,          # WCCI 2022 配置
    SANDBOX_CASE14,     # 沙盒环境配置
    ENV_CONFIGS,        # 所有配置的字典
    get_config,         # 根据名称获取配置
    list_configs,       # 列出所有配置名称
    print_config_info,  # 打印配置详细信息
)

from env_factory import (
    create_env,         # 创建 Grid2Op 环境
    EnvManager,         # 高级环境管理器
    run_episode,        # 运行单个回合
)

__version__ = "1.0.0"
__author__ = "L2RPN Demo Framework"

# 导出的公共接口
__all__ = [
    # 配置相关
    "EnvConfig",            # 环境配置类
    "Competition",          # 比赛枚举
    "NEURIPS_2020_TRACK1",  # NeurIPS 2020 鲁棒性赛道
    "NEURIPS_2020_TRACK2",  # NeurIPS 2020 适应性赛道
    "ICAPS_2021",           # ICAPS 2021 信任赛道
    "WCCI_2022",            # WCCI 2022 碳中和赛道
    "SANDBOX_CASE14",       # 沙盒测试环境
    "ENV_CONFIGS",          # 配置注册表
    "get_config",           # 获取配置函数
    "list_configs",         # 列出配置函数
    "print_config_info",    # 打印配置信息函数
    # 环境工厂相关
    "create_env",           # 创建环境函数
    "EnvManager",           # 环境管理器类
    "run_episode",          # 运行回合函数
]
