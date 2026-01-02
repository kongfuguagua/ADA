#!/usr/bin/env python3
# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

"""
PPO_SB3 评估脚本

约定：
- 模型路径: ./rl_saved_model/{name}/
- 日志路径: ./logs-eval/ppo-sb3-baseline/
"""

import os
import sys

# 添加当前目录到路径（支持本地运行）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import grid2op
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

# 导入评估函数（优先本地，回退到包）
try:
    from evaluate import evaluate
except ImportError:
    try:
        from l2rpn_baselines.PPO_SB3 import evaluate
    except ImportError:
        raise ImportError("无法导入 evaluate 函数")


# ============================================================================
# 约定配置
# ============================================================================
MODEL_NAME = "PPO_SB3"                    # 模型名称（与训练时一致）
MODEL_LOAD_PATH = "./rl_saved_model"      # 模型根目录
LOGS_PATH = "./logs-eval/ppo-sb3-baseline"  # 评估日志目录
NB_EPISODE = 7                            # 评估回合数
# ============================================================================


if __name__ == "__main__":
    print("="*60)
    print("PPO_SB3 模型评估")
    print("="*60)
    
    # 1. 创建环境
    print("\n[1/3] 创建环境...")
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    print("✓ 环境创建成功")
    
    # 2. 评估智能体
    print(f"\n[2/3] 加载模型: {MODEL_LOAD_PATH}/{MODEL_NAME}")
    try:
        trained_agent, res = evaluate(
            env=env,
            load_path=MODEL_LOAD_PATH,
            name=MODEL_NAME,
            logs_path=LOGS_PATH,
            nb_episode=NB_EPISODE,
            verbose=True,
            save_gif=False,  # 调试时可设为 False 以加快速度
        )
        print("\n[3/3] ✓ 评估完成！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到模型文件")
        print(f"   请确保模型已训练并保存在: {os.path.abspath(MODEL_LOAD_PATH)}/{MODEL_NAME}/")
        print(f"   详细错误: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        print("\n建议检查:")
        print("  1. 模型文件是否存在")
        print("  2. 是否安装了 stable-baselines3")
        print("  3. 观察空间/动作空间是否与训练时一致")
        raise
        
    finally:
        env.close()
        print("\n环境已关闭")
