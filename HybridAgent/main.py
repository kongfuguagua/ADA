# -*- coding: utf-8 -*-
"""
HybridAgent 使用示例
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径，以便直接运行时可以导入同目录的模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 添加项目根目录到路径
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import grid2op
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

from evaluate import evaluate

# 示例用法
if __name__ == "__main__":
    # 创建环境
    env = grid2op.make(
        "l2rpn_neurips_2020_track2_small",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    logs_path="./result/neurips-2020/hybrid"
    # 使用 HybridAgent
    print("使用 HybridAgent (OptimCVXPY + LLM Topology)")
    
    # 评估 HybridAgent
    res = evaluate(
        env,
        nb_episode=7,
        verbose=True,
        save_gif=True,
        episode_id=[0,1,2,3,4,5,6],
        max_steps=-1,
        rho_safe=0.85,
        rho_danger=0.95,
        rho_llm_threshold=0.95,
        logs_path=logs_path
    )
    
    print("\n评估完成！")

