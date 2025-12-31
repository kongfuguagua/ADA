# -*- coding: utf-8 -*-
"""
Planner Baseline Agent 使用示例
参考 OptimCVXPY 和 ExpertAgent 的 main.py
"""

import sys
from pathlib import Path

from grid2op.Backend import PandaPowerBackend

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

# 导入 evaluate（支持直接运行和作为模块导入）
try:
    # 如果作为模块导入，使用相对导入
    from .evaluate import evaluate
except ImportError:
    # 如果直接运行，使用绝对导入
    try:
        from Planner.evaluate import evaluate
    except ImportError:
        from evaluate import evaluate

# 示例用法
if __name__ == "__main__":
    # 创建环境
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        reward_class=RedispReward,
        backend=PandaPowerBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    
    # 评估 Agent
    # 注意：需要配置 LLM API Key（通过环境变量或参数）
    # 环境变量: CLOUD_API_KEY, CLOUD_BASE_URL, CLOUD_MODEL
    res = evaluate(
        env,
        nb_episode=7,
        verbose=True,
        save_gif=True,  # 如果需要 GIF，设置为 True（需要 l2rpn_baselines）
        max_react_steps=3,  # Planner 循环最大重试次数
        rho_danger=0.95,  # 启发式策略：当负载率超过 92% 时调用 LLM（预防性调度，避免等到过载）
        llm_temperature=0.7,
    )
    
    print("\n评估完成！")

