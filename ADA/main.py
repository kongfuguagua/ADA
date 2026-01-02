# -*- coding: utf-8 -*-
"""
ADA Agent 主入口脚本
参考 ReAct_Baseline 和 OptLLM 的实现
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

# 导入 evaluate（支持直接运行和作为模块导入）
try:
    # 如果作为模块导入，使用相对导入
    from .evaluate import evaluate
except ImportError:
    # 如果直接运行，使用绝对导入
    try:
        from ADA.evaluate import evaluate
    except ImportError:
        from evaluate import evaluate

# 示例用法
if __name__ == "__main__":
    # 创建环境
    env = grid2op.make(
        "l2rpn_wcci_2022",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    
    # 评估 ADA Agent
    # 注意：需要配置 LLM API Key（通过环境变量或参数）
    # 环境变量: CLOUD_API_KEY, CLOUD_BASE_URL, CLOUD_MODEL
    logs_path="./result/wcci-2022/ada"
    res = evaluate(
        env,
        nb_episode=1,
        verbose=True,
        save_gif=True,  # 如果需要 GIF，设置为 True（需要 l2rpn_baselines）
        grid_name="IEEE118",
        # 场景选择：指定要运行的场景编号（可选）
        episode_id=[0],  # 指定场景编号列表
        # env_seeds=42,
        # env_seeds=[0, 1, 2, 3, 4, 5, 6],   # 指定环境随机种子（可选）
        rho_danger=0.95,  # 危险阈值：当负载率 > 95% 时调用 Judger
        rho_safe=0.85,    # 安全阈值：当负载率 < 85% 时使用快速通道
        max_planner_candidates=5,  # Planner 最大候选数
        max_llm_candidates=3,      # LLM 融合最大候选数
        enable_knowledge_base=True,  # 启用知识库
        llm_temperature=0.7,
        logs_path=logs_path
    )
    
    print("\n评估完成！")

