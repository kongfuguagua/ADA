# -*- coding: utf-8 -*-
"""
OptAgent / HybridAgent 使用示例
参考 OptimCVXPY 和 ExpertAgent 的 main.py
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

# 导入 HybridAgent
try:
    from .hybrid_agent import HybridAgent
except ImportError:
    try:
        from OptLLM.hybrid_agent import HybridAgent
    except ImportError:
        HybridAgent = None

# 导入 LLM 客户端
try:
    from ADA import OpenAIChat
except ImportError:
    OpenAIChat = None

# 示例用法
if __name__ == "__main__":
    # 创建环境
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
    
    # 选择使用 OptAgent 还是 HybridAgent
    use_hybrid = os.getenv("USE_HYBRID_AGENT", "false").lower() == "true"
    
    if use_hybrid and HybridAgent is not None and OpenAIChat is not None:
        # 使用 HybridAgent（混合架构）
        print("使用 HybridAgent (OptimCVXPY + LLM Topology)")
        
        # 初始化 LLM 客户端
        llm_client = OpenAIChat(
            api_key=os.getenv("CLOUD_API_KEY"),
            base_url=os.getenv("CLOUD_BASE_URL"),
            model=os.getenv("CLOUD_MODEL", "gpt-4"),
            temperature=0.7
        )
        
        # 创建 HybridAgent
        agent = HybridAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            env=env,
            llm_client=llm_client,
            rho_safe=0.85,
            rho_danger=0.95,
            rho_llm_threshold=1.05  # 优化后仍超过 105% 才激活 LLM
        )
        
        # 评估（使用自定义 agent）
        from grid2op.Runner import Runner
        runner = Runner(**env.get_params_for_runner(), agentClass=None)
        runner.init_agent(agent)
        
        res = runner.run(
            nb_episode=1,
            path_save="./logs-eval/hybrid-agent",
            max_iter=-1,
            # env_seeds=3,
            episode_id=[3]
        )
        
        # 打印统计信息
        stats = agent.get_stats()
        print("\n=== HybridAgent 统计信息 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    else:
        # 使用 OptAgent（原有模式）
        print("使用 OptAgent (Optimization-Augmented ReAct)")
        
        # 评估 OptAgent
        # 注意：需要配置 LLM API Key（通过环境变量或参数）
        # 环境变量: CLOUD_API_KEY, CLOUD_BASE_URL, CLOUD_MODEL
        res = evaluate(
            env,
            nb_episode=1,
            verbose=True,
            save_gif=True,  # 如果需要 GIF，设置为 True（需要 l2rpn_baselines）
            # 场景选择：指定要运行的场景编号（可选）
            episode_id=[3],  # 指定场景编号列表
            # env_seeds=[0, 1, 2, 3, 4, 5, 6],   # 指定环境随机种子（可选）
            max_react_steps=3,  # OptAgent 最大重试次数
            rho_danger=0.9,  # 启发式策略：当负载率 >= 95% 时调用 LLM（预防性调度，避免等到过载）
            llm_temperature=0.7,
        )
    
    print("\n评估完成！")

