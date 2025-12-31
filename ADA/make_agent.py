# -*- coding: utf-8 -*-
"""
ADA Agent 工厂函数
用于 L2RPN 竞赛或 grid2game
"""

import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from grid2op.Environment import Environment

from ADA.agent import ADA_Agent
from utils import OpenAIChat, get_logger

logger = get_logger("ADA.make_agent")


def make_agent(
    env: Environment,
    dir_path: Optional[os.PathLike] = None,
    grid_name: str = "IEEE14",
    **kwargs
) -> ADA_Agent:
    """
    创建 ADA Agent 实例（用于 L2RPN 竞赛或 grid2game）
    
    Parameters
    ----------
    env : Environment
        The grid2op environment instance.
        
    dir_path : os.PathLike, optional
        Path to directory (用于加载模型，可选).
        
    grid_name : str, optional
        Grid identifier name. Used for local optimization.
        Options: "IEEE14", "IEEE118", "IEEE118_R2". Default: "IEEE14".
        
    **kwargs:
        其他参数传递给 ADA_Agent
        
    Returns
    -------
    ADA_Agent
        An instance of the ADA Agent.
    """
    # 初始化 LLM 客户端
    try:
        llm_client = OpenAIChat()
    except Exception as e:
        logger.warning(f"LLM 客户端初始化失败: {e}，将使用降级模式")
        llm_client = None
    
    # 创建 Agent
    agent = ADA_Agent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        llm_client=llm_client,
        env=env,  # 传递给 Solver 用于读取线路电抗
        grid_name=grid_name,
        **kwargs
    )
    
    # 加载模型（如果有）
    if dir_path:
        agent.load(dir_path)
    
    return agent

