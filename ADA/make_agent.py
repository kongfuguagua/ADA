# -*- coding: utf-8 -*-
"""
创建 ADAgent 实例的工厂函数
参考 ExpertAgent 和 OptimCVXPY 的实现
"""

import os
from typing import Optional
from grid2op.Environment import Environment

from ADAgent import ADAgent
from config import SystemConfig, LLMConfig


def make_agent(
    env: Environment,
    dir_path: os.PathLike,
    name: str = "ADAgent",
    system_config: Optional[SystemConfig] = None,
    llm_config: Optional[LLMConfig] = None,
    rho_safe: float = 0.85,
    rho_danger: float = 0.95,
    **kwargs
) -> ADAgent:
    """
    创建 ADAgent 实例
    
    参考 ExpertAgent.make_agent() 和 OptimCVXPY.make_agent() 的实现
    
    Parameters
    ----------
    env : Environment
        Grid2Op 环境实例
    
    dir_path : os.PathLike
        保存路径（用于知识库等）
    
    name : str, optional
        智能体名称，默认为 "ADAgent"
    
    system_config : SystemConfig, optional
        系统配置（如果为 None，则使用默认配置）
    
    llm_config : LLMConfig, optional
        LLM 配置（如果为 None，则使用默认配置）
    
    rho_safe : float, optional
        安全状态阈值（参考 OptimCVXPY），默认 0.85
    
    rho_danger : float, optional
        危险状态阈值（参考 OptimCVXPY），默认 0.95
    
    **kwargs
        其他参数
    
    Returns
    -------
    ADAgent
        ADAgent 实例
    """
    # 加载配置
    if system_config is None:
        system_config = SystemConfig()
    
    if llm_config is None:
        llm_config = LLMConfig()
    
    # 创建智能体（ADAgent 内部包含完整的 ADA 逻辑）
    agent = ADAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        name=name,
        system_config=system_config,
        llm_config=llm_config,
        rho_safe=rho_safe,
        rho_danger=rho_danger,
        **kwargs
    )
    
    return agent

