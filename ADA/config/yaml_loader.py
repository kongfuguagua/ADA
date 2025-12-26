# -*- coding: utf-8 -*-
"""
YAML 配置加载器
从 YAML 文件加载配置并更新 SystemConfig 和 LLMConfig
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml as pyyaml

from .system_config import SystemConfig
from .llm_config import LLMConfig


def expand_env_vars(value: Any) -> Any:
    """
    展开环境变量引用
    
    支持格式: "${VAR_NAME}" 或 "${VAR_NAME:default_value}"
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            var_expr = value[2:-1]
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
                return os.getenv(var_name.strip(), default.strip())
            else:
                var_name = var_expr.strip()
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(f"环境变量 {var_name} 未设置")
                return env_value
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置
    
    Args:
        yaml_path: YAML 文件路径
    
    Returns:
        配置字典
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config_dict = pyyaml.safe_load(f)
    
    # 展开环境变量
    config_dict = expand_env_vars(config_dict)
    
    return config_dict


def create_system_config_from_yaml(config_dict: Dict[str, Any]) -> SystemConfig:
    """
    从 YAML 配置字典创建 SystemConfig
    
    Args:
        config_dict: YAML 配置字典
    
    Returns:
        SystemConfig 实例
    """
    system_config = config_dict.get("system", {})
    
    config = SystemConfig()
    
    # 更新系统配置
    if "max_retries" in system_config:
        config.max_retries = system_config["max_retries"]
    if "log_level" in system_config:
        config.log_level = system_config["log_level"]
    if "log_dir" in system_config:
        config.log_dir = system_config["log_dir"]
    if "knowledge_storage_path" in system_config:
        config.knowledge_storage_path = system_config["knowledge_storage_path"]
    if "knowledge_top_k" in system_config:
        config.knowledge_top_k = system_config["knowledge_top_k"]
    if "knowledge_chunk_size" in system_config:
        config.knowledge_chunk_size = system_config["knowledge_chunk_size"]
    if "knowledge_chunk_overlap" in system_config:
        config.knowledge_chunk_overlap = system_config["knowledge_chunk_overlap"]
    
    # 更新 Agent 配置
    agents_config = config_dict.get("agents", {})
    
    # 保存完整的 agents 配置到 config 对象（供 orchestrator 使用）
    config._agents_config = agents_config
    
    if "planner" in agents_config:
        planner_config = agents_config["planner"]
        if "max_augmentation_steps" in planner_config:
            config.planner_max_augmentation_steps = planner_config["max_augmentation_steps"]
    
    if "solver" in agents_config:
        solver_config = agents_config["solver"]
        if "timeout" in solver_config:
            config.solver_timeout = solver_config["timeout"]
    
    if "judger" in agents_config:
        judger_config = agents_config["judger"]
        if "alpha" in judger_config:
            config.judger_alpha = judger_config["alpha"]
        if "pass_threshold" in judger_config:
            config.judger_pass_threshold = judger_config["pass_threshold"]
    
    if "summarizer" in agents_config:
        summarizer_config = agents_config["summarizer"]
        if "mcts_exploration_constant" in summarizer_config:
            config.mcts_exploration_constant = summarizer_config["mcts_exploration_constant"]
        if "min_score_threshold" in summarizer_config:
            config.summarizer_min_score_threshold = summarizer_config["min_score_threshold"]
    
    return config


def create_llm_config_from_yaml(config_dict: Dict[str, Any]) -> LLMConfig:
    """
    从 YAML 配置字典创建 LLMConfig
    
    Args:
        config_dict: YAML 配置字典
    
    Returns:
        LLMConfig 实例
    """
    llm_config = config_dict.get("llm", {})
    
    config = LLMConfig()
    
    # Chat 配置
    if "chat" in llm_config:
        chat_config = llm_config["chat"]
        if "model" in chat_config:
            config.model_name = chat_config["model"]
        if "api_key" in chat_config:
            config.api_key = chat_config["api_key"]
        if "base_url" in chat_config:
            config.base_url = chat_config["base_url"]
        if "temperature" in chat_config:
            config.temperature = chat_config["temperature"]
        if "max_tokens" in chat_config:
            config.max_tokens = chat_config["max_tokens"]
    
    # Embedding 配置
    if "embedding" in llm_config:
        embedding_config = llm_config["embedding"]
        if "model" in embedding_config:
            config.embedding_model = embedding_config["model"]
        if "api_key" in embedding_config:
            config.embedding_api_key = embedding_config["api_key"]
        if "base_url" in embedding_config:
            config.embedding_base_url = embedding_config["base_url"]
    
    return config


# ============= 测试代码 =============
if __name__ == "__main__":
    # 测试配置加载
    test_yaml = Path(__file__).parent.parent / "yaml" / "default.yaml"
    
    if test_yaml.exists():
        print(f"加载配置文件: {test_yaml}")
        config_dict = load_config_from_yaml(str(test_yaml))
        
        print("\n系统配置:")
        sys_config = create_system_config_from_yaml(config_dict)
        print(f"  最大重试次数: {sys_config.max_retries}")
        print(f"  Judger Alpha: {sys_config.judger_alpha}")
        
        print("\nLLM 配置:")
        llm_config = create_llm_config_from_yaml(config_dict)
        print(f"  Chat 模型: {llm_config.model_name}")
        print(f"  Embedding 模型: {llm_config.embedding_model}")
    else:
        print(f"测试配置文件不存在: {test_yaml}")

