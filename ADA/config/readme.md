# ADA配置

本文件包括整个agent配置定义，每个组件的通用配置定义

llm_config.py: 统一管理 API Key, Base URL, Model Name（DeepSeek/GPT）。
system_config.py: 定义全局参数，如 Judger 的权重 $\alpha$，MCTS 的探索系数等。
prompts/: 虽然各 Agent 有自己的 prompt，但通用的系统设定（System Prompt）应在此统一管理。