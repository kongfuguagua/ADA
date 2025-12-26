# -*- coding: utf-8 -*-
"""
LLM 配置管理
统一管理 API Key, Base URL, Model Name 等配置

使用方式:
    from config import LLMConfig
    config = LLMConfig()  # 每次创建新实例

环境变量说明：
    Chat 模型（CLOUD_* 前缀）:
        CLOUD_MODEL: Chat 模型名称
        CLOUD_API_KEY: Chat API Key
        CLOUD_BASE_URL: Chat API Base URL
    
    Embedding 模型（OPENAI_* 前缀）:
        OPENAI_API_MODEL: Embedding 模型名称
        OPENAI_API_KEY: Embedding API Key
        OPENAI_BASE_URL: Embedding API Base URL
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())


@dataclass
class LLMConfig:
    """LLM 服务配置类"""
    
    # Chat 模型配置（CLOUD_* 前缀）
    model_name: str = field(default_factory=lambda: os.getenv("CLOUD_MODEL", "deepseek-chat"))
    api_key: str = field(default_factory=lambda: os.getenv("CLOUD_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com"))
    
    # Embedding 模型配置（OPENAI_* 前缀，与 Chat 分离）
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_API_MODEL", "Pro/BAAI/bge-m3"))
    embedding_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"))
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_name": self.model_name,
            "api_key": self.api_key[:8] + "..." if self.api_key else "",  # 隐藏敏感信息
            "base_url": self.base_url,
            "embedding_model": self.embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        if not self.api_key:
            raise ValueError("Chat API Key 未配置，请设置 CLOUD_API_KEY 环境变量")
        if not self.base_url:
            raise ValueError("Chat Base URL 未配置，请设置 CLOUD_BASE_URL 环境变量")
        return True

    def validate_embedding(self) -> bool:
        """验证 Embedding 配置是否完整"""
        if not self.embedding_api_key:
            raise ValueError("Embedding API Key 未配置，请设置 OPENAI_API_KEY 环境变量")
        if not self.embedding_base_url:
            raise ValueError("Embedding Base URL 未配置，请设置 OPENAI_BASE_URL 环境变量")
        return True


# ============= 测试代码 =============
if __name__ == "__main__":
    config = LLMConfig()
    print("LLM 配置信息:")
    print(f"  Chat 模型: {config.model_name}")
    print(f"  Chat Base URL: {config.base_url}")
    print(f"  Embedding 模型: {config.embedding_model}")
    print(f"  Embedding Base URL: {config.embedding_base_url}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Chat 配置验证: {'通过' if config.api_key else '未配置 API Key'}")
    print(f"  Embedding 配置验证: {'通过' if config.embedding_api_key else '未配置 API Key'}")
