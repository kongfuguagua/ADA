# -*- coding: utf-8 -*-
"""
LLM 配置管理
统一管理 API Key, Base URL, Model Name 等配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())


@dataclass
class LLMConfig:
    """LLM 服务配置类"""
    
    # 模型名称
    model_name: str = field(default_factory=lambda: os.getenv("CLOUD_MODEL", "deepseek-chat"))
    
    # API 配置
    api_key: str = field(default_factory=lambda: os.getenv("CLOUD_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com"))
    
    # Embedding 模型配置
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_API_MODEL", "text-embedding-3-small"))
    embedding_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        if not self.api_key:
            raise ValueError("API Key 未配置，请设置 CLOUD_API_KEY 环境变量")
        if not self.base_url:
            raise ValueError("Base URL 未配置，请设置 CLOUD_BASE_URL 环境变量")
        return True


# 全局单例
_llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """获取 LLM 配置单例"""
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config


def reset_llm_config(config: LLMConfig) -> None:
    """重置 LLM 配置（用于测试）"""
    global _llm_config
    _llm_config = config


# ============= 测试代码 =============
if __name__ == "__main__":
    config = get_llm_config()
    print("LLM 配置信息:")
    print(f"  模型: {config.model_name}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Temperature: {config.temperature}")
    print(f"  配置验证: {'通过' if config.api_key else '未配置 API Key'}")

