# -*- coding: utf-8 -*-
"""
Embedding 服务封装
支持 OpenAI 兼容的 Embedding API

注意：此模块不包含 Mock 实现，仅提供真实 Embedding 服务。
如果 API 配置无效，将抛出异常而非静默失败。

环境变量配置：
    OPENAI_API_MODEL: Embedding 模型名称（如 Pro/BAAI/bge-m3）
    OPENAI_API_KEY: Embedding API Key
    OPENAI_BASE_URL: Embedding API Base URL
"""

import os
import math
from typing import List
from abc import ABC, abstractmethod

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class BaseEmbeddings(ABC):
    """Embedding 服务基类"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
        
        Returns:
            向量表示
        
        Raises:
            RuntimeError: API 调用失败
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量表示
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        
        Raises:
            RuntimeError: API 调用失败
        """
        pass
    
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class OpenAIEmbedding(BaseEmbeddings):
    """OpenAI 兼容的 Embedding 实现"""
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None
    ) -> None:
        """
        初始化 OpenAI Embedding
        
        优先级：
        1. 显式传入的参数
        2. OPENAI_* 环境变量（Embedding 专用）
        3. CLOUD_EMBEDDING_* 环境变量（备选）
        
        Args:
            model: 模型名称
            api_key: API Key (必须提供)
            base_url: Base URL
        
        Raises:
            ImportError: openai 模块未安装
            ValueError: API Key 未配置
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 模块未安装，请运行: pip install openai")
        
        # 优先使用 OPENAI_* 配置（Embedding 专用）
        self.model = model or os.getenv("OPENAI_API_MODEL", "Pro/BAAI/bge-m3")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CLOUD_EMBEDDING_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("CLOUD_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
        
        # 验证 API Key
        if not self.api_key:
            raise ValueError(
                "API Key 未配置，请设置 OPENAI_API_KEY 或 CLOUD_EMBEDDING_API_KEY 环境变量\n"
                "参考 env.example 文件配置"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 OpenAI Embedding:")
    
    # 检查 API 配置
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CLOUD_EMBEDDING_API_KEY")
    if not api_key:
        print("错误: 未配置 API Key 环境变量")
        print("请在 .env 文件中设置 OPENAI_API_KEY")
        exit(1)
    
    try:
        embedding = OpenAIEmbedding()
        print(f"模型: {embedding.model}")
        print(f"Base URL: {embedding.base_url}")
        
        v1 = embedding.get_embedding("你好，世界！")
        v2 = embedding.get_embedding("Hello, world!")
        
        sim = OpenAIEmbedding.cosine_similarity(v1, v2)
        
        print(f"向量维度: {len(v1)}")
        print(f"相似度: {sim:.4f}")
    except Exception as e:
        print(f"Embedding 调用失败: {e}")
        exit(1)
