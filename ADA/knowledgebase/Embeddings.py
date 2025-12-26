# -*- coding: utf-8 -*-
"""
Embedding 模型封装
支持 OpenAI 兼容的 Embedding API
"""

import os
import sys
from typing import List, Optional
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class BaseEmbeddings:
    """Embedding 基类"""
    
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        """
        初始化嵌入基类
        
        Args:
            path: 模型或数据的路径
            is_api: 是否使用 API 方式
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        获取文本的嵌入向量表示
        
        Args:
            text: 输入文本
            model: 使用的模型名称
        
        Returns:
            文本的嵌入向量
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_embeddings_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        批量获取文本的嵌入向量
        
        Args:
            texts: 输入文本列表
            model: 使用的模型名称
        
        Returns:
            嵌入向量列表
        """
        return [self.get_embedding(text, model) for text in texts]
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
        
        Returns:
            余弦相似度，范围在 [-1, 1] 之间
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        # 检查向量中是否包含无穷大或 NaN 值
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0
        
        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        
        # 计算向量的范数
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母
        magnitude = norm_v1 * norm_v2
        
        # 处理分母为 0 的特殊情况
        if magnitude == 0:
            return 0.0
        
        return float(dot_product / magnitude)


class OpenAIEmbedding(BaseEmbeddings):
    """OpenAI 兼容的 Embedding 实现"""
    
    def __init__(
        self, 
        path: str = '', 
        is_api: bool = True,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ) -> None:
        """
        初始化 OpenAI Embedding
        
        Args:
            path: 本地模型路径（当 is_api=False 时使用）
            is_api: 是否使用 API
            api_key: API Key（可选，默认从环境变量读取）
            base_url: Base URL（可选，默认从环境变量读取）
            model: 模型名称（可选，默认从环境变量读取）
        """
        super().__init__(path, is_api)
        
        if self.is_api:
            from openai import OpenAI
            
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.default_model = model or os.getenv("OPENAI_API_MODEL", "text-embedding-3-small")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            model: 模型名称
        
        Returns:
            嵌入向量
        """
        if not self.is_api:
            raise NotImplementedError("本地模型暂未实现")
        
        # 预处理文本
        text = text.replace("\n", " ").strip()
        if not text:
            # 返回零向量（维度根据模型确定）
            return [0.0] * 1536
        
        model = model or self.default_model
        
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取 Embedding 失败: {e}")
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        批量获取嵌入向量（更高效）
        
        Args:
            texts: 文本列表
            model: 模型名称
        
        Returns:
            嵌入向量列表
        """
        if not self.is_api:
            raise NotImplementedError("本地模型暂未实现")
        
        # 预处理
        processed_texts = [t.replace("\n", " ").strip() or " " for t in texts]
        model = model or self.default_model
        
        try:
            response = self.client.embeddings.create(
                input=processed_texts,
                model=model
            )
            # 按原始顺序返回
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            return embeddings
        except Exception as e:
            print(f"批量获取 Embedding 失败: {e}")
            return [[0.0] * 1536 for _ in texts]


class MockEmbedding(BaseEmbeddings):
    """
    模拟 Embedding（用于测试）
    使用简单的哈希函数生成固定维度的向量
    """
    
    def __init__(self, dimension: int = 128):
        super().__init__("", is_api=False)
        self.dimension = dimension
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """生成模拟的嵌入向量"""
        import hashlib
        
        # 使用文本哈希生成确定性的向量
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # 归一化到 [-1, 1]
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            vector.append(value)
        
        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 MockEmbedding:")
    mock = MockEmbedding(dimension=8)
    
    v1 = mock.get_embedding("你好，世界！")
    v2 = mock.get_embedding("Hello, world!")
    v3 = mock.get_embedding("你好，世界！")  # 相同文本
    
    print(f"向量1 (中文): {v1[:4]}...")
    print(f"向量2 (英文): {v2[:4]}...")
    print(f"向量3 (中文相同): {v3[:4]}...")
    
    sim_12 = MockEmbedding.cosine_similarity(v1, v2)
    sim_13 = MockEmbedding.cosine_similarity(v1, v3)
    
    print(f"\n相似度(中-英): {sim_12:.4f}")
    print(f"相似度(中-中): {sim_13:.4f}")
    
    # 如果配置了 API，测试真实的 Embedding
    if os.getenv("OPENAI_API_KEY"):
        print("\n测试 OpenAI Embedding:")
        openai_emb = OpenAIEmbedding()
        v1 = openai_emb.get_embedding("你好，世界！")
        v2 = openai_emb.get_embedding("Hello, world!")
        sim = OpenAIEmbedding.cosine_similarity(v1, v2)
        print(f"向量维度: {len(v1)}")
        print(f"相似度: {sim:.4f}")
