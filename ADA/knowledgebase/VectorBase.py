# -*- coding: utf-8 -*-
"""
向量数据库实现
支持向量存储、检索、持久化
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .Embeddings import BaseEmbeddings, OpenAIEmbedding


class VectorStore:
    """
    向量存储类
    支持文档存储、向量检索、持久化
    """
    
    def __init__(self, documents: List[str] = None) -> None:
        """
        初始化向量存储
        
        Args:
            documents: 初始文档列表
        """
        self.documents: List[str] = documents or []
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
    
    def add_documents(
        self, 
        documents: List[str], 
        embedding_model: BaseEmbeddings,
        metadata_list: List[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        添加文档并计算向量
        
        Args:
            documents: 文档列表
            embedding_model: Embedding 模型
            metadata_list: 元数据列表
            show_progress: 是否显示进度条
        
        Returns:
            文档 ID 列表
        """
        new_ids = []
        metadata_list = metadata_list or [{} for _ in documents]
        
        iterator = tqdm(documents, desc="计算嵌入向量") if show_progress else documents
        
        for i, doc in enumerate(iterator):
            doc_id = str(uuid.uuid4())
            vector = embedding_model.get_embedding(doc)
            
            self.documents.append(doc)
            self.vectors.append(vector)
            self.metadata.append(metadata_list[i] if i < len(metadata_list) else {})
            self.ids.append(doc_id)
            new_ids.append(doc_id)
        
        return new_ids
    
    def get_vector(self, embedding_model: BaseEmbeddings, show_progress: bool = True) -> List[List[float]]:
        """
        为所有文档计算向量（兼容旧接口）
        
        Args:
            embedding_model: Embedding 模型
            show_progress: 是否显示进度条
        
        Returns:
            向量列表
        """
        self.vectors = []
        self.ids = []
        self.metadata = []
        
        iterator = tqdm(self.documents, desc="计算嵌入向量") if show_progress else self.documents
        
        for doc in iterator:
            vector = embedding_model.get_embedding(doc)
            self.vectors.append(vector)
            self.ids.append(str(uuid.uuid4()))
            self.metadata.append({})
        
        return self.vectors
    
    def query(
        self, 
        query: str, 
        embedding_model: BaseEmbeddings, 
        k: int = 3,
        type_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        检索相似文档
        
        Args:
            query: 查询文本
            embedding_model: Embedding 模型
            k: 返回数量
            type_filter: 类型过滤（基于 metadata 中的 type 字段）
        
        Returns:
            相似文档列表，每个元素包含 {id, content, score, metadata}
        """
        if not self.vectors:
            return []
        
        # 计算查询向量
        query_vector = embedding_model.get_embedding(query)
        
        # 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 类型过滤
            if type_filter and self.metadata[i].get("type") != type_filter:
                continue
            
            score = BaseEmbeddings.cosine_similarity(query_vector, vector)
            similarities.append((i, score))
        
        # 排序并返回 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:k]:
            results.append({
                "id": self.ids[idx],
                "content": self.documents[idx],
                "score": score,
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def delete(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档 ID
        
        Returns:
            是否删除成功
        """
        if doc_id not in self.ids:
            return False
        
        idx = self.ids.index(doc_id)
        self.documents.pop(idx)
        self.vectors.pop(idx)
        self.metadata.pop(idx)
        self.ids.pop(idx)
        
        return True
    
    def update(self, doc_id: str, content: str, embedding_model: BaseEmbeddings) -> bool:
        """
        更新文档
        
        Args:
            doc_id: 文档 ID
            content: 新内容
            embedding_model: Embedding 模型
        
        Returns:
            是否更新成功
        """
        if doc_id not in self.ids:
            return False
        
        idx = self.ids.index(doc_id)
        self.documents[idx] = content
        self.vectors[idx] = embedding_model.get_embedding(content)
        
        return True
    
    def persist(self, path: str = 'storage') -> None:
        """
        持久化到磁盘
        
        Args:
            path: 存储路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存文档
        with open(f"{path}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # 保存向量
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)
        
        # 保存元数据
        with open(f"{path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # 保存 ID
        with open(f"{path}/ids.json", 'w', encoding='utf-8') as f:
            json.dump(self.ids, f)
    
    def load(self, path: str = 'storage') -> bool:
        """
        从磁盘加载
        
        Args:
            path: 存储路径
        
        Returns:
            是否加载成功
        """
        try:
            # 加载文档
            with open(f"{path}/documents.json", 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            # 加载向量
            vectors_path = f"{path}/vectors.json"
            if os.path.exists(vectors_path):
                with open(vectors_path, 'r', encoding='utf-8') as f:
                    self.vectors = json.load(f)
            
            # 加载元数据
            metadata_path = f"{path}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = [{} for _ in self.documents]
            
            # 加载 ID
            ids_path = f"{path}/ids.json"
            if os.path.exists(ids_path):
                with open(ids_path, 'r', encoding='utf-8') as f:
                    self.ids = json.load(f)
            else:
                self.ids = [str(uuid.uuid4()) for _ in self.documents]
            
            return True
        except Exception as e:
            print(f"加载向量库失败: {e}")
            return False
    
    def load_vector(self, path: str = 'storage') -> None:
        """兼容旧接口"""
        self.load(path)
    
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """计算相似度（兼容旧接口）"""
        return BaseEmbeddings.cosine_similarity(vector1, vector2)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __repr__(self) -> str:
        return f"VectorStore(documents={len(self.documents)}, has_vectors={bool(self.vectors)})"


# ============= 测试代码 =============
if __name__ == "__main__":
    from Embeddings import MockEmbedding
    
    print("测试 VectorStore:")
    
    # 创建测试文档
    docs = [
        "电网调度是指电力系统中对发电和输电的统一调配",
        "潮流计算是电力系统分析的基础",
        "优化调度可以降低发电成本",
        "机器学习在电力系统中有广泛应用",
    ]
    
    # 创建向量存储
    store = VectorStore(docs)
    embedding = MockEmbedding(dimension=64)
    
    # 计算向量
    store.get_vector(embedding, show_progress=False)
    print(f"存储状态: {store}")
    
    # 测试检索
    results = store.query("电力系统调度优化", embedding, k=2)
    print("\n检索结果:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:30]}...")
    
    # 测试持久化
    store.persist("test_storage")
    print("\n持久化完成")
    
    # 测试加载
    new_store = VectorStore()
    new_store.load("test_storage")
    print(f"加载后状态: {new_store}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree("test_storage", ignore_errors=True)
    print("测试完成，已清理临时文件")
