# -*- coding: utf-8 -*-
"""
知识库服务层
提供统一的知识检索和更新接口
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.const import KnowledgeItem, KnowledgeType
from utils.interact import BaseVectorStore, BaseLLM
from config import get_system_config

from .VectorBase import VectorStore
from .Embeddings import BaseEmbeddings, OpenAIEmbedding, MockEmbedding
from .LLM import OpenAIChat, MockLLM


class KnowledgeService:
    """
    知识库服务
    封装向量检索和知识管理功能
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_model: BaseEmbeddings = None,
        llm: BaseLLM = None,
        storage_path: str = None
    ):
        """
        初始化知识服务
        
        Args:
            vector_store: 向量存储实例
            embedding_model: Embedding 模型
            llm: LLM 模型（用于知识提炼）
            storage_path: 持久化路径
        """
        config = get_system_config()
        
        self.storage_path = storage_path or str(config.get_knowledge_path())
        self.top_k = config.knowledge_top_k
        
        # 初始化组件
        self.embedding = embedding_model or self._create_default_embedding()
        self.vector_store = vector_store or VectorStore()
        self.llm = llm
        
        # 尝试加载已有数据
        self._try_load()
    
    def _create_default_embedding(self) -> BaseEmbeddings:
        """创建默认的 Embedding 模型"""
        try:
            if os.getenv("OPENAI_API_KEY"):
                return OpenAIEmbedding()
            else:
                print("警告: 未配置 OPENAI_API_KEY，使用 MockEmbedding")
                return MockEmbedding()
        except Exception as e:
            print(f"创建 Embedding 模型失败: {e}，使用 MockEmbedding")
            return MockEmbedding()
    
    def _try_load(self) -> None:
        """尝试加载已有数据"""
        if os.path.exists(self.storage_path):
            try:
                self.vector_store.load(self.storage_path)
                print(f"已加载知识库: {len(self.vector_store)} 条记录")
            except Exception as e:
                print(f"加载知识库失败: {e}")
    
    def query_task_knowledge(
        self, 
        query: str, 
        k: int = None
    ) -> List[KnowledgeItem]:
        """
        检索任务知识 (TK)
        
        Args:
            query: 查询文本
            k: 返回数量
        
        Returns:
            相关的任务知识列表
        """
        k = k or self.top_k
        results = self.vector_store.query(
            query, 
            self.embedding, 
            k=k,
            type_filter=KnowledgeType.TK.value
        )
        
        return [self._result_to_item(r, KnowledgeType.TK) for r in results]
    
    def query_action_knowledge(
        self, 
        query: str, 
        k: int = None
    ) -> List[KnowledgeItem]:
        """
        检索动作知识 (AK)
        
        Args:
            query: 查询文本
            k: 返回数量
        
        Returns:
            相关的动作知识列表
        """
        k = k or self.top_k
        results = self.vector_store.query(
            query, 
            self.embedding, 
            k=k,
            type_filter=KnowledgeType.AK.value
        )
        
        return [self._result_to_item(r, KnowledgeType.AK) for r in results]
    
    def query(
        self, 
        query: str, 
        k: int = None,
        knowledge_type: KnowledgeType = None
    ) -> List[KnowledgeItem]:
        """
        通用检索接口
        
        Args:
            query: 查询文本
            k: 返回数量
            knowledge_type: 知识类型过滤
        
        Returns:
            相关知识列表
        """
        k = k or self.top_k
        type_filter = knowledge_type.value if knowledge_type else None
        
        results = self.vector_store.query(
            query, 
            self.embedding, 
            k=k,
            type_filter=type_filter
        )
        
        return [self._result_to_item(r) for r in results]
    
    def _result_to_item(
        self, 
        result: Dict[str, Any],
        default_type: KnowledgeType = KnowledgeType.TK
    ) -> KnowledgeItem:
        """将检索结果转换为 KnowledgeItem"""
        metadata = result.get("metadata", {})
        type_str = metadata.get("type", default_type.value)
        
        return KnowledgeItem(
            id=result["id"],
            type=KnowledgeType(type_str) if type_str in [t.value for t in KnowledgeType] else default_type,
            content=result["content"],
            metadata={**metadata, "score": result.get("score", 0.0)}
        )
    
    def add_knowledge(
        self, 
        content: str, 
        knowledge_type: KnowledgeType,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加新知识
        
        Args:
            content: 知识内容
            knowledge_type: 知识类型
            metadata: 附加元数据
        
        Returns:
            知识 ID
        """
        meta = metadata or {}
        meta["type"] = knowledge_type.value
        
        ids = self.vector_store.add_documents(
            [content],
            self.embedding,
            [meta],
            show_progress=False
        )
        
        # 持久化
        self.persist()
        
        return ids[0] if ids else ""
    
    def update_knowledge(
        self, 
        item_id: str, 
        content: str
    ) -> bool:
        """
        更新知识
        
        Args:
            item_id: 知识 ID
            content: 新内容
        
        Returns:
            是否更新成功
        """
        success = self.vector_store.update(item_id, content, self.embedding)
        if success:
            self.persist()
        return success
    
    def delete_knowledge(self, item_id: str) -> bool:
        """
        删除知识
        
        Args:
            item_id: 知识 ID
        
        Returns:
            是否删除成功
        """
        success = self.vector_store.delete(item_id)
        if success:
            self.persist()
        return success
    
    def persist(self) -> None:
        """持久化知识库"""
        self.vector_store.persist(self.storage_path)
    
    def get_context_string(
        self, 
        query: str, 
        knowledge_type: KnowledgeType = None
    ) -> str:
        """
        获取检索结果的上下文字符串（用于 Prompt）
        
        Args:
            query: 查询文本
            knowledge_type: 知识类型过滤
        
        Returns:
            格式化的上下文字符串
        """
        items = self.query(query, knowledge_type=knowledge_type)
        
        if not items:
            return "暂无相关知识"
        
        lines = []
        for i, item in enumerate(items, 1):
            score = item.metadata.get("score", 0.0)
            lines.append(f"[{i}] (相关度: {score:.2f})")
            lines.append(f"    {item.content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def batch_add_knowledge(
        self,
        items: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量添加知识
        
        Args:
            items: 知识列表，每个元素包含 {content, type, metadata}
        
        Returns:
            知识 ID 列表
        """
        contents = []
        metadata_list = []
        
        for item in items:
            contents.append(item["content"])
            meta = item.get("metadata", {})
            meta["type"] = item["type"].value if isinstance(item["type"], KnowledgeType) else item["type"]
            metadata_list.append(meta)
        
        ids = self.vector_store.add_documents(
            contents,
            self.embedding,
            metadata_list,
            show_progress=True
        )
        
        self.persist()
        return ids
    
    def __len__(self) -> int:
        return len(self.vector_store)
    
    def __repr__(self) -> str:
        return f"KnowledgeService(items={len(self)}, path='{self.storage_path}')"


# ============= 测试代码 =============
if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("测试 KnowledgeService:")
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建服务（使用 MockEmbedding）
        service = KnowledgeService(
            embedding_model=MockEmbedding(),
            storage_path=temp_dir
        )
        
        # 添加任务知识
        tk_id = service.add_knowledge(
            "电网调度优化问题通常建模为：min Σc_i*P_i，约束包括功率平衡和线路容量",
            KnowledgeType.TK,
            {"source": "test"}
        )
        print(f"添加 TK: {tk_id}")
        
        # 添加动作知识
        ak_id = service.add_knowledge(
            "当负载预测不确定时，应先调用天气预报工具获取气象数据",
            KnowledgeType.AK,
            {"source": "test"}
        )
        print(f"添加 AK: {ak_id}")
        
        print(f"\n知识库状态: {service}")
        
        # 测试检索
        print("\n检索任务知识:")
        tk_results = service.query_task_knowledge("电网优化调度")
        for item in tk_results:
            print(f"  [{item.metadata.get('score', 0):.4f}] {item.content[:50]}...")
        
        print("\n检索动作知识:")
        ak_results = service.query_action_knowledge("负载预测")
        for item in ak_results:
            print(f"  [{item.metadata.get('score', 0):.4f}] {item.content[:50]}...")
        
        # 测试上下文生成
        print("\n生成上下文:")
        context = service.get_context_string("电网调度")
        print(context)
        
        # 测试更新
        service.update_knowledge(tk_id, "更新后的任务知识内容")
        print("\n更新知识成功")
        
        # 测试删除
        service.delete_knowledge(ak_id)
        print("删除知识成功")
        
        print(f"\n最终状态: {service}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n测试完成，已清理临时文件")
