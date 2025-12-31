#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ADA 知识库服务

与顶层项目的通用 KnowledgeService 相比，这里提供一个
轻量、无额外依赖（如 SystemConfig）的实现，专门给 ADA_Agent
的 Judger / Summarizer 使用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.embeddings import BaseEmbeddings
from ADA.knowledgebase.VectorBase import VectorStore

class KnowledgeService:
    """
    ADA 内部使用的知识库服务封装。

    功能：
    - 向量化存储文本经验
    - 按相似度检索相关经验
    - 简单的增删改查与持久化
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddings,
        storage_dir: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        """
        初始化知识库服务

        Args:
            embedding_model: Embedding 模型实例（必需）
            storage_dir: 持久化目录（默认 ADA/knowledgebase/storage）
            top_k: 默认检索返回的条数
        """
        if embedding_model is None:
            raise ValueError("KnowledgeService: 必须提供 embedding_model 实例")

        self.embedding = embedding_model
        self.vector_store = VectorStore()
        self.top_k = int(top_k)

        # 默认存储目录：ADA/knowledgebase/storage
        default_dir = Path(__file__).parent / "storage"
        self.storage_dir = Path(storage_dir) if storage_dir is not None else default_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # 尝试加载已有数据
        self._try_load()

    # ------------------------------------------------------------------
    # 基础加载 / 持久化
    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        """尝试从磁盘加载已有向量库（如果不存在则忽略）"""
        try:
            self.vector_store.load(str(self.storage_dir))
        except Exception:
            # 加载失败时仅打印信息，不中断主流程
            print(f"[KnowledgeService] 无法从 {self.storage_dir} 加载现有知识库，将从空库开始。")

    def persist(self) -> None:
        """将当前内存中的向量库持久化到磁盘"""
        self.vector_store.persist(str(self.storage_dir))

    # ------------------------------------------------------------------
    # 对外检索接口
    # ------------------------------------------------------------------
    def query(
        self,
        query: str,
        k: Optional[int] = None,
        knowledge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        通用检索接口

        Args:
            query: 查询文本
            k: 返回条数（默认 self.top_k）
            knowledge_type: 过滤类型（如 KnowledgeType.TK / KnowledgeType.AK）

        Returns:
            检索结果列表，每个元素包含:
            - id: 文档 ID
            - content: 文本内容
            - score: 相似度
            - metadata: 元数据（含 type 等）
        """
        if not query:
            return []

        k = int(k or self.top_k)
        type_filter = knowledge_type

        results = self.vector_store.query(
            query=query,
            embedding_model=self.embedding,
            k=k,
            type_filter=type_filter,
        )
        return results


    def get_context_string(
        self,
        query: str,
        knowledge_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> str:
        """
        将检索结果格式化为 Prompt 友好的上下文字符串
        """
        items = self.query(query, k=k, knowledge_type=knowledge_type)
        if not items:
            return "暂无相关知识"

        lines: List[str] = []
        for i, item in enumerate(items, 1):
            score = item.get("score", 0.0)
            content = item.get("content", "")
            lines.append(f"[{i}] (相关度: {score:.2f})")
            lines.append(f"    {content}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 写入 / 更新 / 删除
    # ------------------------------------------------------------------
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加一条知识记录

        Args:
            content: 文本内容   
            metadata: 额外元数据

        Returns:
            新增知识的 ID
        """
        if not content:
            raise ValueError("KnowledgeService.add_knowledge: content 不能为空")

        meta = dict(metadata or {})

        ids = self.vector_store.add_documents(
            [content],
            embedding_model=self.embedding,
            metadata_list=[meta],
            show_progress=False,
        )
        # 持久化
        self.persist()
        return ids[0] if ids else ""

    def update_knowledge(self, item_id: str, content: str) -> bool:
        """更新指定 ID 的知识内容"""
        if not item_id:
            return False

        ok = self.vector_store.update(
            doc_id=item_id,
            content=content,
            embedding_model=self.embedding,
        )
        if ok:
            self.persist()
        return ok

    def delete_knowledge(self, item_id: str) -> bool:
        """删除指定 ID 的知识"""
        if not item_id:
            return False

        ok = self.vector_store.delete(item_id)
        if ok:
            self.persist()
        return ok

    # ------------------------------------------------------------------
    # 兼容 Summarizer 的简易接口（降级路径）
    # ------------------------------------------------------------------
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        兼容 Summarizer._simple_store 使用的接口：
        - 不指定 type，统一按 AK 存储
        """
        return self.add_knowledge(
            content=content,
            metadata=metadata,
        )


__all__ = ["KnowledgeService"]

