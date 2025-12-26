# -*- coding: utf-8 -*-
"""
知识库模块

提供向量检索和知识管理功能
"""

from .service import KnowledgeService
from .VectorBase import VectorStore

__all__ = [
    'KnowledgeService',
    'VectorStore',
]
