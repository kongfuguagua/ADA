# -*- coding: utf-8 -*-
"""
系统全局配置
定义 Judger 权重、MCTS 参数、重试次数等全局参数
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class SystemConfig:
    """系统全局配置类"""
    
    # ============= 系统级配置 =============
    # 项目根目录
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # 最大重试次数
    max_retries: int = 3
    
    # 日志配置
    log_level: str = "INFO"
    log_dir: str = "log"
    
    # ============= Planner 配置 =============
    # 状态增广最大步数
    planner_max_augmentation_steps: int = 5
    
    # ============= Solver 配置 =============
    # 求解超时时间（秒）
    solver_timeout: float = 60.0
    
    # 特征维度
    solver_feature_dim: int = 5
    
    # ============= Judger 配置 =============
    # 物理评分权重 α (R = α * R_ENV + (1-α) * R_LLM)
    judger_alpha: float = 0.7
    
    # 通过阈值
    judger_pass_threshold: float = 0.6
    
    # 评分范围
    judger_score_min: float = 0.0
    judger_score_max: float = 1.0
    
    # ============= Summarizer 配置 =============
    # MCTS 探索系数 (UCB 公式中的 C)
    mcts_exploration_constant: float = 1.414
    
    # 最小入库分数阈值
    summarizer_min_score_threshold: float = 0.7
    
    # ============= 知识库配置 =============
    # 向量数据库存储路径
    knowledge_storage_path: str = "knowledgebase/storage"
    
    # 检索返回数量
    knowledge_top_k: int = 3
    
    # 文档分块大小
    knowledge_chunk_size: int = 600
    knowledge_chunk_overlap: int = 150
    
    def get_log_path(self) -> Path:
        """获取日志目录路径"""
        log_path = self.project_root / self.log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path
    
    def get_knowledge_path(self) -> Path:
        """获取知识库存储路径"""
        kb_path = self.project_root / self.knowledge_storage_path
        kb_path.mkdir(parents=True, exist_ok=True)
        return kb_path
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "max_retries": self.max_retries,
            "judger_alpha": self.judger_alpha,
            "judger_pass_threshold": self.judger_pass_threshold,
            "mcts_exploration_constant": self.mcts_exploration_constant,
            "solver_timeout": self.solver_timeout,
        }


# 全局单例
_system_config: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    """获取系统配置单例"""
    global _system_config
    if _system_config is None:
        _system_config = SystemConfig()
    return _system_config


def reset_system_config(config: SystemConfig) -> None:
    """重置系统配置（用于测试）"""
    global _system_config
    _system_config = config


# ============= 测试代码 =============
if __name__ == "__main__":
    config = get_system_config()
    print("系统配置信息:")
    print(f"  项目根目录: {config.project_root}")
    print(f"  最大重试次数: {config.max_retries}")
    print(f"  Judger Alpha: {config.judger_alpha}")
    print(f"  MCTS 探索系数: {config.mcts_exploration_constant}")
    print(f"  日志目录: {config.get_log_path()}")

