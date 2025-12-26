# -*- coding: utf-8 -*-
"""
ADA Agent 接口定义
定义所有 Agent 的抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

from .const import (
    KnowledgeItem,
    KnowledgeType,
    EnvironmentState,
    OptimizationProblem,
    Solution,
    Feedback,
    ExecutionTrace,
    SolverAlgorithmMeta,
)

# 从 llm 模块重新导出 BaseLLM（保持向后兼容）
from .llm import BaseLLM


# ================= 1. 基础服务接口 =================

class BaseTool(ABC):
    """所有工具（仿真器、查询器）的基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            工具执行结果
        """
        pass
    
    @property
    def schema(self) -> Dict[str, Any]:
        """
        返回 JSON Schema 供 LLM 理解参数
        子类可覆盖此方法提供详细的参数定义
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }


class BaseVectorStore(ABC):
    """向量数据库接口"""
    
    @abstractmethod
    def add(self, items: List[KnowledgeItem]) -> None:
        """
        添加知识项到向量库
        
        Args:
            items: 知识项列表
        """
        pass
    
    @abstractmethod
    def query(
        self, 
        text: str, 
        k: int = 3, 
        type_filter: Optional[KnowledgeType] = None
    ) -> List[KnowledgeItem]:
        """
        检索相似知识
        
        Args:
            text: 查询文本
            k: 返回数量
            type_filter: 知识类型过滤
        
        Returns:
            相似知识项列表
        """
        pass
    
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """
        删除知识项
        
        Args:
            item_id: 知识项 ID
        
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def update(self, item: KnowledgeItem) -> bool:
        """
        更新知识项
        
        Args:
            item: 更新后的知识项
        
        Returns:
            是否更新成功
        """
        pass


class BaseSimulator(ABC):
    """物理仿真器接口"""
    
    @abstractmethod
    def run(self, action_vector: List[float]) -> Dict[str, Any]:
        """
        执行仿真
        
        Args:
            action_vector: 控制动作向量
        
        Returns:
            仿真结果（包含物理指标）
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置仿真环境"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        pass


# ================= 2. Agent 接口 =================

class BasePlanner(ABC):
    """规划智能体接口"""
    
    @abstractmethod
    def plan(
        self, 
        state: EnvironmentState, 
        retry_feedback: Optional[Feedback] = None
    ) -> OptimizationProblem:
        """
        核心规划方法
        
        Args:
            state: 环境状态
            retry_feedback: 重试时的反馈信息（首次运行为 None）
        
        Returns:
            优化问题定义
        """
        pass
    
    @abstractmethod
    def get_tool_chain(self) -> List[Dict[str, Any]]:
        """
        获取本次规划的工具调用链
        
        Returns:
            工具调用记录列表
        """
        pass


class BaseSolverStrategy(ABC):
    """具体算法策略接口（策略模式）"""
    
    @property
    @abstractmethod
    def meta(self) -> SolverAlgorithmMeta:
        """算法的能力描述"""
        pass
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> Solution:
        """
        求解优化问题
        
        Args:
            problem: 优化问题
        
        Returns:
            求解结果
        """
        pass
    
    def get_capability_vector(self) -> List[float]:
        """
        获取算法能力向量 ψ(A)
        
        Returns:
            能力向量 [凸处理, 非凸处理, 约束处理, 速度, 全局最优性]
        """
        caps = self.meta.capabilities
        return [
            caps.get("convex_handling", 0.5),
            caps.get("non_convex_handling", 0.5),
            caps.get("constraint_handling", 0.5),
            caps.get("speed", 0.5),
            caps.get("global_optimality", 0.5),
        ]


class BaseSolver(ABC):
    """求解智能体接口"""
    
    @abstractmethod
    def register_strategy(self, strategy: BaseSolverStrategy) -> None:
        """
        注册求解策略
        
        Args:
            strategy: 求解策略实例
        """
        pass
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> Solution:
        """
        求解优化问题（内部包含：特征提取 -> 算法选择 -> 执行）
        
        Args:
            problem: 优化问题
        
        Returns:
            求解结果
        """
        pass
    
    @abstractmethod
    def get_selected_algorithm(self) -> str:
        """获取选中的算法名称"""
        pass


class BaseJudger(ABC):
    """评估智能体接口"""
    
    @abstractmethod
    def evaluate(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Feedback:
        """
        评估解的质量（包含：物理仿真 + 逻辑校验 + 综合打分）
        
        Args:
            problem: 优化问题
            solution: 求解结果
        
        Returns:
            评估反馈
        """
        pass
    
    @abstractmethod
    def diagnose_error(
        self,
        problem: OptimizationProblem,
        solution: Solution,
        metrics: Dict[str, Any]
    ) -> str:
        """
        诊断错误原因
        
        Args:
            problem: 优化问题
            solution: 求解结果
            metrics: 物理指标
        
        Returns:
            诊断报告
        """
        pass


class BaseSummarizer(ABC):
    """总结智能体接口"""
    
    @abstractmethod
    def summarize(self, trace: ExecutionTrace) -> None:
        """
        异步调用：提取经验 -> 更新 KnowledgeBase
        
        Args:
            trace: 执行轨迹
        """
        pass
    
    @abstractmethod
    def extract_action_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """
        从轨迹中提取动作知识
        
        Args:
            trace: 执行轨迹
        
        Returns:
            提取的知识项（如果值得保存）
        """
        pass
    
    @abstractmethod
    def extract_task_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """
        从轨迹中提取任务知识
        
        Args:
            trace: 执行轨迹
        
        Returns:
            提取的知识项（如果值得保存）
        """
        pass
