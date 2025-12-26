# -*- coding: utf-8 -*-
"""
Summarizer Agent 核心实现
负责经验回溯和知识更新
"""

import sys
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.const import ExecutionTrace, KnowledgeItem, KnowledgeType
from utils.interact import BaseSummarizer
from utils.llm import BaseLLM
from utils.logger import get_logger
from config import SystemConfig

from knowledgebase.service import KnowledgeService
from .knowledge_updater import KnowledgeUpdater

logger = get_logger("Summarizer")


@dataclass
class MCTSNode:
    """MCTS 搜索树节点"""
    state: str  # 状态描述
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: str = ""  # 导致此状态的动作
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """计算 UCB 评分"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        ) if self.parent else 0
        
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """选择最佳子节点"""
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.ucb_score(exploration_constant))
    
    def is_leaf(self) -> bool:
        """是否为叶节点"""
        return len(self.children) == 0


class SummarizerAgent(BaseSummarizer):
    """
    总结智能体
    
    核心职责：
    1. 经验回放 - 收集调度轨迹
    2. MCTS 搜索 - 识别高价值决策路径
    3. 知识提炼 - 提取可复用的知识模式
    4. 知识更新 - 将知识写入知识库
    """
    
    def __init__(
        self,
        kb: KnowledgeService,
        llm: BaseLLM = None,
        exploration_constant: float = None,
        min_score_threshold: float = None
    ):
        """
        初始化 Summarizer
        
        Args:
            kb: 知识库服务
            llm: LLM 服务
            exploration_constant: MCTS 探索系数
            min_score_threshold: 最小入库分数阈值
        """
        config = SystemConfig()
        
        self.kb = kb
        self.llm = llm
        self.exploration_constant = exploration_constant or config.mcts_exploration_constant
        self.min_score_threshold = min_score_threshold or config.summarizer_min_score_threshold
        
        # 知识更新器
        self.knowledge_updater = KnowledgeUpdater(
            kb=kb,
            llm=llm,
            min_score_threshold=self.min_score_threshold
        )
        
        # 历史轨迹缓存
        self._trace_buffer: List[ExecutionTrace] = []
        
        # MCTS 根节点
        self._mcts_root: Optional[MCTSNode] = None
    
    def summarize(self, trace: ExecutionTrace) -> None:
        """
        总结执行轨迹
        
        Args:
            trace: 执行轨迹
        """
        logger.info(f"开始总结轨迹: {trace.trace_id}")
        
        # 1. 添加到缓存
        self._trace_buffer.append(trace)
        
        # 2. 更新 MCTS 树
        self._update_mcts_tree(trace)
        
        # 3. 提炼并更新知识
        result = self.knowledge_updater.update_from_trace(trace)
        
        if result["ak_updated"] or result["tk_updated"]:
            logger.info(f"知识库已更新: {result['details']}")
        
        # 4. 定期进行批量分析
        if len(self._trace_buffer) >= 10:
            self._batch_analysis()
    
    def extract_action_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """从轨迹中提取动作知识"""
        return self.knowledge_updater.extract_action_knowledge(trace)
    
    def extract_task_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """从轨迹中提取任务知识"""
        return self.knowledge_updater.extract_task_knowledge(trace)
    
    def _update_mcts_tree(self, trace: ExecutionTrace) -> None:
        """
        更新 MCTS 搜索树
        将轨迹中的工具调用序列建模为搜索树
        
        Args:
            trace: 执行轨迹
        """
        # 初始化根节点
        if self._mcts_root is None:
            self._mcts_root = MCTSNode(state="root")
        
        # 将工具调用链转换为路径
        current_node = self._mcts_root
        current_node.visits += 1
        
        for step in trace.tool_chain:
            if isinstance(step, dict):
                action = step.get("tool_selected", "unknown")
            else:
                action = getattr(step, "tool_selected", "unknown")
            
            # 查找或创建子节点
            child_node = None
            for child in current_node.children:
                if child.action == action:
                    child_node = child
                    break
            
            if child_node is None:
                child_node = MCTSNode(
                    state=f"{current_node.state}->{action}",
                    parent=current_node,
                    action=action
                )
                current_node.children.append(child_node)
            
            # 更新节点
            child_node.visits += 1
            current_node = child_node
        
        # 反向传播评分
        self._backpropagate(current_node, trace.feedback.score)
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """反向传播更新节点值"""
        while node is not None:
            node.value += value
            node = node.parent
    
    def get_best_tool_sequence(self) -> List[str]:
        """
        获取最佳工具调用序列
        
        Returns:
            最佳工具序列
        """
        if self._mcts_root is None:
            return []
        
        sequence = []
        node = self._mcts_root
        
        while not node.is_leaf():
            # 选择访问次数最多的子节点（利用模式）
            best_child = max(node.children, key=lambda c: c.visits)
            sequence.append(best_child.action)
            node = best_child
        
        return sequence
    
    def _batch_analysis(self) -> None:
        """
        批量分析历史轨迹
        识别成功模式和失败模式
        """
        logger.info("开始批量分析历史轨迹")
        
        # 统计成功和失败案例
        successful_traces = [t for t in self._trace_buffer if t.is_successful()]
        failed_traces = [t for t in self._trace_buffer if not t.is_successful()]
        
        logger.info(f"成功案例: {len(successful_traces)}, 失败案例: {len(failed_traces)}")
        
        # 分析成功模式
        if successful_traces:
            best_trace = max(successful_traces, key=lambda t: t.feedback.score)
            logger.info(f"最佳案例评分: {best_trace.feedback.score:.4f}")
        
        # 清空缓存
        self._trace_buffer = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "buffer_size": len(self._trace_buffer),
            "knowledge_count": len(self.kb),
            "mcts_tree_depth": self._get_tree_depth(),
            "best_sequence": self.get_best_tool_sequence()
        }
        return stats
    
    def _get_tree_depth(self) -> int:
        """获取 MCTS 树深度"""
        if self._mcts_root is None:
            return 0
        
        def depth(node: MCTSNode) -> int:
            if node.is_leaf():
                return 1
            return 1 + max(depth(c) for c in node.children)
        
        return depth(self._mcts_root)
    
    def export_mcts_tree(self) -> Dict[str, Any]:
        """
        导出 MCTS 树结构
        
        Returns:
            树结构字典
        """
        if self._mcts_root is None:
            return {}
        
        def node_to_dict(node: MCTSNode) -> Dict[str, Any]:
            return {
                "action": node.action,
                "visits": node.visits,
                "value": node.value,
                "avg_value": node.value / node.visits if node.visits > 0 else 0,
                "children": [node_to_dict(c) for c in node.children]
            }
        
        return node_to_dict(self._mcts_root)


# ============= 测试代码 =============
if __name__ == "__main__":
    print("Summarizer 模块测试需要配置 LLM API")
    print("请运行 python -m test.test_all 进行完整测试")
