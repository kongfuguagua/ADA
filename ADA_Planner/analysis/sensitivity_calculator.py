# -*- coding: utf-8 -*-
"""
灵敏度计算器 (Sensitivity Calculator)

基于图连接性的启发式分析，识别对过载线路最有效的发电机再调度策略。
"""

from typing import Dict, List, Set
import numpy as np
from grid2op.Observation import BaseObservation


class SensitivityCalculator:
    """
    灵敏度计算器
    
    目标：基于拓扑距离和潮流方向，识别有效的发电机再调度候选。
    """
    
    def __init__(self, max_hop: int = 1):
        """
        初始化计算器
        
        Args:
            max_hop: BFS搜索的最大跳数（默认1跳，即直接连接）
        """
        self.max_hop = max_hop
    
    def analyze_redispatch_strategy(
        self, 
        observation: BaseObservation, 
        line_id: int
    ) -> Dict[str, List[int]]:
        """
        分析再调度策略
        
        基于简单的物理规则：
        - 从过载线路的 Origin 端向外搜索，找到的发电机推荐降出力（Down-regulation）
        - 从过载线路的 Extremity 端向外搜索，找到的发电机推荐升出力（Up-regulation）
        
        Args:
            observation: Grid2Op 观测对象
            line_id: 过载线路ID
            
        Returns:
            包含以下键的字典：
            - decrease_candidates: 推荐降出力的发电机ID列表（位于送端）
            - increase_candidates: 推荐升出力的发电机ID列表（位于受端）
        """
        if not hasattr(observation, 'line_or_to_subid') or not hasattr(observation, 'line_ex_to_subid'):
            return {"decrease_candidates": [], "increase_candidates": []}
        
        # 获取线路两端的变电站
        origin_sub_id = int(observation.line_or_to_subid[line_id])
        extremity_sub_id = int(observation.line_ex_to_subid[line_id])
        
        # 确定潮流方向（简化：基于功率流符号）
        # P_or > 0 表示从 Origin 流向 Extremity
        if hasattr(observation, 'p_or'):
            p_or = float(observation.p_or[line_id])
            # 如果 P_or > 0，Origin 是送端，Extremity 是受端
            # 如果 P_or < 0，方向相反
        else:
            # 默认假设：Origin 是送端
            p_or = 1.0
        
        # BFS 搜索：从 Origin 端搜索（送端区域）
        source_area_subs = self._bfs_search(observation, origin_sub_id, max_hop=self.max_hop)
        
        # BFS 搜索：从 Extremity 端搜索（受端区域）
        sink_area_subs = self._bfs_search(observation, extremity_sub_id, max_hop=self.max_hop)
        
        # 如果潮流方向相反，交换区域
        if p_or < 0:
            source_area_subs, sink_area_subs = sink_area_subs, source_area_subs
        
        # 找出各区域的发电机
        decrease_candidates = self._get_generators_in_subs(observation, source_area_subs)
        increase_candidates = self._get_generators_in_subs(observation, sink_area_subs)
        
        # 过滤：只保留可调度的发电机
        if hasattr(observation, 'gen_redispatchable'):
            decrease_candidates = [
                gen_id for gen_id in decrease_candidates 
                if observation.gen_redispatchable[gen_id]
            ]
            increase_candidates = [
                gen_id for gen_id in increase_candidates 
                if observation.gen_redispatchable[gen_id]
            ]
        
        return {
            "decrease_candidates": decrease_candidates,
            "increase_candidates": increase_candidates
        }
    
    def _bfs_search(
        self, 
        observation: BaseObservation, 
        start_sub_id: int, 
        max_hop: int = 1
    ) -> Set[int]:
        """
        从起始变电站开始进行 BFS 搜索，找出 max_hop 跳内的所有变电站
        
        Args:
            observation: Grid2Op 观测对象
            start_sub_id: 起始变电站ID
            max_hop: 最大跳数
            
        Returns:
            可达变电站ID集合
        """
        visited = set()
        queue = [(start_sub_id, 0)]  # (sub_id, hop_count)
        visited.add(start_sub_id)
        
        while queue:
            current_sub, hop = queue.pop(0)
            
            if hop >= max_hop:
                continue
            
            # 找出所有连接到当前变电站的线路
            connected_subs = set()
            
            # 通过 Origin 端连接的线路
            if hasattr(observation, 'line_or_to_subid'):
                or_lines = np.where(observation.line_or_to_subid == current_sub)[0]
                for line_id in or_lines:
                    # 检查线路是否连接
                    if hasattr(observation, 'line_status') and observation.line_status[line_id]:
                        # 获取线路的另一端（Extremity）
                        if hasattr(observation, 'line_ex_to_subid'):
                            ex_sub = int(observation.line_ex_to_subid[line_id])
                            if ex_sub not in visited:
                                connected_subs.add(ex_sub)
            
            # 通过 Extremity 端连接的线路
            if hasattr(observation, 'line_ex_to_subid'):
                ex_lines = np.where(observation.line_ex_to_subid == current_sub)[0]
                for line_id in ex_lines:
                    # 检查线路是否连接
                    if hasattr(observation, 'line_status') and observation.line_status[line_id]:
                        # 获取线路的另一端（Origin）
                        if hasattr(observation, 'line_or_to_subid'):
                            or_sub = int(observation.line_or_to_subid[line_id])
                            if or_sub not in visited:
                                connected_subs.add(or_sub)
            
            # 将新发现的变电站加入队列
            for next_sub in connected_subs:
                if next_sub not in visited:
                    visited.add(next_sub)
                    queue.append((next_sub, hop + 1))
        
        return visited
    
    def _get_generators_in_subs(
        self, 
        observation: BaseObservation, 
        sub_ids: Set[int]
    ) -> List[int]:
        """
        获取指定变电站集合中的所有发电机ID
        
        Args:
            observation: Grid2Op 观测对象
            sub_ids: 变电站ID集合
            
        Returns:
            发电机ID列表
        """
        if not hasattr(observation, 'gen_to_subid'):
            return []
        
        generator_ids = []
        for gen_id in range(observation.n_gen):
            gen_sub_id = int(observation.gen_to_subid[gen_id])
            if gen_sub_id in sub_ids:
                generator_ids.append(gen_id)
        
        return generator_ids

