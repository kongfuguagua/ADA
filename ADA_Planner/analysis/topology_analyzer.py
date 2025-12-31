# -*- coding: utf-8 -*-
"""
拓扑影响分析器 (Topology Analyzer)

基于图连接性的启发式分析，识别对过载线路最有影响力的变电站。
"""

from typing import List, Dict, Any
import numpy as np
from grid2op.Observation import BaseObservation


class TopologyAnalyzer:
    """
    拓扑影响分析器
    
    目标：识别与过载线路物理上最相关的变电站，用于拓扑调整决策。
    """
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def identify_critical_substations(
        self, 
        observation: BaseObservation, 
        line_id: int
    ) -> List[Dict[str, Any]]:
        """
        识别过载线路两端的关键变电站
        
        Args:
            observation: Grid2Op 观测对象
            line_id: 过载线路ID
            
        Returns:
            关键变电站列表，每个元素包含：
            - sub_id: 变电站ID
            - reason: 原因描述
            - objects: 该变电站连接的设备列表（发电机、负荷等）
        """
        critical_subs = []
        
        # 1. 获取线路两端的变电站ID
        if not hasattr(observation, 'line_or_to_subid') or not hasattr(observation, 'line_ex_to_subid'):
            return critical_subs
        
        origin_sub_id = int(observation.line_or_to_subid[line_id])
        extremity_sub_id = int(observation.line_ex_to_subid[line_id])
        
        # 2. 分析 Origin 端变电站
        origin_info = self._analyze_substation(observation, origin_sub_id, "Origin")
        if origin_info:
            critical_subs.append(origin_info)
        
        # 3. 分析 Extremity 端变电站（如果与 Origin 不同）
        if extremity_sub_id != origin_sub_id:
            extremity_info = self._analyze_substation(observation, extremity_sub_id, "Extremity")
            if extremity_info:
                critical_subs.append(extremity_info)
        
        # 4. 按影响力排序（连接设备多的优先）
        critical_subs.sort(key=lambda x: len(x.get('objects', [])), reverse=True)
        
        return critical_subs
    
    def _analyze_substation(
        self, 
        observation: BaseObservation, 
        sub_id: int, 
        position: str
    ) -> Dict[str, Any]:
        """
        分析单个变电站
        
        Args:
            observation: Grid2Op 观测对象
            sub_id: 变电站ID
            position: 位置描述（"Origin" 或 "Extremity"）
            
        Returns:
            变电站信息字典，如果变电站不可操作则返回 None
        """
        objects = []
        
        # 检查变电站是否在冷却中
        if hasattr(observation, 'time_before_cooldown_sub'):
            cooldown = int(observation.time_before_cooldown_sub[sub_id])
            if cooldown > 0:
                # 冷却中的变电站暂时不可操作，但仍记录
                pass
        
        # 统计连接的发电机
        if hasattr(observation, 'gen_to_subid'):
            gen_ids = np.where(observation.gen_to_subid == sub_id)[0]
            for gen_id in gen_ids:
                gen_p = float(observation.gen_p[gen_id]) if hasattr(observation, 'gen_p') else 0.0
                is_redispatchable = bool(observation.gen_redispatchable[gen_id]) if hasattr(observation, 'gen_redispatchable') else False
                objects.append({
                    "type": "generator",
                    "id": int(gen_id),
                    "power_mw": gen_p,
                    "redispatchable": is_redispatchable
                })
        
        # 统计连接的负荷
        if hasattr(observation, 'load_to_subid'):
            load_ids = np.where(observation.load_to_subid == sub_id)[0]
            for load_id in load_ids:
                load_p = float(observation.load_p[load_id]) if hasattr(observation, 'load_p') else 0.0
                objects.append({
                    "type": "load",
                    "id": int(load_id),
                    "power_mw": load_p
                })
        
        # 统计连接的线路（不包括当前分析的线路）
        connected_lines = []
        if hasattr(observation, 'line_or_to_subid'):
            or_lines = np.where(observation.line_or_to_subid == sub_id)[0]
            connected_lines.extend(or_lines.tolist())
        if hasattr(observation, 'line_ex_to_subid'):
            ex_lines = np.where(observation.line_ex_to_subid == sub_id)[0]
            connected_lines.extend(ex_lines.tolist())
        
        # 去重
        connected_lines = list(set(connected_lines))
        for line_id_conn in connected_lines:
            objects.append({
                "type": "line",
                "id": int(line_id_conn)
            })
        
        # 构建原因描述
        gen_count = sum(1 for obj in objects if obj['type'] == 'generator')
        load_count = sum(1 for obj in objects if obj['type'] == 'load')
        line_count = sum(1 for obj in objects if obj['type'] == 'line')
        
        reason_parts = [f"直接连接过载线路({position}端)"]
        if gen_count > 0:
            reason_parts.append(f"连接{gen_count}个发电机")
        if load_count > 0:
            reason_parts.append(f"连接{load_count}个负荷")
        if line_count > 1:  # 至少2条线路（包括过载线路本身）
            reason_parts.append(f"连接{line_count}条线路")
        
        reason = "，".join(reason_parts)
        
        return {
            "sub_id": sub_id,
            "reason": reason,
            "objects": objects,
            "gen_count": gen_count,
            "load_count": load_count,
            "line_count": line_count
        }

