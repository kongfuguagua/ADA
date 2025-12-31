# -*- coding: utf-8 -*-
"""
观测文本化模块
将 Grid2Op Observation 转换为自然语言描述
"""

from typing import List
import numpy as np
from grid2op.Observation import BaseObservation


class ObservationFormatter:
    """
    将 Grid2Op 观测转换为文本描述
    
    功能：
    - 提取关键信息：最大负载率、过载线路、断开线路、发电机状态
    - 生成简洁明了的自然语言描述
    """
    
    def __init__(self, max_overflow_lines: int = 5):
        """
        初始化格式化器
        
        Args:
            max_overflow_lines: 最多显示多少条过载线路
        """
        self.max_overflow_lines = max_overflow_lines
    
    def format(self, observation: BaseObservation, compact: bool = False) -> str:
        """
        将观测转换为文本描述（精简版：仅输出关键子图）
        
        Args:
            observation: Grid2Op 观测对象
            compact: 是否使用精简模式（仅输出过载线路及其 2-hop 邻居）
            
        Returns:
            文本描述字符串
        """
        if compact:
            return self._format_compact(observation)
        
        # 完整模式（保持向后兼容）
        return self._format_full(observation)
    
    def _format_compact(self, observation: BaseObservation) -> str:
        """
        精简模式：仅输出关键子图（过载线路及其 2-hop 邻居）
        大幅减少 Token 消耗
        """
        lines = []
        
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        lines.append(f"最大负载率: {max_rho:.2%}")
        
        if overflow_count > 0:
            lines.append(f"过载线路数: {overflow_count}")
            overflow_lines = self._get_overflow_lines(observation)
            
            # 提取关键子图：过载线路及其邻居
            critical_subs = set()
            for line_id, rho_val in overflow_lines[:3]:  # 只处理 Top-3
                if hasattr(observation, 'line_or_to_subid') and hasattr(observation, 'line_ex_to_subid'):
                    or_sub_id = int(observation.line_or_to_subid[line_id])
                    ex_sub_id = int(observation.line_ex_to_subid[line_id])
                    critical_subs.add(or_sub_id)
                    critical_subs.add(ex_sub_id)
                    lines.append(f"  线路 {line_id}: {rho_val:.2%} (Sub {or_sub_id}<->{ex_sub_id})")
            
            # 提取关键发电机（连接到关键变电站的）
            if hasattr(observation, 'gen_to_subid'):
                critical_gens = []
                for gen_id in range(min(observation.n_gen, 20)):
                    gen_sub_id = int(observation.gen_to_subid[gen_id])
                    if gen_sub_id in critical_subs:
                        gen_p = float(observation.gen_p[gen_id])
                        is_redispatchable = bool(observation.gen_redispatchable[gen_id]) if hasattr(observation, 'gen_redispatchable') else False
                        if is_redispatchable:
                            critical_gens.append(f"Gen {gen_id}@{gen_sub_id}: {gen_p:.1f}MW")
                
                if critical_gens:
                    lines.append(f"关键发电机: {', '.join(critical_gens[:5])}")
        else:
            lines.append("状态: 安全")
        
        return "\n".join(lines)
    
    def _format_full(self, observation: BaseObservation) -> str:
        """完整模式（原有逻辑）"""
        lines = []
        
        # 1. 全局状态
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        near_overflow_count = int((observation.rho > 0.9).sum())
        total_load = float(observation.load_p.sum())
        total_gen = float(observation.gen_p.sum())
        
        lines.append("=== 当前电网状态 ===")
        lines.append(f"最大负载率: {max_rho:.2%}")
        
        # 判断状态等级
        if max_rho > 1.0:
            lines.append(f"状态: ⚠️ 危险 (存在过载)")
        elif max_rho > 0.95:
            lines.append(f"状态: ⚠️ 警告 (接近过载)")
        elif max_rho > 0.85:
            lines.append(f"状态: ⚡ 注意 (负载较高)")
        else:
            lines.append(f"状态: ✅ 安全")
        
        lines.append(f"总负荷: {total_load:.2f} MW")
        lines.append(f"总发电: {total_gen:.2f} MW")
        lines.append("")
        
        # 2. 过载线路详情
        if overflow_count > 0:
            lines.append(f"过载线路数: {overflow_count}")
            overflow_lines = self._get_overflow_lines(observation)
            for line_id, rho_val in overflow_lines[:self.max_overflow_lines]:
                line_topo_info = ""
                if hasattr(observation, 'line_or_to_subid') and hasattr(observation, 'line_ex_to_subid'):
                    or_sub_id = int(observation.line_or_to_subid[line_id])
                    ex_sub_id = int(observation.line_ex_to_subid[line_id])
                    line_topo_info = f" (变电站 {or_sub_id} <-> {ex_sub_id})"
                lines.append(f"  - 线路 {line_id}{line_topo_info}: 负载率 {rho_val:.2%} (过载)")
            if len(overflow_lines) > self.max_overflow_lines:
                lines.append(f"  ... 还有 {len(overflow_lines) - self.max_overflow_lines} 条过载线路")
            lines.append("")
        
        # 3. 接近过载的线路
        if overflow_count == 0 and near_overflow_count > 0:
            lines.append(f"接近过载的线路数: {near_overflow_count}")
            near_overflow_lines = self._get_near_overflow_lines(observation)
            for line_id, rho_val in near_overflow_lines[:self.max_overflow_lines]:
                lines.append(f"  - 线路 {line_id}: 负载率 {rho_val:.2%}")
            if len(near_overflow_lines) > self.max_overflow_lines:
                lines.append(f"  ... 还有 {len(near_overflow_lines) - self.max_overflow_lines} 条接近过载的线路")
            lines.append("")
        
        # 4. 断开线路
        disconnected_lines = self._get_disconnected_lines(observation)
        if len(disconnected_lines) > 0:
            lines.append(f"断开线路数: {len(disconnected_lines)}")
            for line_id in disconnected_lines[:self.max_overflow_lines]:
                cooldown = int(observation.time_before_cooldown_line[line_id])
                if cooldown > 0:
                    lines.append(f"  - 线路 {line_id}: 断开 (冷却时间: {cooldown} 步)")
                else:
                    lines.append(f"  - 线路 {line_id}: 断开 (可重连)")
            if len(disconnected_lines) > self.max_overflow_lines:
                lines.append(f"  ... 还有 {len(disconnected_lines) - self.max_overflow_lines} 条断开线路")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_overflow_lines(self, observation: BaseObservation) -> List[tuple]:
        """获取过载线路列表（按负载率降序）"""
        overflow_mask = observation.rho > 1.0
        overflow_indices = np.where(overflow_mask)[0]
        overflow_rhos = observation.rho[overflow_indices]
        
        # 按负载率降序排序
        sorted_indices = np.argsort(-overflow_rhos)
        return [(int(overflow_indices[i]), float(overflow_rhos[i])) for i in sorted_indices]
    
    def _get_near_overflow_lines(self, observation: BaseObservation) -> List[tuple]:
        """获取接近过载的线路列表（按负载率降序）"""
        near_overflow_mask = (observation.rho > 0.9) & (observation.rho <= 1.0)
        near_overflow_indices = np.where(near_overflow_mask)[0]
        near_overflow_rhos = observation.rho[near_overflow_indices]
        
        sorted_indices = np.argsort(-near_overflow_rhos)
        return [(int(near_overflow_indices[i]), float(near_overflow_rhos[i])) for i in sorted_indices]
    
    def _get_disconnected_lines(self, observation: BaseObservation) -> List[int]:
        """获取断开线路列表"""
        if not hasattr(observation, 'line_status'):
            return []
        disconnected_mask = ~observation.line_status
        return [int(i) for i in np.where(disconnected_mask)[0]]

