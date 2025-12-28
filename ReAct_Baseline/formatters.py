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
    
    def format(self, observation: BaseObservation) -> str:
        """
        将观测转换为文本描述
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            文本描述字符串
        """
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
                # 获取线路拓扑信息（连接的变电站）
                line_topo_info = ""
                if hasattr(observation, 'line_or_to_subid') and hasattr(observation, 'line_ex_to_subid'):
                    or_sub_id = int(observation.line_or_to_subid[line_id])
                    ex_sub_id = int(observation.line_ex_to_subid[line_id])
                    line_topo_info = f" (变电站 {or_sub_id} <-> {ex_sub_id})"
                lines.append(f"  - 线路 {line_id}{line_topo_info}: 负载率 {rho_val:.2%} (过载)")
            if len(overflow_lines) > self.max_overflow_lines:
                lines.append(f"  ... 还有 {len(overflow_lines) - self.max_overflow_lines} 条过载线路")
            lines.append("")
        
        # 3. 接近过载的线路（如果没有过载）
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
        
        # 5. 关键发电机信息（可选，用于再调度）
        # 只显示可调度的发电机
        if hasattr(observation, 'gen_redispatchable'):
            redispatchable_gens = np.where(observation.gen_redispatchable)[0]
            if len(redispatchable_gens) > 0:
                lines.append(f"可调度发电机数: {len(redispatchable_gens)}")
                # 显示所有可调度发电机的状态（最多显示5个）
                for gen_id in redispatchable_gens[:5]:
                    gen_p = float(observation.gen_p[gen_id])
                    gen_pmax = float(observation.gen_pmax[gen_id])
                    gen_pmin = float(observation.gen_pmin[gen_id])
                    # 计算可调整范围
                    max_increase = gen_pmax - gen_p
                    max_decrease = gen_p - gen_pmin
                    
                    # 获取爬坡速率限制（关键约束）
                    ramp_info = ""
                    if hasattr(observation, 'gen_max_ramp_up') and hasattr(observation, 'gen_max_ramp_down'):
                        max_ramp_up = float(observation.gen_max_ramp_up[gen_id])
                        max_ramp_down = float(observation.gen_max_ramp_down[gen_id])
                        # 实际可调整量受爬坡速率限制
                        actual_max_increase = min(max_increase, max_ramp_up)
                        actual_max_decrease = min(max_decrease, max_ramp_down)
                        ramp_info = f", 爬坡限制: ±{max_ramp_up:.2f} MW/步 (实际可调: +{actual_max_increase:.2f}/-{actual_max_decrease:.2f} MW)"
                    
                    # 获取拓扑信息（发电机连接的变电站）
                    topo_info = ""
                    if hasattr(observation, 'gen_to_subid'):
                        gen_sub_id = int(observation.gen_to_subid[gen_id])
                        topo_info = f" (变电站 {gen_sub_id})"
                    
                    lines.append(f"  - 发电机 {gen_id}{topo_info}: 当前 {gen_p:.2f} MW (范围: {gen_pmin:.2f} ~ {gen_pmax:.2f} MW, 可增: +{max_increase:.2f} MW, 可减: -{max_decrease:.2f} MW{ramp_info})")
                if len(redispatchable_gens) > 5:
                    lines.append(f"  ... 还有 {len(redispatchable_gens) - 5} 个可调度发电机")
                lines.append("")
        
        # 6. 可用线路信息（用于线路操作）
        if hasattr(observation, 'line_status') and hasattr(observation, 'time_before_cooldown_line'):
            # 显示可操作的线路（冷却时间已过）
            available_lines = []
            for line_id in range(min(observation.n_line, 20)):  # 最多显示20条线路
                cooldown = int(observation.time_before_cooldown_line[line_id])
                is_connected = bool(observation.line_status[line_id])
                if cooldown == 0:
                    available_lines.append((line_id, is_connected))
            
            if len(available_lines) > 0:
                lines.append(f"可用线路操作（冷却时间已过）:")
                for line_id, is_connected in available_lines[:5]:
                    status_str = "已连接" if is_connected else "已断开"
                    lines.append(f"  - 线路 {line_id}: {status_str} (可{'断开' if is_connected else '连接'})")
                if len(available_lines) > 5:
                    lines.append(f"  ... 还有 {len(available_lines) - 5} 条可用线路")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_overflow_lines(self, observation: BaseObservation) -> List[tuple]:
        """
        获取过载线路列表（按负载率降序）
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            [(line_id, rho_value), ...] 列表，按负载率降序排列
        """
        overflow_mask = observation.rho > 1.0
        overflow_indices = np.where(overflow_mask)[0]
        overflow_rhos = observation.rho[overflow_indices]
        
        # 按负载率降序排序
        sorted_indices = np.argsort(-overflow_rhos)
        result = [
            (int(overflow_indices[i]), float(overflow_rhos[i]))
            for i in sorted_indices
        ]
        return result
    
    def _get_near_overflow_lines(self, observation: BaseObservation) -> List[tuple]:
        """
        获取接近过载的线路列表（负载率 > 0.9 但 <= 1.0）
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            [(line_id, rho_value), ...] 列表，按负载率降序排列
        """
        near_overflow_mask = (observation.rho > 0.9) & (observation.rho <= 1.0)
        near_overflow_indices = np.where(near_overflow_mask)[0]
        near_overflow_rhos = observation.rho[near_overflow_indices]
        
        # 按负载率降序排序
        sorted_indices = np.argsort(-near_overflow_rhos)
        result = [
            (int(near_overflow_indices[i]), float(near_overflow_rhos[i]))
            for i in sorted_indices
        ]
        return result
    
    def _get_disconnected_lines(self, observation: BaseObservation) -> List[int]:
        """
        获取断开线路列表
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            [line_id, ...] 列表
        """
        disconnected_mask = ~observation.line_status
        disconnected_indices = np.where(disconnected_mask)[0]
        return [int(i) for i in disconnected_indices]

