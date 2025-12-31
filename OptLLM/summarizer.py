# -*- coding: utf-8 -*-
"""
状态摘要器
将复杂的 Observation 压缩为 LLM 决策所需的宏观特征
大幅减少 Token 消耗
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from grid2op.Observation import BaseObservation


class StateSummarizer:
    """
    状态摘要器
    
    将复杂的电网观测压缩为简洁的诊断报告，只包含 LLM 决策所需的关键信息。
    不包含详细的发电机列表、线路列表等静态数据。
    """
    
    def __init__(self):
        """初始化摘要器"""
        pass
    
    def summarize(self, observation: BaseObservation, last_feedback: str = None) -> Dict[str, Any]:
        """
        生成状态摘要
        
        Args:
            observation: 当前电网观测
            last_feedback: 上一次操作的反馈（如果有）
            
        Returns:
            摘要字典，包含：
                - risk_level: 风险等级 (0-4)
                - max_rho: 最大负载率
                - overflow_lines: 过载线路信息
                - gen_capability: 发电机调节能力汇总
                - time_to_overflow: 预估过载时间
        """
        max_rho = float(observation.rho.max())
        overflow_mask = observation.rho > 1.0
        overflow_count = int(overflow_mask.sum())
        
        # 风险等级评估
        risk_level = self._assess_risk_level(max_rho, overflow_count, observation)
        
        # 过载线路信息（只保留最严重的几条）
        overflow_lines = self._extract_overflow_lines(observation, max_lines=3)
        
        # 发电机调节能力汇总（不列出每个发电机）
        gen_capability = self._summarize_gen_capability(observation)
        
        # 预估过载时间（简单启发式）
        time_to_overflow = self._estimate_time_to_overflow(observation)
        
        summary = {
            "risk_level": risk_level,
            "risk_description": self._get_risk_description(risk_level),
            "max_rho": max_rho,
            "max_rho_percent": f"{max_rho:.1%}",
            "overflow_count": overflow_count,
            "overflow_lines": overflow_lines,
            "gen_capability": gen_capability,
            "time_to_overflow": time_to_overflow,
            "last_feedback": last_feedback,
        }
        
        return summary
    
    def _assess_risk_level(
        self, 
        max_rho: float, 
        overflow_count: int,
        observation: BaseObservation
    ) -> int:
        """
        评估风险等级
        
        Returns:
            0: 安全
            1: 注意
            2: 警告
            3: 危险
            4: 极度危险
        """
        if max_rho < 0.85:
            return 0
        elif max_rho < 0.95:
            return 1
        elif max_rho < 1.05:
            return 2
        elif max_rho < 1.15:
            return 3
        else:
            return 4
    
    def _get_risk_description(self, risk_level: int) -> str:
        """获取风险等级描述"""
        descriptions = {
            0: "✅ 安全",
            1: "⚡ 注意",
            2: "⚠️ 警告",
            3: "🔴 危险",
            4: "🔴 极度危险"
        }
        return descriptions.get(risk_level, "未知")
    
    def _extract_overflow_lines(
        self, 
        observation: BaseObservation, 
        max_lines: int = 3
    ) -> List[Dict[str, Any]]:
        """提取过载线路信息（只保留最严重的几条）"""
        overflow_mask = observation.rho > 1.0
        if not np.any(overflow_mask):
            return []
        
        overflow_indices = np.where(overflow_mask)[0]
        overflow_rhos = observation.rho[overflow_indices]
        
        # 按负载率降序排序
        sorted_indices = np.argsort(-overflow_rhos)
        
        lines = []
        for i in sorted_indices[:max_lines]:
            line_id = int(overflow_indices[i])
            rho_val = float(overflow_rhos[i])
            lines.append({
                "line_id": line_id,
                "rho": rho_val,
                "rho_percent": f"{rho_val:.1%}"
            })
        
        return lines
    
    def _summarize_gen_capability(self, observation: BaseObservation) -> Dict[str, float]:
        """
        汇总发电机调节能力（不列出每个发电机）
        
        只返回总量，不返回详细列表
        """
        if not hasattr(observation, 'gen_redispatchable'):
            return {"total_margin_up": 0.0, "total_margin_down": 0.0, "count": 0}
        
        redispatchable_mask = observation.gen_redispatchable
        if not np.any(redispatchable_mask):
            return {"total_margin_up": 0.0, "total_margin_down": 0.0, "count": 0}
        
        total_margin_up = float(observation.gen_margin_up[redispatchable_mask].sum())
        total_margin_down = float(observation.gen_margin_down[redispatchable_mask].sum())
        count = int(redispatchable_mask.sum())
        
        return {
            "total_margin_up": total_margin_up,
            "total_margin_down": total_margin_down,
            "count": count
        }
    
    def _estimate_time_to_overflow(self, observation: BaseObservation) -> str:
        """
        预估过载时间（简单启发式）
        
        这是一个简化的估计，实际应该考虑负荷变化趋势
        """
        max_rho = float(observation.rho.max())
        
        if max_rho < 0.9:
            return "充足"
        elif max_rho < 0.95:
            return "较充足"
        elif max_rho < 1.0:
            return "紧迫"
        elif max_rho < 1.1:
            return "非常紧迫"
        else:
            return "立即"
    
    def format_summary(self, summary: Dict[str, Any]) -> str:
        """
        将摘要格式化为文本（用于 Prompt）
        
        这是一个极简的格式，只包含决策所需的关键信息
        """
        lines = []
        
        # 风险等级
        lines.append(f"【风险等级】{summary['risk_description']} (Level {summary['risk_level']})")
        lines.append(f"【最大负载率】{summary['max_rho_percent']}")
        
        # 过载线路
        if summary['overflow_count'] > 0:
            lines.append(f"【过载线路】{summary['overflow_count']} 条")
            for line_info in summary['overflow_lines']:
                lines.append(f"  - 线路 {line_info['line_id']}: {line_info['rho_percent']}")
        else:
            lines.append("【过载线路】无")
        
        # 发电机能力
        gen_cap = summary['gen_capability']
        lines.append(f"【调节能力】{gen_cap['count']} 台可调度机组，总调节能力: +{gen_cap['total_margin_up']:.1f}MW / -{gen_cap['total_margin_down']:.1f}MW")
        
        # 过载时间
        lines.append(f"【过载时间】{summary['time_to_overflow']}")
        
        # 上次反馈
        if summary.get('last_feedback'):
            lines.append(f"【上次反馈】{summary['last_feedback']}")
        
        return "\n".join(lines)
