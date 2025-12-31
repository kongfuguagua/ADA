# -*- coding: utf-8 -*-
"""
ADA 系统基础数据结构定义
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from grid2op.Action import BaseAction


@dataclass
class CandidateAction:
    """
    候选动作数据结构
    
    用于在 Planner、Solver、Judger 和 Simulator 之间传递动作候选。
    """
    
    source: str
    """动作来源标识，如 "Expert_Topo", "Math_Dispatch", "LLM_Fusion", "LLM_Recovery" """
    
    action_obj: BaseAction
    """Grid2Op 动作对象（核心载体）"""
    
    description: str = ""
    """动作的自然语言或代码描述"""
    
    simulation_result: Optional[Dict[str, Any]] = None
    """仿真结果（由 Simulator 填充）
    
    包含以下字段：
    - is_safe: bool - 是否安全（无异常、无发散、无解列）
    - rho_max: float - 最大线路负载率
    - reward: float - Grid2Op 返回的奖励
    - margin: float - 安全裕度（1.0 - rho_max）
    - exception: Optional[str] - 异常信息（如果有）
    """
    
    priority: int = 0
    """优先级（用于初步排序，数值越大优先级越高）"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """额外的元数据，用于存储来源特定的信息"""
    
    def __post_init__(self):
        """初始化后处理"""
        if self.simulation_result is None:
            self.simulation_result = {
                "is_safe": False,
                "rho_max": float('inf'),
                "reward": float('-inf'),
                "margin": float('-inf'),
                "exception": None
            }
    
    def update_simulation_result(self, **kwargs) -> None:
        """更新仿真结果"""
        if self.simulation_result is None:
            self.simulation_result = {}
        self.simulation_result.update(kwargs)
    
    def is_valid(self) -> bool:
        """检查动作是否有效（有动作对象）"""
        return self.action_obj is not None
    
    def __hash__(self) -> int:
        """基于动作对象的哈希值（用于去重）"""
        # Grid2Op Action 对象应该支持哈希
        try:
            return hash(self.action_obj)
        except TypeError:
            # 如果 Action 对象不支持哈希，使用字符串表示
            return hash(str(self.action_obj))
    
    def __eq__(self, other) -> bool:
        """基于动作对象比较（用于去重）"""
        if not isinstance(other, CandidateAction):
            return False
        # 比较动作对象是否相同
        try:
            return self.action_obj == other.action_obj
        except:
            return str(self.action_obj) == str(other.action_obj)
    
    def __repr__(self) -> str:
        """字符串表示"""
        safe_str = "✓" if (self.simulation_result and self.simulation_result.get("is_safe")) else "✗"
        rho_str = f"{self.simulation_result.get('rho_max', 'N/A'):.3f}" if self.simulation_result else "N/A"
        return f"CandidateAction(source={self.source}, safe={safe_str}, rho_max={rho_str}, priority={self.priority})"

