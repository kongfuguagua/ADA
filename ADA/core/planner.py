# -*- coding: utf-8 -*-
"""
Planner Core Module (Topology Expert) - ExpertAgent State Manager
"""
 
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
from grid2op.Observation import BaseObservation
from grid2op.Action import ActionSpace
from ADA.utils.definitions import CandidateAction
from utils import get_logger
from ADA.analysis.expert_insight import ExpertInsightService

logger = get_logger("ADA.Planner")

class Planner:
    """
    Planner: ExpertAgent 的状态管理者和接口适配器
    """

    def __init__(
        self,
        action_space: ActionSpace,
        observation_space,
        grid_name: str = "IEEE118",
        max_candidates: int = 1, # ExpertAgent 通常只产生一个最佳动作
        **kwargs
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.grid_name = grid_name
        self.max_candidates = max_candidates
        
        # ExpertAgent 状态追踪
        self.sub_2nodes = set()
        self.lines_disconnected = set()
        
        self._expert_service = None
        try:
            self._expert_service = ExpertInsightService(
                action_space=action_space,
                observation_space=observation_space,
                grid_name=grid_name
            )
            logger.info("Planner: ExpertInsightService (Replica) initialized.")
        except Exception as exc:
            logger.error(f"Planner: Failed to init ExpertInsightService: {exc}")

    def suggest_topologies(self, observation: BaseObservation) -> List[CandidateAction]:
        """
        生成拓扑动作建议。
        """
        candidates: List[CandidateAction] = []
        if not self._expert_service:
            return candidates

        # 1. 尝试解决过载
        expert_result = self._expert_service.resolve_overload(
            observation, self.sub_2nodes, self.lines_disconnected
        )
        
        if expert_result["action"] is not None:
            # 找到了解决过载的方案
            action = expert_result["action"]
            
            # === 关键：状态更新 ===
            # ExpertAgent 在决定动作时更新 sub_2nodes
            # 如果是切分变电站 (sub_id_to_split != -1)
            sub_split = expert_result.get("sub_id_to_split", -1)
            if sub_split != -1:
                self.sub_2nodes.add(int(sub_split))
                logger.info(f"Planner: Action implies splitting Substation {sub_split}")
                
            # 注意：恢复操作(Merge)的状态清理通常在执行后或下一次观测确认后
            # 但为了保持一致性，如果 ExpertService 返回了恢复动作，它内部逻辑通常会涉及 sub_2nodes 的处理
            # 简单起见，我们依赖 Observation 的反馈来清理 sub_2nodes (在 reset 或下一帧检测)
            # 或者这里暂不主动 discard，除非 ExpertResult 明确指示
            
            candidates.append(CandidateAction(
                source="Expert_Replica",
                action_obj=action,
                description=expert_result.get("description", "Expert Action"),
                priority=expert_result.get("score", 1)
            ))
            return candidates

        # 2. 如果无过载，尝试恢复或重连
        # 恢复拓扑
        recovery_action = self._expert_service.check_recovery(observation, self.sub_2nodes)
        if recovery_action:
            # 假设恢复动作执行成功，我们需要从 sub_2nodes 中移除
            # 但这里只是 Suggest，真正的移除应该在观察到恢复后。
            # 不过 ExpertAgent 是在 act() 里直接 discard 的。我们这里模拟这种行为。
            # 为了防止反复 suggest 同一个恢复，我们需要确保它有效。
            # 这里的移除只是 Planner 内部状态的预测更新，如果动作失败，可能需要机制加回来（暂忽略）
            candidates.append(CandidateAction(
                source="Expert_Recovery",
                action_obj=recovery_action,
                description="Recover Reference Topology",
                priority=1
            ))
            return candidates

        # 重连线路
        reco_action = self._expert_service.check_reconnection(observation)
        if reco_action:
            candidates.append(CandidateAction(
                source="Expert_Reconnect",
                action_obj=reco_action,
                description="Reconnect Line",
                priority=0
            ))
            
        return candidates

    def reset(self, observation: Optional[BaseObservation] = None) -> None:
        """重置状态"""
        self.sub_2nodes.clear()
        self.lines_disconnected.clear()
        logger.info("Planner: State reset.")
        
    def update_state_from_observation(self, observation: BaseObservation):
        """
        可选：基于实际观测校准 sub_2nodes
        如果外部系统调用此方法，可以增强鲁棒性
        """
        # 检查哪些站实际上是 split 的 (topo_vect 中有 1 和 2)
        # 这可以防止 Planner 状态与环境脱节
        pass