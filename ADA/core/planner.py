# -*- coding: utf-8 -*-
"""
Planner Core Module (Topology Expert) - ExpertAgent State Manager
"""
 
from __future__ import annotations
import logging
import re
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
            
            # 过滤动作：如果动作操作超出grid限制的节点，返回空动作
            if self.should_filter_action(action):
                logger.info(f"Planner: Action filtered: operating on nodes beyond grid size limit for {self.grid_name}")
                return candidates  # 返回空列表（相当于空动作）
            
            # === 关键：状态更新（与 ExpertAgent 完全一致）===
            # 1. 如果是切分变电站 (sub_id_to_split != -1)
            sub_split = expert_result.get("sub_id_to_split", -1)
            if sub_split != -1:
                self.sub_2nodes.add(int(sub_split))
                logger.info(f"Planner: Action implies splitting Substation {sub_split}")
            
            # 2. 如果是恢复操作（从兜底逻辑返回的恢复动作）
            sub_id_to_discard = expert_result.get("sub_id_to_discard", None)
            if sub_id_to_discard is not None:
                self.sub_2nodes.discard(sub_id_to_discard)
                logger.info(f"Planner: Action implies merging Substation {sub_id_to_discard}")
            
            candidates.append(CandidateAction(
                source="Expert_Replica",
                action_obj=action,
                description=expert_result.get("description", "Expert Action"),
                priority=expert_result.get("score", 1)
            ))
            return candidates

        # 2. 如果无过载，尝试恢复或重连
        # 恢复拓扑
        recovery_action, recovery_sub_id = self._expert_service.check_recovery(observation, self.sub_2nodes)
        if recovery_action is not None:
            # 过滤动作：如果动作操作超出grid限制的节点，跳过
            if self.should_filter_action(recovery_action):
                logger.info(f"Planner: Recovery action filtered: operating on nodes beyond grid size limit for {self.grid_name}")
            else:
                # ExpertAgent 逻辑：恢复成功时立即更新状态
                if recovery_sub_id is not None:
                    self.sub_2nodes.discard(recovery_sub_id)
                    logger.info(f"Planner: Recovery action will merge Substation {recovery_sub_id}")
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
            # 过滤动作：如果动作操作超出grid限制的节点，跳过
            if self.should_filter_action(reco_action):
                logger.info(f"Planner: Reconnect action filtered: operating on nodes beyond grid size limit for {self.grid_name}")
            else:
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

    def should_filter_action(self, action):
        """
        检查动作是否应该被过滤（基于grid_name规则）。
        从grid_name中提取数字（如"IEEE9" -> 9），如果动作操作任何ID > (数字-1) 的变电站/节点，则过滤。
        
        参数
        ----------
        action: :class:`grid2op.Action.PlayableAction`
            要检查的动作
            
        返回
        -------
        bool
            True 如果动作应该被过滤（拒绝），False 否则
        """
        # 从grid_name中提取数字（如"IEEE9" -> 9, "IEEE14" -> 14）
        grid_upper = self.grid_name.upper()
        match = re.search(r'IEEE(\d+)', grid_upper)
        if not match:
            return False
        
        max_node_id = int(match.group(1)) - 1  # IEEE9意味着节点0-8，所以max是8 (9-1)
        
        # 检查动作是否操作任何ID > max_node_id 的变电站/节点
        try:
            # 方法1: 检查 set_bus substations_id - 提取变电站ID
            if hasattr(action, 'set_bus'):
                set_bus = action.set_bus
                if set_bus is not None:
                    # 检查 substations_id 格式: [(sub_id, topo_vec), ...]
                    if hasattr(set_bus, 'substations_id'):
                        subs_data = set_bus.substations_id
                        if subs_data is not None:
                            if isinstance(subs_data, list):
                                for item in subs_data:
                                    if isinstance(item, (tuple, list)) and len(item) >= 1:
                                        sub_id = int(item[0])
                                        if sub_id > max_node_id:
                                            return True
                            elif isinstance(subs_data, dict):
                                # 如果是dict，检查所有值
                                for key, value in subs_data.items():
                                    if isinstance(value, list):
                                        for item in value:
                                            if isinstance(item, (tuple, list)) and len(item) >= 1:
                                                sub_id = int(item[0])
                                                if sub_id > max_node_id:
                                                    return True
            
            # 方法2: 尝试 as_dict() 方法获取所有变电站ID
            if hasattr(action, 'as_dict'):
                action_dict = action.as_dict()
                if 'set_bus' in action_dict and action_dict['set_bus'] is not None:
                    set_bus_data = action_dict['set_bus']
                    if isinstance(set_bus_data, dict):
                        # 检查 substations_id
                        if 'substations_id' in set_bus_data:
                            subs_data = set_bus_data['substations_id']
                            if isinstance(subs_data, list):
                                for item in subs_data:
                                    if isinstance(item, (tuple, list)) and len(item) >= 1:
                                        sub_id = int(item[0])
                                        if sub_id > max_node_id:
                                            return True
            
            # 方法3: 尝试 impact_on_substation 获取受影响的变电站ID
            if hasattr(action, 'impact_on_substation'):
                affected_subs = action.impact_on_substation()
                if affected_subs is not None:
                    # affected_subs 是布尔数组，检查索引 > max_node_id
                    for sub_id in range(len(affected_subs)):
                        if sub_id > max_node_id and affected_subs[sub_id]:
                            return True
        except Exception as e:
            logger.warning(f"Planner: Error checking action node IDs: {e}")
            # 如果无法确定，不过滤（更安全地允许动作）
            return False
        
        return False