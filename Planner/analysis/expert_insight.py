# -*- coding: utf-8 -*-
"""
Expert Insight Service (专家洞察服务)

将 ExpertAgent 中经过验证的、基于规则和数学的求解能力完全剥离并封装为一个高阶工具。
在 Planner 循环启动时，强制执行一次专家诊断，将诊断结果作为强提示注入给 LLM。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

try:
    from alphaDeesp.expert_operator import expert_operator
    from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
    _CAN_USE_EXPERT_AGENT = True
except ImportError as exc_:
    _CAN_USE_EXPERT_AGENT = False
    logging.warning(f"ExpertAgent dependencies not available: {exc_}")

logger = logging.getLogger("ExpertInsight")


class ExpertInsightService:
    """
    专家洞察服务
    
    复用 ExpertAgent 的核心算法，但不直接执行动作，而是返回分析数据。
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        grid_name: str = "IEEE14",
        **kwargs
    ):
        """
        初始化专家洞察服务
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            grid_name: 电网名称（用于特定优化，如 IEEE14, IEEE118, IEEE118_R2）
        """
        if not _CAN_USE_EXPERT_AGENT:
            raise ImportError(
                "ExpertInsightService requires alphaDeesp package. "
                "Please install it: pip install alphaDeesp"
            )
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.grid_name = grid_name
        
        # 复用 ExpertAgent 的配置
        self.config = {
            "totalnumberofsimulatedtopos": 25,
            "numberofsimulatedtopospernode": 5,
            "maxUnusedLines": 2,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }
        self.reward_type = "MinMargin_reward"
        
        # 状态持久化：复用模拟器实例（关键优化：减少初始化开销）
        self._cached_simulator = None
        self._cached_observation = None
        
        logger.info(f"ExpertInsightService initialized for grid: {grid_name}")
    
    def generate_insight(
        self, 
        observation: BaseObservation
    ) -> Dict[str, Any]:
        """
        核心方法：生成专家洞察
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            InsightReport 字典，包含：
            - status: "SAFE" 或 "DANGER"
            - critical_line: 最严重的过载线路ID（如果有）
            - solutions: Top-K 推荐方案列表
        """
        # 1. 识别过载 (Ranked Overloads)
        overloaded_lines = self._get_ranked_overloads(observation)
        
        if not overloaded_lines:
            return {
                "status": "SAFE",
                "suggestion": "Do Nothing or Maintenance",
                "solutions": []
            }
        
        # 聚焦最严重的过载
        target_line = overloaded_lines[0]
        target_rho = float(observation.rho[target_line])
        
        logger.info(f"ExpertInsight: Analyzing critical overload on line {target_line} (rho={target_rho:.2%})")
        
        # 2. 运行 ExpertAgent 的核心搜索逻辑
        try:
            # 处理特殊网格的额外线路（如 IEEE118_R2）
            additional_lines_to_cut, lines_considered = self._additional_lines_to_cut(target_line)
            
            # 状态持久化：复用模拟器实例（关键优化：减少初始化开销）
            # 检查是否可以复用缓存的模拟器
            if (self._cached_simulator is not None and 
                self._cached_observation is not None and
                self._can_reuse_simulator(observation, self._cached_observation)):
                # 更新模拟器状态而不是重新创建
                try:
                    # 注意：Grid2opSimulation 可能不支持直接更新状态
                    # 这里我们尝试复用，如果失败则创建新的
                    simulator = self._cached_simulator
                    # 如果模拟器支持状态更新，在这里更新
                    # 否则，我们仍然需要创建新的（但保留这个接口以便未来优化）
                except:
                    simulator = None
            else:
                simulator = None
            
            # 如果无法复用，创建新的模拟器
            if simulator is None:
                simulator = Grid2opSimulation(
                    observation,
                    self.action_space,
                    self.observation_space,
                    param_options=self.config,
                    debug=False,
                    ltc=[target_line],
                    reward_type=self.reward_type
                )
                # 缓存模拟器和观测（注意：Grid2opSimulation 可能包含状态，需要谨慎）
                # 由于 Grid2opSimulation 可能包含不可序列化的状态，这里只缓存引用
                # 实际使用时，如果观测变化较大，仍需要重新创建
                self._cached_simulator = simulator
                self._cached_observation = observation
            
            # 运行专家算子（核心数学计算）
            ranked_combinations, expert_system_results, actions = expert_operator(
                simulator,
                plot=False,
                debug=False
            )
            
            # 3. 提炼 Top-K 方案
            top_k_solutions = []
            
            if expert_system_results is not None and not expert_system_results.empty:
                # 筛选得分高的方案 (Score >= 3 表示能解决问题)
                # Score 含义：
                # 4 - 解决所有过载
                # 3 - 解决目标过载
                # 2 - 部分解决目标过载
                # 1 - 解决目标过载但恶化其他线路
                # 0 - 失败
                
                best_candidates = expert_system_results[
                    expert_system_results['Topology simulated score'] >= 3
                ].head(3)
                
                for idx, row in best_candidates.iterrows():
                    action_obj = actions[idx]
                    sub_id = int(row['Substation ID'])
                    score = int(row['Topology simulated score'])
                    efficacy = float(row['Efficacity']) if 'Efficacity' in row else 0.0
                    
                    # 提取副作用信息（关键优化：让LLM了解方案的代价）
                    side_effects = self._extract_side_effects(row, observation)
                    
                    # 计算成本预估（关键优化：考虑经济性）
                    cost_estimation = self._estimate_cost(action_obj, observation, score)
                    
                    # 生成动作描述
                    action_description = self._describe_action(action_obj, sub_id, score)
                    
                    # 序列化动作（关键优化：解决黑盒化风险）
                    action_serialized = self._serialize_action(action_obj)
                    
                    solution = {
                        "type": "Topology Action",
                        "substation_id": sub_id,
                        "score": score,
                        "efficacy": efficacy,
                        "description": action_description,
                        "expected_outcome": self._get_expected_outcome(score),
                        "side_effects": side_effects,  # 新增：副作用描述
                        "cost_estimation": cost_estimation,  # 新增：成本预估
                        "action_object": action_obj,  # 保存原始动作对象供后续解析
                        "action_serialized": action_serialized,  # 新增：序列化表达
                        "action_code": self._action_to_code(action_obj, sub_id)  # 转化为 LLM 可读的代码
                    }
                    top_k_solutions.append(solution)
            
            # 4. 生成备选方案（如果拓扑无法解决，建议 redispatch）
            if not top_k_solutions:
                logger.info("ExpertInsight: No topology solution found, generating redispatch hint")
                redispatch_hint = self._analyze_sensitivity(observation, target_line)
                if redispatch_hint:
                    top_k_solutions.append(redispatch_hint)
            
            return {
                "status": "DANGER",
                "critical_line": int(target_line),
                "critical_rho": target_rho,
                "solutions": top_k_solutions
            }
            
        except Exception as e:
            logger.error(f"ExpertInsight: Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            # 降级到简单的灵敏度分析
            redispatch_hint = self._analyze_sensitivity(observation, target_line)
            return {
                "status": "DANGER",
                "critical_line": int(target_line),
                "critical_rho": target_rho,
                "solutions": [redispatch_hint] if redispatch_hint else [],
                "error": str(e)
            }
    
    def _get_ranked_overloads(self, observation: BaseObservation) -> List[int]:
        """
        获取排序后的过载线路列表（按严重程度）
        
        Args:
            observation: Grid2Op 观测对象
            
        Returns:
            过载线路ID列表（按严重程度降序）
        """
        timesteps_overflow_allowed = self.observation_space.obs_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED
        
        # 按负载率降序排序
        sort_rho = -np.sort(-observation.rho)  # 降序
        sort_indices = np.argsort(-observation.rho)
        ltc_list = [int(sort_indices[i]) for i in range(len(sort_rho)) if sort_rho[i] >= 1.0]
        
        # 重新排序：关键过载（即将断开）优先
        ltc_critical = [
            l for l in ltc_list 
            if observation.timestep_overflow[l] == timesteps_overflow_allowed
        ]
        ltc_not_critical = [
            l for l in ltc_list 
            if observation.timestep_overflow[l] != timesteps_overflow_allowed
        ]
        
        return ltc_critical + ltc_not_critical
    
    def _additional_lines_to_cut(
        self, 
        line_to_cut: int
    ) -> Tuple[List[int], List[int]]:
        """
        处理特殊网格的额外线路（如 IEEE118_R2 的并行线路）
        
        Args:
            line_to_cut: 目标线路ID
            
        Returns:
            (additional_lines_to_cut, lines_considered) 元组
        """
        additional_lines_to_cut = []
        lines_considered = []
        
        lines_to_consider = []
        pairs = []
        
        if self.grid_name == "IEEE118_R2":
            lines_to_consider = [22, 23, 33, 35, 34, 32]
            pairs = [(22, 23), (33, 35), (34, 32)]
        elif self.grid_name == "IEEE118":
            lines_to_consider = [135, 136, 149, 147, 148, 146]
            pairs = [(135, 136), (149, 147), (148, 146)]
        
        if line_to_cut in lines_to_consider:
            for p in pairs:
                if line_to_cut in p:
                    additional_lines_to_cut = [l for l in p if l != line_to_cut]
                    lines_considered = lines_to_consider
                    break
        
        return additional_lines_to_cut, lines_considered
    
    def _describe_action(
        self, 
        action: BaseAction, 
        sub_id: int, 
        score: int
    ) -> str:
        """
        生成动作的自然语言描述
        
        Args:
            action: Grid2Op 动作对象
            sub_id: 变电站ID
            score: 专家系统评分
            
        Returns:
            动作描述字符串
        """
        score_desc = {
            4: "完美解决（解决所有过载）",
            3: "有效解决（解决目标过载）",
            2: "部分解决",
            1: "解决但可能恶化其他线路"
        }.get(score, "未知")
        
        return f"对变电站 {sub_id} 进行拓扑调整（评分: {score}/4 - {score_desc}）"
    
    def _get_expected_outcome(self, score: int) -> str:
        """
        根据评分获取预期结果描述
        
        Args:
            score: 专家系统评分
            
        Returns:
            预期结果描述
        """
        outcomes = {
            4: "完全解决所有过载问题",
            3: "解决目标过载线路的过载问题",
            2: "部分缓解目标过载",
            1: "解决目标过载但可能恶化其他线路"
        }
        return outcomes.get(score, "效果未知")
    
    def _action_to_code(self, action: BaseAction, sub_id: int) -> str:
        """
        将动作对象转化为 LLM 可读的代码描述
        
        Args:
            action: Grid2Op 动作对象
            sub_id: 变电站ID
            
        Returns:
            代码描述字符串
        """
        action_dict = action.as_dict()
        
        if 'set_bus' in action_dict and action_dict['set_bus'] is not None:
            # 提取 set_bus 信息
            set_bus_data = action_dict['set_bus']
            if isinstance(set_bus_data, dict) and 'substations_id' in set_bus_data:
                # 简化描述：只说明是拓扑调整
                return f"set_bus(substation_id={sub_id}, topology_vector=[...])"
        
        return f"Topology change on substation {sub_id}"
    
    def _analyze_sensitivity(
        self, 
        observation: BaseObservation, 
        line_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        简化的灵敏度分析（当 Expert System 找不到拓扑解时作为保底）
        
        Args:
            observation: Grid2Op 观测对象
            line_id: 过载线路ID
            
        Returns:
            再调度建议字典，如果无法生成则返回 None
        """
        try:
            # 获取线路两端的变电站
            if not hasattr(observation, 'line_or_to_subid') or not hasattr(observation, 'line_ex_to_subid'):
                return None
            
            origin_sub_id = int(observation.line_or_to_subid[line_id])
            extremity_sub_id = int(observation.line_ex_to_subid[line_id])
            
            # 找出各变电站的发电机
            gen_up_candidates = []
            gen_down_candidates = []
            
            if hasattr(observation, 'gen_to_subid'):
                # Origin 端：建议降出力（减少送端发电）
                origin_gens = np.where(observation.gen_to_subid == origin_sub_id)[0]
                for gen_id in origin_gens:
                    if hasattr(observation, 'gen_redispatchable') and observation.gen_redispatchable[gen_id]:
                        gen_down_candidates.append(int(gen_id))
                
                # Extremity 端：建议升出力（增加受端发电）
                extremity_gens = np.where(observation.gen_to_subid == extremity_sub_id)[0]
                for gen_id in extremity_gens:
                    if hasattr(observation, 'gen_redispatchable') and observation.gen_redispatchable[gen_id]:
                        gen_up_candidates.append(int(gen_id))
            
            if not gen_up_candidates and not gen_down_candidates:
                return None
            
            return {
                "type": "Redispatch",
                "description": f"再调度建议：减少送端（变电站 {origin_sub_id}）发电，增加受端（变电站 {extremity_sub_id}）发电",
                "gen_up": gen_up_candidates[:3],  # 最多3个
                "gen_down": gen_down_candidates[:3],  # 最多3个
                "expected_outcome": "通过改变功率流分布来缓解过载",
                "side_effects": {},  # Redispatch 通常副作用较小
                "cost_estimation": {"type": "redispatch", "estimated_cost": 0.1},  # 再调度成本较低
                "action_object": None,  # Redispatch 动作需要 LLM 生成
                "action_serialized": None,
                "action_code": "redispatch(...)"  # 占位符
            }
        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
            return None
    
    def _can_reuse_simulator(
        self, 
        current_obs: BaseObservation, 
        cached_obs: BaseObservation
    ) -> bool:
        """
        判断是否可以复用缓存的模拟器
        
        Args:
            current_obs: 当前观测
            cached_obs: 缓存的观测
            
        Returns:
            如果可以复用则返回True
        """
        # 简单判断：如果观测的关键属性变化不大，可以复用
        # 注意：这是一个保守的判断，实际可能需要更复杂的逻辑
        try:
            # 检查关键状态是否变化
            if hasattr(current_obs, 'rho') and hasattr(cached_obs, 'rho'):
                # 如果负载率分布变化很大，可能需要重新创建
                rho_diff = np.abs(current_obs.rho - cached_obs.rho).max()
                if rho_diff > 0.1:  # 负载率变化超过10%，重新创建
                    return False
            
            # 检查线路状态是否变化
            if hasattr(current_obs, 'line_status') and hasattr(cached_obs, 'line_status'):
                line_status_diff = (current_obs.line_status != cached_obs.line_status).sum()
                if line_status_diff > 0:  # 有线路状态变化，重新创建
                    return False
            
            return True
        except:
            # 如果判断失败，保守地返回False
            return False
    
    def _extract_side_effects(
        self, 
        row: pd.Series, 
        observation: BaseObservation
    ) -> Dict[str, Any]:
        """
        提取方案的副作用信息（关键优化：让LLM了解方案的代价）
        
        Args:
            row: expert_system_results 的一行数据
            observation: 当前观测
            
        Returns:
            副作用描述字典
        """
        side_effects = {
            "worsened_lines": [],
            "max_rho_after": None,
            "overflow_count_after": None,
            "risk_level": "low"
        }
        
        # 提取恶化的线路（如果有）
        if 'Worsened line' in row:
            worsened = row['Worsened line']
            # 安全地检查是否为 NaN 或空值
            try:
                if pd.isna(worsened):
                    worsened = None
                elif isinstance(worsened, (list, np.ndarray)):
                    # 检查数组是否为空
                    if len(worsened) > 0:
                        side_effects["worsened_lines"] = [int(l) for l in worsened if not pd.isna(l)]
                elif isinstance(worsened, (int, np.integer)):
                    side_effects["worsened_lines"] = [int(worsened)]
            except (ValueError, TypeError):
                # 如果检查失败，跳过
                pass
        
        # 如果有模拟后的负载率信息，提取
        if 'Max rho after' in row:
            try:
                max_rho_val = row['Max rho after']
                if not pd.isna(max_rho_val):
                    side_effects["max_rho_after"] = float(max_rho_val)
            except (ValueError, TypeError):
                pass
        
        # 评估风险等级
        if len(side_effects["worsened_lines"]) > 0:
            side_effects["risk_level"] = "medium"
        if side_effects.get("max_rho_after") and side_effects["max_rho_after"] > 0.95:
            side_effects["risk_level"] = "high"
        
        return side_effects
    
    def _estimate_cost(
        self, 
        action: BaseAction, 
        observation: BaseObservation,
        score: int
    ) -> Dict[str, Any]:
        """
        估算动作的经济成本（关键优化：考虑经济性）
        
        Args:
            action: Grid2Op 动作对象
            observation: 当前观测
            score: 专家系统评分
            
        Returns:
            成本预估字典
        """
        cost = {
            "type": "topology",
            "estimated_cost": 0.0,
            "cost_breakdown": {}
        }
        
        action_dict = action.as_dict()
        
        # 拓扑调整成本：主要是操作成本和潜在风险
        if 'set_bus' in action_dict and action_dict['set_bus'] is not None:
            # 拓扑调整的基础成本
            base_cost = 1.0
            
            # 如果评分较低，增加风险成本
            if score <= 2:
                base_cost += 0.5
            
            cost["estimated_cost"] = base_cost
            cost["cost_breakdown"] = {
                "operation_cost": 0.5,
                "risk_cost": 0.5 if score <= 2 else 0.0
            }
        
        # 线路操作成本（如果有）
        if 'set_line_status' in action_dict and action_dict['set_line_status'] is not None:
            cost["estimated_cost"] += 0.3
            cost["cost_breakdown"]["line_operation"] = 0.3
        
        return cost
    
    def _serialize_action(self, action: BaseAction) -> Dict[str, Any]:
        """
        序列化动作对象（关键优化：解决黑盒化风险）
        
        Args:
            action: Grid2Op 动作对象
            
        Returns:
            序列化后的动作字典
        """
        try:
            action_dict = action.as_dict()
            # 提取关键信息用于序列化
            serialized = {
                "action_type": "unknown",
                "parameters": {}
            }
            
            if 'set_bus' in action_dict and action_dict['set_bus'] is not None:
                serialized["action_type"] = "set_bus"
                set_bus_data = action_dict['set_bus']
                if isinstance(set_bus_data, dict) and 'substations_id' in set_bus_data:
                    subs_data = set_bus_data['substations_id']
                    if isinstance(subs_data, list) and len(subs_data) > 0:
                        sub_id, topo_vec = subs_data[0]
                        serialized["parameters"] = {
                            "substation_id": int(sub_id),
                            "topology_vector": topo_vec.tolist() if hasattr(topo_vec, 'tolist') else list(topo_vec)
                        }
            
            elif 'set_line_status' in action_dict and action_dict['set_line_status'] is not None:
                serialized["action_type"] = "set_line_status"
                line_status_data = action_dict['set_line_status']
                if isinstance(line_status_data, list) and len(line_status_data) > 0:
                    line_id, status = line_status_data[0]
                    serialized["parameters"] = {
                        "line_id": int(line_id),
                        "status": int(status)
                    }
            
            elif 'redispatch' in action_dict and action_dict['redispatch'] is not None:
                serialized["action_type"] = "redispatch"
                redispatch_data = action_dict['redispatch']
                if isinstance(redispatch_data, list):
                    serialized["parameters"] = {
                        "redispatch_list": [(int(g), float(a)) for g, a in redispatch_data]
                    }
            
            return serialized
        except Exception as e:
            logger.warning(f"Action serialization failed: {e}")
            return {"action_type": "unknown", "parameters": {}}

