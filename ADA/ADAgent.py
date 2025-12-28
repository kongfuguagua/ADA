# -*- coding: utf-8 -*-
"""
ADA - Agile Dispatch Agent
标准的 Grid2Op 智能体实现

参考 ExpertAgent 和 OptimCVXPY 的优秀实现，实现知识驱动的智能调度。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

# 导入 ADA 核心组件
from utils.const import EnvironmentState, Solution, FeedbackType, Feedback, OptimizationProblem, AgentRole
from utils.logger import get_logger

# 导入各个智能体（ADAgent 内部包含完整的 ADA 逻辑）
from Planner.core import PlannerAgent
from Planner.tools.registry import create_default_registry
from Solver.solver import SolverAgent
from Judger.core import JudgerAgent
from Summarizer.core import SummarizerAgent

# 导入知识库和工具
from knowledgebase.service import KnowledgeService
from utils.llm import OpenAIChat
from utils.embeddings import OpenAIEmbedding
from config import SystemConfig, LLMConfig

logger = get_logger("ADAgent")


class ADAgent(BaseAgent):
    """
    ADA - Agile Dispatch Agent
    
    知识驱动的复杂系统敏捷调度智能体
    
    工作流程（参考 OptimCVXPY 的启发式策略）：
    1. 安全状态（max_rho < 0.85）：使用简单规则（do nothing 或恢复参考状态）
    2. 危险状态（max_rho > 0.95 或过载）：启动完整 ADA 流程（Plan -> Solve -> Judge）
    3. 中间状态：根据配置决定
    
    参考实现：
    - ExpertAgent: 过载检测、拓扑操作策略
    - OptimCVXPY: 状态判断阈值、优化求解策略
    """
    
    def __init__(
        self,
        action_space,
        observation_space,
        name: str = "ADAgent",
        system_config: Optional[SystemConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        rho_safe: float = 0.85,
        rho_danger: float = 0.95,
        **kwargs
    ):
        """
        初始化 ADAgent
        
        Args:
            action_space: Grid2Op 动作空间
            observation_space: Grid2Op 观测空间
            name: 智能体名称
            system_config: 系统配置（如果为 None，使用默认配置）
            llm_config: LLM 配置（如果为 None，使用默认配置）
            rho_safe: 安全状态阈值（参考 OptimCVXPY）
            rho_danger: 危险状态阈值（参考 OptimCVXPY）
            **kwargs: 其他参数
        """
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.rho_safe = rho_safe
        self.rho_danger = rho_danger
        
        # 加载配置
        self.system_config = system_config or SystemConfig()
        self.llm_config = llm_config or LLMConfig()
        
        # 初始化 ADA 核心组件
        self._init_ada_components()
        
        # 内部状态
        self.current_step = 0
        self.last_action = None
        self.last_feedback = None
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "safe_mode_count": 0,
            "full_ada_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }
        
        logger.info(f"ADAgent '{name}' 初始化完成")
        logger.info(f"  安全阈值: rho_safe={rho_safe}, rho_danger={rho_danger}")
    
    def _init_ada_components(self):
        """初始化 ADA 核心组件（Planner, Solver, Judger, Summarizer）"""
        # 初始化 LLM 和 Embedding
        self.llm = OpenAIChat(
            model=self.llm_config.model_name,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            temperature=self.llm_config.temperature
        )
        
        self.embedding = OpenAIEmbedding(
            api_key=self.llm_config.embedding_api_key,
            base_url=self.llm_config.embedding_base_url,
            model=self.llm_config.embedding_model
        )
        
        # 初始化知识库
        storage_path = str(self.system_config.get_knowledge_path())
        self.kb = KnowledgeService(
            embedding_model=self.embedding,
            storage_path=storage_path
        )
        
        # 初始化工具注册表（环境在运行时绑定）
        self.tools = create_default_registry(env=None)
        self._env = None  # 环境实例（用于工具）
        
        # 初始化各智能体
        agents_config = getattr(self.system_config, '_agents_config', {})
        
        planner_config = agents_config.get("planner", {})
        self.planner = PlannerAgent(
            llm=self.llm,
            tools=self.tools,
            kb=self.kb,
            max_augmentation_steps=planner_config.get("max_augmentation_steps", self.system_config.planner_max_augmentation_steps)
        )
        
        solver_config = agents_config.get("solver", {})
        self.solver = SolverAgent(
            llm=self.llm,
            timeout=solver_config.get("timeout", self.system_config.solver_timeout),
            use_llm_features=solver_config.get("use_llm_features", False)
        )
        
        judger_config = agents_config.get("judger", {})
        self.judger = JudgerAgent(
            llm=self.llm,
            alpha=judger_config.get("alpha", self.system_config.judger_alpha),
            pass_threshold=judger_config.get("pass_threshold", self.system_config.judger_pass_threshold)
        )
        
        summarizer_config = agents_config.get("summarizer", {})
        self.summarizer = SummarizerAgent(
            kb=self.kb,
            llm=self.llm,
            exploration_constant=summarizer_config.get("mcts_exploration_constant", self.system_config.mcts_exploration_constant),
            min_score_threshold=summarizer_config.get("min_score_threshold", self.system_config.summarizer_min_score_threshold)
        )
        
        logger.info("ADA 核心组件初始化完成")
    
    def reset(self, observation: BaseObservation):
        """
        重置智能体状态（在每个 episode 开始时调用）
        
        Args:
            observation: 初始观测
        """
        self.current_step = 0
        self.last_action = None
        self.last_feedback = None
        
        # 重置统计信息
        self.stats = {
            "total_steps": 0,
            "safe_mode_count": 0,
            "full_ada_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }
        
        logger.info(f"ADAgent 已重置 (episode start)")
    
    def set_env(self, env):
        """
        设置环境（用于工具绑定）
        
        Args:
            env: Grid2OpEnvironment 实例
        """
        self._env = env
        if hasattr(self.tools, 'set_env'):
            self.tools.set_env(env)
        logger.debug("环境已绑定到工具注册表")
    
    def act(
        self,
        observation: BaseObservation,
        reward: float = 0.0,
        done: bool = False
    ) -> BaseAction:
        """
        智能体的主要方法：根据观测返回动作
        
        工作流程（参考 OptimCVXPY 和 ExpertAgent）：
        1. 判断电网状态（安全/危险/中间）
        2. 安全状态：使用简单规则
        3. 危险状态：启动完整 ADA 流程
        4. 将 Solution 转换为 Grid2Op Action
        
        Args:
            observation: 当前观测
            reward: 上一步的奖励
            done: 是否结束
        
        Returns:
            Grid2Op Action
        """
        self.current_step += 1
        self.stats["total_steps"] += 1
        
        # 判断电网状态（参考 OptimCVXPY 和 ExpertAgent 的启发式）
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        near_overflow_count = int((observation.rho > 0.9).sum())
        
        # 参考 OptimCVXPY：安全/危险状态判断
        is_safe = max_rho < self.rho_safe
        # 参考 ExpertAgent：过载或接近过载都视为危险
        is_danger = max_rho > self.rho_danger or overflow_count > 0 or near_overflow_count > 3
        
        # 安全状态：使用简单规则（参考 OptimCVXPY 的 safe mode）
        if is_safe:
            self.stats["safe_mode_count"] += 1
            logger.debug(f"Step {self.current_step}: 安全状态 (max_rho={max_rho:.3f})，使用简单规则")
            return self._act_safe_mode(observation)
        
        # 危险状态：启动完整 ADA 流程
        if is_danger:
            self.stats["full_ada_count"] += 1
            logger.info(f"Step {self.current_step}: 危险状态 (max_rho={max_rho:.3f}, overflow={overflow_count})，启动 ADA 流程")
            return self._act_full_ada(observation)
        
        # 中间状态：保持当前状态（参考 OptimCVXPY 的 do nothing mode）
        logger.debug(f"Step {self.current_step}: 中间状态 (max_rho={max_rho:.3f})，保持当前状态")
        return self.action_space({})
    
    def _act_safe_mode(self, observation: BaseObservation) -> BaseAction:
        """
        安全模式：使用简单规则（参考 OptimCVXPY 的 safe mode 和 ExpertAgent 的恢复策略）
        
        策略（参考 ExpertAgent 的无过载情况处理）：
        1. 尝试恢复参考拓扑（如果有变电站被分割）
        2. 尝试重连断开的线路（参考 ExpertAgent.reco_line）
        3. 否则执行 do nothing
        """
        action = self.action_space({})
        
        # 1. 尝试重连断开的线路（参考 ExpertAgent.reco_line）
        # 检查线路状态和冷却时间
        can_be_reco = (observation.time_before_cooldown_line == 0) & (~observation.line_status)
        if np.any(can_be_reco):
            # 选择第一个可重连的线路
            line_id = np.where(can_be_reco)[0][0]
            # 模拟重连后的安全性（参考 ExpertAgent 的安全检查）
            test_action = self.action_space({})
            test_action.line_set_status = [(line_id, +1)]
            try:
                sim_obs, _, _, _ = observation.simulate(test_action, time_step=0)
                # 检查是否安全（参考 ExpertAgent 的阈值 0.95）
                if np.all(sim_obs.rho < 0.95):
                    action.line_set_status = [(line_id, +1)]
                    logger.debug(f"安全模式：重连线路 {line_id}")
                    return action
            except Exception as e:
                logger.debug(f"重连线路 {line_id} 模拟失败: {e}")
        
        # 2. 执行 do nothing（参考 OptimCVXPY 的 safe mode）
        return action
    
    def _act_full_ada(self, observation: BaseObservation) -> BaseAction:
        """
        完整 ADA 流程：Plan -> Solve -> Judge
        
        核心逻辑在 Agent 内部，不依赖 orchestrator
        """
        try:
            # 1. 构建环境状态
            env_state = self._observation_to_env_state(observation)
            
            # 2. Plan - 生成优化问题
            problem = self.planner.plan(env_state, retry_feedback=self.last_feedback)
            
            # 快速检查问题定义是否合理
            if len(problem.variables) == 0:
                logger.warning("Planner 生成的问题没有变量")
                self.stats["failure_count"] += 1
                return self.action_space({})
            
            # 3. Solve - 求解优化问题
            solution = self.solver.solve(problem)
            
            # 如果求解器直接失败，返回 do nothing
            if not solution.is_feasible and "问题定义验证失败" in (solution.solver_message or ""):
                logger.warning(f"Solver 检测到问题定义问题: {solution.solver_message}")
                self.stats["failure_count"] += 1
                return self.action_space({})
            
            # 4. Judge - 评估解
            feedback = self.judger.evaluate(problem, solution)
            self.last_feedback = feedback
            
            # 5. 检查是否通过
            if feedback.feedback_type == FeedbackType.PASSED:
                self.stats["success_count"] += 1
                
                # 6. 将 Solution 转换为 Grid2Op Action
                action = self._solution_to_action(
                    solution,
                    observation,
                    "full_ada"
                )
                
                self.last_action = action
                return action
            else:
                self.stats["failure_count"] += 1
                logger.warning(f"ADA 流程失败: {feedback.diagnosis}")
                # 失败时返回 do nothing
                return self.action_space({})
                
        except Exception as e:
            logger.error(f"ADA 流程异常: {e}")
            import traceback
            traceback.print_exc()
            self.stats["failure_count"] += 1
            return self.action_space({})
    
    def _observation_to_env_state(self, observation: BaseObservation) -> EnvironmentState:
        """将 Grid2Op 观测转换为 EnvironmentState"""
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        return EnvironmentState(
            user_instruction="优化电网调度，保持系统稳定运行",
            real_data={
                "total_load": float(observation.load_p.sum()),
                "total_gen": float(observation.gen_p.sum()),
                "max_rho": max_rho,
                "overflow_count": overflow_count,
            },
            extra_context={
                "timestep": observation.current_step,
                "hour_of_day": observation.hour_of_day,
                "load_p": observation.load_p.tolist(),
                "gen_p": observation.gen_p.tolist(),
                "rho": observation.rho.tolist(),
                "line_status": observation.line_status.tolist(),
            }
        )
    
    def _solution_to_action(
        self,
        solution: Solution,
        observation: BaseObservation,
        mode: str = "full_ada"
    ) -> BaseAction:
        """
        将 Solution 转换为 Grid2Op Action
        
        参考 OptimCVXPY.to_grid2op() 的实现
        """
        action = self.action_space({})
        
        if not solution.is_feasible:
            logger.warning("Solution 不可行，返回 do nothing")
            return action
        
        # 从 decision_variables 中提取动作
        # 假设变量命名格式：redispatch_gen_{id}, storage_{id}, curtail_gen_{id}, set_line_{id}
        decision_vars = solution.decision_variables
        
        # 1. 处理再调度（redispatch）
        redispatch_list = []
        for key, value in decision_vars.items():
            if key.startswith("redispatch_gen_") or key.startswith("redispatch_"):
                # 提取发电机 ID
                try:
                    gen_id = int(key.split("_")[-1])
                    if abs(value) > 1e-3:  # 阈值过滤（参考 OptimCVXPY 的 margin_sparse）
                        redispatch_list.append((gen_id, float(value)))
                except (ValueError, IndexError):
                    continue
        
        if redispatch_list:
            action.redispatch = redispatch_list
            logger.debug(f"再调度动作: {len(redispatch_list)} 个发电机")
        
        # 2. 处理储能（storage）
        storage_list = []
        for key, value in decision_vars.items():
            if key.startswith("storage_"):
                try:
                    storage_id = int(key.split("_")[-1])
                    if abs(value) > 1e-3:
                        storage_list.append((storage_id, float(value)))
                except (ValueError, IndexError):
                    continue
        
        if storage_list and hasattr(action, 'storage_p') and action.n_storage > 0:
            storage_array = np.zeros(action.n_storage)
            for storage_id, power in storage_list:
                if 0 <= storage_id < action.n_storage:
                    storage_array[storage_id] = power
            action.storage_p = storage_array
            logger.debug(f"储能动作: {len(storage_list)} 个储能单元")
        
        # 3. 处理弃风（curtailment）
        curtailment_dict = {}
        for key, value in decision_vars.items():
            if key.startswith("curtail_gen_") or key.startswith("curtail_"):
                try:
                    gen_id = int(key.split("_")[-1])
                    if abs(value) > 1e-3:
                        # 注意：curtailment 需要转换为最大允许功率（参考 OptimCVXPY）
                        # 这里简化处理，假设 value 是削减量
                        curtailment_dict[gen_id] = float(value)
                except (ValueError, IndexError):
                    continue
        
        if curtailment_dict and hasattr(action, 'curtail_mw'):
            # 简化处理
            curtailment_array = np.full(observation.n_gen, -1.0)
            for gen_id, value in curtailment_dict.items():
                if 0 <= gen_id < observation.n_gen:
                    # 转换为最大允许功率（简化版）
                    if observation.gen_renewable[gen_id] and observation.gen_p[gen_id] > 0.1:
                        curtailment_array[gen_id] = max(0.0, observation.gen_p[gen_id] - value)
            action.curtail_mw = curtailment_array
            logger.debug(f"弃风动作: {len(curtailment_dict)} 个发电机")
        
        # 4. 处理线路状态（set_line_status）
        line_status_list = []
        for key, value in decision_vars.items():
            if key.startswith("set_line_") or key.startswith("line_status_"):
                try:
                    line_id = int(key.split("_")[-1])
                    status = int(value) if abs(value) > 0.5 else 0
                    if status != 0:
                        line_status_list.append((line_id, status))
                except (ValueError, IndexError):
                    continue
        
        if line_status_list:
            action.set_line_status = line_status_list
            logger.debug(f"线路状态动作: {len(line_status_list)} 条线路")
        
        return action
    
    def load(self, path: Optional[str]):
        """加载智能体状态（如果需要）"""
        if path is None:
            return
        # TODO: 实现加载逻辑
        logger.info(f"从 {path} 加载智能体状态")
    
    def save(self, path: Optional[str]):
        """保存智能体状态（如果需要）"""
        if path is None:
            return
        # TODO: 实现保存逻辑
        logger.info(f"保存智能体状态到 {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["success_count"] / max(1, self.stats["full_ada_count"])
                if self.stats["full_ada_count"] > 0 else 0.0
            ),
        }

