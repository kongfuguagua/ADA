# -*- coding: utf-8 -*-
"""
ADA 系统编排器
协调各智能体完成调度任务

注意：此模块不包含启动逻辑，仅负责核心编排功能。
启动配置、日志、进度打印等应在 main.py 中处理。
"""

import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
from config import SystemConfig, LLMConfig

# 导入数据契约
from utils.const import (
    EnvironmentState,
    OptimizationProblem,
    Solution,
    Feedback,
    FeedbackType,
    ExecutionTrace,
    AgentRole,
)
from utils.logger import get_logger
from utils.llm import OpenAIChat
from utils.embeddings import OpenAIEmbedding

# 导入知识库
from knowledgebase.service import KnowledgeService

# 导入各个智能体
from Planner.core import PlannerAgent
from Planner.tools.registry import create_default_registry

from Solver.solver import SolverAgent

from Judger.core import JudgerAgent

from Summarizer.core import SummarizerAgent

logger = get_logger("Orchestrator")


class ADAOrchestrator:
    """
    ADA 系统编排器
    协调各智能体完成调度任务
    """
    
    def __init__(
        self,
        system_config: SystemConfig,
        llm_config: LLMConfig,
        kb_storage_path: str = None,
        env=None
    ):
        """
        初始化编排器
        
        Args:
            system_config: 系统配置
            llm_config: LLM 配置
            kb_storage_path: 知识库存储路径
            env: Grid2Op 环境实例（可选）
        
        Raises:
            ValueError: API 配置无效
        """
        self.config = system_config
        self.llm_config = llm_config
        self.env = env
        
        # 验证并初始化 LLM 和 Embedding
        self._validate_config()
        self.llm, self.embedding = self._init_llm_services()
        
        # 初始化知识库
        storage_path = kb_storage_path or str(self.config.get_knowledge_path())
        self.kb = KnowledgeService(
            embedding_model=self.embedding,
            storage_path=storage_path
        )
        logger.info(f"知识库已加载: {len(self.kb)} 条记录")
        
        # 初始化工具注册表（绑定环境）
        self.tools = create_default_registry(env=env)
        
        # 初始化各智能体
        self._init_agents()
        
        logger.info("ADA 系统初始化完成")
    
    def _validate_config(self) -> None:
        """验证 API 配置"""
        if not self.llm_config.api_key:
            raise ValueError(
                "未配置 LLM API Key。\n"
                "请在 .env 文件中设置 CLOUD_API_KEY 环境变量或通过 YAML 配置。\n"
                "示例: CLOUD_API_KEY=your-api-key-here"
            )
    
    def _init_llm_services(self):
        """初始化 LLM 和 Embedding 服务"""
        # 创建 LLM
        llm = OpenAIChat(
            model=self.llm_config.model_name,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            temperature=self.llm_config.temperature
        )
        logger.info(f"LLM 已初始化: {self.llm_config.model_name}")
        
        # 创建 Embedding
        embedding = OpenAIEmbedding(
            api_key=self.llm_config.embedding_api_key,
            base_url=self.llm_config.embedding_base_url,
            model=self.llm_config.embedding_model
        )
        logger.info(f"Embedding 已初始化: {self.llm_config.embedding_model}")
        
        return llm, embedding
    
    def _init_agents(self):
        """初始化各智能体"""
        # 从配置中获取 Agent 参数
        agents_config = getattr(self.config, '_agents_config', {})
        
        planner_config = agents_config.get("planner", {})
        solver_config = agents_config.get("solver", {})
        
        self.planner = PlannerAgent(
            llm=self.llm,
            tools=self.tools,
            kb=self.kb,
            max_augmentation_steps=planner_config.get("max_augmentation_steps", self.config.planner_max_augmentation_steps)
        )
        
        self.solver = SolverAgent(
            llm=self.llm,
            timeout=solver_config.get("timeout", self.config.solver_timeout),
            use_llm_features=solver_config.get("use_llm_features", False)
        )
        
        judger_config = agents_config.get("judger", {})
        self.judger = JudgerAgent(
            llm=self.llm,
            alpha=judger_config.get("alpha", self.config.judger_alpha),
            pass_threshold=judger_config.get("pass_threshold", self.config.judger_pass_threshold)
        )
        
        summarizer_config = agents_config.get("summarizer", {})
        self.summarizer = SummarizerAgent(
            kb=self.kb,
            llm=self.llm,
            exploration_constant=summarizer_config.get("mcts_exploration_constant", self.config.mcts_exploration_constant),
            min_score_threshold=summarizer_config.get("min_score_threshold", self.config.summarizer_min_score_threshold)
        )
    
    def set_env(self, env):
        """设置/更新环境"""
        self.env = env
        self.tools.set_env(env)
    
    def run(
        self,
        env_state: EnvironmentState = None,
        max_retries: int = None
    ) -> Dict[str, Any]:
        """
        运行主循环
        
        Args:
            env_state: 环境状态（如果有 Grid2Op 环境则自动获取）
            max_retries: 最大重试次数
        
        Returns:
            运行结果 {success, solution, trace, attempts}
        """
        max_retries = max_retries or self.config.max_retries
        
        # 获取环境状态
        if env_state is None:
            env_state = self._get_env_state()
        
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.trace_start(trace_id, env_state.model_dump())
        
        current_feedback: Optional[Feedback] = None
        problem: Optional[OptimizationProblem] = None
        solution: Optional[Solution] = None
        
        for attempt in range(max_retries):
            logger.info(f"=== 尝试 {attempt + 1}/{max_retries} ===")
            
            try:
                # 1. Plan - 生成优化问题
                logger.info("[Planner] 开始规划...")
                problem = self.planner.plan(env_state, retry_feedback=current_feedback)
                logger.info(f"[Planner] 生成问题: {len(problem.variables)} 个变量, {len(problem.constraints_latex)} 个约束")
                
                # 快速检查问题定义是否合理
                if len(problem.variables) == 0:
                    logger.warning("Planner 生成的问题没有变量，可能是解析失败")
                    current_feedback = Feedback(
                        feedback_type=FeedbackType.LOGICAL_ERROR,
                        score=0.0,
                        diagnosis="问题定义不完整：没有定义变量",
                        error_source=AgentRole.PLANNER,
                        suggested_fix="请重新规划，确保问题定义完整"
                    )
                    continue
                
                # 2. Solve - 求解优化问题
                logger.info("[Solver] 开始求解...")
                solution = self.solver.solve(problem)
                logger.info(f"[Solver] 求解完成: 可行={solution.is_feasible}")
                
                # 如果求解器直接失败（问题验证失败），跳过 Judger，直接生成反馈
                if not solution.is_feasible and "问题定义验证失败" in (solution.solver_message or ""):
                    logger.warning(f"Solver 检测到问题定义问题: {solution.solver_message}")
                    current_feedback = Feedback(
                        feedback_type=FeedbackType.LOGICAL_ERROR,
                        score=0.0,
                        diagnosis=solution.solver_message,
                        error_source=AgentRole.PLANNER,
                        suggested_fix="请重新规划，确保问题定义完整且合理"
                    )
                    continue
                
                # 3. Judge - 评估解
                logger.info("[Judger] 开始评估...")
                current_feedback = self.judger.evaluate(problem, solution)
                logger.info(f"[Judger] 评估结果: {current_feedback.feedback_type.value}, 评分={current_feedback.score:.4f}")
                
                # 4. 决策分支
                if current_feedback.feedback_type == FeedbackType.PASSED:
                    logger.info("✓ 成功！解已通过评估")
                    
                    # 5. 成功后触发总结
                    trace = ExecutionTrace(
                        trace_id=trace_id,
                        environment=env_state,
                        problem=problem,
                        solution=solution,
                        feedback=current_feedback,
                        tool_chain=self.planner.get_tool_chain(),
                        attempt_count=attempt + 1
                    )
                    
                    logger.info("[Summarizer] 开始总结...")
                    self.summarizer.summarize(trace)
                    
                    logger.trace_end(trace_id, success=True, score=current_feedback.score)
                    
                    return {
                        "success": True,
                        "solution": solution,
                        "problem": problem,
                        "feedback": current_feedback,
                        "trace": trace,
                        "attempts": attempt + 1
                    }
                else:
                    logger.warning(f"✗ 失败 ({current_feedback.feedback_type.value}): {current_feedback.diagnosis}")
                    if current_feedback.error_source:
                        logger.info(f"错误来源: {current_feedback.error_source.value}")
                    if current_feedback.suggested_fix:
                        logger.info(f"修复建议: {current_feedback.suggested_fix}")
                    
            except Exception as e:
                logger.error(f"运行异常: {e}")
                import traceback
                traceback.print_exc()
                current_feedback = Feedback(
                    feedback_type=FeedbackType.RUNTIME_ERROR,
                    score=0.0,
                    diagnosis=str(e),
                    error_source=AgentRole.PLANNER,  # 默认归因于 Planner
                    suggested_fix="检查系统配置和输入数据"
                )
        
        logger.error(f"达到最大重试次数 ({max_retries})，任务失败")
        logger.trace_end(trace_id, success=False, score=0.0)
        
        return {
            "success": False,
            "solution": solution,
            "problem": problem,
            "feedback": current_feedback,
            "trace": None,
            "attempts": max_retries
        }
    
    def _get_env_state(self) -> EnvironmentState:
        """从环境获取状态"""
        if self.env is not None and hasattr(self.env, 'get_state_for_planner'):
            grid_state = self.env.get_state_for_planner()
            return EnvironmentState(
                user_instruction="优化电网调度，保持系统稳定运行",
                real_data={
                    "total_load": grid_state.get("total_load_mw", 0),
                    "total_gen": grid_state.get("total_gen_mw", 0),
                    "max_rho": grid_state.get("max_rho", 0),
                },
                extra_context=grid_state
            )
        
        # 默认状态
        return EnvironmentState(
            user_instruction="优化调度任务",
            real_data={"load": 100.0, "generation": 105.0}
        )
    
    def run_episode(
        self,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行一个完整的回合（与 Grid2Op 环境交互）
        
        Args:
            max_steps: 最大步数
            verbose: 是否打印详细信息
        
        Returns:
            回合统计信息
        """
        if self.env is None:
            raise ValueError("需要设置 Grid2Op 环境")
        
        # 重置环境
        self.env.reset()
        
        total_reward = 0.0
        step_count = 0
        success_count = 0
        
        for step in range(max_steps):
            step_count += 1
            
            # 运行 ADA 主循环
            result = self.run(max_retries=1)
            
            if result["success"]:
                success_count += 1
                
                # 将解应用到环境（简化版）
                action = self.env.get_do_nothing_action()
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if verbose and step % 10 == 0:
                    obs_info = self.env.get_observation_info()
                    print(f"Step {step}: reward={reward:.2f}, max_rho={obs_info['max_rho']:.2%}")
                
                if done:
                    if verbose:
                        print(f"回合在第 {step} 步结束")
                    break
            else:
                if verbose:
                    print(f"Step {step}: ADA 规划失败")
        
        return {
            "steps": step_count,
            "total_reward": total_reward,
            "success_rate": success_count / step_count if step_count > 0 else 0,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "knowledge_count": len(self.kb),
            "algorithms": self.solver.list_algorithms(),
            "tools": self.tools.list_tools(),
            "has_env": self.env is not None,
        }

