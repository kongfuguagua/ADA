# -*- coding: utf-8 -*-
"""
ADA (Agile Dispatch Agent) 系统入口
知识驱动的复杂系统敏捷调度智能体

主循环逻辑：
1. 环境观测 - 从 Grid2Op 获取当前状态
2. Planner 规划 - 状态增广 + 问题建模
3. Solver 求解 - 特征提取 + 算法匹配 + 执行
4. Judger 评估 - 物理仿真 + 逻辑校验 + 综合打分
5. 环境执行 - 将解应用到环境
6. Summarizer 总结 - 提炼经验更新知识库

注意：此模块不支持 Mock 模式，必须配置有效的 LLM API。
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

logger = get_logger("Main")


class ADAOrchestrator:
    """
    ADA 系统编排器
    协调各智能体完成调度任务
    """
    
    def __init__(
        self,
        kb_storage_path: str = None,
        env=None
    ):
        """
        初始化编排器
        
        Args:
            kb_storage_path: 知识库存储路径
            env: Grid2Op 环境实例（可选）
        
        Raises:
            ValueError: API 配置无效
        """
        self.config = SystemConfig()
        self.llm_config = LLMConfig()
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
                "请在 .env 文件中设置 CLOUD_API_KEY 环境变量。\n"
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
        self.planner = PlannerAgent(
            llm=self.llm,
            tools=self.tools,
            kb=self.kb,
            config=self.config
        )
        
        self.solver = SolverAgent(
            llm=self.llm,
            config=self.config
        )
        
        self.judger = JudgerAgent(
            llm=self.llm,
            config=self.config
        )
        
        self.summarizer = SummarizerAgent(
            kb=self.kb,
            llm=self.llm,
            config=self.config
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
        
        # 验证环境状态
        if not isinstance(env_state, EnvironmentState):
            raise ValueError(f"env_state 必须是 EnvironmentState 类型，当前类型: {type(env_state)}")
        
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
                logger.info(f"[Planner] 生成问题: {len(problem.variables)} 个变量")
                
                # 2. Solve - 求解优化问题
                logger.info("[Solver] 开始求解...")
                solution = self.solver.solve(problem)
                logger.info(f"[Solver] 求解完成: 可行={solution.is_feasible}")
                
                # 3. Judge - 评估解
                logger.info("[Judger] 开始评估...")
                current_feedback = self.judger.evaluate(problem, solution)
                logger.info(f"[Judger] 评估结果: {current_feedback.feedback_type.value}")
                
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
                    logger.warning(f"✗ 失败: {current_feedback.diagnosis}")
                    
            except Exception as e:
                logger.error(f"运行异常: {e}")
                import traceback
                traceback.print_exc()
                current_feedback = Feedback(
                    feedback_type=FeedbackType.RUNTIME_ERROR,
                    score=0.0,
                    diagnosis=str(e),
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
        if self.env is not None:
            try:
                if hasattr(self.env, 'get_state_for_planner'):
                    grid_state = self.env.get_state_for_planner()
                    if grid_state:
                        return EnvironmentState(
                            user_instruction="优化电网调度，保持系统稳定运行",
                            real_data={
                                "total_load": grid_state.get("total_load_mw", 0),
                                "total_gen": grid_state.get("total_gen_mw", 0),
                                "max_rho": grid_state.get("max_rho", 0),
                            },
                            extra_context=grid_state
                        )
            except Exception as e:
                logger.warning(f"从环境获取状态失败: {e}，使用默认状态")
        
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
            result = self.run(max_retries=self.config.max_retries)
            
            if result["success"]:
                success_count += 1
                
                # 将解应用到环境（简化版）
                # TODO: 这里应该使用 solution 来构建 action，而不是 do_nothing
                try:
                    if hasattr(self.env, 'get_do_nothing_action'):
                        action = self.env.get_do_nothing_action()
                    else:
                        # 如果没有该方法，尝试使用 solution 构建 action
                        action = None
                    
                    if action is not None:
                        obs, reward, done, info = self.env.step(action)
                        total_reward += reward
                    else:
                        # 如果无法构建 action，跳过这一步
                        logger.warning(f"Step {step}: 无法构建环境动作，跳过")
                        continue
                    
                    if verbose and step % 10 == 0:
                        try:
                            if hasattr(self.env, 'get_observation_info'):
                                obs_info = self.env.get_observation_info()
                                max_rho = obs_info.get('max_rho', 0) if isinstance(obs_info, dict) else 0
                                print(f"Step {step}: reward={reward:.2f}, max_rho={max_rho:.2%}")
                            else:
                                print(f"Step {step}: reward={reward:.2f}")
                        except Exception as e:
                            logger.debug(f"获取观察信息失败: {e}")
                            print(f"Step {step}: reward={reward:.2f}")
                    
                    if done:
                        if verbose:
                            print(f"回合在第 {step} 步结束")
                        break
                except Exception as e:
                    logger.error(f"Step {step}: 环境执行失败: {e}")
                    if verbose:
                        print(f"Step {step}: 环境执行失败")
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


def main():
    """主函数"""
    print("=" * 60)
    print("  ADA - Agile Dispatch Agent")
    print("  知识驱动的复杂系统敏捷调度智能体")
    print("=" * 60)
    print()
    
    # 检查 API 配置
    llm_config = LLMConfig()
    if not llm_config.api_key:
        print("错误: 未配置 LLM API Key")
        print("请在 .env 文件中设置 CLOUD_API_KEY 环境变量")
        print()
        print("示例 .env 文件内容:")
        print("  CLOUD_API_KEY=your-api-key-here")
        print("  CLOUD_BASE_URL=https://api.deepseek.com")
        print("  CLOUD_MODEL=deepseek-chat")
        return None
    
    print(f"✓ 使用 LLM: {llm_config.model_name}")
    print()
    
    # 创建编排器
    try:
        orchestrator = ADAOrchestrator()
    except Exception as e:
        print(f"初始化失败: {e}")
        return None
    
    # 创建测试环境状态
    env_state = EnvironmentState(
        user_instruction="优化电网调度，在满足负载需求的前提下最小化发电成本",
        real_data={
            "total_load": 100.0,
            "price_electricity": 0.5,
            "generator_capacity": 150.0
        },
        extra_context={
            "time_horizon": 24,
            "weather": "晴"
        }
    )
    
    print("输入环境状态:")
    print(env_state.to_prompt_string())
    print()
    
    # 运行主循环
    result = orchestrator.run(env_state)
    
    print()
    print("=" * 60)
    print("运行结果:")
    print(f"  成功: {result['success']}")
    print(f"  尝试次数: {result['attempts']}")
    
    if result['success'] and result['solution']:
        solution = result['solution']
        print(f"  算法: {solution.algorithm_used}")
        print(f"  目标值: {solution.objective_value:.4f}")
        print(f"  决策变量: {solution.decision_variables}")
    
    if result['feedback']:
        print(f"  评估类型: {result['feedback'].feedback_type.value}")
        print(f"  评分: {result['feedback'].score:.4f}")
    
    print()
    print("系统状态:")
    status = orchestrator.get_status()
    print(f"  知识库条目: {status['knowledge_count']}")
    print(f"  可用算法: {status['algorithms']}")
    print(f"  可用工具: {status['tools']}")
    
    return result


if __name__ == "__main__":
    result = main()
    
    if result:
        print()
        print("=" * 60)
        print("✓ 运行完成")
        print("=" * 60)
