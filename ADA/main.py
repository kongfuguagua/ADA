# -*- coding: utf-8 -*-
"""
ADA (Agile Dispatch Agent) 系统入口
知识驱动的复杂系统敏捷调度智能体

主循环逻辑：
1. Planner 接收环境状态，生成优化问题
2. Solver 求解优化问题，返回解
3. Judger 评估解的质量，生成反馈
4. 如果失败，Planner 根据反馈重试
5. 成功后，Summarizer 提炼经验更新知识库
"""

import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
from config import get_system_config, get_llm_config

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

# 导入各个智能体
from knowledgebase.service import KnowledgeService
from knowledgebase.LLM import OpenAIChat, MockLLM
from knowledgebase.Embeddings import OpenAIEmbedding, MockEmbedding

from Planner.core import PlannerAgent
from Planner.tools.registry import create_default_registry

from Solver.core import SolverAgent

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
        use_mock: bool = True,
        kb_storage_path: str = None
    ):
        """
        初始化编排器
        
        Args:
            use_mock: 是否使用模拟 LLM（用于测试）
            kb_storage_path: 知识库存储路径
        """
        self.config = get_system_config()
        self.llm_config = get_llm_config()
        
        # 初始化 LLM
        if use_mock:
            self.llm = self._create_mock_llm()
            self.embedding = MockEmbedding()
            logger.info("使用 Mock LLM 模式")
        else:
            self.llm = OpenAIChat(
                model=self.llm_config.model_name,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature
            )
            self.embedding = OpenAIEmbedding()
            logger.info("使用真实 LLM 模式")
        
        # 初始化知识库
        storage_path = kb_storage_path or str(self.config.get_knowledge_path())
        self.kb = KnowledgeService(
            embedding_model=self.embedding,
            storage_path=storage_path
        )
        logger.info(f"知识库已加载: {len(self.kb)} 条记录")
        
        # 初始化工具注册表
        self.tools = create_default_registry()
        
        # 初始化各智能体
        self.planner = PlannerAgent(
            llm=self.llm,
            tools=self.tools,
            kb=self.kb
        )
        
        self.solver = SolverAgent(llm=self.llm)
        
        self.judger = JudgerAgent(llm=self.llm)
        
        self.summarizer = SummarizerAgent(
            kb=self.kb,
            llm=self.llm
        )
        
        logger.info("ADA 系统初始化完成")
    
    def _create_mock_llm(self) -> MockLLM:
        """创建模拟 LLM"""
        mock = MockLLM()
        
        # 设置状态增广响应
        mock.set_response("状态增广", "FINISH")
        mock.set_response("分析当前状态", "FINISH")
        
        # 设置问题建模响应
        mock.set_response("建立数学优化", '''```json
{
    "objective_function_latex": "\\\\min \\\\sum_{i} c_i x_i",
    "objective_function_code": "sum(c[i] * x[i] for i in range(n))",
    "is_minimization": true,
    "constraints_latex": ["\\\\sum x_i \\\\leq B"],
    "constraints_code": ["sum(x) <= B"],
    "variables": [
        {"name": "x1", "type": "continuous", "lower_bound": 0, "upper_bound": 100, "description": "决策变量1"},
        {"name": "x2", "type": "continuous", "lower_bound": 0, "upper_bound": 100, "description": "决策变量2"}
    ],
    "parameters": {"c1": 10, "c2": 20, "B": 150},
    "modeling_rationale": "线性规划模型"
}
```''')
        
        # 设置评估响应
        mock.set_response("评估", '''```json
{
    "is_logical": true,
    "score": 0.8,
    "issues": [],
    "suggestions": [],
    "comment": "解合理"
}
```''')
        
        return mock
    
    def run(
        self,
        env_state: EnvironmentState,
        max_retries: int = None
    ) -> Dict[str, Any]:
        """
        运行主循环
        
        Args:
            env_state: 环境状态
            max_retries: 最大重试次数
        
        Returns:
            运行结果 {success, solution, trace, attempts}
        """
        max_retries = max_retries or self.config.max_retries
        
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.trace_start(trace_id, env_state.model_dump())
        
        current_feedback: Optional[Feedback] = None
        history_trace: List[Dict[str, Any]] = []
        
        for attempt in range(max_retries):
            logger.info(f"=== 尝试 {attempt + 1}/{max_retries} ===")
            
            try:
                # 1. Plan - 生成优化问题
                logger.info("[Planner] 开始规划...")
                problem = self.planner.plan(env_state, retry_feedback=current_feedback)
                logger.info(f"[Planner] 生成问题: {len(problem.variables)} 个变量, {len(problem.constraints_latex)} 个约束")
                
                # 2. Solve - 求解优化问题
                logger.info("[Solver] 开始求解...")
                solution = self.solver.solve(problem)
                logger.info(f"[Solver] 求解完成: 可行={solution.is_feasible}, 目标值={solution.objective_value:.4f}")
                
                # 3. Judge - 评估解
                logger.info("[Judger] 开始评估...")
                current_feedback = self.judger.evaluate(problem, solution)
                logger.info(f"[Judger] 评估结果: {current_feedback.feedback_type.value}, 评分={current_feedback.score:.4f}")
                
                # 记录轨迹
                history_trace.append({
                    'attempt': attempt,
                    'problem': problem.model_dump(),
                    'solution': solution.model_dump(),
                    'feedback': current_feedback.model_dump()
                })
                
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
                        "trace": trace,
                        "attempts": attempt + 1
                    }
                else:
                    logger.warning(f"✗ 失败: {current_feedback.diagnosis}")
                    logger.info(f"建议修复: {current_feedback.suggested_fix}")
                    
            except Exception as e:
                logger.error(f"运行异常: {e}")
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
            "solution": None,
            "trace": history_trace,
            "attempts": max_retries
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "knowledge_count": len(self.kb),
            "algorithms": self.solver.list_algorithms(),
            "tools": self.tools.list_tools(),
            "summarizer_stats": self.summarizer.get_statistics()
        }


def main():
    """主函数"""
    print("=" * 60)
    print("  ADA - Agile Dispatch Agent")
    print("  知识驱动的复杂系统敏捷调度智能体")
    print("=" * 60)
    print()
    
    # 创建编排器（使用 Mock 模式进行测试）
    orchestrator = ADAOrchestrator(use_mock=True)
    
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
    
    if result['success']:
        solution = result['solution']
        print(f"  算法: {solution.algorithm_used}")
        print(f"  目标值: {solution.objective_value:.4f}")
        print(f"  决策变量: {solution.decision_variables}")
    
    print()
    print("系统状态:")
    status = orchestrator.get_status()
    print(f"  知识库条目: {status['knowledge_count']}")
    print(f"  可用算法: {status['algorithms']}")
    print(f"  可用工具: {status['tools']}")
    
    return result


# ============= 测试代码 =============
if __name__ == "__main__":
    result = main()
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
