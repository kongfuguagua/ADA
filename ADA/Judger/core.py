# -*- coding: utf-8 -*-
"""
Judger Agent 核心实现
负责物理仿真评估和逻辑校验
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.const import OptimizationProblem, Solution, Feedback, FeedbackType
from utils.interact import BaseJudger, BaseSimulator
from utils.llm import BaseLLM
from utils.logger import get_logger
from config import SystemConfig

from .Reward.phy_reward import PhysicalReward
from .Reward.llm_reward import LLMReward
from .Debug.feedback import FeedbackGenerator

logger = get_logger("Judger")


class JudgerAgent(BaseJudger):
    """
    评估智能体
    
    核心职责：
    1. 物理仿真 - 检查解是否满足物理约束
    2. 逻辑评估 - 使用 LLM 评估解的合理性
    3. 综合打分 - 加权计算最终评分
    4. 故障诊断 - 分析失败原因并生成反馈
    """
    
    def __init__(
        self,
        simulator: BaseSimulator = None,
        llm: BaseLLM = None,
        alpha: float = None,
        pass_threshold: float = None
    ):
        """
        初始化 Judger
        
        Args:
            simulator: 仿真器实例
            llm: LLM 服务
            alpha: 物理评分权重
            pass_threshold: 通过阈值
        """
        config = SystemConfig()
        
        self.alpha = alpha if alpha is not None else config.judger_alpha
        self.pass_threshold = pass_threshold if pass_threshold is not None else config.judger_pass_threshold
        
        # 初始化评分器
        self.phy_reward = PhysicalReward(simulator=simulator)
        self.llm_reward = LLMReward(llm=llm)
        
        # 初始化反馈生成器
        self.feedback_generator = FeedbackGenerator(llm=llm)
        
        # 保存最后的评估详情
        self._last_phy_details: Dict[str, Any] = {}
        self._last_llm_details: Dict[str, Any] = {}
    
    def evaluate(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Feedback:
        """
        评估解的质量
        
        Args:
            problem: 优化问题
            solution: 求解结果
        
        Returns:
            评估反馈
        """
        logger.info("开始评估", 
                   algorithm=solution.algorithm_used,
                   feasible=solution.is_feasible)
        
        # 1. 物理评分
        phy_score, phy_details = self.phy_reward(problem, solution)
        self._last_phy_details = phy_details
        logger.debug(f"物理评分: {phy_score:.4f}")
        
        # 2. 逻辑评分
        llm_score, llm_details = self.llm_reward(problem, solution)
        self._last_llm_details = llm_details
        logger.debug(f"逻辑评分: {llm_score:.4f}")
        
        # 3. 生成反馈
        feedback = self.feedback_generator.generate_feedback(
            problem=problem,
            solution=solution,
            phy_score=phy_score,
            phy_details=phy_details,
            llm_score=llm_score,
            llm_details=llm_details,
            alpha=self.alpha,
            pass_threshold=self.pass_threshold
        )
        
        logger.info(f"评估完成: {feedback.feedback_type.value}, 评分={feedback.score:.4f}")
        
        return feedback
    
    def diagnose_error(
        self,
        problem: OptimizationProblem,
        solution: Solution,
        metrics: Dict[str, Any]
    ) -> str:
        """
        诊断错误原因
        
        Args:
            problem: 优化问题
            solution: 求解结果
            metrics: 物理指标
        
        Returns:
            诊断报告
        """
        diagnosis = self.feedback_generator.diagnose_with_llm(
            problem, solution, metrics
        )
        
        report_lines = [
            "=== 错误诊断报告 ===",
            f"错误来源: {diagnosis.get('error_source', '未知')}",
            f"错误类型: {diagnosis.get('error_type', '未知')}",
            f"根本原因: {diagnosis.get('root_cause', '未知')}",
            f"修复建议: {diagnosis.get('suggested_fix', '无')}",
        ]
        
        return "\n".join(report_lines)
    
    def get_evaluation_details(self) -> Dict[str, Any]:
        """获取最后一次评估的详细信息"""
        return {
            "physical": self._last_phy_details,
            "logical": self._last_llm_details
        }
    
    def quick_check(self, solution: Solution) -> bool:
        """
        快速检查解是否可行（不进行完整评估）
        
        Args:
            solution: 求解结果
        
        Returns:
            是否可行
        """
        if not solution.is_feasible:
            return False
        
        if solution.objective_value == float('inf'):
            return False
        
        return True


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 JudgerAgent:")
    
    # 创建 Judger
    judger = JudgerAgent()
    
    # 创建测试问题
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        constraints_latex=[r"x_1 + x_2 \geq 1"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ],
        modeling_rationale="最小化二次目标"
    )
    
    # 测试好的解
    print("\n测试1: 好的解")
    good_solution = Solution(
        is_feasible=True,
        algorithm_used="ConvexOptimizer",
        decision_variables={"x1": 0.5, "x2": 0.5},
        objective_value=0.5,
        solving_time=0.1
    )
    
    feedback = judger.evaluate(problem, good_solution)
    print(f"类型: {feedback.feedback_type}")
    print(f"评分: {feedback.score:.4f}")
    print(f"需要重试: {feedback.needs_retry()}")
    
    # 测试失败的解
    print("\n测试2: 失败的解")
    bad_solution = Solution(
        is_feasible=False,
        algorithm_used="FailedOptimizer",
        decision_variables={},
        objective_value=float('inf'),
        solving_time=10.0,
        solver_message="求解失败"
    )
    
    feedback = judger.evaluate(problem, bad_solution)
    print(f"类型: {feedback.feedback_type}")
    print(f"评分: {feedback.score:.4f}")
    print(f"诊断: {feedback.diagnosis}")
    print(f"建议: {feedback.suggested_fix}")
    
    # 测试边界违规
    print("\n测试3: 边界违规")
    violation_solution = Solution(
        is_feasible=True,
        algorithm_used="BadOptimizer",
        decision_variables={"x1": 15.0, "x2": 0.5},  # x1 超出边界
        objective_value=225.25,
        solving_time=0.1
    )
    
    feedback = judger.evaluate(problem, violation_solution)
    print(f"类型: {feedback.feedback_type}")
    print(f"评分: {feedback.score:.4f}")
    print(f"错误来源: {feedback.error_source}")
    print(f"诊断: {feedback.diagnosis}")
    
    print("\n测试完成")
