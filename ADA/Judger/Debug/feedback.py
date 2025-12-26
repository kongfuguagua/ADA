# -*- coding: utf-8 -*-
"""
反馈生成器
分析错误原因并生成结构化反馈
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import (
    OptimizationProblem, 
    Solution, 
    Feedback, 
    FeedbackType,
    AgentRole,
    PhysicalMetrics
)
from utils.interact import BaseLLM
from utils.logger import get_logger

from ..prompt import JudgerPrompts

logger = get_logger("Feedback")


class FeedbackGenerator:
    """
    反馈生成器
    分析求解失败的原因并生成结构化反馈
    """
    
    def __init__(self, llm: BaseLLM = None):
        """
        初始化反馈生成器
        
        Args:
            llm: LLM 服务
        """
        self.llm = llm
    
    def generate_feedback(
        self,
        problem: OptimizationProblem,
        solution: Solution,
        phy_score: float,
        phy_details: Dict[str, Any],
        llm_score: float,
        llm_details: Dict[str, Any],
        alpha: float = 0.7,
        pass_threshold: float = 0.6
    ) -> Feedback:
        """
        生成综合反馈
        
        Args:
            problem: 优化问题
            solution: 求解结果
            phy_score: 物理评分
            phy_details: 物理评分详情
            llm_score: LLM 评分
            llm_details: LLM 评分详情
            alpha: 物理评分权重
            pass_threshold: 通过阈值
        
        Returns:
            反馈对象
        """
        # 计算综合评分
        final_score = alpha * phy_score + (1 - alpha) * llm_score
        
        # 构建物理指标
        physical_metrics = PhysicalMetrics(
            is_safe=phy_details.get("is_safe", True),
            cost=phy_details.get("cost", 0.0),
            efficiency=phy_details.get("efficiency", 0.0),
            stability_margin=phy_details.get("stability_margin", 0.0),
            violation_details=phy_details.get("violation_details", {})
        )
        
        # 判断是否通过
        if final_score >= pass_threshold and solution.is_feasible:
            return Feedback(
                feedback_type=FeedbackType.PASSED,
                score=final_score,
                diagnosis="解满足所有要求",
                physical_metrics=physical_metrics,
                suggested_fix=""
            )
        
        # 分析失败原因
        feedback_type, error_source, diagnosis, suggested_fix = self._analyze_failure(
            problem, solution, phy_details, llm_details
        )
        
        return Feedback(
            feedback_type=feedback_type,
            score=final_score,
            diagnosis=diagnosis,
            physical_metrics=physical_metrics,
            error_source=error_source,
            suggested_fix=suggested_fix
        )
    
    def _analyze_failure(
        self,
        problem: OptimizationProblem,
        solution: Solution,
        phy_details: Dict[str, Any],
        llm_details: Dict[str, Any]
    ) -> tuple:
        """
        分析失败原因
        
        Returns:
            (feedback_type, error_source, diagnosis, suggested_fix)
        """
        # 优先检查物理违规
        if not phy_details.get("is_safe", True):
            violations = phy_details.get("violation_details", {})
            diagnosis = f"物理约束违规: {violations}"
            
            # 判断是模型问题还是求解器问题
            if self._is_model_error(problem, violations):
                return (
                    FeedbackType.PHYSICAL_ERROR,
                    AgentRole.PLANNER,
                    diagnosis,
                    "请检查约束条件是否完整，变量边界是否正确"
                )
            else:
                return (
                    FeedbackType.SOLVER_ERROR,
                    AgentRole.SOLVER,
                    diagnosis,
                    "求解器可能陷入局部最优，尝试使用全局优化算法"
                )
        
        # 检查求解器失败
        if not solution.is_feasible:
            return (
                FeedbackType.RUNTIME_ERROR,
                AgentRole.SOLVER,
                f"求解器未找到可行解: {solution.solver_message}",
                "检查问题是否有可行解，或尝试其他算法"
            )
        
        # 检查逻辑问题
        if not llm_details.get("is_logical", True):
            issues = llm_details.get("issues", [])
            return (
                FeedbackType.LOGICAL_ERROR,
                AgentRole.PLANNER,
                f"逻辑问题: {issues}",
                llm_details.get("suggestions", ["检查建模逻辑"])[0] if llm_details.get("suggestions") else "检查建模逻辑"
            )
        
        # 默认：评分过低
        return (
            FeedbackType.LOGICAL_ERROR,
            AgentRole.PLANNER,
            "解的质量未达到要求",
            "请优化目标函数或调整约束条件"
        )
    
    def _is_model_error(
        self, 
        problem: OptimizationProblem, 
        violations: Dict[str, Any]
    ) -> bool:
        """
        判断是否为模型错误
        
        如果违规的变量边界与问题定义不一致，则是模型错误
        """
        bounds = problem.get_variable_bounds()
        
        for var_name in violations:
            if var_name in bounds:
                # 检查边界是否过于宽松
                lb, ub = bounds[var_name]
                if lb == float('-inf') or ub == float('inf'):
                    return True  # 边界未定义，模型问题
        
        return False
    
    def diagnose_with_llm(
        self,
        problem: OptimizationProblem,
        solution: Solution,
        physical_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行深度诊断
        
        Args:
            problem: 优化问题
            solution: 求解结果
            physical_metrics: 物理指标
        
        Returns:
            诊断结果
        """
        if not self.llm:
            return {
                "error_source": "unknown",
                "error_type": "未知",
                "root_cause": "无法诊断（LLM 不可用）",
                "suggested_fix": "请检查问题定义和求解器配置"
            }
        
        try:
            prompt = JudgerPrompts.build_error_diagnosis_prompt(
                objective_function=problem.objective_function_latex,
                constraints=str(problem.constraints_latex),
                variables=str([v.name for v in problem.variables]),
                algorithm=solution.algorithm_used,
                solver_status="成功" if solution.is_feasible else "失败",
                error_message=solution.solver_message,
                physical_metrics=str(physical_metrics)
            )
            
            response = self.llm.chat(prompt, system_prompt=JudgerPrompts.SYSTEM_PROMPT)
            
            # 解析响应
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"LLM 诊断失败: {e}")
            return {
                "error_source": "unknown",
                "error_type": "诊断失败",
                "root_cause": str(e),
                "suggested_fix": "请手动检查问题"
            }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 FeedbackGenerator:")
    
    generator = FeedbackGenerator()
    
    # 创建测试问题
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1 + x_2",
        constraints_latex=[r"x_1 + x_2 \geq 5"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ]
    )
    
    # 测试通过情况
    print("\n测试1: 解通过")
    solution = Solution(
        is_feasible=True,
        decision_variables={"x1": 3.0, "x2": 3.0},
        objective_value=6.0
    )
    
    feedback = generator.generate_feedback(
        problem, solution,
        phy_score=0.8, phy_details={"is_safe": True, "efficiency": 0.8},
        llm_score=0.9, llm_details={"is_logical": True}
    )
    
    print(f"类型: {feedback.feedback_type}")
    print(f"评分: {feedback.score:.4f}")
    print(f"诊断: {feedback.diagnosis}")
    
    # 测试物理违规
    print("\n测试2: 物理违规")
    feedback = generator.generate_feedback(
        problem, solution,
        phy_score=0.0, 
        phy_details={"is_safe": False, "violation_details": {"x1": "超出边界"}},
        llm_score=0.5, llm_details={"is_logical": True}
    )
    
    print(f"类型: {feedback.feedback_type}")
    print(f"错误来源: {feedback.error_source}")
    print(f"诊断: {feedback.diagnosis}")
    print(f"建议: {feedback.suggested_fix}")

