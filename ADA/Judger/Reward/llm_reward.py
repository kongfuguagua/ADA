# -*- coding: utf-8 -*-
"""
LLM 评分器
使用 LLM 评估解的逻辑合理性
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, Solution
from utils.interact import BaseLLM
from utils.logger import get_logger

from .base_reward import BaseReward
from ..prompt import JudgerPrompts

logger = get_logger("LLMReward")


class LLMReward(BaseReward):
    """
    LLM 评分器
    使用 LLM 评估解的逻辑合理性和可解释性
    """
    
    def __init__(
        self, 
        llm: BaseLLM = None,
        weight: float = 1.0
    ):
        """
        初始化 LLM 评分器
        
        Args:
            llm: LLM 服务
            weight: 评分权重
        """
        super().__init__(weight)
        self.llm = llm
    
    def forward(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算 LLM 评分
        
        Args:
            problem: 优化问题
            solution: 求解结果
        
        Returns:
            (评分, 评估详情)
        """
        # 如果没有 LLM，使用规则评分
        if self.llm is None:
            return self._rule_based_evaluation(problem, solution)
        
        try:
            # 构建提示
            prompt = JudgerPrompts.build_logical_evaluation_prompt(
                objective_function=problem.objective_function_latex,
                constraints=str(problem.constraints_latex),
                modeling_rationale=problem.modeling_rationale,
                algorithm=solution.algorithm_used,
                decision_variables=str(solution.decision_variables),
                objective_value=str(solution.objective_value)
            )
            
            # 调用 LLM
            response = self.llm.chat(prompt, system_prompt=JudgerPrompts.SYSTEM_PROMPT)
            
            # 解析响应
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM 评估失败: {e}")
            return self._rule_based_evaluation(problem, solution)
    
    def _parse_llm_response(self, response: str) -> Tuple[float, Dict[str, Any]]:
        """解析 LLM 响应"""
        try:
            # 提取 JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            score = float(data.get("score", 0.5))
            details = {
                "is_logical": data.get("is_logical", True),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
                "comment": data.get("comment", "")
            }
            
            return score, details
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"解析 LLM 响应失败: {e}")
            return 0.5, {"error": str(e), "raw_response": response[:200]}
    
    def _rule_based_evaluation(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """
        规则评估（无 LLM 时使用）
        """
        details = {
            "is_logical": True,
            "issues": [],
            "suggestions": [],
            "comment": ""
        }
        
        score = 0.5  # 基础分
        
        # 检查解是否可行
        if solution.is_feasible:
            score += 0.2
        else:
            details["is_logical"] = False
            details["issues"].append("解不可行")
            return 0.0, details
        
        # 检查目标值是否合理
        obj_value = solution.objective_value
        if obj_value != float('inf') and obj_value != float('-inf'):
            score += 0.1
        else:
            details["issues"].append("目标值异常")
            score -= 0.2
        
        # 检查变量数量
        if len(solution.decision_variables) == len(problem.variables):
            score += 0.1
        else:
            details["issues"].append("变量数量不匹配")
            score -= 0.1
        
        # 检查求解时间
        if solution.solving_time < 60:
            score += 0.1
        else:
            details["suggestions"].append("求解时间较长，考虑优化算法")
        
        details["comment"] = f"规则评估完成，发现 {len(details['issues'])} 个问题"
        
        return min(1.0, max(0.0, score)), details


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 LLMReward:")
    
    # 创建测试问题和解
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        constraints_latex=[r"x_1 + x_2 \geq 1"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ],
        modeling_rationale="最小化二次目标，满足线性约束"
    )
    
    solution = Solution(
        is_feasible=True,
        algorithm_used="ConvexOptimizer",
        decision_variables={"x1": 0.5, "x2": 0.5},
        objective_value=0.5,
        solving_time=0.1
    )
    
    # 使用规则评估
    reward = LLMReward()
    score, details = reward(problem, solution)
    
    print(f"\n评分: {score:.4f}")
    print(f"是否合理: {details['is_logical']}")
    print(f"问题: {details['issues']}")
    print(f"建议: {details['suggestions']}")
    print(f"评语: {details['comment']}")

