# -*- coding: utf-8 -*-
"""
物理评分器
基于仿真环境的物理指标评分
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.const import OptimizationProblem, Solution, PhysicalMetrics
from utils.interact import BaseSimulator
from utils.logger import get_logger

from .base_reward import BaseReward

logger = get_logger("PhyReward")


class PhysicalReward(BaseReward):
    """
    物理评分器
    通过仿真环境评估解的物理可行性
    """
    
    def __init__(
        self, 
        simulator: BaseSimulator = None,
        weight: float = 1.0,
        safety_threshold: float = 0.95,
        efficiency_weight: float = 0.4,
        cost_weight: float = 0.3,
        stability_weight: float = 0.3
    ):
        """
        初始化物理评分器
        
        Args:
            simulator: 仿真器实例
            weight: 评分权重
            safety_threshold: 安全阈值
            efficiency_weight: 效率权重
            cost_weight: 成本权重
            stability_weight: 稳定性权重
        """
        super().__init__(weight)
        self.simulator = simulator
        self.safety_threshold = safety_threshold
        self.efficiency_weight = efficiency_weight
        self.cost_weight = cost_weight
        self.stability_weight = stability_weight
    
    def forward(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算物理评分
        
        Args:
            problem: 优化问题
            solution: 求解结果
        
        Returns:
            (评分, 物理指标详情)
        """
        # 如果没有仿真器，使用模拟评分
        if self.simulator is None:
            return self._mock_evaluation(problem, solution)
        
        try:
            # 执行仿真
            action_vector = solution.to_action_vector()
            sim_result = self.simulator.run(action_vector)
            
            # 解析仿真结果
            metrics = self._parse_simulation_result(sim_result)
            
            # 计算评分
            score = self._calculate_score(metrics)
            
            return score, metrics.model_dump()
            
        except Exception as e:
            logger.error(f"仿真执行失败: {e}")
            return 0.0, {"error": str(e), "is_safe": False}
    
    def _mock_evaluation(
        self, 
        problem: OptimizationProblem, 
        solution: Solution
    ) -> Tuple[float, Dict[str, Any]]:
        """
        模拟评估（无仿真器时使用）
        基于约束满足度和目标值进行评分
        """
        details = {
            "is_safe": True,
            "cost": 0.0,
            "efficiency": 0.0,
            "stability_margin": 0.0,
            "violation_details": {}
        }
        
        # 检查解是否可行
        if not solution.is_feasible:
            details["is_safe"] = False
            return 0.0, details
        
        # 检查变量边界
        bounds = problem.get_variable_bounds()
        violations = {}
        
        for var_name, value in solution.decision_variables.items():
            if var_name in bounds:
                lb, ub = bounds[var_name]
                if value < lb:
                    violations[var_name] = f"低于下界 {lb}"
                    details["is_safe"] = False
                elif value > ub:
                    violations[var_name] = f"高于上界 {ub}"
                    details["is_safe"] = False
        
        details["violation_details"] = violations
        
        if not details["is_safe"]:
            return 0.0, details
        
        # 计算效率评分（基于目标值）
        obj_value = solution.objective_value
        if obj_value != float('inf') and obj_value != float('-inf'):
            # 归一化目标值到 [0, 1]
            # 假设目标值越小越好（最小化问题）
            if problem.is_minimization:
                details["efficiency"] = max(0, 1 - abs(obj_value) / 1000)
            else:
                details["efficiency"] = min(1, abs(obj_value) / 1000)
        else:
            details["efficiency"] = 0.0
        
        # 模拟成本和稳定性
        details["cost"] = abs(obj_value) if obj_value != float('inf') else 1000
        details["stability_margin"] = 0.8  # 模拟稳定裕度
        
        # 计算总分
        score = (
            self.efficiency_weight * details["efficiency"] +
            self.cost_weight * (1 - min(1, details["cost"] / 1000)) +
            self.stability_weight * details["stability_margin"]
        )
        
        return min(1.0, max(0.0, score)), details
    
    def _parse_simulation_result(self, result: Dict[str, Any]) -> PhysicalMetrics:
        """解析仿真结果为物理指标"""
        return PhysicalMetrics(
            is_safe=result.get("is_safe", True),
            cost=result.get("cost", 0.0),
            efficiency=result.get("efficiency", 0.0),
            stability_margin=result.get("stability_margin", 0.0),
            violation_details=result.get("violations", {})
        )
    
    def _calculate_score(self, metrics: PhysicalMetrics) -> float:
        """计算综合评分"""
        if not metrics.is_safe:
            return 0.0
        
        score = (
            self.efficiency_weight * metrics.efficiency +
            self.cost_weight * (1 - min(1, metrics.cost / 1000)) +
            self.stability_weight * metrics.stability_margin
        )
        
        return min(1.0, max(0.0, score))


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("测试 PhysicalReward:")
    
    # 创建测试问题和解
    problem = OptimizationProblem(
        objective_function_latex=r"\min x_1 + x_2",
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ]
    )
    
    # 可行解
    feasible_solution = Solution(
        is_feasible=True,
        decision_variables={"x1": 5.0, "x2": 3.0},
        objective_value=8.0
    )
    
    # 不可行解
    infeasible_solution = Solution(
        is_feasible=True,
        decision_variables={"x1": 15.0, "x2": 3.0},  # x1 超出边界
        objective_value=18.0
    )
    
    reward = PhysicalReward()
    
    # 测试可行解
    score, details = reward(problem, feasible_solution)
    print(f"\n可行解评分: {score:.4f}")
    print(f"详情: {details}")
    
    # 测试不可行解
    score, details = reward(problem, infeasible_solution)
    print(f"\n不可行解评分: {score:.4f}")
    print(f"详情: {details}")

