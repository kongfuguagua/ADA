# -*- coding: utf-8 -*-
"""
ADA 数据契约定义
定义所有 Agent 之间通信的标准数据结构
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


# ================= 0. 枚举定义 =================

class AgentRole(str, Enum):
    """智能体角色枚举"""
    PLANNER = "planner"
    SOLVER = "solver"
    JUDGER = "judger"
    SUMMARIZER = "summarizer"


class KnowledgeType(str, Enum):
    """知识类型枚举"""
    TK = "task_knowledge"    # 任务知识：问题建模模版
    AK = "action_knowledge"  # 动作知识：工具使用策略


class FeedbackType(str, Enum):
    """反馈类型枚举"""
    PASSED = "passed"                  # 通过
    PHYSICAL_ERROR = "physical_error"  # 硬约束违规（电压越限等）
    LOGICAL_ERROR = "logical_error"    # 建模逻辑错误（变量未定义等）
    RUNTIME_ERROR = "runtime_error"    # 求解器运行时错误
    SOLVER_ERROR = "solver_error"      # 求解器算法错误（陷入局部最优等）


class VariableType(str, Enum):
    """决策变量类型枚举"""
    CONTINUOUS = "continuous"  # 连续变量
    BINARY = "binary"          # 二元变量 (0/1)
    INTEGER = "integer"        # 整数变量


# ================= 1. 环境与输入 =================

class EnvironmentState(BaseModel):
    """
    系统的原始输入状态
    Planner 的输入
    """
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    real_data: Dict[str, float] = Field(default_factory=dict)  # 实际数据
    user_instruction: str = ""  # 自然语言指令
    extra_context: Dict[str, Any] = Field(default_factory=dict)  # 允许动态扩展

    def to_prompt_string(self) -> str:
        """转换为 Prompt 可用的字符串格式"""
        lines = [
            f"时间戳: {self.timestamp}",
            f"用户指令: {self.user_instruction}",
            "实时数据:",
        ]
        for key, value in self.real_data.items():
            lines.append(f"  - {key}: {value}")
        if self.extra_context:
            lines.append("附加上下文:")
            for key, value in self.extra_context.items():
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)


# ================= 2. 知识库数据 =================

class KnowledgeItem(BaseModel):
    """RAG 检索或存储的最小单元"""
    id: str
    type: KnowledgeType
    content: str  # 文本内容
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 来源、时间、成功率等
    
    def to_context_string(self) -> str:
        """转换为上下文字符串"""
        return f"[{self.type.value}] {self.content}"


# ================= 3. Planner 内部类型 =================

class ToolAction(BaseModel):
    """工具调用动作"""
    tool_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    is_finish: bool = False  # 是否结束增广循环


class AugmentationStep(BaseModel):
    """记录一次工具调用的思考过程"""
    thought: str  # 思考过程
    tool_selected: str  # 选择的工具
    tool_input: Dict[str, Any] = Field(default_factory=dict)  # 工具输入
    tool_output: str = ""  # 工具返回的观察结果
    updated_knowledge: str = ""  # 基于观察对自己认知的更新


# ================= 4. Planner -> Solver 协议 =================

class VariableDefinition(BaseModel):
    """决策变量定义"""
    name: str
    type: VariableType = VariableType.CONTINUOUS
    lower_bound: float = float('-inf')
    upper_bound: float = float('inf')
    description: str = ""
    
    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.lower_bound, self.upper_bound)


class OptimizationProblem(BaseModel):
    """
    Planner 的最终产出，Solver 的输入
    优化问题的完整定义
    """
    # 数学表达式
    objective_function_latex: str  # 目标函数的数学表达
    objective_function_code: str = ""  # 目标函数的 Python 代码表达
    constraints_latex: List[str] = Field(default_factory=list)  # 约束条件的数学表达
    constraints_code: List[str] = Field(default_factory=list)  # 约束条件的代码表达
    
    # 结构化定义
    variables: List[VariableDefinition] = Field(default_factory=list)
    parameters: Dict[str, float] = Field(default_factory=dict)  # 常数项
    
    # 元信息
    is_minimization: bool = True  # 是否为最小化问题
    modeling_rationale: str = ""  # 建模理由说明
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_variable_names(self) -> List[str]:
        """获取所有变量名"""
        return [v.name for v in self.variables]
    
    def get_variable_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取所有变量的边界"""
        return {v.name: v.bounds for v in self.variables}


# ================= 5. Solver -> Judger 协议 =================

class SolverAlgorithmMeta(BaseModel):
    """描述求解算法的元数据"""
    name: str
    description: str = ""
    capabilities: Dict[str, float] = Field(default_factory=dict)  # 能力向量


class Solution(BaseModel):
    """求解结果"""
    is_feasible: bool = False  # 是否找到可行解
    algorithm_used: str = ""  # 使用的算法名称
    decision_variables: Dict[str, float] = Field(default_factory=dict)  # 最优解
    objective_value: float = float('inf')  # 目标函数值
    solving_time: float = 0.0  # 求解时间（秒）
    convergence_curve: List[float] = Field(default_factory=list)  # 收敛曲线
    solver_message: str = ""  # 求解器消息
    
    def to_action_vector(self) -> List[float]:
        """转换为动作向量"""
        return list(self.decision_variables.values())


# ================= 6. Judger -> Planner/Summarizer 协议 =================

class PhysicalMetrics(BaseModel):
    """仿真环境返回的物理指标"""
    is_safe: bool = True
    cost: float = 0.0
    stability_margin: float = 0.0
    efficiency: float = 0.0
    violation_details: Dict[str, float] = Field(default_factory=dict)


class Feedback(BaseModel):
    """系统的"奖惩信号"""
    feedback_type: FeedbackType
    score: float = 0.0  # 综合得分 R ∈ [0, 1]
    diagnosis: str = ""  # LLM 给出的诊断建议
    physical_metrics: Optional[PhysicalMetrics] = None
    error_source: Optional[AgentRole] = None  # 错误来源
    suggested_fix: str = ""  # 建议的修复方案
    
    def needs_retry(self) -> bool:
        """是否需要重试"""
        return self.feedback_type != FeedbackType.PASSED
    
    def is_model_error(self) -> bool:
        """是否为模型错误（Planner 责任）"""
        return self.feedback_type in [FeedbackType.LOGICAL_ERROR, FeedbackType.PHYSICAL_ERROR]
    
    def is_solver_error(self) -> bool:
        """是否为求解器错误（Solver 责任）"""
        return self.feedback_type in [FeedbackType.SOLVER_ERROR, FeedbackType.RUNTIME_ERROR]


# ================= 7. Summarizer 历史轨迹 =================

class ExecutionTrace(BaseModel):
    """一次完整的运行记录，用于复盘"""
    trace_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    environment: EnvironmentState
    problem: OptimizationProblem
    solution: Solution
    feedback: Feedback
    tool_chain: List[AugmentationStep] = Field(default_factory=list)  # Planner 的工具调用链
    attempt_count: int = 1  # 尝试次数
    
    def is_successful(self) -> bool:
        """是否成功"""
        return self.feedback.feedback_type == FeedbackType.PASSED


# ============= 测试代码 =============
if __name__ == "__main__":
    # 测试 EnvironmentState
    env = EnvironmentState(
        user_instruction="优化电网调度",
        real_data={"load": 100.0, "generation": 120.0}
    )
    print("环境状态:")
    print(env.to_prompt_string())
    print()
    
    # 测试 OptimizationProblem
    problem = OptimizationProblem(
        objective_function_latex=r"\min \sum_{i} c_i x_i",
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=100),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=50),
        ],
        parameters={"c1": 10.0, "c2": 20.0}
    )
    print("优化问题:")
    print(f"  变量: {problem.get_variable_names()}")
    print(f"  边界: {problem.get_variable_bounds()}")
    print()
    
    # 测试 Feedback
    feedback = Feedback(
        feedback_type=FeedbackType.PASSED,
        score=0.85,
        diagnosis="解满足所有约束"
    )
    print("反馈:")
    print(f"  需要重试: {feedback.needs_retry()}")
    print(f"  评分: {feedback.score}")
