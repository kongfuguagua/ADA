# -*- coding: utf-8 -*-
"""
Planner Agent 核心实现
负责主动状态增广和问题建模
"""

import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.const import (
    EnvironmentState,
    OptimizationProblem,
    VariableDefinition,
    VariableType,
    Feedback,
    ToolAction,
    AugmentationStep,
)
from utils.interact import BasePlanner
from utils.llm import BaseLLM
from utils.logger import get_logger
from config import SystemConfig

from knowledgebase.service import KnowledgeService
from .prompt import PlannerPrompts
from .tools.registry import ToolRegistry, create_default_registry

logger = get_logger("Planner")


class PlannerAgent(BasePlanner):
    """
    规划智能体
    
    核心职责：
    1. 主动状态增广 - 通过工具调用收集信息
    2. 问题建模 - 将需求转化为数学优化问题
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry = None,
        kb: KnowledgeService = None,
        max_augmentation_steps: int = None,
        config: SystemConfig = None
    ):
        """
        初始化 Planner
        
        Args:
            llm: LLM 服务
            tools: 工具注册表
            kb: 知识库服务
            max_augmentation_steps: 最大增广步数
            config: 系统配置（如果为 None 则创建新实例）
        """
        self.llm = llm
        self.tools = tools or create_default_registry()
        self.kb = kb
        
        self.config = config or SystemConfig()
        self.max_steps = max_augmentation_steps or self.config.planner_max_augmentation_steps
        
        # 记录本次规划的工具调用链
        self._tool_chain: List[AugmentationStep] = []
        self._augmented_context: str = ""
    
    def plan(
        self, 
        state: EnvironmentState, 
        retry_feedback: Optional[Feedback] = None
    ) -> OptimizationProblem:
        """
        核心规划方法
        
        Args:
            state: 环境状态
            retry_feedback: 重试时的反馈信息
        
        Returns:
            优化问题定义
        """
        logger.info("开始规划", instruction=state.user_instruction[:50])
        
        # 重置工具链
        self._tool_chain = []
        
        # 1. 检索动作知识 (AK)
        ak_context = self._get_action_knowledge(state)
        
        # 2. 主动状态增广
        self._augmented_context = self._augment_state(state, ak_context)
        
        # 3. 检索任务知识 (TK)
        tk_context = self._get_task_knowledge(self._augmented_context)
        
        # 4. 问题建模
        problem = self._formulate_problem(state, tk_context, retry_feedback)
        
        logger.info("规划完成", 
                   variables=len(problem.variables),
                   constraints=len(problem.constraints_latex))
        
        return problem
    
    def get_tool_chain(self) -> List[Dict[str, Any]]:
        """获取本次规划的工具调用链"""
        return [step.model_dump() for step in self._tool_chain]
    
    def _get_action_knowledge(self, state: EnvironmentState) -> str:
        """检索动作知识"""
        if not self.kb:
            return "暂无动作知识"
        
        try:
            items = self.kb.query_action_knowledge(state.user_instruction)
            if items:
                return "\n".join([item.content for item in items])
            return "暂无相关动作知识"
        except Exception as e:
            logger.warning(f"检索动作知识失败: {e}")
            return "知识库暂不可用"
    
    def _get_task_knowledge(self, context: str) -> str:
        """检索任务知识"""
        if not self.kb:
            return "暂无任务知识"
        
        try:
            items = self.kb.query_task_knowledge(context)
            if items:
                return "\n".join([item.content for item in items])
            return "暂无相关任务知识"
        except Exception as e:
            logger.warning(f"检索任务知识失败: {e}")
            return "知识库暂不可用"
    
    def _augment_state(self, state: EnvironmentState, ak_context: str) -> str:
        """
        主动状态增广
        通过工具调用收集必要信息
        
        Args:
            state: 初始环境状态
            ak_context: 动作知识上下文
        
        Returns:
            增广后的状态上下文
        """
        current_context = state.to_prompt_string()
        tool_history = "暂无"
        
        for step in range(self.max_steps):
            logger.debug(f"状态增广步骤 {step + 1}/{self.max_steps}")
            
            # 构建提示
            prompt = PlannerPrompts.build_augmentation_prompt(
                current_state=current_context,
                tool_descriptions=self.tools.get_tool_descriptions(),
                action_knowledge=ak_context,
                tool_history=tool_history
            )
            
            # 调用 LLM 决策
            response = self.llm.chat(prompt, system_prompt=PlannerPrompts.SYSTEM_PROMPT)
            
            # 解析响应
            action = self._parse_augmentation_response(response)
            
            if action.is_finish:
                logger.info(f"状态增广完成，共 {step} 步")
                break
            
            # 执行工具调用
            result = self.tools.execute(action.tool_name, **action.params)
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            
            # 记录步骤
            aug_step = AugmentationStep(
                thought=action.params.get("thought", ""),
                tool_selected=action.tool_name,
                tool_input=action.params,
                tool_output=result_str,
                updated_knowledge=""
            )
            self._tool_chain.append(aug_step)
            
            # 更新上下文
            current_context += f"\n\n## 工具调用结果 ({action.tool_name})\n{result_str}"
            tool_history = self._format_tool_history()
        
        return current_context
    
    def _parse_augmentation_response(self, response: str) -> ToolAction:
        """解析增广响应"""
        response = response.strip()
        
        # 检查是否结束
        if "FINISH" in response.upper():
            return ToolAction(tool_name="", is_finish=True)
        
        # 尝试解析 JSON
        try:
            # 提取 JSON 块
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            return ToolAction(
                tool_name=data.get("tool_name", ""),
                params=data.get("tool_params", {}),
                is_finish=False
            )
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"解析增广响应失败: {e}")
            return ToolAction(tool_name="", is_finish=True)
    
    def _format_tool_history(self) -> str:
        """格式化工具调用历史"""
        if not self._tool_chain:
            return "暂无"
        
        lines = []
        for i, step in enumerate(self._tool_chain, 1):
            lines.append(f"{i}. {step.tool_selected}")
            lines.append(f"   输入: {json.dumps(step.tool_input, ensure_ascii=False)}")
            lines.append(f"   输出: {step.tool_output[:100]}...")
        
        return "\n".join(lines)
    
    def _formulate_problem(
        self, 
        state: EnvironmentState, 
        tk_context: str,
        feedback: Optional[Feedback]
    ) -> OptimizationProblem:
        """
        问题建模
        
        Args:
            state: 环境状态
            tk_context: 任务知识上下文
            feedback: 重试反馈
        
        Returns:
            优化问题
        """
        prompt = PlannerPrompts.build_formulation_prompt(
            environment_state=state.to_prompt_string(),
            augmented_state=self._augmented_context,
            task_knowledge=tk_context,
            feedback=feedback
        )
        
        # 调用 LLM 生成问题定义
        response = self.llm.chat(prompt, system_prompt=PlannerPrompts.SYSTEM_PROMPT)
        
        # 解析响应
        return self._parse_problem_response(response)
    
    def _parse_problem_response(self, response: str) -> OptimizationProblem:
        """解析问题建模响应"""
        try:
            # 提取 JSON 块
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            # 解析变量
            variables = []
            for var_data in data.get("variables", []):
                var_type = var_data.get("type", "continuous")
                if var_type in ["continuous", "binary", "integer"]:
                    var_type = VariableType(var_type)
                else:
                    var_type = VariableType.CONTINUOUS
                
                variables.append(VariableDefinition(
                    name=var_data.get("name", "x"),
                    type=var_type,
                    lower_bound=var_data.get("lower_bound", float('-inf')),
                    upper_bound=var_data.get("upper_bound", float('inf')),
                    description=var_data.get("description", "")
                ))
            
            return OptimizationProblem(
                objective_function_latex=data.get("objective_function_latex", ""),
                objective_function_code=data.get("objective_function_code", ""),
                is_minimization=data.get("is_minimization", True),
                constraints_latex=data.get("constraints_latex", []),
                constraints_code=data.get("constraints_code", []),
                variables=variables,
                parameters=data.get("parameters", {}),
                modeling_rationale=data.get("modeling_rationale", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析问题定义失败: {e}")
            # 返回默认问题
            return OptimizationProblem(
                objective_function_latex=r"\min x",
                variables=[VariableDefinition(name="x", lower_bound=0, upper_bound=100)],
                modeling_rationale=f"解析失败，使用默认问题: {str(e)}"
            )


# ============= 测试代码 =============
if __name__ == "__main__":
    print("Planner 模块测试需要配置 LLM API")
    print("请运行 python -m test.test_all 进行完整测试")
