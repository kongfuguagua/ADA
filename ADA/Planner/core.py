# -*- coding: utf-8 -*-
"""
Planner Agent 核心实现
负责主动状态增广和问题建模
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入安全的 JSON 序列化工具
from utils.json_utils import safe_json_dumps

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
        max_augmentation_steps: int = None
    ):
        """
        初始化 Planner
        
        Args:
            llm: LLM 服务
            tools: 工具注册表
            kb: 知识库服务
            max_augmentation_steps: 最大增广步数
        """
        self.llm = llm
        self.tools = tools or create_default_registry()
        self.kb = kb
        
        config = SystemConfig()
        self.max_steps = max_augmentation_steps or config.planner_max_augmentation_steps
        
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
            result_str = safe_json_dumps(result, ensure_ascii=False, indent=2)
            
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
            lines.append(f"   输入: {safe_json_dumps(step.tool_input, ensure_ascii=False)}")
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
        # 获取环境特征信息（用于指导建模）
        env_features = self._get_environment_features()
        
        prompt = PlannerPrompts.build_formulation_prompt(
            environment_state=state.to_prompt_string(),
            augmented_state=self._augmented_context,
            task_knowledge=tk_context,
            environment_features=env_features,
            feedback=feedback
        )
        
        # 调用 LLM 生成问题定义
        response = self.llm.chat(prompt, system_prompt=PlannerPrompts.SYSTEM_PROMPT)
        
        # 解析响应
        return self._parse_problem_response(response)
    
    def _get_environment_features(self) -> str:
        """获取环境特征信息（用于指导建模）"""
        try:
            # 尝试调用环境特征分析工具
            if "environment_feature_analysis" in self.tools:
                result = self.tools.execute("environment_feature_analysis")
                if "error" not in result:
                    return self._format_environment_features(result)
        except Exception as e:
            logger.warning(f"获取环境特征失败: {e}")
        
        return "环境特征信息不可用，请基于收集到的信息进行建模"
    
    def _format_environment_features(self, features: Dict[str, Any]) -> str:
        """格式化环境特征信息为提示词文本"""
        lines = []
        
        # 环境配置
        env_config = features.get("environment_config", {})
        lines.append("## 环境配置特征")
        lines.append(f"- 比赛类型: {env_config.get('competition', 'unknown')}")
        lines.append(f"- 支持储能: {'是' if env_config.get('has_storage') else '否'}")
        lines.append(f"- 支持可再生能源: {'是' if env_config.get('has_renewable') else '否'}")
        lines.append(f"- 支持再调度: {'是' if env_config.get('has_redispatch') else '否'}")
        lines.append(f"- 环境描述: {env_config.get('description', 'N/A')}")
        lines.append("")
        
        # 电网结构
        grid_structure = features.get("grid_structure", {})
        lines.append("## 电网规模")
        n_gen = grid_structure.get("n_generators", "unknown")
        n_load = grid_structure.get("n_loads", "unknown")
        n_line = grid_structure.get("n_lines", "unknown")
        lines.append(f"- 发电机数量 (n_g): {n_gen}")
        lines.append(f"- 负载数量 (n_l): {n_load}")
        lines.append(f"- 线路数量 (n_line): {n_line}")
        
        # 发电机信息
        gen_info = grid_structure.get("generator_info", {})
        if gen_info:
            lines.append("\n## 发电机参数")
            gen_pmin = gen_info.get("gen_pmin", [])
            gen_pmax = gen_info.get("gen_pmax", [])
            gen_cost = gen_info.get("gen_cost_per_mw", [])
            if gen_pmin and gen_pmax:
                lines.append(f"- 出力下限范围: [{min(gen_pmin):.1f}, {max(gen_pmin):.1f}] MW")
                lines.append(f"- 出力上限范围: [{min(gen_pmax):.1f}, {max(gen_pmax):.1f}] MW")
            if gen_cost:
                lines.append(f"- 成本系数范围: [{min(gen_cost):.3f}, {max(gen_cost):.3f}] /MW")
        lines.append("")
        
        # 决策空间
        decision_space = features.get("decision_space", {})
        lines.append("## 决策变量空间")
        primary_vars = decision_space.get("primary_variables", {})
        if "generator_power" in primary_vars:
            gen_power = primary_vars["generator_power"]
            dim = gen_power.get("dimension", "unknown")
            lines.append(f"- 主要变量: 发电机出力向量 p")
            lines.append(f"  * 维度: {dim} (等于发电机数量)")
            lines.append(f"  * 类型: 连续变量")
            lines.append(f"  * 约束: 边界约束 (p_min[i] <= p[i] <= p_max[i])")
        var_count = decision_space.get("variable_count_estimate", "unknown")
        lines.append(f"- 估计变量总数: {var_count}")
        lines.append("")
        
        # 约束摘要
        constraint_summary = features.get("constraint_summary", {})
        lines.append("## 约束条件摘要")
        constraint_types = constraint_summary.get("constraint_types", {})
        for name, info in constraint_types.items():
            lines.append(f"- {name}: {info.get('description', 'N/A')}")
            lines.append(f"  类型: {info.get('type', 'N/A')}, 复杂度: {info.get('complexity', 'N/A')}")
        total_count = constraint_summary.get("total_constraint_count_estimate", "unknown")
        lines.append(f"- 估计约束总数: {total_count}")
        
        return "\n".join(lines)
    
    def _parse_problem_response(self, response: str) -> OptimizationProblem:
        """
        解析问题建模响应
        
        使用多种策略尝试解析JSON，提高健壮性：
        1. 直接解析
        2. 提取JSON代码块
        3. 使用正则表达式提取JSON
        4. 尝试修复常见JSON格式问题
        """
        original_response = response
        response = response.strip()
        
        # 策略1: 尝试直接解析（如果响应已经是纯JSON）
        data = self._try_parse_json(response)
        if data is not None:
            return self._build_optimization_problem(data)
        
        # 策略2: 提取 ```json 代码块
        json_str = self._extract_json_block(response)
        if json_str:
            data = self._try_parse_json(json_str)
            if data is not None:
                return self._build_optimization_problem(data)
        
        # 策略3: 使用正则表达式查找JSON对象
        json_str = self._extract_json_with_regex(response)
        if json_str:
            data = self._try_parse_json(json_str)
            if data is not None:
                return self._build_optimization_problem(data)
        
        # 策略4: 尝试修复常见问题后解析
        json_str = self._extract_json_block(response) or response
        fixed_json = self._fix_common_json_issues(json_str)
        if fixed_json:
            data = self._try_parse_json(fixed_json)
            if data is not None:
                logger.warning("通过修复JSON格式问题成功解析")
                return self._build_optimization_problem(data)
        
        # 所有策略都失败，记录详细错误并返回默认问题
        logger.error(
            "解析问题定义失败",
            error_type="JSON_PARSE_FAILED",
            response_preview=original_response[:500] if len(original_response) > 500 else original_response,
            response_length=len(original_response)
        )
        
        # 尝试从响应中提取部分信息（即使JSON不完整）
        partial_data = self._extract_partial_info(original_response)
        if partial_data:
            logger.info("提取到部分信息，尝试构建问题")
            return self._build_optimization_problem(partial_data, is_partial=True)
        
        # 返回默认问题
        return OptimizationProblem(
            objective_function_latex=r"\min x",
            variables=[VariableDefinition(name="x", lower_bound=0, upper_bound=100)],
            modeling_rationale=f"解析失败，使用默认问题。原始响应长度: {len(original_response)}"
        )
    
    def _try_parse_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """尝试解析JSON字符串"""
        if not json_str or not json_str.strip():
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return None
    
    def _extract_json_block(self, response: str) -> Optional[str]:
        """提取代码块中的JSON"""
        # 尝试提取 ```json ... ``` 块（更灵活的模式）
        # 支持 ```json 后直接跟内容，或跟换行符
        pattern1 = r'```json\s*\n?(.*?)```'
        match = re.search(pattern1, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content.startswith('{'):
                return content
        
        # 尝试提取 ``` ... ``` 块（可能是JSON）
        pattern2 = r'```\s*\n?(.*?)```'
        match = re.search(pattern2, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 如果内容看起来像JSON（以 { 开头）
            if content.startswith('{'):
                return content
        
        return None
    
    def _extract_json_with_regex(self, response: str) -> Optional[str]:
        """使用正则表达式提取JSON对象"""
        # 查找第一个 { 到最后一个 } 之间的内容
        # 尝试匹配平衡的大括号
        start_idx = response.find('{')
        if start_idx == -1:
            return None
        
        # 从后往前找最后一个 }
        end_idx = response.rfind('}')
        if end_idx == -1 or end_idx <= start_idx:
            return None
        
        json_candidate = response[start_idx:end_idx + 1]
        
        # 简单验证：检查大括号是否平衡
        if json_candidate.count('{') == json_candidate.count('}'):
            return json_candidate
        
        return None
    
    def _fix_common_json_issues(self, json_str: str) -> Optional[str]:
        """修复常见的JSON格式问题"""
        if not json_str:
            return None
        
        fixed = json_str.strip()
        original = fixed
        
        # 移除尾随逗号（在 } 或 ] 之前）
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # 尝试修复未闭合的字符串引号（简单情况）
        # 如果最后一个字段缺少引号，尝试修复
        
        # 尝试修复单引号（JSON要求双引号）
        # 但要注意不要破坏字符串内容中的单引号
        # 这里只处理键名的情况：'key': -> "key":
        fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', fixed)
        
        # 尝试修复注释（JSON不支持注释）
        # 移除 // 和 /* */ 注释
        fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, re.DOTALL)
        
        return fixed if fixed != original else None
    
    def _extract_partial_info(self, response: str) -> Optional[Dict[str, Any]]:
        """从响应中提取部分信息（即使JSON不完整）"""
        partial = {}
        
        # 尝试提取目标函数
        latex_match = re.search(r'"objective_function_latex"\s*:\s*"([^"]*)"', response)
        if latex_match:
            partial["objective_function_latex"] = latex_match.group(1).replace('\\n', '\n').replace('\\\\', '\\')
        
        # 尝试提取变量信息
        variables = []
        var_pattern = r'"variables"\s*:\s*\[(.*?)\]'
        var_match = re.search(var_pattern, response, re.DOTALL)
        if var_match:
            # 尝试提取变量定义
            var_blocks = re.findall(r'\{[^}]*"name"\s*:\s*"([^"]*)"[^}]*\}', var_match.group(1))
            for var_name in var_blocks:
                variables.append({
                    "name": var_name,
                    "type": "continuous",
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                    "description": ""
                })
        
        if partial or variables:
            partial["variables"] = variables
            partial["constraints_latex"] = []
            partial["constraints_code"] = []
            partial["parameters"] = {}
            partial["is_minimization"] = True
            partial["objective_function_code"] = ""
            partial["modeling_rationale"] = "从部分解析的响应中提取"
            return partial
        
        return None
    
    def _build_optimization_problem(
        self, 
        data: Dict[str, Any], 
        is_partial: bool = False
    ) -> OptimizationProblem:
        """从解析的数据构建OptimizationProblem对象"""
        try:
            # 解析变量
            variables = []
            for var_data in data.get("variables", []):
                if not isinstance(var_data, dict):
                    continue
                    
                var_type = var_data.get("type", "continuous")
                if var_type in ["continuous", "binary", "integer"]:
                    try:
                        var_type = VariableType(var_type)
                    except ValueError:
                        var_type = VariableType.CONTINUOUS
                else:
                    var_type = VariableType.CONTINUOUS
                
                # 安全地获取边界值
                lower_bound = var_data.get("lower_bound")
                upper_bound = var_data.get("upper_bound", float('inf'))
                
                # 处理可能的None值
                if lower_bound is None:
                    lower_bound = float('-inf')
                if upper_bound is None:
                    upper_bound = float('inf')
                
                # 确保是数值类型
                try:
                    lower_bound = float(lower_bound) if lower_bound != float('-inf') else float('-inf')
                    upper_bound = float(upper_bound) if upper_bound != float('inf') else float('inf')
                except (ValueError, TypeError):
                    lower_bound = float('-inf')
                    upper_bound = float('inf')
                
                variables.append(VariableDefinition(
                    name=str(var_data.get("name", "x")),
                    type=var_type,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    description=str(var_data.get("description", ""))
                ))
            
            # 如果没有变量，添加默认变量
            if not variables:
                variables = [VariableDefinition(name="x", lower_bound=0, upper_bound=100)]
            
            # 安全地获取其他字段
            objective_latex = str(data.get("objective_function_latex", ""))
            objective_code = str(data.get("objective_function_code", ""))
            constraints_latex = data.get("constraints_latex", [])
            constraints_code = data.get("constraints_code", [])
            parameters = data.get("parameters", {})
            rationale = str(data.get("modeling_rationale", ""))
            
            # 确保列表类型
            if not isinstance(constraints_latex, list):
                constraints_latex = []
            if not isinstance(constraints_code, list):
                constraints_code = []
            if not isinstance(parameters, dict):
                parameters = {}
            
            # 确保约束是字符串列表
            constraints_latex = [str(c) for c in constraints_latex]
            constraints_code = [str(c) for c in constraints_code]
            
            # 处理参数中的非数值类型（递归处理嵌套结构）
            def clean_parameter_value(value):
                """递归清理参数值，支持嵌套的列表和字典"""
                if isinstance(value, dict):
                    return {k: clean_parameter_value(v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    return [clean_parameter_value(x) for x in value]
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return value
            
            cleaned_parameters = {}
            for k, v in parameters.items():
                try:
                    cleaned_parameters[k] = clean_parameter_value(v)
                except (ValueError, TypeError) as e:
                    logger.warning(f"清理参数 {k} 时出错: {e}，保留原始值")
                    cleaned_parameters[k] = v
            
            if is_partial:
                rationale = f"[部分解析] {rationale}"
            
            return OptimizationProblem(
                objective_function_latex=objective_latex,
                objective_function_code=objective_code,
                is_minimization=bool(data.get("is_minimization", True)),
                constraints_latex=constraints_latex,
                constraints_code=constraints_code,
                variables=variables,
                parameters=cleaned_parameters,
                modeling_rationale=rationale
            )
        except Exception as e:
            logger.error(f"构建OptimizationProblem失败: {e}", exc_info=True)
            # 返回最小可用的问题
            return OptimizationProblem(
                objective_function_latex=r"\min x",
                variables=[VariableDefinition(name="x", lower_bound=0, upper_bound=100)],
                modeling_rationale=f"构建失败: {str(e)}"
            )


# ============= 测试代码 =============
if __name__ == "__main__":
    print("Planner 模块测试需要配置 LLM API")
    print("请运行 python -m test.test_all 进行完整测试")
