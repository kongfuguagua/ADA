# -*- coding: utf-8 -*-
"""
知识更新器
负责将提炼的知识写入知识库
"""

import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.const import KnowledgeItem, KnowledgeType, ExecutionTrace
from utils.interact import BaseLLM
from utils.logger import get_logger

from knowledgebase.service import KnowledgeService
from .prompt import SummarizerPrompts

logger = get_logger("KnowledgeUpdater")


class KnowledgeUpdater:
    """
    知识更新器
    从执行轨迹中提炼知识并更新知识库
    """
    
    def __init__(
        self,
        kb: KnowledgeService,
        llm: BaseLLM = None,
        min_score_threshold: float = 0.7
    ):
        """
        初始化知识更新器
        
        Args:
            kb: 知识库服务
            llm: LLM 服务
            min_score_threshold: 最小入库分数阈值
        """
        self.kb = kb
        self.llm = llm
        self.min_score_threshold = min_score_threshold
    
    def update_from_trace(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """
        从执行轨迹更新知识库
        
        Args:
            trace: 执行轨迹
        
        Returns:
            更新结果 {ak_updated, tk_updated, details}
        """
        result = {
            "ak_updated": False,
            "tk_updated": False,
            "details": {}
        }
        
        # 检查是否值得入库
        if trace.feedback.score < self.min_score_threshold:
            logger.info(f"评分 {trace.feedback.score:.4f} 低于阈值，跳过知识更新")
            result["details"]["skipped"] = "评分低于阈值"
            return result
        
        if not trace.is_successful():
            logger.info("案例未成功，跳过知识更新")
            result["details"]["skipped"] = "案例未成功"
            return result
        
        # 提炼动作知识
        ak_item = self.extract_action_knowledge(trace)
        if ak_item:
            ak_id = self.kb.add_knowledge(
                ak_item.content,
                KnowledgeType.AK,
                ak_item.metadata
            )
            result["ak_updated"] = True
            result["details"]["ak_id"] = ak_id
            logger.info(f"添加动作知识: {ak_id}")
        
        # 提炼任务知识
        tk_item = self.extract_task_knowledge(trace)
        if tk_item:
            tk_id = self.kb.add_knowledge(
                tk_item.content,
                KnowledgeType.TK,
                tk_item.metadata
            )
            result["tk_updated"] = True
            result["details"]["tk_id"] = tk_id
            logger.info(f"添加任务知识: {tk_id}")
        
        return result
    
    def extract_action_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """
        从轨迹中提取动作知识
        
        Args:
            trace: 执行轨迹
        
        Returns:
            动作知识项（如果值得保存）
        """
        # 检查是否有工具调用
        if not trace.tool_chain:
            return None
        
        # 格式化工具链
        tool_chain_str = self._format_tool_chain(trace.tool_chain)
        
        if self.llm:
            # 使用 LLM 提炼
            content = self._extract_ak_with_llm(trace, tool_chain_str)
        else:
            # 使用规则提炼
            content = self._extract_ak_with_rules(trace, tool_chain_str)
        
        if not content:
            return None
        
        return KnowledgeItem(
            id=str(uuid.uuid4()),
            type=KnowledgeType.AK,
            content=content,
            metadata={
                "source_trace": trace.trace_id,
                "score": trace.feedback.score,
                "timestamp": datetime.now().isoformat(),
                "tool_count": len(trace.tool_chain)
            }
        )
    
    def extract_task_knowledge(self, trace: ExecutionTrace) -> Optional[KnowledgeItem]:
        """
        从轨迹中提取任务知识
        
        Args:
            trace: 执行轨迹
        
        Returns:
            任务知识项（如果值得保存）
        """
        if self.llm:
            content = self._extract_tk_with_llm(trace)
        else:
            content = self._extract_tk_with_rules(trace)
        
        if not content:
            return None
        
        return KnowledgeItem(
            id=str(uuid.uuid4()),
            type=KnowledgeType.TK,
            content=content,
            metadata={
                "source_trace": trace.trace_id,
                "score": trace.feedback.score,
                "timestamp": datetime.now().isoformat(),
                "algorithm": trace.solution.algorithm_used,
                "objective_value": trace.solution.objective_value
            }
        )
    
    def _format_tool_chain(self, tool_chain: list) -> str:
        """格式化工具调用链"""
        lines = []
        for i, step in enumerate(tool_chain, 1):
            if isinstance(step, dict):
                tool_name = step.get("tool_selected", "unknown")
                thought = step.get("thought", "")
                output = step.get("tool_output", "")[:100]
            else:
                tool_name = getattr(step, "tool_selected", "unknown")
                thought = getattr(step, "thought", "")
                output = getattr(step, "tool_output", "")[:100]
            
            lines.append(f"{i}. 工具: {tool_name}")
            if thought:
                lines.append(f"   思考: {thought}")
            if output:
                lines.append(f"   结果: {output}...")
        
        return "\n".join(lines)
    
    def _extract_ak_with_llm(self, trace: ExecutionTrace, tool_chain_str: str) -> str:
        """使用 LLM 提炼动作知识"""
        prompt = SummarizerPrompts.build_action_knowledge_prompt(
            user_instruction=trace.environment.user_instruction,
            score=trace.feedback.score,
            tool_chain=tool_chain_str
        )
        
        try:
            response = self.llm.chat(prompt, system_prompt=SummarizerPrompts.SYSTEM_PROMPT)
            # 提取知识条目
            if "```" in response:
                content = response.split("```")[1].strip()
            else:
                content = response.strip()
            return content
        except Exception as e:
            logger.error(f"LLM 提炼动作知识失败: {e}")
            return self._extract_ak_with_rules(trace, tool_chain_str)
    
    def _extract_ak_with_rules(self, trace: ExecutionTrace, tool_chain_str: str) -> str:
        """使用规则提炼动作知识"""
        tools_used = []
        for step in trace.tool_chain:
            if isinstance(step, dict):
                tools_used.append(step.get("tool_selected", "unknown"))
            else:
                tools_used.append(getattr(step, "tool_selected", "unknown"))
        
        if not tools_used:
            return ""
        
        instruction = trace.environment.user_instruction[:50]
        tools_str = " -> ".join(tools_used)
        
        return f"当处理「{instruction}...」类问题时，建议按顺序调用 {tools_str}。评分: {trace.feedback.score:.2f}"
    
    def _extract_tk_with_llm(self, trace: ExecutionTrace) -> str:
        """使用 LLM 提炼任务知识"""
        prompt = SummarizerPrompts.build_task_knowledge_prompt(
            user_instruction=trace.environment.user_instruction,
            score=trace.feedback.score,
            objective_function=trace.problem.objective_function_latex,
            constraints=str(trace.problem.constraints_latex),
            modeling_rationale=trace.problem.modeling_rationale,
            algorithm=trace.solution.algorithm_used,
            objective_value=str(trace.solution.objective_value)
        )
        
        try:
            response = self.llm.chat(prompt, system_prompt=SummarizerPrompts.SYSTEM_PROMPT)
            if "```" in response:
                content = response.split("```")[1].strip()
            else:
                content = response.strip()
            return content
        except Exception as e:
            logger.error(f"LLM 提炼任务知识失败: {e}")
            return self._extract_tk_with_rules(trace)
    
    def _extract_tk_with_rules(self, trace: ExecutionTrace) -> str:
        """使用规则提炼任务知识"""
        instruction = trace.environment.user_instruction[:50]
        obj_func = trace.problem.objective_function_latex[:50]
        algorithm = trace.solution.algorithm_used
        
        return (
            f"对于「{instruction}...」类问题，"
            f"目标函数可建模为 {obj_func}...，"
            f"使用 {algorithm} 算法求解，"
            f"评分: {trace.feedback.score:.2f}"
        )


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import (
        EnvironmentState, OptimizationProblem, Solution, 
        Feedback, FeedbackType, VariableDefinition, AugmentationStep
    )
    from knowledgebase.Embeddings import MockEmbedding
    import tempfile
    import shutil
    
    print("测试 KnowledgeUpdater:")
    
    # 创建临时知识库
    temp_dir = tempfile.mkdtemp()
    kb = KnowledgeService(
        embedding_model=MockEmbedding(),
        storage_path=temp_dir
    )
    
    updater = KnowledgeUpdater(kb=kb)
    
    # 创建测试轨迹
    trace = ExecutionTrace(
        trace_id="test_001",
        environment=EnvironmentState(
            user_instruction="优化电网调度，最小化发电成本"
        ),
        problem=OptimizationProblem(
            objective_function_latex=r"\min \sum c_i P_i",
            constraints_latex=[r"\sum P_i = D"],
            variables=[
                VariableDefinition(name="P1", lower_bound=0, upper_bound=100),
                VariableDefinition(name="P2", lower_bound=0, upper_bound=80),
            ],
            modeling_rationale="最小化发电成本"
        ),
        solution=Solution(
            is_feasible=True,
            algorithm_used="ConvexOptimizer",
            decision_variables={"P1": 60, "P2": 40},
            objective_value=1000
        ),
        feedback=Feedback(
            feedback_type=FeedbackType.PASSED,
            score=0.85
        ),
        tool_chain=[
            AugmentationStep(
                thought="需要获取负载预测",
                tool_selected="load_forecast",
                tool_input={"hours": 24},
                tool_output="峰值负载: 100MW"
            )
        ]
    )
    
    # 测试更新
    result = updater.update_from_trace(trace)
    
    print(f"\n更新结果:")
    print(f"  AK 更新: {result['ak_updated']}")
    print(f"  TK 更新: {result['tk_updated']}")
    print(f"  详情: {result['details']}")
    
    print(f"\n知识库状态: {kb}")
    
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n测试完成")

