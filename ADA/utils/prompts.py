# -*- coding: utf-8 -*-
"""
提示词管理模块
定义 Judger 的 System Prompt 和提示词模板
"""

from typing import List, Dict, Any
from ADA.utils.definitions import CandidateAction


class PromptManager:
    """
    提示词管理器
    
    负责构建 Judger 的 System Prompt 和用户提示
    """
    
    def __init__(self):
        """初始化提示词管理器"""
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        构建 System Prompt
        
        Returns:
            System Prompt 字符串
        """
        prompt = """你是电网调度 AI 的战略指挥官。你的任务是分析多个候选方案，并生成融合策略。

## 输入信息

1. **Planner 方案**：基于物理规则的拓扑动作（节点分裂、线路开关）
2. **Solver 方案**：基于数学优化的重调度/削减/储能动作
3. **历史参考**：类似场景下的成功案例
4. **当前状态**：电网的过载详情

## 你的任务

分析上述输入，生成**融合策略**：
- **组合模式**：将 Planner 的拓扑动作与 Solver 的重调度动作叠加
- **历史迁移**：参考历史案例，补充 Planner/Solver 未考虑的动作
- **优化改进**：在现有方案基础上进行微调

## 输出格式

必须输出 JSON 格式：

{
  "thought": "分析过程...",
  "action_type": "redispatch|set_line_status|set_bus|do_nothing",
  "params": {
    "gen_id": 0,              // redispatch: 发电机ID
    "amount_mw": 10.5,        // redispatch: 调整量(MW)
    "line_id": 3,             // set_line_status: 线路ID
    "status": 1,              // set_line_status: +1开启, -1关闭
    "substation_id": 1,       // set_bus: 变电站ID
    "topology_vector": [1,1]  // set_bus: 拓扑向量
  },
  "reasoning": "选择原因..."
}

## 核心原则

1. **安全性优先**：优先消除过载（rho < 100%）
2. **融合优势**：结合拓扑和重调度的优势
3. **成本最小**：在安全前提下选择成本最低方案
"""
        return prompt
    
    def build_fusion_prompt(
        self,
        observation_text: str,
        planner_candidates: List[CandidateAction],
        solver_candidates: List[CandidateAction],
        history_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        构建融合提示词
        
        Args:
            observation_text: 当前观测的文本描述
            planner_candidates: Planner 生成的候选动作
            solver_candidates: Solver 生成的候选动作
            history_context: 历史参考上下文
            
        Returns:
            消息列表（用于 LLM 调用）
        """
        messages = []
        
        # System Prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 构建用户提示
        user_parts = []
        user_parts.append("=== 当前电网状态 ===")
        user_parts.append(observation_text)
        user_parts.append("")
        
        # Planner 方案
        if planner_candidates:
            user_parts.append("=== Planner 方案（拓扑专家）===")
            for i, candidate in enumerate(planner_candidates[:3]):  # 最多显示3个
                user_parts.append(f"[{i}] {candidate.description}")
                if candidate.simulation_result:
                    rho_max = candidate.simulation_result.get('rho_max', 'N/A')
                    user_parts.append(f"    预期效果: rho_max={rho_max}")
            user_parts.append("")
        
        # Solver 方案
        if solver_candidates:
            user_parts.append("=== Solver 方案（优化专家）===")
            for i, candidate in enumerate(solver_candidates[:3]):  # 最多显示3个
                user_parts.append(f"[{i}] {candidate.description}")
                if candidate.simulation_result:
                    rho_max = candidate.simulation_result.get('rho_max', 'N/A')
                    user_parts.append(f"    预期效果: rho_max={rho_max}")
            user_parts.append("")
        
        # 历史参考
        if history_context:
            user_parts.append("=== 历史参考案例 ===")
            user_parts.append(history_context)
            user_parts.append("")
        
        user_parts.append("请分析上述方案，生成融合策略（可以组合多个动作）。")
        
        messages.append({
            "role": "user",
            "content": "\n".join(user_parts)
        })
        
        return messages

