# -*- coding: utf-8 -*-
"""
提示词管理模块
定义 ADA_Planner 的 System Prompt 和 Few-shot Examples
"""

from typing import List, Dict, Any


class PromptManager:
    """
    提示词管理器
    
    负责：
    - 构建 System Prompt
    - 维护 ADA_Planner 历史上下文
    - 提供 Few-shot Examples
    - 管理环境概况信息
    """
    
    def __init__(self):
        """初始化提示词管理器"""
        self.env_info = None  # 环境静态信息
        self.system_prompt = self._build_system_prompt()
        self.few_shot_examples = self._build_few_shot_examples()
    
    def set_env_info(self, env_info: Dict[str, Any]):
        """
        设置环境静态信息
        
        Args:
            env_info: 环境信息字典（从 ADA_Planner._extract_env_info 获取）
        """
        self.env_info = env_info
        # 重新构建 system prompt（包含环境概况）
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        构建精简的 System Prompt（Less is More）
        
        Returns:
            System Prompt 字符串
        """
        # 核心 Prompt：只保留必要信息
        base_prompt = """你是电网调度 AI。任务：消除过载（优先级1），降低成本（优先级2）。

## 输出格式（必须严格遵守 JSON）

仅输出一个 JSON 对象，不要包含 Markdown 标记（如 ```json）：

{
  "thought": "分析过程...",
  "action_type": "redispatch|set_line_status|execute_expert_solution|do_nothing",
  "params": {
    "gen_id": 0,        // redispatch: 发电机ID
    "amount_mw": 10.5,   // redispatch: 调整量(MW, 正数增加, 负数减少)
    "line_id": 3,       // set_line_status: 线路ID
    "status": 1,        // set_line_status: +1开启, -1关闭
    "index": 0          // execute_expert_solution: 专家方案索引
  },
  "reasoning": "选择原因..."
}

## 核心原则
1. 安全：优先消除过载（rho < 100%）
2. 成本：在安全前提下选择成本最低方案
3. 防震荡：避免与上一步动作逆操作"""
        
        return base_prompt
    
    def _build_env_overview(self, observation_text: str = "") -> str:
        """
        构建精简的环境概况（表格化数据，移除废话）
        
        Args:
            observation_text: 当前观测文本（用于判断是否需要显示某些信息）
            
        Returns:
            精简的环境概况文本
        """
        if not self.env_info:
            return ""
        
        lines = []
        
        # 只显示可调度发电机的表格（精简）
        generators = self.env_info.get("generators", [])
        redispatchable_gens = [g for g in generators if g.get("redispatchable", False)]
        
        if redispatchable_gens:
            lines.append("## 可调度发电机（ID | P | Pmin~Pmax | Ramp | Sub）")
            for gen in redispatchable_gens[:10]:  # 最多显示10个
                gen_id = gen.get("gen_id", -1)
                p = gen.get("p", 0.0)
                pmin = gen.get("pmin", 0.0)
                pmax = gen.get("pmax", 0.0)
                ramp = gen.get("max_ramp_up", 0.0)
                sub_id = gen.get("sub_id", -1)
                lines.append(f"  {gen_id} | {p:.1f} | {pmin:.1f}~{pmax:.1f} | ±{ramp:.1f} | {sub_id}")
            if len(redispatchable_gens) > 10:
                lines.append(f"  ... 还有 {len(redispatchable_gens) - 10} 个")
        
        return "\n".join(lines) if lines else ""
    
    def _build_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        构建精简的 Few-shot Examples（移除，Zero-shot 已足够）
        
        Returns:
            Few-shot Examples 列表（空列表）
        """
        # 移除 Few-shot Examples，Zero-shot 已足够
        return []
    
    def build(
        self,
        observation_text: str,
        history: List[Dict[str, str]] = None,
        physics_hint: str = "",
        expert_insight: Dict[str, Any] = None,
        last_action: str = ""
    ) -> List[Dict[str, str]]:
        """
        构建完整的对话历史（用于 LLM 调用）
        
        Args:
            observation_text: 当前观测的文本描述
            history: 之前的对话历史（用于 ADA_Planner 循环）
            physics_hint: Physics Hint（物理分析建议），如果有过载则提供
            expert_insight: 专家洞察报告（原始字典，用于 ActionParser）
            
        Returns:
            完整的消息列表（包含 system, few-shot examples, history, current observation）
        """
        messages = []
        
        # 判断是否为第一次调用（history 为空或长度为 0）
        is_first_call = history is None or len(history) == 0
        
        # 1. System Prompt（动态裁剪，只给必要信息）
        if is_first_call:
            # 基础 System Prompt
            system_prompt = self.system_prompt
            
            # 动态添加环境概况（精简表格）
            if self.env_info:
                env_overview = self._build_env_overview(observation_text)
                if env_overview:
                    system_prompt = system_prompt + "\n\n" + env_overview
            
            # 动态添加工具说明（根据场景）
            tools_note = self._build_tools_note(expert_insight, observation_text)
            if tools_note:
                system_prompt = system_prompt + "\n\n" + tools_note
            
            # 动态添加场景说明（精简版）
            scenario_note = self._build_scenario_note(expert_insight, observation_text)
            if scenario_note:
                system_prompt = system_prompt + "\n\n" + scenario_note
            
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 2. 历史对话（ADA_Planner 循环中的 Thought-Action-Observation）
        if history:
            messages.extend(history)
        
        # 3. 当前观测 + Expert Insight（精简） + 上一步动作（防震荡）
        observation_parts = [observation_text]
        
        # 精简 Expert Insight 显示
        if expert_insight and expert_insight.get("status") == "DANGER":
            expert_summary = self._format_expert_insight_compact(expert_insight)
            if expert_summary:
                observation_parts.append("\n" + expert_summary)
        
        if last_action:
            observation_parts.append(f"\n【上一步】: {last_action} (避免逆操作)")
        
        full_observation = "\n".join(observation_parts)
        
        messages.append({
            "role": "user",
            "content": full_observation
        })
        
        return messages
    
    def add_observation_feedback(
        self,
        history: List[Dict[str, str]],
        feedback: str
    ) -> List[Dict[str, str]]:
        """
        添加环境反馈到历史（用于 ADA_Planner 循环）
        
        Args:
            history: 当前对话历史
            feedback: 环境反馈（如 "动作非法" 或 "模拟显示该动作会导致危险"）
            
        Returns:
            更新后的对话历史
        """
        # 添加 Observation 消息
        history.append({
            "role": "user",
            "content": f"Observation: {feedback}"
        })
        return history
    
    def _build_tools_note(self, expert_insight: Dict[str, Any] = None, observation_text: str = "") -> str:
        """
        动态构建工具说明（只给必要信息）
        
        Args:
            expert_insight: 专家洞察报告
            observation_text: 观测文本
            
        Returns:
            工具说明文本
        """
        lines = []
        
        # 如果有专家建议，添加 execute_expert_solution 说明
        if expert_insight and expert_insight.get("status") == "DANGER":
            lines.append("## 可用动作")
            lines.append("- execute_expert_solution(index): 执行专家方案（推荐，Score>=3时优先）")
            lines.append("- redispatch(gen_id, amount_mw): 调整发电机出力")
            lines.append("- set_line_status(line_id, status): 改变线路状态 (+1开启, -1关闭)")
            lines.append("- do_nothing(): 保持现状")
        else:
            # 没有专家建议时，只给基本工具
            lines.append("## 可用动作")
            lines.append("- redispatch(gen_id, amount_mw): 调整发电机出力")
            lines.append("- set_line_status(line_id, status): 改变线路状态 (+1开启, -1关闭)")
            lines.append("- do_nothing(): 保持现状")
        
        return "\n".join(lines)
    
    def _build_scenario_note(self, expert_insight: Dict[str, Any] = None, observation_text: str = "") -> str:
        """
        动态构建场景说明（精简版）
        
        Args:
            expert_insight: 专家洞察报告
            observation_text: 观测文本
            
        Returns:
            场景说明文本
        """
        if expert_insight and expert_insight.get("status") == "DANGER":
            solutions = expert_insight.get("solutions", [])
            has_topology_solution = any(
                sol.get("type") == "Topology Action" and sol.get("score", 0) >= 3
                for sol in solutions
            )
            
            if has_topology_solution:
                return "## 场景：危急（有专家方案）\n优先使用 execute_expert_solution，在安全前提下选择成本最低方案。"
            else:
                return "## 场景：复杂决策（无完美方案）\n考虑组合动作（拓扑+再调度）或优化专家部分方案。"
        
        # 预警场景
        if self._is_preventive_scenario(observation_text):
            return "## 场景：预警（接近过载）\n考虑预防性调度：温和 Redispatch 降低负载，避免未来过载。"
        
        return ""
    
    def _format_expert_insight_compact(self, insight_report: Dict[str, Any]) -> str:
        """
        精简版 Expert Insight 格式化（只显示关键信息）
        
        Args:
            insight_report: 专家洞察报告
            
        Returns:
            精简的文本字符串
        """
        if insight_report.get("status") != "DANGER":
            return ""
        
        lines = []
        lines.append("## 专家方案（按推荐度排序）")
        
        solutions = insight_report.get("solutions", [])
        if not solutions:
            return ""
        
        for i, solution in enumerate(solutions[:3]):  # 最多显示3个
            sol_type = solution.get("type", "Unknown")
            score = solution.get("score", 0)
            cost = solution.get("cost_estimation", {}).get("estimated_cost", 0.0)
            
            if sol_type == "Topology Action":
                sub_id = solution.get("substation_id", -1)
                lines.append(f"{i}. [推荐] 方案#{i} (Topo Sub{sub_id}, Score:{score}/4, Cost:{cost:.2f}) → execute_expert_solution({i})")
            elif sol_type == "Redispatch":
                gen_up = solution.get("gen_up", [])
                gen_down = solution.get("gen_down", [])
                gen_str = f"Up:{gen_up[:2]}" if gen_up else ""
                if gen_down:
                    gen_str += f" Down:{gen_down[:2]}" if gen_str else f"Down:{gen_down[:2]}"
                lines.append(f"{i}. 方案#{i} (Redispatch {gen_str}, Cost:{cost:.2f})")
        
        lines.append("指令：优先选择 Score=4 且 Cost 最低的方案。若无完美方案，结合 Redispatch。")
        
        return "\n".join(lines)
    
    def _is_preventive_scenario(self, observation_text: str) -> bool:
        """
        判断是否为预警场景（预防性调度）
        
        Args:
            observation_text: 观测文本描述
            
        Returns:
            如果是预警场景则返回 True
        """
        import re
        # 从文本中提取最大负载率
        match = re.search(r'最大负载率:\s*(\d+\.?\d*)%', observation_text)
        if match:
            max_rho = float(match.group(1)) / 100.0
            # 预警场景：负载率在 0.85-0.95 之间，且未过载
            if 0.85 <= max_rho < 1.0:
                if "危险" not in observation_text and "过载" not in observation_text:
                    return True
        return False

