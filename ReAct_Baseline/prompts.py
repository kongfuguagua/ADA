# -*- coding: utf-8 -*-
"""
提示词管理模块
定义 ReAct 的 System Prompt 和 Few-shot Examples
"""

from typing import List, Dict, Any


class PromptManager:
    """
    提示词管理器
    
    负责：
    - 构建 System Prompt
    - 维护 ReAct 历史上下文
    - 提供 Few-shot Examples
    - 管理环境概况信息
    """
    
    def __init__(self):
        """初始化提示词管理器"""
        self.env_info = None  # 环境静态信息
        self.rag_context = ""  # RAG 检索到的上下文
        self.system_prompt = self._build_system_prompt()
        self.few_shot_examples = self._build_few_shot_examples()
    
    def set_env_info(self, env_info: Dict[str, Any]):
        """
        设置环境静态信息
        
        Args:
            env_info: 环境信息字典（从 ReActAgent._extract_env_info 获取）
        """
        self.env_info = env_info
        # 重新构建 system prompt（包含环境概况）
        self.system_prompt = self._build_system_prompt()
    
    def set_rag_context(self, context: str):
        """
        设置 RAG 上下文
        
        Args:
            context: RAG 检索到的历史经验文本
        """
        self.rag_context = context
        # 上下文变更后，重新构建 system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        构建 System Prompt
        
        Returns:
            System Prompt 字符串
        """
        # 基础 system prompt（精简版）
        base_prompt = """你是电网调度员，目标是确保电网安全运行，避免线路过载。

## 任务
根据当前电网状态，分析问题并采取控制动作，确保电网安全（无过载）。

## 可用动作
1. **redispatch(gen_id, amount_mw)**: 调整发电机出力
   - 只能对可调度发电机操作，调整量不能超过爬坡速率限制
   - 示例: redispatch(2, 10.5)

2. **set_line_status(line_id, status)**: 改变线路状态（+1 开启，-1 关闭）
   - 只能对冷却时间已过的线路操作
   - 示例: set_line_status(3, -1)

3. **do_nothing()**: 保持现状

## 工作流程（ReAct 格式）
**Thought**: 分析问题，制定方案
**Action**: 选择动作指令
**Observation**: （环境反馈，无需填写）

**提示**：如果很自信，可直接输出 Action，无需冗长 Thought。

## 核心原则
1. **安全第一**: 优先消除过载（负载率 < 100%）
2. **最小干预**: 优先选择调整量最小、操作步数最少的方案（提高奖励）
3. **缓解策略**: 允许渐进式改善，不要求一步到位消除过载
4. **物理理解**: 过载由功率流不平衡引起，通过再调度或拓扑调整改变功率流分布
5. **拓扑感知**: 调整与过载线路连接的变电站附近的发电机，通常更有效

## 关键约束
- **爬坡速率**：单步调整量 ≤ 发电机最大爬坡速率（最重要！）
- **出力范围**：调整后出力必须在 Pmin 和 Pmax 之间
- **冷却时间**：线路操作后需等待冷却时间
- **动作格式**：必须严格按照格式，参数为数字

## 缓解策略
当电网过载时，系统接受缓解性动作：
- ✅ 过载线路数量减少
- ✅ 最大负载率明显下降（即使仍过载）
- ❌ 情况恶化或导致极度过载（>150%）

**关键**：只要动作能让情况变好，就是有效的。可分多步逐步缓解。"""
        
        # 动态插入 RAG 信息
        rag_section = ""
        if self.rag_context and self.rag_context != "暂无相关知识":
            rag_section = f"""
## 🧠 知识库回忆 (Historical Insights)

以下是根据当前电网状态检索到的相似历史场景及处理方案。

请**批判性地参考**这些信息：

1. **匹配**: 观察历史场景中的过载线路是否与当前一致。
2. **迁移**: 如果历史方案有效，尝试将其应用到当前发电机/线路，但要注意 ID 是否对应。
3. **验证**: 生成指令前，请思考该历史动作在当前拓扑下是否仍然安全。

[检索内容开始]

{self.rag_context}

[检索内容结束]

"""

        # 组合 Prompt
        # 建议放在环境概况之后，"## 你的任务" 之前
        if self.env_info:
            env_overview = self._build_env_overview()
            return env_overview + "\n\n" + rag_section + "\n" + base_prompt
        else:
            return rag_section + "\n" + base_prompt
    
    def _build_env_overview(self) -> str:
        """
        构建环境概况文本（基于环境静态信息）
        
        Returns:
            环境概况文本字符串
        """
        if not self.env_info:
            return ""
        
        lines = []
        lines.append("## 电网环境概况")
        lines.append("")
        
        # 基本信息
        grid_name = self.env_info.get("grid_name", "Unknown")
        n_gen = self.env_info.get("n_gen", 0)
        n_load = self.env_info.get("n_load", 0)
        n_line = self.env_info.get("n_line", 0)
        n_sub = self.env_info.get("n_sub", 0)
        
        lines.append(f"**电网**: {grid_name} | {n_sub} 变电站, {n_gen} 发电机, {n_load} 负荷, {n_line} 条线路")
        lines.append("")
        
        # 只显示关键约束摘要，不显示所有发电机详细信息（观测中已有）
        generators = self.env_info.get("generators", [])
        if generators:
            # 统计可调度发电机数量
            redispatchable_count = sum(1 for gen in generators if gen.get("redispatchable", False))
            lines.append(f"**可调度发电机**: {redispatchable_count}/{len(generators)}")
            lines.append("**关键约束**: 爬坡速率限制（最重要）、出力范围限制")
            lines.append("")
        
        # 线路信息简化
        lines_info = self.env_info.get("lines", [])
        if lines_info:
            lines.append(f"**线路总数**: {len(lines_info)} 条（负载率 > 100% 时过载）")
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        构建 Few-shot Examples
        
        Returns:
            Few-shot Examples 列表
        """
        return [
            {
                "user": """=== 当前电网状态 ===
最大负载率: 82.50%
状态: ✅ 安全
总负荷: 3200.00 MW
总发电: 3200.00 MW

可调度发电机数: 3
  - 发电机 0: 150.00 MW (范围: 0.00 ~ 200.00 MW)
  - 发电机 1: 200.00 MW (范围: 50.00 ~ 250.00 MW)
  - 发电机 2: 180.00 MW (范围: 100.00 ~ 200.00 MW)""",
                "assistant": """Thought: 当前电网状态安全，最大负载率为 82.50%，没有过载。所有线路都在安全范围内。此时不需要采取任何动作，保持现状即可。

Action: do_nothing()"""
            },
            {
                "user": """=== 当前电网状态 ===
最大负载率: 105.20%
状态: ⚠️ 危险 (存在过载)
总负荷: 3500.00 MW
总发电: 3500.00 MW

过载线路数: 2
  - 线路 3: 负载率 105.20% (过载)
  - 线路 7: 负载率 98.50% (过载)

可调度发电机数: 3
  - 发电机 0: 150.00 MW (范围: 0.00 ~ 200.00 MW)
  - 发电机 1: 200.00 MW (范围: 50.00 ~ 250.00 MW)
  - 发电机 2: 180.00 MW (范围: 100.00 ~ 200.00 MW)""",
                "assistant": """Thought: 检测到线路 3 和线路 7 过载。线路 3 的负载率为 105.20%，线路 7 为 98.50%。过载通常是由于功率流分布不均导致的。我需要通过再调度来改变功率流分布，减轻过载线路的负担。

策略：减少向过载线路输送功率的发电机出力，或增加其他发电机的出力来分流。由于线路 3 过载更严重，我应该优先处理。

尝试：减少发电机 0 的出力 20 MW，同时增加发电机 1 的出力 20 MW，这样可以改变功率流分布。

Action: redispatch(0, -20)"""
            }
        ]
    
    def build(
        self,
        observation_text: str,
        history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        构建完整的对话历史（用于 LLM 调用）
        
        Args:
            observation_text: 当前观测的文本描述
            history: 之前的对话历史（用于 ReAct 循环）
            
        Returns:
            完整的消息列表（包含 system, few-shot examples, history, current observation）
        """
        messages = []
        
        # 1. System Prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 2. Few-shot Examples（只在第一次调用时添加）
        if history is None or len(history) == 0:
            for example in self.few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": example["user"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["assistant"]
                })
        
        # 3. 历史对话（ReAct 循环中的 Thought-Action-Observation）
        if history:
            messages.extend(history)
        
        # 4. 当前观测
        messages.append({
            "role": "user",
            "content": observation_text
        })
        
        return messages
    
    def add_observation_feedback(
        self,
        history: List[Dict[str, str]],
        feedback: str
    ) -> List[Dict[str, str]]:
        """
        添加环境反馈到历史（用于 ReAct 循环）
        
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
    
    def compress_history(
        self,
        history: List[Dict[str, str]],
        max_preserved: int = 2
    ) -> List[Dict[str, str]]:
        """
        压缩历史对话，保留最近的 N 条，将之前的摘要化
        
        用于减少 Token 消耗，当 ReAct 步数过多时调用
        
        Args:
            history: 当前对话历史
            max_preserved: 保留的最近对话条数（默认 2，即保留最近一轮 Thought-Action-Observation）
            
        Returns:
            压缩后的对话历史
        """
        if len(history) <= max_preserved * 3:  # 如果历史不长，不需要压缩
            return history
        
        # 保留最近的对话
        preserved = history[-max_preserved * 3:]
        
        # 将之前的对话摘要化
        compressed_count = len(history) - len(preserved)
        if compressed_count > 0:
            summary = {
                "role": "user",
                "content": f"[历史摘要]: 之前进行了 {compressed_count // 3} 轮尝试，均未成功。请参考最新的反馈信息重新规划。"
            }
            return [summary] + preserved
        
        return preserved

