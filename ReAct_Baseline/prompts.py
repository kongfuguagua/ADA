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
    
    def _build_system_prompt(self) -> str:
        """
        构建 System Prompt
        
        Returns:
            System Prompt 字符串
        """
        # 基础 system prompt
        base_prompt = """你是一个专业的电网调度员。你的目标是确保电网安全运行，避免线路过载。

## 你的任务
根据当前电网状态，分析问题并采取合适的控制动作，确保电网安全（无过载）。

## 可用工具（动作指令格式）
1. **redispatch(gen_id, amount_mw)**: 调整发电机出力
   - gen_id: 发电机编号（整数，必须是可调度发电机）
   - amount_mw: 调整量（MW，正数表示增加，负数表示减少）
   - **重要约束**:
     * 只能对可调度发电机（在观测中标记为"可调度"）进行操作
     * 调整后的出力必须在发电机的最大/最小出力范围内
     * 观测中会显示每个发电机的当前出力和可调整范围
   - 示例: redispatch(2, 10.5) 表示将发电机 2 的出力增加 10.5 MW

2. **set_line_status(line_id, status)**: 改变线路状态
   - line_id: 线路编号（整数）
   - status: 状态值（+1 表示开启/连接，-1 表示关闭/断开）
   - **重要约束**:
     * 只能对冷却时间已过的线路进行操作（观测中会显示可用线路）
     * 不能对已连接的线路再次连接，也不能对已断开的线路再次断开
     * 如果线路正在冷却中，无法改变其状态
   - 示例: set_line_status(3, -1) 表示关闭线路 3

3. **do_nothing()**: 保持现状，不执行任何动作
   - 示例: do_nothing()

## 工作流程（ReAct 格式）
请严格遵循以下格式：

**Thought**: 分析当前电网状态，识别问题（如过载线路），通过物理知识判断过载原因，制定解决方案。

**Action**: 选择一个具体的动作指令（必须是上述三种格式之一）。

**Observation**: （由环境反馈结果，你不需要填写）

## 重要原则
1. **安全第一**: 优先消除过载，确保所有线路负载率 < 100%
2. **最小干预**: 在保证安全的前提下，尽量少改变系统状态
3. **物理理解**: 理解电力系统的基本原理：
   - 过载通常由功率流不平衡引起
   - 再调度可以改变功率流分布
   - 断开线路可以改变网络拓扑，影响功率流
4. **拓扑感知**: 利用拓扑信息进行智能调度：
   - 观测中会显示发电机和线路连接的变电站信息
   - **策略提示**：
     * 如果某条线路过载，尝试调整**与该线路连接的变电站附近**的发电机出力
     * 增加位于**负荷中心附近**的发电机出力，或减少位于**拥堵线路上游**的发电机出力，通常可以缓解过载
     * 利用变电站信息判断发电机和线路的物理位置关系，做出更合理的调度决策
5. **缓解策略**: 允许渐进式改善（重要！）
   - 如果当前已经过载，**不要求一步到位完全消除过载**
   - 只要动作能**改善情况**（减少过载线路数，或降低最大负载率），就是有效的
   - 可以分多步逐步缓解，最终达到安全状态
6. **验证动作**: 如果动作导致危险（过载加剧或游戏结束），环境会反馈错误，你需要重新思考并尝试其他策略

## 注意事项
- **动作指令必须严格按照格式**，参数必须是数字
- **检查动作合法性**：
  * 确保发电机 ID 在有效范围内且是可调度的
  * 确保再调度量不会超出发电机的最大/最小出力范围
  * 确保线路 ID 在有效范围内且冷却时间已过
  * 不要对已连接线路再次连接，不要对已断开线路再次断开
- **如果动作非法或导致危险**，环境会返回详细的错误信息，你需要：
  * 仔细阅读错误信息，理解问题所在
  * 根据错误信息调整参数（例如，减少再调度量，或选择其他发电机/线路）
  * 如果错误提示"冷却中"，选择其他线路
  * 如果错误提示"超出范围"，减小调整量
- **如果多次尝试后仍无法找到安全动作**，可以返回 do_nothing()

## 缓解策略说明（重要）
当电网已经过载时，系统会接受**缓解性动作**（Mitigation Actions）：
- ✅ **接受**：动作后过载线路数量减少
- ✅ **接受**：动作后最大负载率明显下降（即使仍有过载）
- ❌ **拒绝**：动作后情况恶化或没有改善
- ❌ **拒绝**：动作导致极度过载（>150%）或电网崩溃

**关键理解**：在过载情况下，不要求一步到位完全消除过载。只要动作能让情况变好，就是有效的。可以分多步逐步缓解。"""
        
        # 如果有环境信息，添加环境概况
        if self.env_info:
            env_overview = self._build_env_overview()
            return env_overview + "\n\n" + base_prompt
        else:
            return base_prompt
    
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
        
        lines.append(f"**电网名称**: {grid_name}")
        lines.append(f"**系统规模**: {n_sub} 个变电站, {n_gen} 台发电机, {n_load} 个负荷, {n_line} 条线路")
        lines.append("")
        
        # 发电机详细信息
        generators = self.env_info.get("generators", [])
        if generators:
            lines.append("### 发电机信息（物理硬约束）")
            lines.append("")
            lines.append("每台发电机的关键约束：")
            lines.append("- **Pmax/Pmin**: 最大/最小出力限制（MW）")
            lines.append("- **Max Ramp Up/Down**: 最大爬坡速率限制（MW/步）- **这是最重要的约束之一**")
            lines.append("  - 每次再调度时，调整量不能超过该发电机的爬坡速率限制")
            lines.append("  - 例如：如果最大爬坡速率为 5 MW，则单步调整量不能超过 5 MW")
            lines.append("- **Redispatchable**: 是否可调度（只有可调度的发电机才能进行再调度）")
            lines.append("")
            
            # 显示所有发电机的信息
            for gen in generators:
                gen_id = gen.get("gen_id", -1)
                pmax = gen.get("pmax", 0.0)
                pmin = gen.get("pmin", 0.0)
                max_ramp_up = gen.get("max_ramp_up", None)
                max_ramp_down = gen.get("max_ramp_down", None)
                redispatchable = gen.get("redispatchable", False)
                
                lines.append(f"**发电机 {gen_id}**:")
                lines.append(f"  - 出力范围: {pmin:.2f} ~ {pmax:.2f} MW")
                if max_ramp_up is not None and max_ramp_down is not None:
                    lines.append(f"  - 最大爬坡速率: +{max_ramp_up:.2f} MW/步 (向上), -{max_ramp_down:.2f} MW/步 (向下)")
                else:
                    lines.append(f"  - 最大爬坡速率: 未知（请谨慎操作）")
                lines.append(f"  - 可调度: {'是' if redispatchable else '否'}")
                lines.append("")
        
        # 线路信息
        lines_info = self.env_info.get("lines", [])
        if lines_info:
            lines.append("### 线路信息")
            lines.append("")
            lines.append("每条线路的热限（Thermal Limit）决定了线路的最大传输容量。")
            lines.append("当线路负载率（rho）超过 100% 时，线路过载，需要立即处理。")
            lines.append("")
            
            # 显示前10条线路的信息（避免太长）
            display_lines = lines_info[:10]
            for line in display_lines:
                line_id = line.get("line_id", -1)
                thermal_limit = line.get("thermal_limit", 0.0)
                lines.append(f"  - 线路 {line_id}: 热限 {thermal_limit:.2f} MW")
            
            if len(lines_info) > 10:
                lines.append(f"  ... 还有 {len(lines_info) - 10} 条线路")
            lines.append("")
        
        # 重要提醒
        lines.append("### ⚠️ 重要物理约束提醒")
        lines.append("")
        lines.append("1. **爬坡速率限制（Ramp Rate）是最严格的约束**：")
        lines.append("   - 每次再调度时，单步调整量绝对不能超过该发电机的最大爬坡速率")
        lines.append("   - 如果当前发电机出力为 P，最大爬坡速率为 R，则：")
        lines.append("     * 向上调整：调整量 ≤ R")
        lines.append("     * 向下调整：调整量 ≤ R（绝对值）")
        lines.append("   - 违反此约束会导致动作被拒绝")
        lines.append("")
        lines.append("2. **出力范围限制**：")
        lines.append("   - 再调度后的出力必须在 Pmin 和 Pmax 之间")
        lines.append("   - 即：Pmin ≤ 当前出力 + 调整量 ≤ Pmax")
        lines.append("")
        lines.append("3. **线路冷却时间**：")
        lines.append("   - 断开或连接的线路需要等待冷却时间后才能再次操作")
        lines.append("   - 观测中会显示每条线路的冷却时间")
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

