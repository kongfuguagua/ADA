# -*- coding: utf-8 -*-
"""
简化的提示词管理模块
专注于优化器参数配置，移除冗余的静态信息
"""

from typing import List, Dict, Any


class PromptManager:
    """
    简化的提示词管理器
    
    专注于让 LLM 配置优化器参数，不包含详细的发电机列表等静态信息。
    """
    
    def __init__(self):
        """初始化提示词管理器"""
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        构建简化的 System Prompt
        
        采用"意图模式"抽象，只解释模式选择逻辑，不包含具体参数数值
        """
        return """你是电网调度决策专家 (OptAgent)。

## 你的任务
根据电网状态摘要，选择最合适的**运行模式 (Mode)**，让底层优化器执行具体计算。

## 可选模式

1. **ECONOMIC (经济模式)**
   - **适用场景**：电网安全 (Max Rho < 95%)，无异常。
   - **策略**：禁止切负荷，允许极短时轻微过载以降低成本。

2. **CAUTIOUS (谨慎模式)**
   - **适用场景**：轻微过载趋势 (95% < Max Rho < 105%) 或 预测有风险。
   - **策略**：收紧安全裕度，防止 AC/DC 误差导致的越限。

3. **EMERGENCY (紧急模式)**
   - **适用场景**：明显过载 (Max Rho > 105%) 或 收到"优化无解(INFEASIBLE)"反馈。
   - **策略**：放宽对切负荷的限制，优先消除线路过载。

4. **SURVIVAL (生存模式)** ⭐ 最高优先级
   - **适用场景**：严重过载 (Max Rho > 110%)、**潮流发散**、**模拟失败**、或**电网解列**。
   - **策略**：**不惜一切代价**（大量切负荷）确保系统存活。

## 决策逻辑

- **安全第一**：如果收到"模拟失败"或"发散"的反馈，**必须**立即选择 **SURVIVAL**。
- **逐级升级**：如果当前模式无法解决问题（如上一轮反馈 INFEASIBLE），请升级到更激进的模式（如 EMERGENCY -> SURVIVAL）。
- **反馈优先**：如果【上次反馈】包含"模拟失败"、"发散"、"电压崩溃"或"电网解列"，请强制忽略当前状态，直接输出 SURVIVAL 模式。

## 输出格式

请直接输出 JSON，不要包含 Markdown 代码块：

{
    "mode": "模式名称",
    "reasoning": "简短的决策理由"
}

模式名称必须是以下之一（不区分大小写）：
- ECONOMIC
- CAUTIOUS
- EMERGENCY
- SURVIVAL

## 示例输出

### 场景 A：电网安全
{
    "mode": "ECONOMIC",
    "reasoning": "系统安全，Max Rho < 95%，采用经济模式"
}

### 场景 B：严重过载
{
    "mode": "EMERGENCY",
    "reasoning": "Max Rho > 105%，需要紧急消除过载"
}

### 场景 C：收到"发散"反馈（系统危急）
{
    "mode": "SURVIVAL",
    "reasoning": "检测到潮流发散，系统危急，必须切换到生存模式"
}"""
    
    def build(
        self,
        state_summary: str,
        feedback: str = None
    ) -> List[Dict[str, str]]:
        """
        构建简化的对话历史
        
        Args:
            state_summary: 状态摘要文本（由 StateSummarizer 生成）
            feedback: 上一次操作的反馈（如果有）
            
        Returns:
            消息列表（只包含 system 和当前状态，不包含历史对话）
        """
        messages = []
        
        # System Prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 当前状态（极简格式）
        user_content = state_summary
        if feedback:
            user_content += f"\n\n【上次反馈】{feedback}"
        
        user_content += "\n\n请输出优化器参数配置（JSON 格式）："
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
