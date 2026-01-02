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
    
    def build_tuning_prompt(
        self,
        state_summary: str
    ) -> List[Dict[str, str]]:
        """
        构建参数调优 Prompt (Workflow 2)
        
        Args:
            state_summary: 状态摘要文本
            
        Returns:
            消息列表
        """
        system_prompt = """你是电网优化器参数调优专家。

## 你的任务
根据当前电网状态，动态调整优化器的超参数，以提高求解成功率和效果。

## 可调参数

1. **margin_th_limit** (热极限安全裕度)
   - 范围: 0.5 ~ 1.0
   - 默认: 0.9
   - **调优策略**:
     - 如果优化器无解 (INFEASIBLE)，**提高**此值（如 0.95 或 0.98）以放宽约束
     - 如果当前过载严重 (Max Rho > 110%)，可以提高到 1.0 以允许满载运行
     - 如果 AC/DC 误差导致模拟失败，**降低**此值（如 0.85）以留出更多缓冲

2. **penalty_curtailment** (切负荷惩罚)
   - 范围: 0.001 ~ 10.0
   - 默认: 0.1
   - **调优策略**:
     - 如果过载严重且优化器无解，**降低**此值（如 0.01 或 0.001）以允许更多切负荷
     - 如果系统安全，**提高**此值（如 1.0 或 10.0）以禁止切负荷

3. **penalty_redispatch** (再调度惩罚)
   - 范围: 0.01 ~ 1.0
   - 默认: 0.03
   - 通常保持默认值，除非需要优先使用再调度

4. **penalty_storage** (储能惩罚)
   - 范围: 0.1 ~ 1.0
   - 默认: 0.3
   - 通常保持默认值

## 决策逻辑

- **紧急情况** (Max Rho > 110%): 
  - margin_th_limit → 0.95~1.0
  - penalty_curtailment → 0.001~0.01
  
- **优化器无解**:
  - margin_th_limit → 提高 0.05~0.1
  - penalty_curtailment → 降低 10 倍

- **模拟失败** (AC/DC 误差):
  - margin_th_limit → 降低 0.05~0.1

## 输出格式

请直接输出 JSON，不要包含 Markdown 代码块：

{
    "margin_th_limit": 0.95,
    "penalty_curtailment": 0.01,
    "penalty_redispatch": 0.03,
    "penalty_storage": 0.3,
    "reasoning": "简要说明调优理由"
}

## 示例

### 场景 A：严重过载，优化器无解
{
    "margin_th_limit": 0.98,
    "penalty_curtailment": 0.001,
    "penalty_redispatch": 0.03,
    "penalty_storage": 0.3,
    "reasoning": "严重过载，放宽约束并允许大量切负荷"
}

### 场景 B：轻微过载，但优化器无解
{
    "margin_th_limit": 0.95,
    "penalty_curtailment": 0.01,
    "penalty_redispatch": 0.03,
    "penalty_storage": 0.3,
    "reasoning": "优化器无解，适度放宽约束"
}"""
        
        user_content = f"""当前电网状态：

{state_summary}

请根据上述状态，输出优化器参数配置（JSON 格式）："""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def build_enhancement_prompt(
        self,
        state_summary: str,
        base_action_desc: str
    ) -> List[Dict[str, str]]:
        """
        构建拓扑增强 Prompt (Workflow 1)
        
        Args:
            state_summary: 状态摘要文本
            base_action_desc: 优化器建议的动作描述
            
        Returns:
            消息列表
        """
        system_prompt = """你是电网拓扑调整专家。

## 你的任务
优化器只能处理连续变量（发电机调整、削减、储能），无法处理离散拓扑动作。
你的任务是观察网络拓扑，判断是否可以通过**切换母线 (Bus Splitting)** 或 **开关线路** 来进一步降低负载率。

## 可用拓扑动作

1. **set_line_status(line_id, status)**
   - 开启/关闭线路
   - line_id: 线路编号（从 0 开始）
   - status: +1 (开启) 或 -1 (关闭)

2. **set_bus(substation_id, bus_id)**
   - 设置变电站的母线连接
   - 需要了解变电站拓扑结构

3. **change_bus(substation_id)**
   - 切换变电站的母线（如果变电站有两条母线）

## 决策逻辑

- **优先考虑**: 过载线路附近的拓扑调整
- **安全第一**: 不要断开关键线路，避免导致电网解列
- **组合策略**: 可以结合优化器的调度动作，进一步降低负载率

## 输出格式

如果不需要拓扑调整，输出：
```
do_nothing()
```

如果需要拓扑调整，输出具体的动作：
```
set_line_status(3, -1)
set_line_status(5, +1)
```

或者使用 JSON 格式：
{
    "actions": [
        {"type": "set_line_status", "line_id": 3, "status": -1},
        {"type": "set_line_status", "line_id": 5, "status": 1}
    ],
    "reasoning": "关闭过载线路 3，开启备用线路 5 以分流"
}

## 示例

### 场景 A：优化器已尽力，但线路 4 依然过载
```
set_line_status(4, -1)
set_line_status(6, +1)
```
说明：关闭过载线路 4，开启备用线路 6 以分流

### 场景 B：无需拓扑调整
```
do_nothing()
```
说明：优化器的调度动作已足够，无需拓扑调整"""
        
        user_content = f"""当前电网状态：

{state_summary}

优化器建议的动作：
{base_action_desc}

请观察网络拓扑，判断是否需要额外的拓扑调整来进一步降低负载率。
如果需要，请输出具体的拓扑动作；如果不需要，请输出 do_nothing()。"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]