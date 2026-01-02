# -*- coding: utf-8 -*-
"""
拓扑提示词生成器 (Topology Prompter)

为 LLM 生成关于母线分裂（Bus Splitting）的提示词。
采用 RAG 思路，只提供局部信息（过载线路连接的变电站）。
"""

from typing import Dict, Any, List, Optional
import numpy as np
from grid2op.Observation import BaseObservation

from .grid_utils import get_substation_elements


class TopologyPrompter:
    """
    拓扑提示词生成器
    
    功能：
    - 提取过载线路连接的变电站信息
    - 生成简洁的提示词，让 LLM 建议母线分裂配置
    """
    
    def __init__(self):
        """初始化提示词生成器"""
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是电网拓扑调度专家。

## 你的任务

当优化器（OptimCVXPY）无法通过数值优化（再调度、切负荷）解决线路过载时，你需要通过**母线分裂（Bus Splitting）**来改变电网拓扑，从而缓解过载。

## 母线分裂原理

每个变电站（Substation）内部有多个母线（Bus）。默认情况下，变电站内的所有设备（发电机、负荷、线路）都连接在同一母线上。

通过**母线分裂**，你可以将变电站内的设备分成两组，分别连接到两个不同的母线。这样可以改变潮流分布，从而缓解过载线路的压力。

## 决策策略

1. **识别过载源**：找出导致过载线路流量过大的设备（通常是发电机或流入线路）
2. **分离策略**：将过载线路与重载设备分离到不同母线
3. **平衡考虑**：确保分裂后两个母线都有足够的发电/负荷平衡

## 输出格式

请直接输出 JSON，不要包含 Markdown 代码块：

{
    "substation_id": <变电站ID>,
    "bus_1": ["element_type_id", ...],
    "bus_2": ["element_type_id", ...],
    "reasoning": "简短的决策理由"
}

### 元素命名规则

- 发电机: "gen_<id>" (例如: "gen_0", "gen_5")
- 负荷: "load_<id>" (例如: "load_2", "load_10")
- 线路: "line_<id>" (例如: "line_3", "line_15")

### 约束条件

1. **必须包含过载线路**：过载线路必须放在 bus_1 或 bus_2 中
2. **完整性**：变电站内的所有设备必须被分配到 bus_1 或 bus_2（不能遗漏）
3. **非空性**：bus_1 和 bus_2 都不能为空
4. **互斥性**：每个设备只能出现在 bus_1 或 bus_2 中（不能重复）

## 示例

### 场景：线路 3 过载，变电站 5 连接了发电机 1 和负荷 2

输入：
- 过载线路: Line 3 (连接变电站 5 <-> 变电站 7)
- 变电站 5 配置:
  - Line 3 (过载线路)
  - Line 5
  - Gen 1 (100 MW)
  - Load 2 (50 MW)

输出：
{
    "substation_id": 5,
    "bus_1": ["line_3", "load_2"],
    "bus_2": ["line_5", "gen_1"],
    "reasoning": "将过载线路与重载发电机分离，减少线路 3 的流量"
}"""
    
    def build_prompt(
        self,
        observation: BaseObservation,
        overloaded_line_id: int
    ) -> List[Dict[str, str]]:
        """
        构建拓扑调整提示词
        
        Args:
            observation: 当前电网观测
            overloaded_line_id: 过载线路ID
            
        Returns:
            消息列表（system + user prompt）
        """
        # 1. 提取过载线路信息
        line_info = self._extract_line_info(observation, overloaded_line_id)
        
        # 2. 提取连接的变电站信息
        substation_info = self._extract_substation_info(observation, overloaded_line_id)
        
        # 3. 构建用户提示词
        user_prompt = self._build_user_prompt(line_info, substation_info)
        
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        return messages
    
    def _extract_line_info(
        self,
        observation: BaseObservation,
        line_id: int
    ) -> Dict[str, Any]:
        """提取过载线路信息"""
        rho = float(observation.rho[line_id])
        
        # 获取线路连接的变电站
        if hasattr(observation, 'line_or_to_subid') and hasattr(observation, 'line_ex_to_subid'):
            or_sub_id = int(observation.line_or_to_subid[line_id])
            ex_sub_id = int(observation.line_ex_to_subid[line_id])
        else:
            or_sub_id = None
            ex_sub_id = None
        
        # 获取线路功率
        if hasattr(observation, 'p_or'):
            p_or = float(observation.p_or[line_id])
        else:
            p_or = 0.0
        
        # 获取热限
        if hasattr(observation, 'thermal_limit'):
            thermal_limit = float(observation.thermal_limit[line_id])
        else:
            thermal_limit = 0.0
        
        return {
            "line_id": line_id,
            "rho": rho,
            "rho_percent": f"{rho:.1%}",
            "or_sub_id": or_sub_id,
            "ex_sub_id": ex_sub_id,
            "power_mw": p_or,
            "thermal_limit_mw": thermal_limit
        }
    
    def _extract_substation_info(
        self,
        observation: BaseObservation,
        line_id: int
    ) -> List[Dict[str, Any]]:
        """
        提取过载线路连接的变电站信息
        
        返回两个变电站的信息（Origin 和 Extremity）
        """
        substations = []
        
        if not hasattr(observation, 'line_or_to_subid') or not hasattr(observation, 'line_ex_to_subid'):
            return substations
        
        or_sub_id = int(observation.line_or_to_subid[line_id])
        ex_sub_id = int(observation.line_ex_to_subid[line_id])
        
        # 分析 Origin 端变电站
        or_sub_info = self._analyze_substation(observation, or_sub_id, line_id, "Origin")
        if or_sub_info:
            substations.append(or_sub_info)
        
        # 分析 Extremity 端变电站（如果不同）
        if ex_sub_id != or_sub_id:
            ex_sub_info = self._analyze_substation(observation, ex_sub_id, line_id, "Extremity")
            if ex_sub_info:
                substations.append(ex_sub_info)
        
        return substations
    
    def _analyze_substation(
        self,
        observation: BaseObservation,
        sub_id: int,
        target_line_id: int,
        position: str
    ) -> Optional[Dict[str, Any]]:
        """
        分析单个变电站，提取所有连接的设备
        
        Args:
            observation: 电网观测
            sub_id: 变电站ID
            target_line_id: 目标过载线路ID（用于标记）
            position: 位置描述（"Origin" 或 "Extremity"）
            
        Returns:
            变电站信息字典，包含所有连接的设备
        """
        elements = []
        
        # 检查变电站冷却时间
        cooldown = 0
        if hasattr(observation, 'time_before_cooldown_sub'):
            cooldown = int(observation.time_before_cooldown_sub[sub_id])
        
        # 使用共享工具函数获取变电站元件（包含功率和负载率信息）
        sub_elements_dict, _ = get_substation_elements(
            observation, sub_id, include_power=True, include_rho=True
        )
        
        # 1. 处理发电机
        for gen_info in sub_elements_dict['generators']:
            elements.append({
                "type": "generator",
                "id": gen_info["id"],
                "name": gen_info["name"],
                "power_mw": gen_info.get("p", 0.0),
                "redispatchable": gen_info.get("redispatchable", False)
            })
        
        # 2. 处理负荷
        for load_info in sub_elements_dict['loads']:
            elements.append({
                "type": "load",
                "id": load_info["id"],
                "name": load_info["name"],
                "power_mw": load_info.get("p", 0.0)
            })
        
        # 3. 处理线路（包括过载标记）
        for line_info in sub_elements_dict['lines']:
            line_id_conn = line_info["id"]
            is_overloaded = (line_id_conn == target_line_id)
            elements.append({
                "type": "line",
                "id": line_id_conn,
                "name": line_info["name"],
                "is_overloaded": is_overloaded,
                "rho": line_info.get("rho", 0.0)
            })
        
        # 如果变电站没有设备，返回 None
        if len(elements) == 0:
            return None
        
        return {
            "sub_id": sub_id,
            "position": position,
            "elements": elements,
            "cooldown": cooldown,
            "can_operate": (cooldown == 0)
        }
    
    def _build_user_prompt(
        self,
        line_info: Dict[str, Any],
        substation_info: List[Dict[str, Any]]
    ) -> str:
        """构建用户提示词"""
        lines = []
        
        # 1. 过载线路信息
        lines.append("## 当前情况")
        lines.append(f"过载线路: Line {line_info['line_id']}")
        lines.append(f"负载率: {line_info['rho_percent']} (过载)")
        lines.append(f"功率: {line_info['power_mw']:.2f} MW / 热限: {line_info['thermal_limit_mw']:.2f} MW")
        lines.append("")
        
        # 2. 连接的变电站信息
        if len(substation_info) == 0:
            lines.append("⚠️ 无法获取变电站信息，请返回空动作。")
            return "\n".join(lines)
        
        lines.append("## 连接的变电站")
        for sub_info in substation_info:
            sub_id = sub_info['sub_id']
            position = sub_info['position']
            can_operate = sub_info['can_operate']
            elements = sub_info['elements']
            
            lines.append(f"### 变电站 {sub_id} ({position}端)")
            
            if not can_operate:
                lines.append(f"⚠️ 该变电站正在冷却中（冷却时间: {sub_info['cooldown']} 步），无法操作。")
                lines.append("")
                continue
            
            # 按类型分组显示
            gens = [e for e in elements if e['type'] == 'generator']
            loads = [e for e in elements if e['type'] == 'load']
            lines_conn = [e for e in elements if e['type'] == 'line']
            
            if gens:
                lines.append("发电机:")
                for gen in gens:
                    redisp_info = " (可调度)" if gen.get('redispatchable', False) else ""
                    lines.append(f"  - {gen['name']}: {gen['power_mw']:.2f} MW{redisp_info}")
            
            if loads:
                lines.append("负荷:")
                for load in loads:
                    lines.append(f"  - {load['name']}: {load['power_mw']:.2f} MW")
            
            if lines_conn:
                lines.append("线路:")
                for line in lines_conn:
                    overload_mark = " ⚠️ 过载" if line.get('is_overloaded', False) else ""
                    lines.append(f"  - {line['name']}: 负载率 {line.get('rho', 0):.2%}{overload_mark}")
            
            lines.append("")
        
        # 3. 任务说明
        lines.append("## 任务")
        lines.append("请为上述变电站（优先选择冷却时间已过的变电站）设计母线分裂方案，以缓解过载线路的压力。")
        lines.append("")
        lines.append("请输出 JSON 格式的配置（参考系统提示词中的格式）。")
        
        return "\n".join(lines)

