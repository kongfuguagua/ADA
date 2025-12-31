# -*- coding: utf-8 -*-
"""
动作解析模块
将 LLM 生成的文本指令解析为 Grid2Op Action 对象
"""

import re
from typing import Optional, Dict, Any
from grid2op.Action import BaseAction


class ActionParser:
    """
    文本指令解析器
    
    支持解析以下格式的文本指令：
    - redispatch(gen_id, amount_mw)
    - set_line_status(line_id, status)
    - set_bus(substation_id, topology_vector)
    - do_nothing()
    """
    
    def __init__(self):
        """初始化解析器"""
        # 编译正则表达式以提高性能
        self.redispatch_pattern = re.compile(
            r'redispatch\s*\(\s*(\d+)\s*,\s*([+-]?\d+\.?\d*)\s*\)',
            re.IGNORECASE
        )
        self.set_line_status_pattern = re.compile(
            r'set_line_status\s*\(\s*(\d+)\s*,\s*([+-]?\d+)\s*\)',
            re.IGNORECASE
        )
        self.set_bus_pattern = re.compile(
            r'set_bus\s*\(\s*substation_id\s*=\s*(\d+)\s*,\s*topology_vector\s*=\s*\[(.*?)\]\s*\)',
            re.IGNORECASE | re.DOTALL
        )
        self.do_nothing_pattern = re.compile(
            r'do_nothing\s*\(\)',
            re.IGNORECASE
        )
    
    def parse(
        self,
        action_text: str,
        action_space
    ) -> BaseAction:
        """
        解析文本指令为 Grid2Op Action
        
        Args:
            action_text: LLM 生成的文本指令
            action_space: Grid2Op 动作空间
            
        Returns:
            Grid2Op Action 对象
            
        Raises:
            ValueError: 如果解析失败或动作非法
        """
        # 清理文本（移除多余空白）
        action_text = action_text.strip()
        
        # 检查是否为 do_nothing
        if self.do_nothing_pattern.search(action_text):
            return action_space({})
        
        # 创建空动作
        action = action_space({})
        
        # 解析 redispatch
        redispatch_matches = self.redispatch_pattern.findall(action_text)
        if redispatch_matches:
            redispatch_list = []
            for gen_id_str, amount_str in redispatch_matches:
                try:
                    gen_id = int(gen_id_str)
                    amount = float(amount_str)
                    redispatch_list.append((gen_id, amount))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"redispatch 参数解析失败: gen_id={gen_id_str}, amount={amount_str}, 错误={e}")
            
            if redispatch_list:
                action.redispatch = redispatch_list
        
        # 解析 set_line_status
        line_status_matches = self.set_line_status_pattern.findall(action_text)
        if line_status_matches:
            line_status_list = []
            for line_id_str, status_str in line_status_matches:
                try:
                    line_id = int(line_id_str)
                    status = int(status_str)
                    # 验证状态值（+1 开启，-1 关闭）
                    if status not in [-1, 1]:
                        raise ValueError(f"set_line_status 状态值无效: {status} (应为 +1 或 -1)")
                    line_status_list.append((line_id, status))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"set_line_status 参数解析失败: line_id={line_id_str}, status={status_str}, 错误={e}")
            
            if line_status_list:
                action.set_line_status = line_status_list
        
        # 解析 set_bus（拓扑调整）
        bus_matches = self.set_bus_pattern.findall(action_text)
        if bus_matches:
            for sub_id_str, topo_vec_str in bus_matches:
                try:
                    sub_id = int(sub_id_str)
                    # 解析拓扑向量
                    topo_vec = [int(x.strip()) for x in topo_vec_str.split(',') if x.strip()]
                    if topo_vec:
                        action.set_bus = {"substations_id": [(sub_id, topo_vec)]}
                except (ValueError, TypeError) as e:
                    raise ValueError(f"set_bus 参数解析失败: sub_id={sub_id_str}, topo_vec={topo_vec_str}, 错误={e}")
        
        # 如果没有匹配到任何动作，检查是否是格式错误
        if (not redispatch_matches and not line_status_matches and not bus_matches and
            not self.do_nothing_pattern.search(action_text)):
            # 尝试提取可能的动作关键词
            if ('redispatch' in action_text.lower() or 
                'set_line_status' in action_text.lower() or
                'set_bus' in action_text.lower()):
                raise ValueError(f"无法解析动作格式: {action_text[:100]}...")
            # 如果完全没有动作关键词，返回 do_nothing
            return action_space({})
        
        return action
    
    def extract_action_from_response(self, llm_response: str) -> str:
        """
        从 LLM 的完整响应中提取动作指令
        
        优先使用 JSON 格式提取，然后使用锚点匹配（Action: 标记），最后使用"Last Match"原则。
        
        Args:
            llm_response: LLM 的完整响应
            
        Returns:
            提取出的动作指令文本
        """
        # 策略1：JSON 格式提取
        json_action = self._extract_from_json(llm_response)
        if json_action:
            return json_action
        
        # 策略2：锚点匹配
        strict_patterns = [
            r'(?:\*\*)?Action(?:\*\*)?\s*:\s*(.+?)(?=\n\s*(?:\*\*)?(?:Thought|Observation|$))',
            r'(?:\*\*)?Action(?:\*\*)?\s*:\s*(.+?)$',
        ]
        
        for pattern in strict_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                action_part = match.group(1).strip()
                action_part = action_part.strip('`').strip()
                action_part = re.sub(r'\s+', ' ', action_part)
                
                if self._is_valid_action(action_part):
                    return action_part
        
        # 策略3：Last Match 原则
        all_matches = []
        
        valid_patterns = [
            (self.redispatch_pattern, lambda m: f"redispatch({m.group(1)}, {m.group(2)})"),
            (self.set_line_status_pattern, lambda m: f"set_line_status({m.group(1)}, {m.group(2)})"),
            (self.set_bus_pattern, lambda m: f"set_bus(substation_id={m.group(1)}, topology_vector=[{m.group(2)}])"),
        ]
        
        do_nothing_matches = list(self.do_nothing_pattern.finditer(llm_response))
        for match in do_nothing_matches:
            start_pos = match.start()
            context_before = llm_response[max(0, start_pos-50):start_pos]
            code_block_before = llm_response[:start_pos].count('```') % 2 == 1
            if code_block_before:
                continue
            if re.search(r'(?:上一步|previous|last\s+action|回顾|历史)', context_before, re.IGNORECASE):
                continue
            all_matches.append((start_pos, "do_nothing()"))
        
        for pattern, formatter in valid_patterns:
            for match in pattern.finditer(llm_response):
                start_pos = match.start()
                context_before = llm_response[max(0, start_pos-50):start_pos]
                code_block_before = llm_response[:start_pos].count('```') % 2 == 1
                if code_block_before:
                    continue
                if re.search(r'(?:上一步|previous|last\s+action|回顾|历史|建议|建议列表)', context_before, re.IGNORECASE):
                    continue
                all_matches.append((match.start(), formatter(match)))
        
        if all_matches:
            last_match = sorted(all_matches, key=lambda x: x[0])[-1]
            return last_match[1]
        
        return ""
    
    def _is_valid_action(self, action_text: str) -> bool:
        """验证提取的文本是否包含合法的动作指令"""
        return (
            self.redispatch_pattern.search(action_text) is not None or
            self.set_line_status_pattern.search(action_text) is not None or
            self.set_bus_pattern.search(action_text) is not None or
            self.do_nothing_pattern.search(action_text) is not None
        )
    
    def _extract_from_json(self, llm_response: str) -> Optional[str]:
        """从 JSON 格式的响应中提取动作指令"""
        import json
        
        # 策略1：尝试从代码块中提取 JSON
        json_block_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 策略2：寻找最外层的 JSON 对象（支持嵌套）
            # 使用堆栈方法找到第一个 { 和最后一个匹配的 }
            start_idx = llm_response.find('{')
            if start_idx == -1:
                return None
            
            # 从第一个 { 开始，使用堆栈找到匹配的 }
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(llm_response)):
                if llm_response[i] == '{':
                    brace_count += 1
                elif llm_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if end_idx == -1:
                return None
            
            json_str = llm_response[start_idx:end_idx + 1]
        
        try:
            data = json.loads(json_str)
            action_type = data.get("action_type", "").lower()
            params = data.get("params", {})
            
            if action_type == "redispatch":
                gen_id = params.get("gen_id")
                amount = params.get("amount_mw")
                if gen_id is not None and amount is not None:
                    # 注意：转化为 float 避免类型错误
                    return f"redispatch({gen_id}, {float(amount)})"
            
            elif action_type == "set_line_status":
                line_id = params.get("line_id")
                status = params.get("status")
                if line_id is not None and status is not None:
                    return f"set_line_status({line_id}, {status})"
            
            elif action_type == "set_bus":
                sub_id = params.get("substation_id")
                topo_vec = params.get("topology_vector")
                if sub_id is not None and topo_vec is not None:
                    topo_str = ','.join([str(x) for x in topo_vec])
                    return f"set_bus(substation_id={sub_id}, topology_vector=[{topo_str}])"
            
            elif action_type == "do_nothing":
                return "do_nothing()"
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # 可选：添加日志用于调试
            # logger.warning(f"JSON 解析失败: {e}")
            return None
        
        return None

