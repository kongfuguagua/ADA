# -*- coding: utf-8 -*-
"""
动作解析模块
将 LLM 生成的文本指令解析为 Grid2Op Action 对象
"""

import re
from typing import Optional, Tuple, Dict, Any
from grid2op.Action import BaseAction


class ActionParser:
    """
    文本指令解析器
    
    支持解析以下格式的文本指令：
    - redispatch(gen_id, amount_mw)
    - set_line_status(line_id, status)
    - do_nothing()
    - execute_expert_solution(index)  # 执行专家系统推荐的方案
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
        self.do_nothing_pattern = re.compile(
            r'do_nothing\s*\(\)',
            re.IGNORECASE
        )
        self.execute_expert_solution_pattern = re.compile(
            r'execute_expert_solution\s*\(\s*(\d+)\s*\)',
            re.IGNORECASE
        )
    
    def parse(
        self,
        action_text: str,
        action_space,
        expert_insight_report: Dict[str, Any] = None
    ) -> BaseAction:
        """
        解析文本指令为 Grid2Op Action
        
        Args:
            action_text: LLM 生成的文本指令
            action_space: Grid2Op 动作空间
            expert_insight_report: 专家洞察报告（用于 execute_expert_solution）
            
        Returns:
            Grid2Op Action 对象
            
        Raises:
            ValueError: 如果解析失败或动作非法
        """
        # 清理文本（移除多余空白）
        action_text = action_text.strip()
        
        # 检查是否为 execute_expert_solution（优先处理）
        expert_match = self.execute_expert_solution_pattern.search(action_text)
        if expert_match:
            if expert_insight_report is None:
                raise ValueError("execute_expert_solution 需要专家洞察报告，但未提供")
            
            solution_idx = int(expert_match.group(1))
            solutions = expert_insight_report.get("solutions", [])
            
            if solution_idx < 0 or solution_idx >= len(solutions):
                raise ValueError(
                    f"专家方案索引 {solution_idx} 超出范围 (共有 {len(solutions)} 个方案)"
                )
            
            solution = solutions[solution_idx]
            action_object = solution.get("action_object")
            
            if action_object is None:
                raise ValueError(
                    f"专家方案 {solution_idx} 没有关联的动作对象（可能是再调度建议，需要手动生成）"
                )
            
            # 直接返回专家系统计算好的动作对象（零错误率）
            return action_object
        
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
        
        # 如果没有匹配到任何动作，检查是否是格式错误
        if (not redispatch_matches and not line_status_matches and 
            not self.do_nothing_pattern.search(action_text) and
            not expert_match):
            # 尝试提取可能的动作关键词
            if ('redispatch' in action_text.lower() or 
                'set_line_status' in action_text.lower() or
                'execute_expert_solution' in action_text.lower()):
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
        # ===== 策略0：JSON 结构化输出（优先） =====
        # 尝试从 JSON 格式中提取动作
        json_action = self._extract_from_json(llm_response)
        if json_action:
            return json_action
        # ===== 策略1：锚点匹配（优先） =====
        # 匹配 "Action:" 或 "**Action**:" 后面的内容
        # 支持多种格式：Action:、**Action**:、Action: 等
        strict_patterns = [
            r'(?:\*\*)?Action(?:\*\*)?\s*:\s*(.+?)(?=\n\s*(?:\*\*)?(?:Thought|Observation|$))',
            r'(?:\*\*)?Action(?:\*\*)?\s*:\s*(.+?)$',  # 如果 Action 在最后
        ]
        
        for pattern in strict_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                action_part = match.group(1).strip()
                # 移除可能的反引号包裹
                action_part = action_part.strip('`').strip()
                # 清理空白
                action_part = re.sub(r'\s+', ' ', action_part)
                
                # 验证提取的内容是否包含合法的函数调用
                if self._is_valid_action(action_part):
                    return action_part
        
        # ===== 策略2：Last Match 原则（保底） =====
        # 如果无法通过锚点提取，则提取整段文本中最后一次出现的合法动作指令
        # 这样可以避免匹配到 Thought 部分提到的历史动作
        
        all_matches = []  # 初始化匹配列表
        
        # 定义所有合法的动作模式（带格式化函数）
        valid_patterns = [
            (self.redispatch_pattern, lambda m: f"redispatch({m.group(1)}, {m.group(2)})"),
            (self.set_line_status_pattern, lambda m: f"set_line_status({m.group(1)}, {m.group(2)})"),
            (self.execute_expert_solution_pattern, lambda m: f"execute_expert_solution({m.group(1)})"),
        ]
        
        # do_nothing 需要特殊处理（因为它没有捕获组）
        do_nothing_matches = list(self.do_nothing_pattern.finditer(llm_response))
        for match in do_nothing_matches:
            start_pos = match.start()
            context_before = llm_response[max(0, start_pos-50):start_pos]
            
            # 检查是否在代码块中
            code_block_before = llm_response[:start_pos].count('```') % 2 == 1
            if code_block_before:
                continue
            
            # 检查是否在"上一步动作"等回顾性描述中
            if re.search(r'(?:上一步|previous|last\s+action|回顾|历史)', context_before, re.IGNORECASE):
                continue
            
            all_matches.append((start_pos, "do_nothing()"))
        
        # 处理其他动作模式
        for pattern, formatter in valid_patterns:
            for match in pattern.finditer(llm_response):
                # 过滤：如果匹配在代码块中（```...```）或前面有"上一步"等关键词，跳过
                start_pos = match.start()
                context_before = llm_response[max(0, start_pos-50):start_pos]
                
                # 检查是否在代码块中
                code_block_before = llm_response[:start_pos].count('```') % 2 == 1
                if code_block_before:
                    continue
                
                # 检查是否在"上一步动作"等回顾性描述中
                if re.search(r'(?:上一步|previous|last\s+action|回顾|历史|建议|建议列表)', context_before, re.IGNORECASE):
                    continue
                
                all_matches.append((match.start(), formatter(match)))
        
        if all_matches:
            # 按在文本中出现的位置排序，取最后一个（最可能是最终决策）
            last_match = sorted(all_matches, key=lambda x: x[0])[-1]
            return last_match[1]
        
        # 如果都没找到，返回空字符串
        return ""
    
    def _is_valid_action(self, action_text: str) -> bool:
        """
        验证提取的文本是否包含合法的动作指令
        
        Args:
            action_text: 待验证的文本
            
        Returns:
            如果包含合法动作则返回 True
        """
        # 检查是否包含任何合法的动作模式
        return (
            self.redispatch_pattern.search(action_text) is not None or
            self.set_line_status_pattern.search(action_text) is not None or
            self.execute_expert_solution_pattern.search(action_text) is not None or
            self.do_nothing_pattern.search(action_text) is not None
        )
    
    def _extract_from_json(self, llm_response: str) -> Optional[str]:
        """
        从 JSON 格式的响应中提取动作指令（策略三：结构化输出）
        
        Args:
            llm_response: LLM 的完整响应
            
        Returns:
            提取出的动作指令文本，如果未找到则返回 None
        """
        import json
        import re
        
        # 尝试提取 JSON 代码块
        json_block_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试提取纯 JSON 对象（不在代码块中）
            json_obj_pattern = r'\{[^{}]*"action_type"[^{}]*\}'
            json_match = re.search(json_obj_pattern, llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        try:
            data = json.loads(json_str)
            action_type = data.get("action_type", "").lower()
            params = data.get("params", {})
            
            # 根据 action_type 构造动作指令
            if action_type == "redispatch":
                gen_id = params.get("gen_id")
                amount = params.get("amount_mw")
                if gen_id is not None and amount is not None:
                    return f"redispatch({gen_id}, {amount})"
            
            elif action_type == "set_line_status":
                line_id = params.get("line_id")
                status = params.get("status")
                if line_id is not None and status is not None:
                    return f"set_line_status({line_id}, {status})"
            
            elif action_type == "execute_expert_solution":
                index = params.get("index")
                if index is not None:
                    return f"execute_expert_solution({index})"
            
            elif action_type == "do_nothing":
                return "do_nothing()"
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # JSON 解析失败，降级到其他策略
            return None
        
        return None

