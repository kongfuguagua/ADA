# -*- coding: utf-8 -*-
"""
动作解析模块
将 LLM 生成的文本指令解析为 Grid2Op Action 对象
"""

import re
from typing import Optional, Tuple
from grid2op.Action import BaseAction


class ActionParser:
    """
    文本指令解析器
    
    支持解析以下格式的文本指令：
    - redispatch(gen_id, amount_mw)
    - set_line_status(line_id, status)
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
        
        # 如果没有匹配到任何动作，检查是否是格式错误
        if not redispatch_matches and not line_status_matches and not self.do_nothing_pattern.search(action_text):
            # 尝试提取可能的动作关键词
            if 'redispatch' in action_text.lower() or 'set_line_status' in action_text.lower():
                raise ValueError(f"无法解析动作格式: {action_text[:100]}...")
            # 如果完全没有动作关键词，返回 do_nothing
            return action_space({})
        
        return action
    
    def extract_action_from_response(self, llm_response: str) -> str:
        """
        从 LLM 的完整响应中提取动作指令
        
        LLM 可能生成包含 Thought 和 Action 的完整响应，需要提取 Action 部分
        
        Args:
            llm_response: LLM 的完整响应
            
        Returns:
            提取出的动作指令文本（可能包含多个动作，用换行分隔）
        """
        # 查找 Action: 标记后的内容（直到下一个 Thought 或 Observation）
        action_pattern = re.compile(
            r'Action\s*:\s*(.+?)(?=\n\s*(?:Thought|Observation|$))',
            re.IGNORECASE | re.DOTALL
        )
        
        match = action_pattern.search(llm_response)
        if match:
            action_text = match.group(1).strip()
            # 清理可能的额外空白和换行
            action_text = re.sub(r'\s+', ' ', action_text)
            return action_text
        
        # 如果没有找到 Action: 标记，尝试查找函数调用模式
        # 查找所有可能的动作（redispatch, set_line_status, do_nothing）
        actions = []
        
        # 查找所有 redispatch
        redispatch_matches = self.redispatch_pattern.findall(llm_response)
        if redispatch_matches:
            for gen_id_str, amount_str in redispatch_matches:
                actions.append(f"redispatch({gen_id_str}, {amount_str})")
        
        # 查找所有 set_line_status
        line_status_matches = self.set_line_status_pattern.findall(llm_response)
        if line_status_matches:
            for line_id_str, status_str in line_status_matches:
                actions.append(f"set_line_status({line_id_str}, {status_str})")
        
        # 查找 do_nothing
        if self.do_nothing_pattern.search(llm_response):
            actions.append("do_nothing()")
        
        if actions:
            # 返回所有动作（用换行分隔，parse 方法会处理）
            return "\n".join(actions)
        
        # 如果都没找到，返回空字符串（将由 parse 处理）
        return ""
    
    def extract_action_from_text(self, text: str) -> Optional[str]:
        """
        从文本中提取动作指令（用于 RAG 上下文解析）
        
        与 extract_action_from_response 类似，但专门用于从 RAG 检索到的历史经验中提取动作
        
        Args:
            text: 包含动作指令的文本
            
        Returns:
            提取出的动作指令文本，如果未找到则返回 None
        """
        # 查找 Action: 标记后的内容
        action_pattern = re.compile(
            r'Action\s*:\s*(.+?)(?=\n\s*(?:Thought|Observation|$))',
            re.IGNORECASE | re.DOTALL
        )
        
        match = action_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        # 如果没有找到 Action: 标记，尝试查找函数调用模式
        actions = []
        
        redispatch_matches = self.redispatch_pattern.findall(text)
        if redispatch_matches:
            for gen_id_str, amount_str in redispatch_matches:
                actions.append(f"redispatch({gen_id_str}, {amount_str})")
        
        line_status_matches = self.set_line_status_pattern.findall(text)
        if line_status_matches:
            for line_id_str, status_str in line_status_matches:
                actions.append(f"set_line_status({line_id_str}, {status_str})")
        
        if self.do_nothing_pattern.search(text):
            actions.append("do_nothing()")
        
        if actions:
            return "\n".join(actions)
        
        return None

