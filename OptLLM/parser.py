# -*- coding: utf-8 -*-
"""
动作解析模块
将 LLM 生成的文本指令解析为 Grid2Op Action 对象
"""

import re
import json
import logging
from typing import Optional, Tuple, Dict, Any
from grid2op.Action import BaseAction

logger = logging.getLogger(__name__)

# 导入模式配置
try:
    from .config_modes import get_mode_config, get_mode_description, DEFAULT_MODE
except ImportError:
    # 容错：如果导入失败，定义默认函数
    def get_mode_config(mode_name: str) -> Dict[str, Any]:
        return {"penalty_curtailment": 0.1, "penalty_overflow": 1.0, "margin_th_limit": 0.9}
    def get_mode_description(mode_name: str) -> str:
        return "默认模式"
    DEFAULT_MODE = "EMERGENCY"


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
        
        # 获取线路数量用于验证
        n_line = getattr(action_space, 'n_line', None)
        if n_line is None:
            # 尝试从 observation_space 获取（如果 action_space 没有）
            if hasattr(action_space, 'observation_space'):
                n_line = getattr(action_space.observation_space, 'n_line', None)
        
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
                    # 验证线路 ID 范围
                    if n_line is not None and (line_id < 0 or line_id >= n_line):
                        raise ValueError(f"set_line_status 线路 ID 无效: {line_id} (有效范围: 0-{n_line-1})")
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
    
    def extract_optimization_config_json(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应中提取优化器配置（JSON 模式）
        
        优先尝试解析模式（mode-based），如果失败则回退到直接参数格式（向后兼容）
        
        Args:
            llm_response: LLM 的完整响应（应该是 JSON 格式）
            
        Returns:
            配置字典，如果未找到则返回 None
        """
        # 方法1: 尝试解析模式格式（新模式）
        try:
            # 尝试提取 JSON 块
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(llm_response)
            for match in matches:
                try:
                    data = json.loads(match)
                    
                    # 检查是否为模式格式（包含 "mode" 字段）
                    if "mode" in data:
                        mode_name = data.get("mode", "").strip()
                        reasoning = data.get("reasoning", "")
                        
                        # 从模式配置表获取参数
                        config = get_mode_config(mode_name)
                        
                        # 添加策略描述（包含模式和推理）
                        config["strategy_description"] = f"[{mode_name.upper()}] {reasoning}"
                        
                        return self._normalize_config(config)
                    
                    # 检查是否为直接参数格式（向后兼容）
                    if "penalty_curtailment" in data or "margin_th_limit" in data or "penalty_overflow" in data:
                        return self._normalize_config(data)
                        
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # 方法2: 尝试从文本中提取模式名称（容错处理）
        mode_pattern = re.compile(r'"mode"\s*:\s*"([^"]+)"', re.IGNORECASE)
        match = mode_pattern.search(llm_response)
        if match:
            mode_name = match.group(1).strip()
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', llm_response, re.IGNORECASE)
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            
            config = get_mode_config(mode_name)
            config["strategy_description"] = f"[{mode_name.upper()}] {reasoning}"
            return self._normalize_config(config)
        
        # 方法3: 回退到文本解析（兼容旧格式）
        return self.extract_optimization_config(llm_response)
    
    def extract_optimization_config(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应中提取优化器配置
        
        支持以下格式：
        1. JSON 格式：{"penalty_curtailment": 0.1, "margin_th_limit": 0.9, ...}
        2. 函数调用格式：run_optimization_solver(penalty_curtailment=0.1, margin_th_limit=0.9, ...)
        3. 自然语言描述（尝试提取关键参数）
        
        Args:
            llm_response: LLM 的完整响应
            
        Returns:
            配置字典，如果未找到则返回 None
        """
        # 方法1: 查找 JSON 格式
        json_pattern = re.compile(
            r'\{[^{}]*"penalty_curtailment"[^{}]*\}',
            re.IGNORECASE | re.DOTALL
        )
        match = json_pattern.search(llm_response)
        if match:
            try:
                config = json.loads(match.group(0))
                # 验证必要字段
                if "penalty_curtailment" in config or "margin_th_limit" in config:
                    return self._normalize_config(config)
            except json.JSONDecodeError:
                pass
        
        # 方法2: 查找函数调用格式
        func_pattern = re.compile(
            r'run_optimization_solver\s*\([^)]*\)',
            re.IGNORECASE
        )
        match = func_pattern.search(llm_response)
        if match:
            func_call = match.group(0)
            # 提取参数
            params = {}
            # 提取 penalty_curtailment
            pc_match = re.search(r'penalty_curtailment\s*=\s*([0-9.]+)', func_call, re.IGNORECASE)
            if pc_match:
                params["penalty_curtailment"] = float(pc_match.group(1))
            # 提取 penalty_redispatch
            pr_match = re.search(r'penalty_redispatch\s*=\s*([0-9.]+)', func_call, re.IGNORECASE)
            if pr_match:
                params["penalty_redispatch"] = float(pr_match.group(1))
            # 提取 margin_th_limit
            mtl_match = re.search(r'margin_th_limit\s*=\s*([0-9.]+)', func_call, re.IGNORECASE)
            if mtl_match:
                params["margin_th_limit"] = float(mtl_match.group(1))
            # 提取 strategy_description
            sd_match = re.search(r'strategy_description\s*=\s*["\']([^"\']+)["\']', func_call, re.IGNORECASE)
            if sd_match:
                params["strategy_description"] = sd_match.group(1)
            
            if params:
                return self._normalize_config(params)
        
        # 方法3: 从自然语言中提取关键参数（更宽松的匹配）
        # 查找 "penalty_curtailment" 或 "切负荷惩罚" 等关键词
        config = {}
        
        # 提取 penalty_curtailment
        pc_patterns = [
            r'penalty_curtailment[:\s=]+([0-9.]+)',
            r'切负荷惩罚[:\s=]+([0-9.]+)',
            r'curtailment.*?([0-9.]+)',
        ]
        for pattern in pc_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["penalty_curtailment"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取 margin_th_limit
        mtl_patterns = [
            r'margin_th_limit[:\s=]+([0-9.]+)',
            r'安全裕度[:\s=]+([0-9.]+)',
            r'margin[:\s=]+([0-9.]+)',
        ]
        for pattern in mtl_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["margin_th_limit"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取 strategy_description
        sd_patterns = [
            r'策略[:\s]+([^\n]+)',
            r'strategy[:\s]+([^\n]+)',
        ]
        for pattern in sd_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                config["strategy_description"] = match.group(1).strip()
                break
        
        if config:
            return self._normalize_config(config)
        
        return None
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化配置字典，确保所有必要字段存在
        
        Args:
            config: 原始配置字典
            
        Returns:
            规范化后的配置字典
        """
        normalized = {}
        
        # 必需字段（如果没有则使用默认值）
        normalized["penalty_curtailment"] = config.get("penalty_curtailment", 0.1)
        normalized["penalty_overflow"] = config.get("penalty_overflow", 1.0)  # 新增字段
        normalized["margin_th_limit"] = config.get("margin_th_limit", 0.9)
        
        # 可选字段
        if "penalty_redispatch" in config:
            normalized["penalty_redispatch"] = config["penalty_redispatch"]
        else:
            normalized["penalty_redispatch"] = 0.03  # 默认值
        
        if "penalty_storage" in config:
            normalized["penalty_storage"] = config["penalty_storage"]
        else:
            normalized["penalty_storage"] = 0.3  # 默认值
        
        if "strategy_description" in config:
            normalized["strategy_description"] = config["strategy_description"]
        else:
            normalized["strategy_description"] = "未指定策略"
        
        return normalized
    
    def extract_tuning_config(self, llm_response: str) -> Optional[Dict[str, float]]:
        """
        从 LLM 响应中提取参数调优配置
        
        Args:
            llm_response: LLM 的完整响应（应该是 JSON 格式）
            
        Returns:
            配置字典，如果未找到则返回 None
        """
        # 方法1: 尝试解析 JSON 格式
        try:
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(llm_response)
            for match in matches:
                try:
                    data = json.loads(match)
                    
                    # 检查是否包含调优参数
                    if "margin_th_limit" in data or "penalty_curtailment" in data:
                        config = {}
                        if "margin_th_limit" in data:
                            config["margin_th_limit"] = float(data["margin_th_limit"])
                        if "penalty_curtailment" in data:
                            config["penalty_curtailment"] = float(data["penalty_curtailment"])
                        if "penalty_redispatch" in data:
                            config["penalty_redispatch"] = float(data["penalty_redispatch"])
                        if "penalty_storage" in data:
                            config["penalty_storage"] = float(data["penalty_storage"])
                        
                        if config:
                            return config
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # 方法2: 从文本中提取参数
        config = {}
        
        # 提取 margin_th_limit
        mtl_patterns = [
            r'"margin_th_limit"\s*:\s*([0-9.]+)',
            r'margin_th_limit[:\s=]+([0-9.]+)',
        ]
        for pattern in mtl_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["margin_th_limit"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取 penalty_curtailment
        pc_patterns = [
            r'"penalty_curtailment"\s*:\s*([0-9.]+)',
            r'penalty_curtailment[:\s=]+([0-9.]+)',
        ]
        for pattern in pc_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["penalty_curtailment"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取 penalty_redispatch
        pr_patterns = [
            r'"penalty_redispatch"\s*:\s*([0-9.]+)',
            r'penalty_redispatch[:\s=]+([0-9.]+)',
        ]
        for pattern in pr_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["penalty_redispatch"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取 penalty_storage
        ps_patterns = [
            r'"penalty_storage"\s*:\s*([0-9.]+)',
            r'penalty_storage[:\s=]+([0-9.]+)',
        ]
        for pattern in ps_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    config["penalty_storage"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        return config if config else None
    
    def parse_topology_action(
        self,
        action_text: str,
        action_space
    ) -> Optional[BaseAction]:
        """
        解析拓扑动作
        
        Args:
            action_text: LLM 生成的拓扑动作文本
            action_space: Grid2Op 动作空间
            
        Returns:
            Grid2Op Action 对象，如果解析失败则返回 None
        """
        # 清理文本
        action_text = action_text.strip()
        
        # 检查是否为 do_nothing
        if self.do_nothing_pattern.search(action_text):
            return action_space({})
        
        # 获取线路数量用于验证
        n_line = getattr(action_space, 'n_line', None)
        if n_line is None:
            # 尝试从 observation_space 获取（如果 action_space 没有）
            if hasattr(action_space, 'observation_space'):
                n_line = getattr(action_space.observation_space, 'n_line', None)
        
        # 创建空动作
        action = action_space({})
        
        # 方法1: 尝试解析 JSON 格式
        try:
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(action_text)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "actions" in data and isinstance(data["actions"], list):
                        for act in data["actions"]:
                            if act.get("type") == "set_line_status":
                                line_id = int(act.get("line_id"))
                                status = int(act.get("status"))
                                # 验证线路 ID 范围
                                if n_line is not None and (line_id < 0 or line_id >= n_line):
                                    logger.warning(f"跳过无效的线路 ID: {line_id} (有效范围: 0-{n_line-1})")
                                    continue  # 跳过无效的线路 ID
                                if status in [-1, 1]:
                                    if not hasattr(action, 'set_line_status') or action.set_line_status is None:
                                        action.set_line_status = []
                                    action.set_line_status.append((line_id, status))
                        if hasattr(action, 'set_line_status') and action.set_line_status:
                            return action
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        except Exception:
            pass
        
        # 方法2: 解析文本格式（set_line_status, set_bus, change_bus）
        has_action = False
        
        # 解析 set_line_status
        line_status_matches = self.set_line_status_pattern.findall(action_text)
        if line_status_matches:
            line_status_list = []
            for line_id_str, status_str in line_status_matches:
                try:
                    line_id = int(line_id_str)
                    status = int(status_str)
                    # 验证线路 ID 范围
                    if n_line is not None and (line_id < 0 or line_id >= n_line):
                        logger.warning(f"跳过无效的线路 ID: {line_id} (有效范围: 0-{n_line-1})")
                        continue  # 跳过无效的线路 ID
                    if status not in [-1, 1]:
                        continue
                    line_status_list.append((line_id, status))
                    has_action = True
                except (ValueError, TypeError):
                    continue
            
            if line_status_list:
                action.set_line_status = line_status_list
        
        # 解析 set_bus (格式: set_bus(substation_id, bus_id))
        set_bus_pattern = re.compile(
            r'set_bus\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
            re.IGNORECASE
        )
        set_bus_matches = set_bus_pattern.findall(action_text)
        if set_bus_matches:
            set_bus_list = []
            for sub_id_str, bus_id_str in set_bus_matches:
                try:
                    sub_id = int(sub_id_str)
                    bus_id = int(bus_id_str)
                    if bus_id in [1, 2]:
                        set_bus_list.append((sub_id, bus_id))
                        has_action = True
                except (ValueError, TypeError):
                    continue
            
            if set_bus_list:
                action.set_bus = set_bus_list
        
        # 解析 change_bus (格式: change_bus(substation_id))
        change_bus_pattern = re.compile(
            r'change_bus\s*\(\s*(\d+)\s*\)',
            re.IGNORECASE
        )
        change_bus_matches = change_bus_pattern.findall(action_text)
        if change_bus_matches:
            change_bus_list = []
            for sub_id_str in change_bus_matches:
                try:
                    sub_id = int(sub_id_str)
                    change_bus_list.append(sub_id)
                    has_action = True
                except (ValueError, TypeError):
                    continue
            
            if change_bus_list:
                action.change_bus = change_bus_list
        
        if has_action:
            return action
        
        return None

