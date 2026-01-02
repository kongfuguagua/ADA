# -*- coding: utf-8 -*-
"""
拓扑动作解析器 (Topology Parser)

将 LLM 返回的 JSON（母线分裂配置）转换为 Grid2Op Action。
"""

import json
import re
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation

from .grid_utils import get_substation_elements


class TopologyParser:
    """
    拓扑动作解析器
    
    功能：
    - 解析 LLM 返回的 JSON（母线分裂配置）
    - 将元素名称映射到 Grid2Op 内部 ID
    - 构建 Grid2Op Action（set_bus）
    """
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    def parse_bus_splitting(
        self,
        llm_response: str,
        observation: BaseObservation,
        action_space
    ) -> Optional[BaseAction]:
        """
        解析 LLM 返回的母线分裂配置
        
        Args:
            llm_response: LLM 的响应（应该是 JSON 格式）
            observation: 当前电网观测（用于验证和映射）
            action_space: Grid2Op 动作空间
            
        Returns:
            Grid2Op Action 对象，如果解析失败则返回 None
        """
        # 1. 提取 JSON
        config = self._extract_json(llm_response)
        if config is None:
            return None
        
        # 2. 验证配置格式
        if not self._validate_config(config):
            return None
        
        # 3. 构建动作
        try:
            action = self._build_action(config, observation, action_space)
            return action
        except Exception as e:
            # 解析失败，返回 None
            return None
    
    def _extract_json(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应中提取 JSON
        
        支持以下格式：
        1. 纯 JSON 块
        2. Markdown 代码块中的 JSON
        3. 文本中的 JSON 片段
        """
        # 方法1: 尝试直接解析整个响应
        try:
            data = json.loads(llm_response.strip())
            if isinstance(data, dict) and "substation_id" in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # 方法2: 提取 JSON 代码块
        json_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
        match = json_pattern.search(llm_response)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "substation_id" in data:
                    return data
            except json.JSONDecodeError:
                pass
        
        # 方法3: 提取 JSON 对象（更宽松的匹配）
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
        matches = json_pattern.findall(llm_response)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "substation_id" in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置格式
        
        检查：
        1. 必需字段存在
        2. bus_1 和 bus_2 非空
        3. bus_1 和 bus_2 互斥（无重复元素）
        """
        # 检查必需字段
        if "substation_id" not in config:
            return False
        if "bus_1" not in config or "bus_2" not in config:
            return False
        
        bus_1 = config["bus_1"]
        bus_2 = config["bus_2"]
        
        # 检查非空
        if not isinstance(bus_1, list) or len(bus_1) == 0:
            return False
        if not isinstance(bus_2, list) or len(bus_2) == 0:
            return False
        
        # 检查互斥性（不能有重复元素）
        set_1 = set(bus_1)
        set_2 = set(bus_2)
        if len(set_1) != len(bus_1) or len(set_2) != len(bus_2):
            # 有重复元素
            return False
        if set_1 & set_2:
            # 有交集（元素重复）
            return False
        
        return True
    
    def _build_action(
        self,
        config: Dict[str, Any],
        observation: BaseObservation,
        action_space
    ) -> BaseAction:
        """
        构建 Grid2Op Action
        
        Args:
            config: 解析后的配置字典
            observation: 当前观测
            action_space: 动作空间
            
        Returns:
            Grid2Op Action 对象
        """
        sub_id = int(config["substation_id"])
        bus_1_elements = config["bus_1"]  # 元素名称列表
        bus_2_elements = config["bus_2"]  # 元素名称列表
        
        # 1. 获取变电站的所有设备（使用共享工具函数）
        _, sub_elements = get_substation_elements(observation, sub_id, include_power=False, include_rho=False)
        
        # 2. 将元素名称映射到 Grid2Op 内部 ID
        bus_1_ids = self._map_element_names_to_ids(bus_1_elements, sub_elements)
        bus_2_ids = self._map_element_names_to_ids(bus_2_elements, sub_elements)
        
        # 3. 验证完整性（所有设备都被分配）
        all_assigned = set(bus_1_ids.keys()) | set(bus_2_ids.keys())
        all_elements = set(sub_elements.keys())
        if all_assigned != all_elements:
            # 有设备未被分配，使用默认分配（保持原状）
            missing = all_elements - all_assigned
            # 将缺失的设备分配到 bus_1（保持原状）
            for elem_key in missing:
                bus_1_ids[elem_key] = sub_elements[elem_key]
        
        # 4. 构建 set_bus 字典（使用循环消除重复逻辑）
        # Grid2Op 的 set_bus 动作格式：
        # action.set_bus = {
        #     "loads_id": [(load_id, bus_id), ...],
        #     "generators_id": [(gen_id, bus_id), ...],
        #     "lines_or_id": [(line_id, bus_id), ...],
        #     "lines_ex_id": [(line_id, bus_id), ...],
        # }
        
        # 使用字典存储各类型设备的分配
        topo_dict = {
            "loads": [],
            "generators": [],
            "lines_or": [],
            "lines_ex": []
        }
        
        # 遍历 bus_1 (bus=1) 和 bus_2 (bus=2)，统一处理
        assignments = [(bus_1_ids, 1), (bus_2_ids, 2)]
        processed_ids = set()  # 防止重复分配
        
        for elem_dict, bus_idx in assignments:
            for elem_key, elem_info in elem_dict.items():
                elem_type = elem_info["type"]
                elem_id = elem_info["id"]
                
                # 防止重复分配
                if (elem_type, elem_id) in processed_ids:
                    continue
                processed_ids.add((elem_type, elem_id))
                
                if elem_type == "load":
                    topo_dict["loads"].append((elem_id, bus_idx))
                elif elem_type == "generator":
                    topo_dict["generators"].append((elem_id, bus_idx))
                elif elem_type == "line":
                    # 判断是 origin 还是 extremity
                    line_id = elem_id
                    if hasattr(observation, 'line_or_to_subid'):
                        if observation.line_or_to_subid[line_id] == sub_id:
                            topo_dict["lines_or"].append((line_id, bus_idx))
                    if hasattr(observation, 'line_ex_to_subid'):
                        if observation.line_ex_to_subid[line_id] == sub_id:
                            topo_dict["lines_ex"].append((line_id, bus_idx))
        
        # 构建 Grid2Op 动作字典
        set_bus_dict = {}
        if topo_dict["loads"]:
            set_bus_dict["loads_id"] = topo_dict["loads"]
        if topo_dict["generators"]:
            set_bus_dict["generators_id"] = topo_dict["generators"]
        if topo_dict["lines_or"]:
            set_bus_dict["lines_or_id"] = topo_dict["lines_or"]
        if topo_dict["lines_ex"]:
            set_bus_dict["lines_ex_id"] = topo_dict["lines_ex"]
        
        # 如果没有有效的动作，返回空动作
        if not set_bus_dict:
            return action_space({})
        
        # 创建动作
        action = action_space({"set_bus": set_bus_dict})
        return action
    
    
    def _map_element_names_to_ids(
        self,
        element_names: List[str],
        sub_elements: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        将元素名称列表映射到元素信息
        
        Args:
            element_names: 元素名称列表（如 ["gen_0", "load_2"]）
            sub_elements: 变电站的所有元素字典
            
        Returns:
            映射后的元素信息字典
        """
        mapped = {}
        for elem_name in element_names:
            # 标准化名称（去除空格，转为小写）
            elem_name_clean = elem_name.strip().lower()
            
            # 直接匹配
            if elem_name_clean in sub_elements:
                mapped[elem_name_clean] = sub_elements[elem_name_clean]
                continue
            
            # 尝试模糊匹配（如果名称格式略有不同）
            # 例如 "gen0" vs "gen_0"
            for key, value in sub_elements.items():
                key_clean = key.lower()
                # 检查是否匹配（忽略下划线）
                if elem_name_clean.replace("_", "") == key_clean.replace("_", ""):
                    mapped[key] = value
                    break
        
        return mapped

