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
        
        # 1. 获取变电站的所有设备
        sub_elements = self._get_substation_elements(observation, sub_id)
        
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
        
        # 4. 构建 set_bus 字典
        set_bus_dict = {}
        
        # 对于每个设备类型，构建对应的 set_bus 配置
        # Grid2Op 的 set_bus 格式：
        # - 对于线路: {"lines_or_id": [line_id, ...], "lines_ex_id": [line_id, ...]}
        # - 对于发电机: {"generators_id": [gen_id, ...]}
        # - 对于负荷: {"loads_id": [load_id, ...]}
        
        # 但是，更简单的方式是使用统一的 set_bus 格式：
        # action.set_bus = {
        #     "substations_id": [sub_id],
        #     "substations_bus": [[bus_config]]
        # }
        
        # 或者使用更直接的方式：为每个设备单独设置
        # 我们需要构建一个字典，键是设备类型和ID，值是目标母线（1 或 2）
        
        # 实际上，Grid2Op 的 set_bus 动作格式是：
        # action.set_bus = {
        #     "loads_id": [load_id, ...],
        #     "loads_bus": [bus_id, ...],
        #     "generators_id": [gen_id, ...],
        #     "generators_bus": [bus_id, ...],
        #     "lines_or_id": [line_id, ...],
        #     "lines_or_bus": [bus_id, ...],
        #     "lines_ex_id": [line_id, ...],
        #     "lines_ex_bus": [bus_id, ...],
        # }
        
        loads_id = []
        loads_bus = []
        generators_id = []
        generators_bus = []
        lines_or_id = []
        lines_or_bus = []
        lines_ex_id = []
        lines_ex_bus = []
        
        # 处理 bus_1 的设备（分配到母线 1）
        for elem_key, elem_info in bus_1_ids.items():
            elem_type = elem_info["type"]
            elem_id = elem_info["id"]
            
            if elem_type == "load":
                loads_id.append(elem_id)
                loads_bus.append(1)
            elif elem_type == "generator":
                generators_id.append(elem_id)
                generators_bus.append(1)
            elif elem_type == "line":
                # 需要判断是 origin 还是 extremity
                line_id = elem_id
                if hasattr(observation, 'line_or_to_subid'):
                    if observation.line_or_to_subid[line_id] == sub_id:
                        lines_or_id.append(line_id)
                        lines_or_bus.append(1)
                if hasattr(observation, 'line_ex_to_subid'):
                    if observation.line_ex_to_subid[line_id] == sub_id:
                        lines_ex_id.append(line_id)
                        lines_ex_bus.append(1)
        
        # 处理 bus_2 的设备（分配到母线 2）
        for elem_key, elem_info in bus_2_ids.items():
            elem_type = elem_info["type"]
            elem_id = elem_info["id"]
            
            if elem_type == "load":
                loads_id.append(elem_id)
                loads_bus.append(2)
            elif elem_type == "generator":
                generators_id.append(elem_id)
                generators_bus.append(2)
            elif elem_type == "line":
                line_id = elem_id
                if hasattr(observation, 'line_or_to_subid'):
                    if observation.line_or_to_subid[line_id] == sub_id:
                        lines_or_id.append(line_id)
                        lines_or_bus.append(2)
                if hasattr(observation, 'line_ex_to_subid'):
                    if observation.line_ex_to_subid[line_id] == sub_id:
                        lines_ex_id.append(line_id)
                        lines_ex_bus.append(2)
        
        # 构建动作字典（Grid2Op 格式：每个元素类型使用 (id, bus) 元组列表）
        set_bus_dict = {}
        
        if loads_id:
            # 格式: [(load_id, bus_id), ...]
            set_bus_dict["loads_id"] = [(loads_id[i], loads_bus[i]) for i in range(len(loads_id))]
        
        if generators_id:
            # 格式: [(gen_id, bus_id), ...]
            set_bus_dict["generators_id"] = [(generators_id[i], generators_bus[i]) for i in range(len(generators_id))]
        
        if lines_or_id:
            # 格式: [(line_id, bus_id), ...]
            set_bus_dict["lines_or_id"] = [(lines_or_id[i], lines_or_bus[i]) for i in range(len(lines_or_id))]
        
        if lines_ex_id:
            # 格式: [(line_id, bus_id), ...]
            set_bus_dict["lines_ex_id"] = [(lines_ex_id[i], lines_ex_bus[i]) for i in range(len(lines_ex_id))]
        
        # 如果没有有效的动作，返回空动作
        if not set_bus_dict:
            return action_space({})
        
        # 创建动作
        action = action_space({"set_bus": set_bus_dict})
        return action
    
    def _get_substation_elements(
        self,
        observation: BaseObservation,
        sub_id: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取变电站的所有设备
        
        Returns:
            字典，键为元素名称（如 "gen_0"），值为元素信息
        """
        elements = {}
        
        # 1. 发电机
        if hasattr(observation, 'gen_to_subid'):
            gen_ids = np.where(observation.gen_to_subid == sub_id)[0]
            for gen_id in gen_ids:
                elem_name = f"gen_{gen_id}"
                elements[elem_name] = {
                    "type": "generator",
                    "id": int(gen_id),
                    "name": elem_name
                }
        
        # 2. 负荷
        if hasattr(observation, 'load_to_subid'):
            load_ids = np.where(observation.load_to_subid == sub_id)[0]
            for load_id in load_ids:
                elem_name = f"load_{load_id}"
                elements[elem_name] = {
                    "type": "load",
                    "id": int(load_id),
                    "name": elem_name
                }
        
        # 3. 线路
        connected_lines = []
        if hasattr(observation, 'line_or_to_subid'):
            or_lines = np.where(observation.line_or_to_subid == sub_id)[0]
            connected_lines.extend(or_lines.tolist())
        if hasattr(observation, 'line_ex_to_subid'):
            ex_lines = np.where(observation.line_ex_to_subid == sub_id)[0]
            connected_lines.extend(ex_lines.tolist())
        
        connected_lines = list(set(connected_lines))
        for line_id in connected_lines:
            elem_name = f"line_{line_id}"
            elements[elem_name] = {
                "type": "line",
                "id": int(line_id),
                "name": elem_name
            }
        
        return elements
    
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

