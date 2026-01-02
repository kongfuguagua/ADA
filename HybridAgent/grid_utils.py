# -*- coding: utf-8 -*-
"""
Grid2Op 工具函数

提供共享的电网操作工具函数，避免代码重复。
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from grid2op.Observation import BaseObservation


def get_substation_elements(
    observation: BaseObservation,
    sub_id: int,
    include_power: bool = False,
    include_rho: bool = False
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    统一提取变电站连接元件的逻辑
    
    Args:
        observation: Grid2Op 观测对象
        sub_id: 变电站ID
        include_power: 是否包含功率信息（用于 Prompter）
        include_rho: 是否包含线路负载率信息（用于 Prompter）
        
    Returns:
        (elements_dict, lookup_dict)
        - elements_dict: 按类型分组的元素字典 {'generators': [...], 'loads': [...], 'lines': [...]}
        - lookup_dict: 按名称快速查找的字典 {element_name: element_info}
    """
    elements = {'generators': [], 'loads': [], 'lines': []}
    lookup = {}
    
    # 1. 发电机
    if hasattr(observation, 'gen_to_subid'):
        gen_ids = np.where(observation.gen_to_subid == sub_id)[0]
        for gen_id in gen_ids:
            info = {
                "type": "generator",
                "id": int(gen_id),
                "name": f"gen_{gen_id}"
            }
            if include_power and hasattr(observation, 'gen_p'):
                info['p'] = float(observation.gen_p[gen_id])
            if include_power and hasattr(observation, 'gen_redispatchable'):
                info['redispatchable'] = bool(observation.gen_redispatchable[gen_id])
            
            elements['generators'].append(info)
            lookup[info['name']] = info
    
    # 2. 负荷
    if hasattr(observation, 'load_to_subid'):
        load_ids = np.where(observation.load_to_subid == sub_id)[0]
        for load_id in load_ids:
            info = {
                "type": "load",
                "id": int(load_id),
                "name": f"load_{load_id}"
            }
            if include_power and hasattr(observation, 'load_p'):
                info['p'] = float(observation.load_p[load_id])
            
            elements['loads'].append(info)
            lookup[info['name']] = info
    
    # 3. 线路
    line_ids = []
    if hasattr(observation, 'line_or_to_subid'):
        line_ids.extend(np.where(observation.line_or_to_subid == sub_id)[0])
    if hasattr(observation, 'line_ex_to_subid'):
        line_ids.extend(np.where(observation.line_ex_to_subid == sub_id)[0])
    
    # 去重
    line_ids = list(set(line_ids))
    for line_id in line_ids:
        info = {
            "type": "line",
            "id": int(line_id),
            "name": f"line_{line_id}"
        }
        if include_rho and hasattr(observation, 'rho'):
            info['rho'] = float(observation.rho[line_id])
        
        elements['lines'].append(info)
        lookup[info['name']] = info
    
    return elements, lookup

