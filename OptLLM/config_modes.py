# -*- coding: utf-8 -*-
"""
优化器模式配置模块

定义不同的运行模式及其对应的参数配置。
采用"意图模式"抽象，让 LLM 只需选择模式，而不需要记忆具体参数数值。
"""

from typing import Dict, Any

# 优化器模式配置表
OPTIMIZER_MODES: Dict[str, Dict[str, Any]] = {
    "ECONOMIC": {
        "description": "经济模式：系统安全，优先考虑成本，禁止切负荷",
        "params": {
            "penalty_curtailment": 10.0,      # 禁止切负荷
            "penalty_overflow": 1.0,          # 允许轻微过载换取成本
            "penalty_redispatch": 0.03,       # 标准再调度惩罚
            "margin_th_limit": 0.90           # 标准裕度
        }
    },
    "CAUTIOUS": {
        "description": "谨慎模式：有潜在风险，收紧约束，允许极少量切负荷",
        "params": {
            "penalty_curtailment": 0.1,       # 标准惩罚
            "penalty_overflow": 100.0,       # 更加厌恶过载
            "penalty_redispatch": 0.03,       # 标准再调度惩罚
            "margin_th_limit": 0.85           # 留更多缓冲 (AC/DC误差)
        }
    },
    "EMERGENCY": {
        "description": "紧急模式：检测到明显过载，必须消除越限",
        "params": {
            "penalty_curtailment": 0.01,      # 允许切负荷
            "penalty_overflow": 100.0,       # 严厉禁止过载
            "penalty_redispatch": 0.03,       # 标准再调度惩罚
            "margin_th_limit": 0.95           # 稍微放宽裕度以求有解
        }
    },
    "SURVIVAL": {
        "description": "生存模式：系统即将崩溃（发散/解列），不惜一切代价求生",
        "params": {
            "penalty_curtailment": 0.001,     # 拼命切负荷
            "penalty_overflow": 1000.0,      # 绝对禁止过载
            "penalty_redispatch": 0.03,       # 标准再调度惩罚
            "margin_th_limit": 1.0            # 甚至允许满载运行
        }
    }
}

# 模式优先级（用于容错时的回退顺序）
MODE_PRIORITY = ["ECONOMIC", "CAUTIOUS", "EMERGENCY", "SURVIVAL"]

# 默认模式（当无法解析时使用）
DEFAULT_MODE = "EMERGENCY"


def get_mode_config(mode_name: str) -> Dict[str, Any]:
    """
    根据模式名称获取配置参数
    
    Args:
        mode_name: 模式名称（不区分大小写）
        
    Returns:
        配置参数字典，如果模式不存在则返回默认模式（EMERGENCY）的配置
    """
    mode_name_upper = mode_name.upper().strip()
    
    if mode_name_upper in OPTIMIZER_MODES:
        return OPTIMIZER_MODES[mode_name_upper]["params"].copy()
    else:
        # 容错：返回默认模式
        return OPTIMIZER_MODES[DEFAULT_MODE]["params"].copy()


def get_mode_description(mode_name: str) -> str:
    """
    获取模式的描述信息
    
    Args:
        mode_name: 模式名称（不区分大小写）
        
    Returns:
        模式描述字符串
    """
    mode_name_upper = mode_name.upper().strip()
    
    if mode_name_upper in OPTIMIZER_MODES:
        return OPTIMIZER_MODES[mode_name_upper]["description"]
    else:
        return OPTIMIZER_MODES[DEFAULT_MODE]["description"]


def list_available_modes() -> list:
    """
    列出所有可用的模式名称
    
    Returns:
        模式名称列表
    """
    return list(OPTIMIZER_MODES.keys())

