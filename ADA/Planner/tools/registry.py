# -*- coding: utf-8 -*-
"""
Planner 分析工具注册表
管理为 Planner 决策服务的分析工具

区别于 env/tools（环境交互工具）：
- 分析工具：统计数据、规则检测、趋势分析、决策建议
- 环境工具：发送命令、获取原始数据
"""

import sys
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.interact import BaseTool
from utils.logger import get_logger

logger = get_logger("ToolRegistry")


class ToolRegistry:
    """
    工具注册表
    管理所有可用工具的注册、执行和描述生成
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")
    
    def register_function(
        self, 
        name: str, 
        func: Callable, 
        description: str,
        parameters: Dict[str, Any] = None
    ) -> None:
        """注册函数为工具"""
        tool = FunctionTool(name, func, description, parameters)
        self.register(tool)
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self._tools:
            error_msg = f"工具 '{tool_name}' 未注册"
            logger.error(error_msg)
            return {"error": error_msg}
        
        tool = self._tools[tool_name]
        
        try:
            logger.info(f"执行工具: {tool_name}", params=str(kwargs)[:100])
            result = tool.execute(**kwargs)
            logger.info(f"工具执行完成: {tool_name}")
            return result
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)
    
    def get_tool_descriptions(self) -> str:
        """生成给 LLM 看的工具列表描述"""
        if not self._tools:
            return "暂无可用工具"
        
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
            
            schema = tool.schema
            if schema and "parameters" in schema:
                props = schema["parameters"].get("properties", {})
                if props:
                    lines.append("  参数:")
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        lines.append(f"    - {param_name} ({param_type}): {param_desc}")
        
        return "\n".join(lines)
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """获取 OpenAI 格式的工具定义"""
        tools = []
        for tool in self._tools.values():
            tools.append({
                "type": "function",
                "function": tool.schema
            })
        return tools
    
    def list_tools(self) -> List[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())
    
    def set_env(self, env) -> None:
        """为所有支持的工具设置环境"""
        for tool in self._tools.values():
            if hasattr(tool, 'set_env'):
                tool.set_env(env)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


class FunctionTool(BaseTool):
    """将普通函数包装为工具"""
    
    def __init__(
        self, 
        name: str, 
        func: Callable, 
        description: str,
        parameters: Dict[str, Any] = None
    ):
        self._name = name
        self._func = func
        self._description = description
        self._parameters = parameters or {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def execute(self, **kwargs) -> Any:
        return self._func(**kwargs)
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": self._parameters,
                "required": list(self._parameters.keys())
            }
        }


# ============= Planner 分析工具 =============

class FinishTool(BaseTool):
    """结束状态增广的工具"""
    
    @property
    def name(self) -> str:
        return "finish"
    
    @property
    def description(self) -> str:
        return "当收集到足够信息时调用此工具，结束状态增广阶段"
    
    def execute(self, summary: str = "", **kwargs) -> Dict[str, Any]:
        return {"status": "finished", "summary": summary}
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "状态增广的总结"
                    }
                },
                "required": []
            }
        }


class GridStatusAnalysisTool(BaseTool):
    """电网状态分析工具（为 Planner 提供分析结论）"""
    
    def __init__(self, env=None):
        self._env = env
    
    def set_env(self, env):
        self._env = env
    
    @property
    def name(self) -> str:
        return "grid_status_analysis"
    
    @property
    def description(self) -> str:
        return "分析当前电网状态，提供安全评估和风险识别结论"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """分析电网状态"""
        if self._env is None or not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            return self._default_analysis()
        
        obs = self._env.current_obs
        
        # 统计分析
        max_rho = float(obs.rho.max())
        avg_rho = float(obs.rho.mean())
        overflow_count = int((obs.rho > 1.0).sum())
        near_overflow = int((obs.rho > 0.9).sum())
        
        # 风险评估
        if overflow_count > 0:
            risk_level = "危险"
            risk_score = 1.0
        elif near_overflow > 0:
            risk_level = "警告"
            risk_score = 0.7
        elif max_rho > 0.7:
            risk_level = "注意"
            risk_score = 0.4
        else:
            risk_level = "安全"
            risk_score = 0.1
        
        # 生成建议
        recommendations = []
        if overflow_count > 0:
            recommendations.append("立即采取再调度或拓扑调整措施")
        if near_overflow > 0:
            recommendations.append("关注高负载线路，准备应急方案")
        if max_rho > 0.7:
            recommendations.append("监控负载变化趋势")
        
        return {
            "summary": f"电网状态: {risk_level}",
            "risk_level": risk_level,
            "risk_score": risk_score,
            "statistics": {
                "max_rho": max_rho,
                "avg_rho": avg_rho,
                "overflow_count": overflow_count,
                "near_overflow_count": near_overflow,
            },
            "recommendations": recommendations,
        }
    
    def _default_analysis(self) -> Dict[str, Any]:
        return {
            "summary": "电网状态: 正常（模拟数据）",
            "risk_level": "安全",
            "risk_score": 0.2,
            "statistics": {
                "max_rho": 0.75,
                "avg_rho": 0.45,
                "overflow_count": 0,
                "near_overflow_count": 2,
            },
            "recommendations": ["监控负载变化趋势"],
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}
        }


class OverflowRiskAnalysisTool(BaseTool):
    """过载风险分析工具"""
    
    def __init__(self, env=None):
        self._env = env
    
    def set_env(self, env):
        self._env = env
    
    @property
    def name(self) -> str:
        return "overflow_risk_analysis"
    
    @property
    def description(self) -> str:
        return "分析线路过载风险，识别高风险线路并提供处理建议"
    
    def execute(self, threshold: float = 0.9, **kwargs) -> Dict[str, Any]:
        """分析过载风险"""
        if self._env is None or not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            return self._default_analysis(threshold)
        
        obs = self._env.current_obs
        env = self._env.env
        
        # 识别高风险线路
        high_risk_lines = []
        for i, rho in enumerate(obs.rho):
            if rho > threshold:
                line_name = env.name_line[i] if hasattr(env, 'name_line') else f"line_{i}"
                high_risk_lines.append({
                    "line_id": i,
                    "line_name": str(line_name),
                    "rho": float(rho),
                    "is_overflow": bool(rho > 1.0),
                    "risk_category": "过载" if rho > 1.0 else "高负载",
                })
        
        # 按风险排序
        high_risk_lines.sort(key=lambda x: x["rho"], reverse=True)
        
        # 生成处理建议
        suggestions = []
        overflow_lines = [l for l in high_risk_lines if l["is_overflow"]]
        if overflow_lines:
            suggestions.append({
                "priority": "紧急",
                "action": "再调度",
                "target": [l["line_name"] for l in overflow_lines[:3]],
                "reason": "线路过载，需立即降低负载"
            })
        
        near_overflow = [l for l in high_risk_lines if not l["is_overflow"]]
        if near_overflow:
            suggestions.append({
                "priority": "预防",
                "action": "拓扑调整",
                "target": [l["line_name"] for l in near_overflow[:3]],
                "reason": "线路接近过载，建议预防性调整"
            })
        
        return {
            "threshold": threshold,
            "total_high_risk": len(high_risk_lines),
            "overflow_count": len(overflow_lines),
            "high_risk_lines": high_risk_lines[:10],  # 限制返回数量
            "suggestions": suggestions,
        }
    
    def _default_analysis(self, threshold: float) -> Dict[str, Any]:
        return {
            "threshold": threshold,
            "total_high_risk": 2,
            "overflow_count": 0,
            "high_risk_lines": [
                {"line_id": 5, "line_name": "line_5", "rho": 0.95, "is_overflow": False, "risk_category": "高负载"},
                {"line_id": 12, "line_name": "line_12", "rho": 0.92, "is_overflow": False, "risk_category": "高负载"},
            ],
            "suggestions": [
                {"priority": "预防", "action": "拓扑调整", "target": ["line_5", "line_12"], "reason": "线路接近过载"}
            ],
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "风险阈值 (0-1)，默认 0.9"
                    }
                },
                "required": []
            }
        }


class GeneratorCapacityAnalysisTool(BaseTool):
    """发电机容量分析工具"""
    
    def __init__(self, env=None):
        self._env = env
    
    def set_env(self, env):
        self._env = env
    
    @property
    def name(self) -> str:
        return "generator_capacity_analysis"
    
    @property
    def description(self) -> str:
        return "分析发电机出力和可调容量，评估调度灵活性"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """分析发电机容量"""
        if self._env is None or not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            return self._default_analysis()
        
        obs = self._env.current_obs
        env = self._env.env
        
        total_gen = float(obs.gen_p.sum())
        total_load = float(obs.load_p.sum())
        
        # 分析可调度发电机
        redispatchable_analysis = []
        total_upward = 0.0
        total_downward = 0.0
        
        if hasattr(env, 'gen_redispatchable'):
            for i in range(env.n_gen):
                if env.gen_redispatchable[i]:
                    p_max = float(obs.gen_pmax[i]) if hasattr(obs, 'gen_pmax') else 100.0
                    p_min = float(obs.gen_pmin[i]) if hasattr(obs, 'gen_pmin') else 0.0
                    current_p = float(obs.gen_p[i])
                    
                    upward = p_max - current_p
                    downward = current_p - p_min
                    
                    total_upward += upward
                    total_downward += downward
                    
                    redispatchable_analysis.append({
                        "gen_id": i,
                        "gen_name": str(env.name_gen[i]) if hasattr(env, 'name_gen') else f"gen_{i}",
                        "current_p": current_p,
                        "upward_margin": upward,
                        "downward_margin": downward,
                    })
        
        # 评估灵活性
        flexibility_score = min(1.0, (total_upward + total_downward) / (total_gen + 1e-6))
        
        return {
            "summary": f"发电总量: {total_gen:.1f}MW, 负荷: {total_load:.1f}MW",
            "total_generation_mw": total_gen,
            "total_load_mw": total_load,
            "power_balance": total_gen - total_load,
            "flexibility": {
                "total_upward_margin_mw": total_upward,
                "total_downward_margin_mw": total_downward,
                "flexibility_score": flexibility_score,
            },
            "redispatchable_generators": redispatchable_analysis[:5],  # 限制返回数量
        }
    
    def _default_analysis(self) -> Dict[str, Any]:
        return {
            "summary": "发电总量: 105.0MW, 负荷: 100.0MW",
            "total_generation_mw": 105.0,
            "total_load_mw": 100.0,
            "power_balance": 5.0,
            "flexibility": {
                "total_upward_margin_mw": 50.0,
                "total_downward_margin_mw": 30.0,
                "flexibility_score": 0.76,
            },
            "redispatchable_generators": [
                {"gen_id": 0, "gen_name": "gen_0", "current_p": 40.0, "upward_margin": 20.0, "downward_margin": 15.0},
            ],
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}
        }


class LoadTrendAnalysisTool(BaseTool):
    """负荷趋势分析工具"""
    
    def __init__(self, env=None):
        self._env = env
    
    def set_env(self, env):
        self._env = env
    
    @property
    def name(self) -> str:
        return "load_trend_analysis"
    
    @property
    def description(self) -> str:
        return "分析负荷变化趋势，预测峰谷时段"
    
    def execute(self, forecast_hours: int = 24, **kwargs) -> Dict[str, Any]:
        """分析负荷趋势"""
        import math
        
        if self._env is None or not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            current_load = 100.0
            current_hour = 12
        else:
            obs = self._env.current_obs
            current_load = float(obs.load_p.sum())
            current_hour = obs.hour_of_day if hasattr(obs, 'hour_of_day') else 12
        
        # 生成日负荷曲线预测
        forecast = []
        for h in range(forecast_hours):
            hour = (current_hour + h) % 24
            # 典型日负荷曲线
            factor = 0.8 + 0.4 * math.sin((hour - 6) * math.pi / 12)
            predicted_load = current_load * factor
            forecast.append({
                "hour": hour,
                "predicted_load_mw": predicted_load,
            })
        
        # 识别峰谷
        peak = max(forecast, key=lambda x: x["predicted_load_mw"])
        valley = min(forecast, key=lambda x: x["predicted_load_mw"])
        
        # 趋势判断
        next_3h = forecast[:3]
        trend = "上升" if next_3h[-1]["predicted_load_mw"] > next_3h[0]["predicted_load_mw"] else "下降"
        
        return {
            "current_load_mw": current_load,
            "current_hour": current_hour,
            "trend": trend,
            "peak": {
                "hour": peak["hour"],
                "load_mw": peak["predicted_load_mw"],
                "hours_until": (peak["hour"] - current_hour) % 24,
            },
            "valley": {
                "hour": valley["hour"],
                "load_mw": valley["predicted_load_mw"],
                "hours_until": (valley["hour"] - current_hour) % 24,
            },
            "forecast_summary": forecast[:6],  # 只返回前6小时
            "planning_advice": f"负荷{trend}趋势，预计{(peak['hour'] - current_hour) % 24}小时后达到峰值",
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_hours": {
                        "type": "integer",
                        "description": "预测小时数，默认 24"
                    }
                },
                "required": []
            }
        }


class EnvironmentFeatureAnalysisTool(BaseTool):
    """环境特征分析工具（提取环境配置和结构信息）"""
    
    def __init__(self, env=None):
        self._env = env
    
    def set_env(self, env):
        self._env = env
    
    @property
    def name(self) -> str:
        return "environment_feature_analysis"
    
    @property
    def description(self) -> str:
        return "分析环境特征，提供电网规模、决策变量维度、约束范围等结构化信息，用于指导问题建模"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """分析环境特征"""
        try:
            # 从环境配置获取信息
            env_config_info = self._get_env_config_info()
            
            # 从实际环境对象获取信息
            env_structure_info = self._get_env_structure_info()
            
            # 合并信息
            return {
                "environment_config": env_config_info,
                "grid_structure": env_structure_info,
                "decision_space": self._analyze_decision_space(),
                "constraint_summary": self._analyze_constraints(),
            }
        except Exception as e:
            logger.warning(f"环境特征分析失败: {e}")
            return self._default_features()
    
    def _get_env_config_info(self) -> Dict[str, Any]:
        """从env/config.py获取环境配置信息"""
        try:
            from env.config import get_env_config, ENV_CONFIGS
            
            # 尝试从环境对象获取配置名称
            env_name = None
            if self._env is not None:
                # 尝试从环境属性推断
                if hasattr(self._env, 'name'):
                    env_name = self._env.name
                elif hasattr(self._env, 'env_name'):
                    env_name = self._env.env_name
            
            # 如果无法确定，返回通用信息
            if env_name is None:
                return {
                    "competition": "unknown",
                    "has_storage": False,
                    "has_renewable": False,
                    "has_curtailment": False,
                    "has_redispatch": True,
                    "has_alarm": False,
                    "description": "环境配置信息不可用"
                }
            
            # 尝试匹配配置
            for config_name, config in ENV_CONFIGS.items():
                if config.env_name == env_name or config_name in str(env_name):
                    return {
                        "competition": config.competition.value,
                        "name": config.name,
                        "has_storage": config.has_storage,
                        "has_renewable": config.has_renewable,
                        "has_curtailment": config.has_curtailment,
                        "has_redispatch": config.has_redispatch,
                        "has_alarm": config.has_alarm,
                        "description": config.description
                    }
            
            return {
                "competition": "unknown",
                "has_storage": False,
                "has_renewable": False,
                "has_curtailment": False,
                "has_redispatch": True,
                "has_alarm": False,
                "description": f"环境: {env_name}"
            }
        except Exception as e:
            logger.warning(f"获取环境配置失败: {e}")
            return {
                "competition": "unknown",
                "has_storage": False,
                "has_renewable": False,
                "has_curtailment": False,
                "has_redispatch": True,
                "has_alarm": False,
                "description": "环境配置信息不可用"
            }
    
    def _get_env_structure_info(self) -> Dict[str, Any]:
        """从实际环境对象获取电网结构信息"""
        if self._env is None or not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            return {
                "n_generators": "unknown",
                "n_loads": "unknown",
                "n_lines": "unknown",
                "n_substations": "unknown",
                "note": "环境对象不可用，使用默认值"
            }
        
        try:
            obs = self._env.current_obs
            
            return {
                "n_generators": int(obs.n_gen) if hasattr(obs, 'n_gen') else "unknown",
                "n_loads": int(obs.n_load) if hasattr(obs, 'n_load') else "unknown",
                "n_lines": int(obs.n_line) if hasattr(obs, 'n_line') else "unknown",
                "n_substations": int(obs.n_sub) if hasattr(obs, 'n_sub') else "unknown",
                "generator_info": {
                    "gen_pmin": [float(x) for x in obs.gen_pmin] if hasattr(obs, 'gen_pmin') else [],
                    "gen_pmax": [float(x) for x in obs.gen_pmax] if hasattr(obs, 'gen_pmax') else [],
                    "gen_cost_per_mw": [float(x) for x in obs.gen_cost_per_MW] if hasattr(obs, 'gen_cost_per_MW') else [],
                } if hasattr(obs, 'gen_pmin') else {},
                "load_info": {
                    "current_load": [float(x) for x in obs.load_p] if hasattr(obs, 'load_p') else [],
                } if hasattr(obs, 'load_p') else {},
            }
        except Exception as e:
            logger.warning(f"获取环境结构信息失败: {e}")
            return {
                "n_generators": "unknown",
                "n_loads": "unknown",
                "n_lines": "unknown",
                "n_substations": "unknown",
                "note": f"获取失败: {str(e)}"
            }
    
    def _analyze_decision_space(self) -> Dict[str, Any]:
        """分析决策空间"""
        structure = self._get_env_structure_info()
        n_gen = structure.get("n_generators", 0)
        
        if isinstance(n_gen, int) and n_gen > 0:
            return {
                "primary_variables": {
                    "generator_power": {
                        "dimension": n_gen,
                        "description": "发电机出力向量，维度等于发电机数量",
                        "typical_range": "每个发电机的出力上下限由gen_pmin和gen_pmax决定",
                        "constraint_type": "box_constraint"
                    }
                },
                "optional_variables": {
                    "redispatch_delta": {
                        "dimension": n_gen,
                        "description": "再调度调整量（如果支持再调度）",
                        "typical_range": "通常限制在[-30, 50] MW范围内",
                        "constraint_type": "box_constraint"
                    }
                },
                "variable_count_estimate": n_gen,
                "note": "主要决策变量是发电机出力，必须满足功率平衡和线路容量约束"
            }
        else:
            return {
                "primary_variables": {
                    "generator_power": {
                        "dimension": "unknown",
                        "description": "发电机出力向量",
                        "typical_range": "由环境决定",
                        "constraint_type": "box_constraint"
                    }
                },
                "variable_count_estimate": "unknown",
                "note": "环境信息不可用，需要从其他工具获取"
            }
    
    def _analyze_constraints(self) -> Dict[str, Any]:
        """分析约束条件"""
        structure = self._get_env_structure_info()
        n_gen = structure.get("n_generators", 0)
        n_load = structure.get("n_loads", 0)
        n_line = structure.get("n_lines", 0)
        
        constraints = {
            "power_balance": {
                "type": "equality",
                "description": "功率平衡约束：总发电量 = 总负载",
                "formula": "sum(p_i) = sum(load_k)",
                "complexity": "linear"
            },
            "generator_limits": {
                "type": "inequality",
                "description": "发电机出力上下限约束",
                "formula": "p_min[i] <= p[i] <= p_max[i]",
                "complexity": "linear",
                "count": n_gen if isinstance(n_gen, int) else "unknown"
            },
            "line_loading": {
                "type": "inequality",
                "description": "线路负载率约束：rho_j <= 1.0",
                "formula": "rho_j = f_j / S_j <= 1.0",
                "complexity": "nonlinear",
                "count": n_line if isinstance(n_line, int) else "unknown"
            }
        }
        
        return {
            "constraint_types": constraints,
            "total_constraint_count_estimate": (
                (1 + n_gen + n_line) 
                if isinstance(n_gen, int) and isinstance(n_line, int) 
                else "unknown"
            ),
            "constraint_complexity": "混合（线性+非线性）",
            "note": "必须包含功率平衡、发电机边界和线路容量约束"
        }
    
    def _default_features(self) -> Dict[str, Any]:
        """默认环境特征（当无法获取真实信息时）"""
        return {
            "environment_config": {
                "competition": "unknown",
                "has_storage": False,
                "has_renewable": False,
                "has_curtailment": False,
                "has_redispatch": True,
                "has_alarm": False,
                "description": "默认配置"
            },
            "grid_structure": {
                "n_generators": "unknown",
                "n_loads": "unknown",
                "n_lines": "unknown",
                "note": "使用默认值，建议调用其他工具获取详细信息"
            },
            "decision_space": {
                "primary_variables": {
                    "generator_power": {
                        "dimension": "unknown",
                        "description": "发电机出力向量",
                        "typical_range": "由环境决定"
                    }
                }
            },
            "constraint_summary": {
                "constraint_types": {
                    "power_balance": {"type": "equality", "description": "功率平衡"},
                    "generator_limits": {"type": "inequality", "description": "发电机边界"},
                    "line_loading": {"type": "inequality", "description": "线路容量"}
                }
            }
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}
        }


def create_default_registry(env=None) -> ToolRegistry:
    """
    创建包含默认工具的注册表
    
    Args:
        env: Grid2Op 环境实例（可选）
    
    Returns:
        配置好的工具注册表
    """
    registry = ToolRegistry()
    
    # 注册结束工具
    registry.register(FinishTool())
    
    # 注册分析工具
    registry.register(EnvironmentFeatureAnalysisTool(env))  # 新增：环境特征分析
    registry.register(GridStatusAnalysisTool(env))
    registry.register(OverflowRiskAnalysisTool(env))
    registry.register(GeneratorCapacityAnalysisTool(env))
    registry.register(LoadTrendAnalysisTool(env))
    
    return registry
