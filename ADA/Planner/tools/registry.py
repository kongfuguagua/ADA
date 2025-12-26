# -*- coding: utf-8 -*-
"""
工具注册表
管理 Planner 可用的所有工具
"""

import sys
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from abc import ABC, abstractmethod

# 添加项目根目录到路径
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
        """
        注册工具
        
        Args:
            tool: 工具实例
        """
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")
    
    def register_function(
        self, 
        name: str, 
        func: Callable, 
        description: str,
        parameters: Dict[str, Any] = None
    ) -> None:
        """
        注册函数为工具
        
        Args:
            name: 工具名称
            func: 可调用函数
            description: 工具描述
            parameters: 参数定义
        """
        tool = FunctionTool(name, func, description, parameters)
        self.register(tool)
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
        
        Returns:
            工具执行结果
        """
        if tool_name not in self._tools:
            error_msg = f"工具 '{tool_name}' 未注册"
            logger.error(error_msg)
            return {"error": error_msg}
        
        tool = self._tools[tool_name]
        
        try:
            logger.info(f"执行工具: {tool_name}", params=str(kwargs)[:100])
            result = tool.execute(**kwargs)
            logger.info(f"工具执行完成: {tool_name}", result=str(result)[:100])
            return result
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)
    
    def get_tool_descriptions(self) -> str:
        """
        生成给 LLM 看的工具列表描述
        
        Returns:
            格式化的工具描述字符串
        """
        if not self._tools:
            return "暂无可用工具"
        
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
            
            # 添加参数说明
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
        """
        获取 OpenAI 格式的工具定义
        
        Returns:
            工具定义列表
        """
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


# ============= 预定义工具 =============

class WeatherForecastTool(BaseTool):
    """天气预报工具（模拟）"""
    
    @property
    def name(self) -> str:
        return "weather_forecast"
    
    @property
    def description(self) -> str:
        return "获取指定位置的天气预报信息，用于预测负载变化"
    
    def execute(self, location: str = "default", hours: int = 24) -> Dict[str, Any]:
        """模拟天气预报"""
        return {
            "location": location,
            "forecast": [
                {"hour": i, "temperature": 20 + (i % 12), "humidity": 60 + (i % 20)}
                for i in range(hours)
            ],
            "summary": f"{location} 未来 {hours} 小时天气预报：晴转多云，气温 18-32°C"
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "位置名称"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "预报小时数"
                    }
                },
                "required": []
            }
        }


class PowerFlowTool(BaseTool):
    """潮流计算工具（模拟）"""
    
    @property
    def name(self) -> str:
        return "power_flow"
    
    @property
    def description(self) -> str:
        return "计算电力系统潮流，返回各节点电压和线路功率"
    
    def execute(self, load_data: Dict[str, float] = None) -> Dict[str, Any]:
        """模拟潮流计算"""
        load_data = load_data or {"bus_1": 100.0, "bus_2": 80.0}
        
        return {
            "converged": True,
            "iterations": 5,
            "bus_voltages": {
                bus: 1.0 + (load % 10) / 100 
                for bus, load in load_data.items()
            },
            "line_flows": {
                "line_1_2": sum(load_data.values()) * 0.6,
                "line_2_3": sum(load_data.values()) * 0.4
            },
            "total_loss": sum(load_data.values()) * 0.02
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "load_data": {
                        "type": "object",
                        "description": "各节点负载数据 {节点名: 负载值}"
                    }
                },
                "required": []
            }
        }


class LoadForecastTool(BaseTool):
    """负载预测工具（模拟）"""
    
    @property
    def name(self) -> str:
        return "load_forecast"
    
    @property
    def description(self) -> str:
        return "预测未来时段的电力负载"
    
    def execute(self, base_load: float = 100.0, hours: int = 24) -> Dict[str, Any]:
        """模拟负载预测"""
        import math
        
        forecast = []
        for h in range(hours):
            # 模拟日负载曲线
            factor = 0.8 + 0.4 * math.sin((h - 6) * math.pi / 12)
            forecast.append({
                "hour": h,
                "load": base_load * factor,
                "confidence": 0.95 - h * 0.01
            })
        
        return {
            "base_load": base_load,
            "forecast": forecast,
            "peak_hour": 14,
            "peak_load": base_load * 1.2,
            "valley_hour": 4,
            "valley_load": base_load * 0.6
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "base_load": {
                        "type": "number",
                        "description": "基础负载值 (MW)"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "预测小时数"
                    }
                },
                "required": []
            }
        }


def create_default_registry() -> ToolRegistry:
    """创建包含默认工具的注册表"""
    registry = ToolRegistry()
    registry.register(WeatherForecastTool())
    registry.register(PowerFlowTool())
    registry.register(LoadForecastTool())
    return registry


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 ToolRegistry:")
    
    # 创建注册表
    registry = create_default_registry()
    
    print(f"已注册工具数量: {len(registry)}")
    print(f"工具列表: {registry.list_tools()}")
    
    print("\n工具描述:")
    print(registry.get_tool_descriptions())
    
    print("\n测试工具执行:")
    
    # 天气预报
    result = registry.execute("weather_forecast", location="北京", hours=6)
    print(f"天气预报: {result['summary']}")
    
    # 潮流计算
    result = registry.execute("power_flow", load_data={"bus_1": 100, "bus_2": 80})
    print(f"潮流计算: 收敛={result['converged']}, 损耗={result['total_loss']:.2f}MW")
    
    # 负载预测
    result = registry.execute("load_forecast", base_load=100, hours=24)
    print(f"负载预测: 峰值={result['peak_load']:.1f}MW@{result['peak_hour']}时")
    
    # 测试不存在的工具
    result = registry.execute("unknown_tool")
    print(f"未知工具: {result}")
