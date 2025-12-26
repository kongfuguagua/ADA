# -*- coding: utf-8 -*-
"""
Grid2Op 环境交互工具
负责与 Grid2Op 环境进行直接交互：发送命令、获取原始数据

区别于 Planner/tools（分析工具）：
- 环境工具：发送命令、获取原始数据
- 分析工具：统计数据、规则检测、趋势分析
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.interact import BaseTool


class BaseEnvTool(BaseTool):
    """环境交互工具基类"""
    
    def __init__(self, env=None):
        """
        初始化工具
        
        Args:
            env: Grid2OpEnvironment 实例
        """
        self._env = env
    
    def set_env(self, env):
        """设置环境实例"""
        self._env = env
    
    @property
    def env(self):
        """获取环境实例"""
        return self._env
    
    def _check_env(self) -> Optional[str]:
        """检查环境是否可用"""
        if self._env is None:
            return "环境未初始化"
        if not hasattr(self._env, 'current_obs') or self._env.current_obs is None:
            return "环境未 reset"
        return None


class GetObservationTool(BaseEnvTool):
    """获取当前观测数据"""
    
    @property
    def name(self) -> str:
        return "get_observation"
    
    @property
    def description(self) -> str:
        return "获取当前电网的完整观测数据"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """获取观测"""
        error = self._check_env()
        if error:
            return {"error": error}
        
        obs = self._env.current_obs
        
        return {
            "timestep": int(obs.current_step),
            "load_p": obs.load_p.tolist(),
            "load_q": obs.load_q.tolist(),
            "gen_p": obs.gen_p.tolist(),
            "gen_q": obs.gen_q.tolist(),
            "rho": obs.rho.tolist(),
            "line_status": obs.line_status.tolist(),
            "topo_vect": obs.topo_vect.tolist(),
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}
        }


class SimulateActionTool(BaseEnvTool):
    """模拟动作效果（不实际执行）"""
    
    @property
    def name(self) -> str:
        return "simulate_action"
    
    @property
    def description(self) -> str:
        return "模拟执行动作后的电网状态，返回预测结果"
    
    def execute(
        self,
        redispatch: Dict[int, float] = None,
        set_line_status: Dict[int, int] = None,
        change_bus: Dict[str, int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """模拟动作"""
        error = self._check_env()
        if error:
            return {"error": error}
        
        obs = self._env.current_obs
        action_space = self._env.env.action_space
        action = action_space({})
        
        # 构建动作
        try:
            if redispatch:
                for gen_id, delta_p in redispatch.items():
                    action.redispatch = [(int(gen_id), float(delta_p))]
            
            if set_line_status:
                for line_id, status in set_line_status.items():
                    action.set_line_status = [(int(line_id), int(status))]
            
            if change_bus:
                # change_bus 格式: {"load_0": 2, "gen_1": 1}
                for element, bus in change_bus.items():
                    parts = element.split("_")
                    if len(parts) == 2:
                        elem_type, elem_id = parts[0], int(parts[1])
                        if elem_type == "load":
                            action.set_bus = {"loads_id": [(elem_id, int(bus))]}
                        elif elem_type == "gen":
                            action.set_bus = {"generators_id": [(elem_id, int(bus))]}
            
            # 模拟
            sim_obs, sim_reward, sim_done, sim_info = obs.simulate(action)
            
            return {
                "success": True,
                "reward": float(sim_reward),
                "done": bool(sim_done),
                "max_rho_before": float(obs.rho.max()),
                "max_rho_after": float(sim_obs.rho.max()),
                "overflow_before": int((obs.rho > 1.0).sum()),
                "overflow_after": int((sim_obs.rho > 1.0).sum()),
                "is_safe": bool(sim_obs.rho.max() <= 1.0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "redispatch": {
                        "type": "object",
                        "description": "再调度 {发电机ID: 调整量MW}"
                    },
                    "set_line_status": {
                        "type": "object",
                        "description": "线路状态 {线路ID: 状态(1开/0关)}"
                    },
                    "change_bus": {
                        "type": "object",
                        "description": "母线切换 {元件名_ID: 目标母线}"
                    }
                },
                "required": []
            }
        }


class ExecuteActionTool(BaseEnvTool):
    """执行动作（实际改变环境）"""
    
    @property
    def name(self) -> str:
        return "execute_action"
    
    @property
    def description(self) -> str:
        return "在环境中执行动作，返回执行结果"
    
    def execute(
        self,
        redispatch: Dict[int, float] = None,
        set_line_status: Dict[int, int] = None,
        do_nothing: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """执行动作"""
        error = self._check_env()
        if error:
            return {"error": error}
        
        action_space = self._env.env.action_space
        
        if do_nothing:
            action = action_space({})
        else:
            action = action_space({})
            
            if redispatch:
                for gen_id, delta_p in redispatch.items():
                    action.redispatch = [(int(gen_id), float(delta_p))]
            
            if set_line_status:
                for line_id, status in set_line_status.items():
                    action.set_line_status = [(int(line_id), int(status))]
        
        # 执行
        try:
            obs, reward, done, info = self._env.env.step(action)
            self._env.current_obs = obs
            
            return {
                "success": True,
                "reward": float(reward),
                "done": bool(done),
                "max_rho": float(obs.rho.max()),
                "overflow_count": int((obs.rho > 1.0).sum()),
                "info": str(info) if info else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "redispatch": {
                        "type": "object",
                        "description": "再调度 {发电机ID: 调整量MW}"
                    },
                    "set_line_status": {
                        "type": "object",
                        "description": "线路状态 {线路ID: 状态(1开/0关)}"
                    },
                    "do_nothing": {
                        "type": "boolean",
                        "description": "是否执行空动作"
                    }
                },
                "required": []
            }
        }


class GetGridInfoTool(BaseEnvTool):
    """获取电网拓扑信息"""
    
    @property
    def name(self) -> str:
        return "get_grid_info"
    
    @property
    def description(self) -> str:
        return "获取电网的静态拓扑信息"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """获取电网信息"""
        if self._env is None:
            return {"error": "环境未初始化"}
        
        env = self._env.env
        
        return {
            "n_gen": int(env.n_gen),
            "n_load": int(env.n_load),
            "n_line": int(env.n_line),
            "n_sub": int(env.n_sub),
            "gen_names": list(env.name_gen),
            "load_names": list(env.name_load),
            "line_names": list(env.name_line),
            "gen_redispatchable": env.gen_redispatchable.tolist() if hasattr(env, 'gen_redispatchable') else [],
            "thermal_limits": env.get_thermal_limit().tolist(),
        }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}
        }


class GetForecastTool(BaseEnvTool):
    """获取负荷/发电预测数据"""
    
    @property
    def name(self) -> str:
        return "get_forecast"
    
    @property
    def description(self) -> str:
        return "获取未来时段的负荷和发电预测数据"
    
    def execute(self, horizon: int = 12, **kwargs) -> Dict[str, Any]:
        """获取预测"""
        error = self._check_env()
        if error:
            return {"error": error}
        
        obs = self._env.current_obs
        
        # Grid2Op 预测数据
        result = {
            "horizon": horizon,
            "current_total_load": float(obs.load_p.sum()),
            "current_total_gen": float(obs.gen_p.sum()),
        }
        
        # 如果有预测数据
        if hasattr(obs, 'get_forecasted_inj'):
            try:
                forecast_load = []
                forecast_gen = []
                for h in range(horizon):
                    f_load_p, f_load_q, f_prod_p, f_prod_v = obs.get_forecasted_inj(h)
                    forecast_load.append(float(f_load_p.sum()))
                    forecast_gen.append(float(f_prod_p.sum()))
                
                result["forecast_load"] = forecast_load
                result["forecast_gen"] = forecast_gen
            except:
                result["forecast_available"] = False
        
        return result
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "integer",
                        "description": "预测时间范围（步数）"
                    }
                },
                "required": []
            }
        }


def create_env_tools(env=None) -> List[BaseEnvTool]:
    """创建环境交互工具集"""
    return [
        GetObservationTool(env),
        SimulateActionTool(env),
        ExecuteActionTool(env),
        GetGridInfoTool(env),
        GetForecastTool(env),
    ]
