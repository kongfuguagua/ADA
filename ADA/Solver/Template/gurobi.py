# -*- coding: utf-8 -*-
"""
Gurobi 优化求解器
适用于线性规划、二次规划、混合整数规划等问题
"""

from typing import Dict, Any, List, Optional

from utils.const import OptimizationProblem, SolverAlgorithmMeta, VariableType
from utils.logger import get_logger
from .base import BaseAlgorithm

logger = get_logger("Gurobi")


class GurobiOptimizer(BaseAlgorithm):
    """
    Gurobi 优化求解器
    
    支持问题类型：
    - 线性规划 (LP)
    - 二次规划 (QP)
    - 混合整数线性规划 (MILP)
    - 混合整数二次规划 (MIQP)
    
    需要安装: pip install gurobipy
    并配置有效的 Gurobi 许可证
    """
    
    def __init__(
        self,
        time_limit: float = 300.0,
        mip_gap: float = 0.01,
        threads: int = 0,
        verbose: bool = False
    ):
        """
        初始化 Gurobi 优化器
        
        Args:
            time_limit: 求解时间限制（秒）
            mip_gap: MIP 相对间隙阈值
            threads: 线程数（0 表示自动）
            verbose: 是否输出求解日志
        """
        super().__init__()
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.verbose = verbose
        
        # 检查 Gurobi 是否可用
        self._gurobi_available = self._check_gurobi()
    
    def _check_gurobi(self) -> bool:
        """检查 Gurobi 是否可用"""
        try:
            import gurobipy as gp
            # 尝试创建一个简单模型测试许可证
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
            return True
        except ImportError:
            logger.warning("gurobipy 未安装，请运行: pip install gurobipy")
            return False
        except Exception as e:
            logger.warning(f"Gurobi 许可证无效或未配置: {e}")
            return False
    
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="GurobiOptimizer",
            description="Gurobi 商业求解器，适用于 LP/QP/MILP/MIQP 问题",
            capabilities={
                "convex_handling": 1.0,       # 凸问题处理能力极强
                "non_convex_handling": 0.4,   # 非凸问题需要特殊处理
                "constraint_handling": 1.0,   # 约束处理能力极强
                "speed": 0.95,                # 求解速度极快
                "global_optimality": 0.9,     # LP/MILP 保证全局最优
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Gurobi 求解实现"""
        
        # 检查 Gurobi 可用性
        if not self._gurobi_available:
            return self._fallback_solve(problem)
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return {
                "feasible": False,
                "message": "gurobipy 未安装，请运行: pip install gurobipy"
            }
        
        var_names = problem.get_variable_names()
        n_vars = len(var_names)
        
        if n_vars == 0:
            return {"feasible": False, "message": "无决策变量"}
        
        try:
            # 创建模型
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 1 if self.verbose else 0)
                env.start()
                
                with gp.Model("optimization", env=env) as model:
                    # 设置参数
                    model.Params.TimeLimit = self.time_limit
                    model.Params.MIPGap = self.mip_gap
                    if self.threads > 0:
                        model.Params.Threads = self.threads
                    
                    # 创建变量
                    gurobi_vars = self._create_variables(model, problem, var_names, GRB)
                    
                    # 设置目标函数
                    obj_expr = self._build_objective(problem, gurobi_vars, var_names)
                    if problem.is_minimization:
                        model.setObjective(obj_expr, GRB.MINIMIZE)
                    else:
                        model.setObjective(obj_expr, GRB.MAXIMIZE)
                    
                    # 添加约束
                    self._add_constraints(model, problem, gurobi_vars, var_names)
                    
                    # 求解
                    model.optimize()
                    
                    # 处理结果
                    return self._extract_solution(model, gurobi_vars, var_names, problem, GRB)
        
        except Exception as e:
            logger.error(f"Gurobi 求解异常: {e}")
            return {
                "feasible": False,
                "message": f"Gurobi 求解异常: {str(e)}"
            }
    
    def _create_variables(
        self,
        model,
        problem: OptimizationProblem,
        var_names: List[str],
        GRB
    ) -> Dict[str, Any]:
        """创建 Gurobi 变量"""
        gurobi_vars = {}
        
        for var_def in problem.variables:
            name = var_def.name
            lb = var_def.lower_bound if var_def.lower_bound != float('-inf') else -1e10
            ub = var_def.upper_bound if var_def.upper_bound != float('inf') else 1e10
            
            # 确定变量类型
            if var_def.type == VariableType.BINARY:
                vtype = GRB.BINARY
                lb, ub = 0, 1
            elif var_def.type == VariableType.INTEGER:
                vtype = GRB.INTEGER
            else:
                vtype = GRB.CONTINUOUS
            
            gurobi_vars[name] = model.addVar(
                lb=lb, ub=ub, vtype=vtype, name=name
            )
        
        model.update()
        return gurobi_vars
    
    def _build_objective(
        self,
        problem: OptimizationProblem,
        gurobi_vars: Dict[str, Any],
        var_names: List[str]
    ):
        """构建目标函数表达式"""
        code = problem.objective_function_code
        
        if not code:
            # 默认：变量之和
            import gurobipy as gp
            return gp.quicksum(gurobi_vars.values())
        
        # 尝试解析简单的线性/二次表达式
        try:
            return self._parse_expression(code, gurobi_vars, problem.parameters)
        except Exception as e:
            logger.warning(f"目标函数解析失败，使用默认: {e}")
            import gurobipy as gp
            return gp.quicksum(gurobi_vars.values())
    
    def _parse_expression(
        self,
        expr_code: str,
        gurobi_vars: Dict[str, Any],
        parameters: Dict[str, float]
    ):
        """
        解析表达式为 Gurobi 表达式
        支持简单的线性和二次表达式
        """
        import gurobipy as gp
        
        # 创建安全的执行环境
        safe_env = {
            **gurobi_vars,
            **parameters,
            'sum': lambda x: gp.quicksum(x),
            'quicksum': gp.quicksum,
        }
        
        # 尝试直接求值
        try:
            result = eval(expr_code, {"__builtins__": {}}, safe_env)
            return result
        except Exception:
            # 如果失败，返回简单的线性组合
            return gp.quicksum(gurobi_vars.values())
    
    def _add_constraints(
        self,
        model,
        problem: OptimizationProblem,
        gurobi_vars: Dict[str, Any],
        var_names: List[str]
    ) -> None:
        """添加约束条件"""
        import gurobipy as gp
        
        for i, constraint_code in enumerate(problem.constraints_code):
            if not constraint_code.strip():
                continue
            
            try:
                # 解析约束类型
                if '<=' in constraint_code:
                    lhs, rhs = constraint_code.split('<=')
                    sense = '<='
                elif '>=' in constraint_code:
                    lhs, rhs = constraint_code.split('>=')
                    sense = '>='
                elif '==' in constraint_code:
                    lhs, rhs = constraint_code.split('==')
                    sense = '=='
                elif '=' in constraint_code and '<' not in constraint_code and '>' not in constraint_code:
                    lhs, rhs = constraint_code.split('=')
                    sense = '=='
                else:
                    continue
                
                # 解析左右表达式
                lhs_expr = self._parse_expression(lhs.strip(), gurobi_vars, problem.parameters)
                rhs_expr = self._parse_expression(rhs.strip(), gurobi_vars, problem.parameters)
                
                # 添加约束
                if sense == '<=':
                    model.addConstr(lhs_expr <= rhs_expr, name=f"c{i}")
                elif sense == '>=':
                    model.addConstr(lhs_expr >= rhs_expr, name=f"c{i}")
                else:
                    model.addConstr(lhs_expr == rhs_expr, name=f"c{i}")
                    
            except Exception as e:
                logger.warning(f"约束 {i} 解析失败: {e}")
    
    def _extract_solution(
        self,
        model,
        gurobi_vars: Dict[str, Any],
        var_names: List[str],
        problem: OptimizationProblem,
        GRB
    ) -> Dict[str, Any]:
        """提取求解结果"""
        
        status = model.Status
        
        if status == GRB.OPTIMAL:
            variables = {name: var.X for name, var in gurobi_vars.items()}
            return {
                "feasible": True,
                "variables": variables,
                "objective": model.ObjVal,
                "message": "最优解"
            }
        
        elif status == GRB.INFEASIBLE:
            return {
                "feasible": False,
                "message": "问题不可行"
            }
        
        elif status == GRB.UNBOUNDED:
            return {
                "feasible": False,
                "message": "问题无界"
            }
        
        elif status == GRB.TIME_LIMIT:
            # 超时但可能有可行解
            if model.SolCount > 0:
                variables = {name: var.X for name, var in gurobi_vars.items()}
                return {
                    "feasible": True,
                    "variables": variables,
                    "objective": model.ObjVal,
                    "message": f"求解超时，返回当前最优解 (Gap: {model.MIPGap:.2%})"
                }
            return {
                "feasible": False,
                "message": "求解超时，未找到可行解"
            }
        
        else:
            return {
                "feasible": False,
                "message": f"求解状态: {status}"
            }
    
    def _fallback_solve(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """
        Gurobi 不可用时的备选求解方案
        使用 scipy 进行求解
        """
        logger.info("Gurobi 不可用，使用 scipy 备选方案")
        
        try:
            from scipy.optimize import linprog, minimize
            import numpy as np
        except ImportError:
            return {
                "feasible": False,
                "message": "Gurobi 和 scipy 均不可用"
            }
        
        var_names = problem.get_variable_names()
        n_vars = len(var_names)
        
        if n_vars == 0:
            return {"feasible": False, "message": "无决策变量"}
        
        # 获取边界
        bounds_dict = problem.get_variable_bounds()
        bounds = [
            (
                b[0] if b[0] != float('-inf') else -1e10,
                b[1] if b[1] != float('inf') else 1e10
            )
            for b in [bounds_dict[v] for v in var_names]
        ]
        
        # 构建目标函数
        obj_func = self.parse_objective_function(problem)
        
        def scipy_objective(x):
            x_dict = {var_names[i]: x[i] for i in range(n_vars)}
            value = obj_func(x_dict)
            self.record_convergence(value)
            return value if problem.is_minimization else -value
        
        # 初始点
        x0 = np.array([
            (bounds[i][0] + bounds[i][1]) / 2
            for i in range(n_vars)
        ])
        
        # 求解
        result = minimize(
            scipy_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={"maxiter": 1000}
        )
        
        variables = {var_names[i]: float(result.x[i]) for i in range(n_vars)}
        objective_value = result.fun if problem.is_minimization else -result.fun
        
        return {
            "feasible": result.success,
            "variables": variables,
            "objective": objective_value,
            "message": f"备选求解器 (scipy): {result.message}"
        }


# ============= 测试代码 =============
if __name__ == "__main__":
    from utils.const import VariableDefinition
    
    print("=" * 60)
    print("测试 GurobiOptimizer")
    print("=" * 60)
    
    optimizer = GurobiOptimizer(verbose=False)
    print(f"\nGurobi 可用: {optimizer._gurobi_available}")
    print(f"算法能力: {optimizer.get_capability_vector()}")
    
    # 测试1: 简单线性规划
    print("\n" + "-" * 40)
    print("测试1: 线性规划 min 2x1 + 3x2")
    print("-" * 40)
    
    lp_problem = OptimizationProblem(
        objective_function_latex=r"\min 2x_1 + 3x_2",
        objective_function_code="2*x1 + 3*x2",
        constraints_latex=[r"x_1 + x_2 \geq 1"],
        constraints_code=["x1 + x2 >= 1"],
        variables=[
            VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
        ]
    )
    
    solution = optimizer.solve(lp_problem)
    print(f"可行: {solution.is_feasible}")
    print(f"变量: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value}")
    print(f"消息: {solution.solver_message}")
    
    # 测试2: 混合整数规划
    print("\n" + "-" * 40)
    print("测试2: 混合整数规划")
    print("-" * 40)
    
    mip_problem = OptimizationProblem(
        objective_function_latex=r"\min x_1 + 2x_2 + 3x_3",
        objective_function_code="x1 + 2*x2 + 3*x3",
        constraints_latex=[r"x_1 + x_2 + x_3 \geq 1"],
        constraints_code=["x1 + x2 + x3 >= 1"],
        variables=[
            VariableDefinition(name="x1", type=VariableType.BINARY),
            VariableDefinition(name="x2", type=VariableType.BINARY),
            VariableDefinition(name="x3", type=VariableType.INTEGER, lower_bound=0, upper_bound=5),
        ]
    )
    
    solution = optimizer.solve(mip_problem)
    print(f"可行: {solution.is_feasible}")
    print(f"变量: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value}")
    print(f"消息: {solution.solver_message}")
    
    # 测试3: 二次规划
    print("\n" + "-" * 40)
    print("测试3: 二次规划 min x1² + x2²")
    print("-" * 40)
    
    qp_problem = OptimizationProblem(
        objective_function_latex=r"\min x_1^2 + x_2^2",
        objective_function_code="x1*x1 + x2*x2",
        constraints_latex=[r"x_1 + x_2 = 1"],
        constraints_code=["x1 + x2 == 1"],
        variables=[
            VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
            VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
        ]
    )
    
    solution = optimizer.solve(qp_problem)
    print(f"可行: {solution.is_feasible}")
    print(f"变量: {solution.decision_variables}")
    print(f"目标值: {solution.objective_value}")
    print(f"消息: {solution.solver_message}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

