# Solver Agent 设计规范

## 1. 核心职责

Solver 是系统的**计算引擎**。它接收 Planner 定义的数学问题，通过特征分析自动匹配最适合的优化算法进行求解。

## 2. 核心逻辑：问题-算法对齐 (Problem-Algorithm Alignment)

### 2.1 特征提取 (Feature Extraction)

将输入的优化元组 $\langle \mathcal{J}, \Theta \rangle$ 映射为特征向量 $\phi$：

$$\phi(\langle \mathcal{J}, \Theta \rangle) = [f_1, \dots, f_n]^\top$$

其中 $f_i \in [0, 1]$ 量化了优化问题的各项特征：

| 特征维度 | 含义 | 计算方式 |
|---------|------|---------|
| $f_1$ | 非凸性 (Non-convexity) | 基于表达式模式分析 |
| $f_2$ | 非线性程度 (Non-linearity) | 非线性项占比 |
| $f_3$ | 约束紧迫度 (Constraint Stiffness) | 约束密度 + 边界紧度 |
| $f_4$ | 离散性 (Discreteness) | 整数/二元变量占比 |
| $f_5$ | 规模复杂度 (Scale) | 变量数归一化 |

### 2.2 算法匹配 (Algorithm Matching)

每个算法 $A \in \mathcal{A}$ 维护能力向量 $\psi(A)$ 表征其归纳偏置。对齐评分函数：

$$G(A, \phi) = \sum_{f_i \in \phi} w(f_i) \cdot \frac{\psi_i(A) \cdot (k_1 + 1)}{\psi_i(A) + K(A)}$$

其中特征权重采用 IDF 风格计算：

$$w(f_i) = \ln \left( 1 + \frac{\sum_{A' \in \mathcal{A}} \sum_{j=1}^n \psi_j(A')}{\sum_{A \in \mathcal{A}} \psi_i(A) + \epsilon} \right)$$

### 2.3 执行求解 (Execution)

实例化选定算法，运行求解过程，记录收敛轨迹。

## 3. 模块架构

```
Solver/
├── __init__.py          # 模块导出
├── readme.md            # 本文档
│
├── solver.py            # 【主入口】SolverAgent 类
├── matcher.py           # 【算法匹配】对齐评分计算
├── feature.py           # 【特征提取】问题特征分析
├── prompt.py            # 【Prompt】LLM 提示模板
│
└── Template/            # 【算法库】策略模式实现
    ├── __init__.py      # 算法导出
    ├── base.py          # 基类 BaseAlgorithm
    ├── convex.py        # 凸优化 (scipy)
    ├── gurobi.py        # 混合整数规划 (Gurobi)
    ├── pso.py           # 粒子群优化
    ├── bayesian.py      # 贝叶斯优化
    ├── genetic.py       # 遗传算法
    └── gradient.py      # 梯度下降
```

## 4. 接口数据定义

### 4.1 输入: OptimizationProblem

```python
class OptimizationProblem(BaseModel):
    objective_function_latex: str   # 目标函数 LaTeX
    objective_function_code: str    # 目标函数 Python 代码
    constraints_latex: List[str]    # 约束条件 LaTeX
    constraints_code: List[str]     # 约束条件代码
    variables: List[VariableDefinition]  # 变量定义
    parameters: Dict[str, float]    # 常数参数
    is_minimization: bool = True    # 是否最小化
```

### 4.2 输出: Solution

```python
class Solution(BaseModel):
    is_feasible: bool               # 是否可行
    algorithm_used: str             # 使用的算法
    decision_variables: Dict[str, float]  # 最优解
    objective_value: float          # 目标值
    solving_time: float             # 求解时间
    convergence_curve: List[float]  # 收敛曲线
    solver_message: str             # 求解器消息
```

## 5. 算法能力向量定义

每个算法需定义 5 维能力向量 $\psi(A) \in [0,1]^5$：

| 维度 | 能力 | 说明 |
|-----|------|------|
| $\psi_0$ | convex_handling | 凸问题处理能力 |
| $\psi_1$ | non_convex_handling | 非凸问题处理能力 |
| $\psi_2$ | constraint_handling | 约束处理能力 |
| $\psi_3$ | speed | 求解速度 |
| $\psi_4$ | global_optimality | 全局最优性保证 |

## 6. 算法库概览

| 算法 | 适用场景 | 依赖 |
|-----|---------|------|
| ConvexOptimizer | 凸优化、光滑问题 | scipy |
| GurobiOptimizer | 线性规划、混合整数规划 | gurobipy |
| PSOOptimizer | 非凸连续优化 | 无 |
| BayesianOptimizer | 黑盒优化、昂贵函数 | scikit-optimize |
| GeneticOptimizer | 组合优化、离散问题 | 无 |
| GradientOptimizer | 可微分问题 | scipy |

## 7. 使用示例

```python
from Solver import SolverAgent
from utils.const import OptimizationProblem, VariableDefinition

# 创建 Solver
solver = SolverAgent()

# 定义问题
problem = OptimizationProblem(
    objective_function_latex=r"\min x_1^2 + x_2^2",
    objective_function_code="x1**2 + x2**2",
    variables=[
        VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
        VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
    ]
)

# 求解（自动选择算法）
solution = solver.solve(problem)

print(f"算法: {solution.algorithm_used}")
print(f"最优解: {solution.decision_variables}")
print(f"目标值: {solution.objective_value}")
```

## 8. 扩展新算法

继承 `BaseAlgorithm` 并实现 `meta` 和 `_solve_impl`：

```python
from Solver.Template.base import BaseAlgorithm

class MyOptimizer(BaseAlgorithm):
    @property
    def meta(self) -> SolverAlgorithmMeta:
        return SolverAlgorithmMeta(
            name="MyOptimizer",
            description="我的优化算法",
            capabilities={
                "convex_handling": 0.8,
                "non_convex_handling": 0.6,
                "constraint_handling": 0.7,
                "speed": 0.9,
                "global_optimality": 0.5,
            }
        )
    
    def _solve_impl(self, problem: OptimizationProblem) -> Dict[str, Any]:
        # 实现求解逻辑
        return {
            "feasible": True,
            "variables": {...},
            "objective": 0.0,
            "message": "求解完成"
        }
```
