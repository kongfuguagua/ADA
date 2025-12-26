# Solver Agent 设计规范

## 1. 核心职责

Solver 是系统的**计算引擎**，负责解决**调度目标优先级与求解算法的场景适应性**问题（论文 2.2 节）。它接收 Planner 定义的数学问题，通过特征分析自动匹配最适合的优化算法进行求解。

## 2. 论文需求响应表

| 论文需求 | 实现状态 | 实现位置 |
|---------|---------|---------|
| 问题-算法对齐 (Problem-Algorithm Alignment) | ✅ 已实现 | `matcher.py: AlgorithmMatcher` |
| 特征向量提取 φ | ✅ 已实现 | `feature.py: ProblemFeatureExtractor` |
| 算法能力向量 ψ(A) | ✅ 已实现 | `Template/base.py: get_capability_vector()` |
| BM25 风格对齐评分 G(A,φ) | ✅ 已实现 | `matcher.py: _alignment_score()` |
| IDF 风格特征权重 w(fᵢ) | ✅ 已实现 | `matcher.py: _get_feature_weights()` |
| 多算法策略模式 | ✅ 已实现 | `Template/*.py` |
| 收敛轨迹记录 | ✅ 已实现 | `Solution.convergence_curve` |

## 3. 核心逻辑：问题-算法对齐

### 3.1 特征提取

将输入的优化元组 $\langle \mathcal{J}, \Theta \rangle$ 映射为 **5 维特征向量** $\phi \in [0,1]^5$：

| 维度 | 特征 | 含义 | 计算方式 |
|-----|------|------|---------|
| $f_1$ | non_convexity_score | 非凸性程度 | 表达式模式分析 |
| $f_2$ | non_linearity_score | 非线性程度 | 非线性项占比 |
| $f_3$ | constraint_stiffness | 约束紧迫度 | 约束密度 + 边界紧度 |
| $f_4$ | discreteness_score | 离散性程度 | 整数/二元变量占比 |
| $f_5$ | scale_score | 规模复杂度 | 变量数归一化 |

### 3.2 算法匹配

每个算法 $A$ 维护 **5 维能力向量** $\psi(A) \in [0,1]^5$：

| 维度 | 能力 | 说明 |
|-----|------|------|
| $\psi_0$ | convex_handling | 凸问题处理能力 |
| $\psi_1$ | non_convex_handling | 非凸问题处理能力 |
| $\psi_2$ | constraint_handling | 约束处理能力 |
| $\psi_3$ | speed | 求解速度 |
| $\psi_4$ | global_optimality | 全局最优性保证 |

对齐评分函数（BM25 风格）：

$$G(A, \phi) = \sum_{i} w(f_i) \cdot \frac{\psi_j(A) \cdot (k_1 + 1)}{\psi_j(A) + k_1 \cdot (1 - b + b \cdot f_i)}$$

特征权重（IDF 风格）：

$$w(f_i) = \ln \left( 1 + \frac{\sum_{A'} \sum_j \psi_j(A')}{\sum_A \psi_i(A) + \epsilon} \right)$$

## 4. 模块架构

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

## 5. 接口定义

### 输入: OptimizationProblem

```python
class OptimizationProblem(BaseModel):
    objective_function_latex: str
    objective_function_code: str
    constraints_latex: List[str]
    constraints_code: List[str]
    variables: List[VariableDefinition]
    parameters: Dict[str, float]
    is_minimization: bool = True
```

### 输出: Solution

```python
class Solution(BaseModel):
    is_feasible: bool
    algorithm_used: str
    decision_variables: Dict[str, float]
    objective_value: float
    solving_time: float
    convergence_curve: List[float]
    solver_message: str
```

## 6. 算法库概览

| 算法 | 适用场景 | 能力向量 | 依赖 |
|-----|---------|---------|------|
| ConvexOptimizer | 凸优化、光滑问题 | [0.9, 0.2, 0.7, 0.8, 0.9] | scipy |
| GurobiOptimizer | LP、MILP、QP | [0.9, 0.5, 0.9, 0.7, 0.95] | gurobipy |
| PSOOptimizer | 非凸连续优化 | [0.3, 0.8, 0.5, 0.5, 0.6] | 无 |
| BayesianOptimizer | 黑盒优化 | [0.4, 0.7, 0.4, 0.3, 0.5] | scikit-optimize |
| GeneticOptimizer | 组合优化、离散问题 | [0.2, 0.7, 0.6, 0.4, 0.7] | 无 |
| GradientOptimizer | 可微分问题 | [0.7, 0.3, 0.4, 0.9, 0.4] | scipy |

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
