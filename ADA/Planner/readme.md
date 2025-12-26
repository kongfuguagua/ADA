# Planner Agent 设计规范

## 1. 核心职责
Planner 是系统的“大脑”，负责解决动态场景下的歧义性问题。它不直接求解，而是定义问题。它通过主动调用工具链（Tools）来消除环境不确定性，生成严格的数学规划模型。

## 2. 核心逻辑：主动状态增广 (Active State Augmentation)
过程建模为链式推导：$x_0 \xrightarrow{t_0} x_1 \dots \xrightarrow{t_n} x_n$。

1.  **感知 (Perception)**: 接收原始环境状态 $x_0$。
2.  **检索 (Retrieval)**: 从 `knowledgebase` 中检索相关的动作知识 ($\mathcal{AK}$)。
3.  **增广 (Augmentation)**:
    * 基于 $\mathcal{AK}$ 决策是否需要调用工具（如天气预报、潮流计算）。
    * 执行工具调用 $t_i$，更新状态向量 $x_i$。
4.  **建模 (Formulation)**:
    * 基于最终状态 $x_n$ 和任务知识 ($\mathcal{TK}$)，映射出优化元组 $\langle \mathcal{J}, \Theta \rangle$。
    * 输出标准化的 `OptimizationProblem` 对象。

## 3. 模块实现细节
* **`planner.py`**: 主类，维护状态机。
* **`prompt.py`**:
    * `STATE_AUGMENTATION_PROMPT`: 指导 LLM 选择工具。
    * `FORMULATION_PROMPT`: 指导 LLM 生成数学公式（Latex/Python 格式）。
* **交互接口**:
    * 输入: `SystemStatus`
    * 依赖: `KnowledgeService` (查 TK/AK), `ToolServer` (调工具)
    * 输出: `OptimizationProblem`

## 4. 异常处理
* 若收到 Judger 的 `MODEL_ERROR` 反馈，Planner 需读取反馈中的 `error_diagnosis`，在下一次 Prompt 中增加“修正约束”的指令，进行 Self-Correction。


## 5.关键结构体定义
优化问题模板（待完善）
```python
class OptPbm:
    Object:iter
    constant:iter
```

求解算法模板（待完善）
```python
class Tool(ABC):
    def __self__():
        pass

    def invoke(x):
        pass
```


## 6.参考材料

Planner 采用一种主动状态增广（Active State Augmentation）机制。依据动作知识库 $\mathcal{AK}$ 激活工具$t_i$进行状态校准。我们将此过程建模为一个受控的状态演化链条：
\begin{equation}
  x_0 \xrightarrow{t_0} x_1 \xrightarrow{t_1} x_2 \xrightarrow{t_2} \dots \xrightarrow{t_n} x_n
\end{equation}
其中 $t_i \in \mathcal{T}$ 代表对潮流分析器、气象标定工具或历史数据库的调用，调用的逻辑和依据由动作知识 $\mathcal{AK}$ 决定，而 $x_i$ 是扩充后的状态向量。

在获得增强状态 $x_n$ 后，Planner在任务知识 $\mathcal{TK}$ 的指导下动态选择最适配当前运行方式的优化目标分量 $\mathcal{J}$ 和约束条件 $\Theta$。具体的转化逻辑遵循如下映射映射：
\begin{equation}
\mathcal{X} \xrightarrow{t_{\mathcal{AK}} \subseteq \mathcal{T}} \mathcal{X}_{\mathcal{AK}} \xrightarrow{\mathcal{TK}} \langle \mathcal{J}, \Theta \rangle
\end{equation}