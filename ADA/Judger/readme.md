# Judger Agent 设计规范

## 1. 核心职责
Judger 是系统的“质检员”。它建立物理安全与逻辑合理的融合评价体系，并负责运行时诊断。

## 2. 核心逻辑：混合评分与反馈
1.  **物理仿真 (Physical Simulation)**:
    * 将 Solver 的解输入仿真环境（如电网仿真器）。
    * 检查是否违反硬约束（如电压越限）。
    * 计算物理指标 $R_{ENV}$（成本、稳定性）。
2.  **逻辑评估 (LLM-as-a-Judge)**:
    * LLM 评估解的可解释性和逻辑合理性，得出 $R_{LLM}$。
3.  **综合打分**:
    * 计算 $R = \alpha \cdot W \cdot R_{ENV} + (1-\alpha) \cdot R_{LLM}$。
4.  **故障溯源 (Error Diagnosis)**:
    * 若不可行（Infeasible），判断是 Planner 约束漏写（Model Error）还是 Solver 陷入局部最优（Solver Error）。

## 3. 模块实现
* **`Reward/`**:
    * `phy_reward.py`: 封装仿真环境接口。
    * `llm_reward.py`: 封装 LLM 评估 Prompt。
* **`Debug/Feedback.py`**:
    * 构建结构化反馈消息。若物理违规，提取具体的违规数值和约束项，传回 Planner。

## 4. 交互接口
* 输入: `OptimizationProblem`, `Solution`
* 输出: `Feedback` 对象 (含 `score`, `passed`, `diagnosis_report`)

## 5.代码结构为（待完善）：
```
--Reward
    --base_reward.py 评分基础类
    --phy_reward.py 物理评分组
    --llm_reward.py LLM-as-a-Judge评分组
--Debug
    --Feedback.py Planner报错和Solver报错时捕获异常，并分析原因，提供答案给对应错误智能体
--config.py 配置项，包括llm接口配置（openai）、奖励最大最小配置、judge配置
--judger.py judger主函数，获得solver结果后处理、打分、记录、返回
--prompt.py judger的prompt模板库，包括提示自己的和发送给Planner和Solver的
```

## 6.关键结构体定义
评分基础类
```python
class Reward(ABC):
    def __self__():
        pass

    def forward(x):
    ```
    对输入x计算分数y
    ```
        pass
        return y
```