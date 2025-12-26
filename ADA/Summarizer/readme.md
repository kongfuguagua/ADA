# Summarizer Agent 设计规范

## 1. 核心职责
Summarizer 是系统的“进化引擎”。它不参与实时调度，而在幕后通过经验回溯（Experience Retrospective）来更新知识库（TK/AK），防止系统重犯错误。

## 2. 核心逻辑：MCTS 驱动的知识更新
1.  **经验回放**:
    * 收集本轮调度的完整轨迹：$\mathcal{E} = \{(R_{before}, R_{after}, t_i, \dots)\}$。
2.  **蒙特卡洛树搜索 (MCTS)**:
    * 将 Planner 的工具调用序列建模为搜索树。
    * 利用 UCB 准则识别高价值（高分）的决策路径 $N_*$。
3.  **知识提炼**:
    * **AK 更新**: 归纳成功的工具调用模式，更新 Action Knowledge。
    * **TK 更新**: 归纳高分的问题建模模版，更新 Task Knowledge。

## 3. 模块实现
* **`summarizer.py`**: 实现 MCTS 算法逻辑。
* **`knowledge_updater.py`**: 调用 `KnowledgeService` 的写入接口，将提炼出的文本存入向量数据库。