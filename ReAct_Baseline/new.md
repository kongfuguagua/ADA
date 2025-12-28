Grid2Op ReAct Agent v2.0: Neuro-Symbolic 增强版设计文档
1. 设计理念 (Design Philosophy)
核心思想：Math for Precision, LLM for Strategy (数学提供精度，大模型负责策略)。

现状 (v1.0): LLM 瞎猜 "我觉得应该调整发电机 1"，然后靠 simulate 试错。效率低，盲目性大。

目标 (v2.0):

预分析 (Pre-Analysis): 在 LLM 思考前，先运行确定性的数学算法（参考 ExpertAgent），计算出“关键子图”。

工具增强 (Tool-Augmented): 给 LLM 配备 topology_analyzer 和 sensitivity_calculator 工具，而不是让它凭空推理物理规律。

引导式思考: 强制 LLM 先调用工具获取数据，再基于数据生成动作。

2. 新增核心模块设计
我们需要在 ReAct_Baseline/ 下新增一个 analysis 包，包含两个核心数学工具类。

2.1. 拓扑影响分析器 (topology_analyzer.py)
目标：实现 Expert System 中的 "Influence Graph" 简化版。给定一条过载线路，通过图搜索算法找出物理上最相关的变电站。

数学/逻辑原理：

1-Hop Connectivity: 直接连接过载线路两端的变电站是最高优先级的操作对象。

Load/Gen Distribution: 如果过载线路的一端连接着大负荷或大发电，该变电站的拓扑操作（母线分裂）最可能改变潮流。

接口定义:

Python

class TopologyAnalyzer:
    def get_influential_substations(self, observation, line_id: int) -> List[Dict]:
        """
        输入: 当前观测, 过载线路ID
        输出: 有影响力的变电站列表 (按影响力排序)
        逻辑:
          1. 找出线路直接连接的 origin_sub 和 extremity_sub。
          2. 检查这些变电站是否"可操作" (有多个物体连接，且未冷却)。
          3. (进阶) 搜索 2-hop 内的大型发电厂变电站。
        返回格式:
          [
            {"sub_id": 3, "reason": "直接连接过载线路(Origin)，连接3个发电机", "objects": [...]},
            {"sub_id": 8, "reason": "直接连接过载线路(Extremity)", "objects": [...]}
          ]
        """
2.2. 灵敏度计算器 (sensitivity_calculator.py)
目标：替代 LLM 的 "直觉"，用简单的物理规则指导再调度（Redispatching）。

数学/逻辑原理：

拓扑距离 (Topological Distance): 距离过载线路越近的发电机，对该线路潮流影响越大。

潮流方向 (Flow Direction):

位于线路送端 (Source) 的发电机：减少出力有助于减载。

位于线路受端 (Sink) 的发电机：增加出力有助于减载（通过就地平衡负荷）。

接口定义:

Python

class SensitivityCalculator:
    def get_effective_generators(self, observation, line_id: int) -> Dict[str, List[int]]:
        """
        输入: 当前观测, 过载线路ID
        输出: {"decrease_candidates": [], "increase_candidates": []}
        逻辑:
          1. 确定线路的实际流向 (P_or, P_ex 的符号)。
          2. 找出送端变电站集合(Source Area)和受端变电站集合(Sink Area) (通过 BFS 搜索 1-2 跳)。
          3. 位于 Source Area 的发电机 -> 推荐降出力 (Decrease)。
          4. 位于 Sink Area 的发电机 -> 推荐升出力 (Increase)。
        """
3. ReAct 流程改造 (Workflow Refactoring)
修改 ReActAgent.act() 的循环逻辑，引入**"分析-思考-行动" (Analyze-Think-Act)** 模式。

3.1. 强制预分析 (Mandatory Pre-Analysis)
在将 observation 喂给 LLM 之前，先运行 Python 代码生成一段 "专家分析报告"，并拼接到 Prompt 中。

代码逻辑伪代码:

Python

# 在 agent.py 中
def act(self, observation):
    # ... 检测过载 ...
    overloaded_lines = ...
    
    analysis_report = ""
    if overloaded_lines:
        target_line = overloaded_lines[0] # 聚焦最严重的
        
        # 1. 调用数学工具
        topo_candidates = self.topo_analyzer.get_influential_substations(observation, target_line)
        gen_candidates = self.sensitivity_calc.get_effective_generators(observation, target_line)
        
        # 2. 生成报告文本
        analysis_report += f"【专家辅助分析报告】\n"
        analysis_report += f"针对过载线路 {target_line}，物理分析建议：\n"
        analysis_report += f"1. 拓扑调整候选: 变电站 {topo_candidates} 最有可能改变该线路潮流。\n"
        analysis_report += f"2. 发电调整建议: \n"
        analysis_report += f"   - 尝试 降低 发电机 {gen_candidates['decrease']} 的出力 (位于送端)。\n"
        analysis_report += f"   - 尝试 增加 发电机 {gen_candidates['increase']} 的出力 (位于受端)。\n"

    # 3. 将 report 放入 Prompt
    prompt = self.prompt_manager.build(obs_text, analysis_report, history)
    # ... LLM 生成动作 ...
3.2. 辅助工具 (Auxiliary Tools for LLM)
除了直接给分析报告，还可以给 LLM 提供一个**"虚拟仿真工具"**。

工具名: simulate_topology(sub_id)

功能: LLM 不直接输出 Action，而是先请求 Tool: simulate_topology(3)。

Agent 响应: Agent 在内部模拟将变电站 3 的总线分裂（随机或基于规则分裂），然后告诉 LLM：“模拟结果：线路 X 负载率从 105% 降到了 98%”。

LLM 决策: 收到好结果后，LLM 再输出最终 Action。