# ADA 知识库策略

基于 ExpertAgent 和 OptimCVXPY 的经验，定义任务知识（TK）和动作知识（AK）的存储格式和策略。

## 1. 知识类型

### 1.1 任务知识 (Task Knowledge, TK)

**定义**：描述电网调度问题的建模模板、问题特征识别、约束条件等。

**来源**：
- ExpertAgent：过载检测、优先级排序、拓扑操作策略
- OptimCVXPY：安全/危险状态判断、优化目标定义、约束条件

**格式**：Markdown 或 JSON

### 1.2 动作知识 (Action Knowledge, AK)

**定义**：描述具体动作的执行策略、工具使用经验、成功案例等。

**来源**：
- ExpertAgent：拓扑操作经验、线路重连策略、变电站分割策略
- OptimCVXPY：再调度策略、储能管理、弃风策略

**格式**：Markdown 或 JSON

## 2. 知识格式规范

### 2.1 Markdown 格式（推荐用于人类可读）

```markdown
# 知识标题

## 问题描述
[描述该知识适用的场景]

## 关键特征
- 特征1: 描述
- 特征2: 描述

## 解决方案
[解决方案的详细描述]

## 参考案例
- 案例1: 描述
- 案例2: 描述

## 元数据
- 来源: ExpertAgent/OptimCVXPY
- 成功率: 0.85
- 适用场景: [场景描述]
- 更新时间: 2024-01-01
```

### 2.2 JSON 格式（推荐用于程序处理）

```json
{
  "id": "knowledge_001",
  "type": "task_knowledge|action_knowledge",
  "title": "知识标题",
  "content": "知识内容（Markdown 格式）",
  "metadata": {
    "source": "ExpertAgent|OptimCVXPY",
    "success_rate": 0.85,
    "scenarios": ["场景1", "场景2"],
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "tags": ["标签1", "标签2"],
    "grid_type": "IEEE14|IEEE118|WCCI2022"
  }
}
```

## 3. 任务知识 (TK) 模板

### 3.1 过载检测与优先级排序（来自 ExpertAgent）

**Markdown 格式**：

```markdown
# 过载线路检测与优先级排序

## 问题描述
当电网出现多条线路过载时，需要识别最关键的过载线路并优先处理。

## 关键特征
- 线路负载率 (rho) > 1.0 表示过载
- 需要考虑 `timestep_overflow`（剩余允许过载时间步数）
- 临界过载（`timestep_overflow == NB_TIMESTEP_OVERFLOW_ALLOWED`）需要立即处理

## 解决方案
1. 按负载率降序排序所有过载线路
2. 将临界过载线路（`timestep_overflow == max`）排在前面
3. 优先处理临界过载线路

## 参考案例
- ExpertAgent.getRankedOverloads(): 实现过载排序逻辑
- 优先级：临界过载 > 非临界过载

## 元数据
- 来源: ExpertAgent
- 成功率: 0.90
- 适用场景: 多线路过载情况
- 更新时间: 2024-01-01
```

**JSON 格式**：

```json
{
  "id": "tk_overload_detection",
  "type": "task_knowledge",
  "title": "过载线路检测与优先级排序",
  "content": "# 过载线路检测与优先级排序\n\n## 问题描述\n当电网出现多条线路过载时...",
  "metadata": {
    "source": "ExpertAgent",
    "success_rate": 0.90,
    "scenarios": ["多线路过载"],
    "tags": ["过载检测", "优先级排序"],
    "grid_type": "all"
  }
}
```

### 3.2 安全/危险状态判断（来自 OptimCVXPY）

**Markdown 格式**：

```markdown
# 电网安全状态判断

## 问题描述
根据线路负载率判断电网当前状态，决定采用何种策略。

## 关键特征
- `max_rho < 0.85`: 安全状态（Safe Grid）
- `max_rho > 0.95`: 危险状态（Unsafe Grid）
- `0.85 <= max_rho <= 0.95`: 中间状态（Intermediate）

## 解决方案
- **安全状态**：尝试恢复参考状态（取消再调度、恢复储能目标、重连线路）
- **危险状态**：启动优化求解，最小化过载
- **中间状态**：保持当前状态，避免振荡

## 参考案例
- OptimCVXPY.act(): 根据 rho_danger 和 rho_safe 阈值判断
- 安全模式：compute_optimum_safe()
- 危险模式：compute_optimum_unsafe()

## 元数据
- 来源: OptimCVXPY
- 成功率: 0.85
- 适用场景: 所有电网状态判断
- 更新时间: 2024-01-01
```

### 3.3 优化问题建模模板

**Markdown 格式**：

```markdown
# 电网调度优化问题建模

## 问题描述
将电网调度问题建模为优化问题，包括目标函数和约束条件。

## 关键特征
- 决策变量：再调度量、储能功率、弃风量
- 目标函数：最小化过载、最小化操作成本
- 约束条件：功率平衡、线路容量、发电机限制

## 解决方案
1. **危险状态优化**：
   - 目标：最小化过载（`sum_squares(pos(abs(flow) - margin * limit))`）
   - 约束：功率平衡、发电机限制、储能限制
   
2. **安全状态优化**：
   - 目标：恢复参考状态（最小化再调度、储能目标偏差）
   - 约束：保持安全（`flow <= margin * limit`）

## 参考案例
- OptimCVXPY.compute_optimum_unsafe(): 危险状态优化
- OptimCVXPY.compute_optimum_safe(): 安全状态优化

## 元数据
- 来源: OptimCVXPY
- 成功率: 0.80
- 适用场景: 需要优化求解的场景
- 更新时间: 2024-01-01
```

## 4. 动作知识 (AK) 模板

### 4.1 拓扑操作策略（来自 ExpertAgent）

**Markdown 格式**：

```markdown
# 变电站拓扑操作策略

## 问题描述
当检测到过载时，通过改变变电站拓扑来缓解过载。

## 关键特征
- 识别过载线路相关的变电站
- 评估不同拓扑配置的效果
- 优先选择能解决所有过载的拓扑（score=4）

## 解决方案
1. 识别过载线路及其相关变电站
2. 生成候选拓扑配置（参考拓扑、分割拓扑）
3. 模拟评估每个配置的效果
4. 选择最佳配置（优先级：score=4 > score=3 > score=1）

## 参考案例
- ExpertAgent.act(): 拓扑操作主流程
- expert_operator(): 拓扑评估和排序
- 评分标准：4=解决所有过载，3=解决目标过载，1=部分解决

## 元数据
- 来源: ExpertAgent
- 成功率: 0.75
- 适用场景: 过载缓解、拓扑优化
- 更新时间: 2024-01-01
```

### 4.2 再调度策略（来自 OptimCVXPY）

**Markdown 格式**：

```markdown
# 发电机再调度策略

## 问题描述
通过调整发电机出力来缓解过载或恢复参考状态。

## 关键特征
- 危险状态：最小化过载，允许再调度
- 安全状态：恢复参考状态（再调度量趋于0）
- 考虑发电机爬坡限制（ramp_up/ramp_down）

## 解决方案
1. **危险状态**：
   - 目标：最小化过载
   - 允许再调度，但惩罚再调度量（penalty_redispatching_unsafe=0.03）
   
2. **安全状态**：
   - 目标：恢复参考状态（target_dispatch=0）
   - 惩罚再调度偏差（weight_redisp_target=1.0）

## 参考案例
- OptimCVXPY.compute_optimum_unsafe(): 危险状态再调度
- OptimCVXPY.compute_optimum_safe(): 安全状态再调度
- 再调度分配：按发电机可用容量比例分配

## 元数据
- 来源: OptimCVXPY
- 成功率: 0.80
- 适用场景: 过载缓解、参考状态恢复
- 更新时间: 2024-01-01
```

### 4.3 储能管理策略（来自 OptimCVXPY）

**Markdown 格式**：

```markdown
# 储能单元管理策略

## 问题描述
管理储能单元的充放电，以缓解过载或维持目标状态。

## 关键特征
- 危险状态：允许充放电以缓解过载
- 安全状态：维持目标状态（0.5 * Emax）
- 考虑能量限制（当前电量、最大容量）

## 解决方案
1. **危险状态**：
   - 目标：最小化过载
   - 允许储能操作，但惩罚储能功率（penalty_storage_unsafe=0.3）
   
2. **安全状态**：
   - 目标：维持目标状态（0.5 * Emax）
   - 惩罚状态偏差（weight_storage_target=1.0）

## 参考案例
- OptimCVXPY._add_storage_const(): 储能约束计算
- 储能限制：考虑功率限制和能量限制

## 元数据
- 来源: OptimCVXPY
- 成功率: 0.75
- 适用场景: 储能管理、过载缓解
- 更新时间: 2024-01-01
```

### 4.4 线路重连策略（来自 ExpertAgent）

**Markdown 格式**：

```markdown
# 断开线路重连策略

## 问题描述
在电网安全时，尝试重连因维护或攻击而断开的线路。

## 关键特征
- 仅在电网安全时执行（`max_rho < 0.95`）
- 检查线路冷却时间（`time_before_cooldown_line == 0`）
- 模拟重连后的安全性（`rho < 0.95`）

## 解决方案
1. 识别断开且可重连的线路（`line_status == False` 且 `cooldown == 0`）
2. 模拟重连后的状态
3. 如果安全（`max_rho < 0.95`），执行重连

## 参考案例
- ExpertAgent.reco_line(): 线路重连逻辑
- 仅在安全状态执行，避免引入新的过载

## 元数据
- 来源: ExpertAgent
- 成功率: 0.90
- 适用场景: 电网恢复、线路维护后
- 更新时间: 2024-01-01
```

## 5. 知识检索策略

### 5.1 任务知识检索

**查询场景**：
- "如何检测过载线路？"
- "如何判断电网安全状态？"
- "如何建模优化问题？"

**检索策略**：
1. 使用语义相似度检索（向量检索）
2. 根据场景标签过滤（grid_type, scenarios）
3. 按成功率排序

### 5.2 动作知识检索

**查询场景**：
- "如何缓解线路过载？"
- "如何执行再调度？"
- "如何管理储能？"

**检索策略**：
1. 使用语义相似度检索（向量检索）
2. 根据场景标签过滤
3. 结合当前电网状态（max_rho, overflow_count）进行匹配

## 6. 知识更新策略

### 6.1 自动更新

- 当 Summarizer 总结成功案例时，自动添加知识
- 更新成功率统计
- 合并相似知识

### 6.2 手动更新

- 支持手动添加/编辑知识
- 支持批量导入（从 JSON/Markdown 文件）

## 7. 示例知识条目

### 7.1 完整示例（JSON）

```json
{
  "id": "ak_topology_operation",
  "type": "action_knowledge",
  "title": "变电站拓扑操作策略",
  "content": "# 变电站拓扑操作策略\n\n## 问题描述\n当检测到过载时...",
  "metadata": {
    "source": "ExpertAgent",
    "success_rate": 0.75,
    "scenarios": ["过载缓解", "拓扑优化"],
    "tags": ["拓扑操作", "过载缓解"],
    "grid_type": "all",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "usage_count": 10,
    "last_used": "2024-01-15T10:30:00"
  }
}
```

## 8. 实施建议

1. **初始知识库**：从 ExpertAgent 和 OptimCVXPY 的代码中提取关键策略，转换为上述格式
2. **知识验证**：在真实场景中测试知识有效性，更新成功率
3. **知识维护**：定期审查和更新知识，删除过时知识
4. **知识扩展**：随着系统运行，不断积累新的成功案例

