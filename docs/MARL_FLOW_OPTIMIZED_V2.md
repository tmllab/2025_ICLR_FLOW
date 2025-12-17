# MARL-Flow 顶会优化方案 v2.0

## 目录
1. [计算资源评估](#1-计算资源评估)
2. [方案优化与精简](#2-方案优化与精简)
3. [顶会创新点设计](#3-顶会创新点设计)
4. [系统架构设计](#4-系统架构设计)
5. [核心模块详解](#5-核心模块详解)
6. [训练流程设计](#6-训练流程设计)
7. [实验设计](#7-实验设计)
8. [扩展性设计](#8-扩展性设计)

---

## 1. 计算资源评估

### 1.1 当前方案资源需求分析

| 组件 | 参数量 | 显存需求 | 说明 |
|------|--------|----------|------|
| Text Encoder (BERT-base) | 110M | ~2GB | 可用小模型替代 |
| Meta-Agent Policy | ~5M | ~100MB | 轻量级MLP |
| Meta-Agent Value | ~3M | ~60MB | 轻量级MLP |
| Executor Agent (×3) | ~4M × 3 | ~250MB | 可共享参数 |
| QMIX Mixer | ~1M | ~20MB | 轻量级 |
| Workflow GNN Encoder | ~10M | ~200MB | 可选组件 |
| **训练时峰值（单GPU）** | - | **~8-10GB** | 含梯度和优化器状态 |

### 1.2 与8×H200对比

```
8×H200 总显存: 8 × 80GB = 640GB
当前方案需求: ~10GB (单GPU) / ~30GB (数据并行)

结论: 资源严重过剩，8×H200可支持更大规模实验
```

### 1.3 资源优化建议

**实际推荐配置**：
- **开发调试**: 1× RTX 4090 (24GB) ✓ 足够
- **正式训练**: 1× A100 (40GB) ✓ 充足
- **大规模实验**: 2× A100 或 1× H100 ✓ 绑定

**8×H200 的合理利用方式**：
1. **并行超参搜索**: 8个不同配置同时训练
2. **多任务并行**: 不同benchmark同时运行
3. **大规模消融实验**: 同时跑多个消融组合
4. **Scaling实验**: 测试Agent数量从3到50的扩展性

---

## 2. 方案优化与精简

### 2.1 核心简化原则

```
原方案问题:
├── 过度工程化: 太多组件，实现复杂
├── 创新点分散: 多个小创新不如一个大创新
└── 实验难度: 需要大量调参和debug

优化方向:
├── 聚焦核心创新: 1-2个清晰的贡献点
├── 简化架构: 能用简单方法解决的不用复杂方法
└── 易于复现: 代码清晰，实验可重复
```

### 2.2 精简后的方案对比

| 维度 | 原方案 | 优化方案 |
|------|--------|----------|
| Agent数量 | 动态 (3-10) | 固定 3个 (先验证) |
| 网络架构 | Transformer+GNN | MLP + Attention |
| 训练算法 | PPO + QMIX | 仅PPO (简化版) |
| 状态编码 | BERT嵌入 | 预计算的GPT嵌入 |
| 信用分配 | QMIX | 简单比例分配 + 可选QMIX |
| 实现周期 | 6-8周 | 3-4周 |

---

## 3. 顶会创新点设计

### 3.1 主创新点: Learning to Decompose (L2D)

**核心Idea**: 学习如何将复杂任务分解为最优的子任务DAG结构

```
传统方法: LLM启发式分解 → 固定规则评估 → 执行
我们的方法: LLM生成候选 → RL学习选择/优化 → 自适应执行

创新性:
├── 首次将任务分解建模为可学习的决策问题
├── 端到端优化: 分解质量由最终执行效果驱动
└── 泛化能力: 从历史任务学习分解模式
```

### 3.2 副创新点: Adaptive Execution Policy

**核心Idea**: Agent学习根据任务特征和上下文选择最优执行策略

```
学习内容:
├── 何时请求更多context
├── 何时重试 vs 放弃
├── 如何调整prompt风格
└── 如何利用历史经验
```

### 3.3 创新点的ICLR/NeurIPS定位

| 会议 | 定位角度 | 强调点 |
|------|----------|--------|
| **ICLR** | 表示学习 | 任务分解的可学习表示 |
| **NeurIPS** | 算法创新 | 层次化RL + 信用分配 |
| **ICML** | 理论+实践 | 收敛性分析 + 实验 |
| **ACL/EMNLP** | NLP应用 | LLM-Agent协作 |

---

## 4. 系统架构设计

### 4.1 简化后的整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     MARL-Flow v2 Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Task Input (Natural Language)           │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Workflow Generator (LLM-based)              │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  GPT/Claude generates N candidate workflows      │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │ N candidates                    │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Workflow Selector (RL Policy π_select)         │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  State: task_emb + candidate_features            │    │    │
│  │  │  Action: select one from N candidates            │    │    │
│  │  │  Reward: final workflow success rate             │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │ selected workflow               │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Execution Controller (RL Policy π_exec)       │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  For each task in DAG:                           │    │    │
│  │  │    State: task_obj + context + history           │    │    │
│  │  │    Action: execution_strategy ∈ {s1, s2, ...}    │    │    │
│  │  │    Reward: task_success + downstream_bonus       │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   LLM Task Executor                      │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  Execute task with selected strategy             │    │    │
│  │  │  Validate result                                 │    │    │
│  │  │  Retry if needed (policy-controlled)             │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      Final Output                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 模块职责划分

| 模块 | 职责 | 是否用RL | 复杂度 |
|------|------|----------|--------|
| Workflow Generator | 生成候选DAG | ❌ (用LLM) | 低 |
| Workflow Selector | 选择最优DAG | ✅ (核心) | 中 |
| Execution Controller | 执行策略选择 | ✅ (核心) | 中 |
| Task Executor | 调用LLM执行 | ❌ (用LLM) | 低 |
| Validator | 验证结果 | ❌ (规则) | 低 |

---

## 5. 核心模块详解

### 5.1 模块一: Workflow Selector (核心RL模块)

**目标**: 从LLM生成的N个候选workflow中选择最优

```
输入:
├── task_description: 任务描述文本
├── candidates: List[Workflow], N个候选DAG
└── history_stats: 历史执行统计 (可选)

状态编码 S:
├── task_embedding: 任务描述的向量表示 [d_task]
├── candidate_features: 每个候选的结构特征 [N, d_cand]
│   ├── num_tasks: 任务数量
│   ├── max_depth: DAG最大深度
│   ├── avg_parallelism: 平均并行度
│   ├── dependency_complexity: 依赖复杂度
│   └── estimated_difficulty: 估计难度
└── global_context: 历史成功模式 [d_ctx]

动作空间 A:
├── 离散选择: a ∈ {0, 1, ..., N-1}
└── 选择第a个候选workflow

奖励设计 R:
├── 终局奖励: workflow执行完成后
│   ├── success_rate × 10 (主奖励)
│   ├── + efficiency_bonus (时间越短越好)
│   └── - complexity_penalty (过于复杂的惩罚)
└── 无中间奖励 (延迟奖励)

网络架构:
┌─────────────────────────────────────────┐
│  Task Embedding (pre-computed GPT emb)  │
│              [batch, d_task]            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Candidate Encoder (MLP)                │
│  Input: [batch, N, d_cand]              │
│  Output: [batch, N, d_hidden]           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Cross-Attention                        │
│  Query: task_emb                        │
│  Key/Value: candidate_embs              │
│  Output: attended_features              │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Policy Head  │  │ Value Head   │
│ π(a|s)       │  │ V(s)         │
│ [N] softmax  │  │ [1] scalar   │
└──────────────┘  └──────────────┘
```

### 5.2 模块二: Execution Controller (核心RL模块)

**目标**: 为每个任务选择最优执行策略

```
输入:
├── task: 当前待执行任务
├── context: 上游任务的结果
├── downstream_objectives: 下游任务的目标
└── execution_history: 本任务的历史尝试

状态编码 S:
├── task_embedding: 任务目标的向量 [d_task]
├── context_embedding: 上下文摘要 [d_ctx]
├── history_features: 历史尝试特征 [d_hist]
│   ├── num_attempts: 已尝试次数
│   ├── last_feedback_type: 上次失败类型
│   └── improvement_trend: 改进趋势
└── downstream_features: 下游需求特征 [d_down]

动作空间 A (离散):
├── strategy_0: "concise" - 简洁输出
├── strategy_1: "detailed" - 详细输出
├── strategy_2: "step_by_step" - 分步输出
├── strategy_3: "example_based" - 示例驱动
└── (可扩展更多策略)

重试动作空间 A_retry:
├── action_0: "retry" - 直接重试
├── action_1: "modify" - 修改prompt重试
├── action_2: "simplify" - 简化要求重试
└── action_3: "give_up" - 放弃 (标记失败)

奖励设计 R:
├── 即时奖励:
│   ├── +1.0 验证通过
│   ├── -0.1 每次重试
│   └── -0.5 最终失败
└── 延迟奖励:
    └── +0.5 × 下游任务成功数 (downstream bonus)

网络架构:
┌─────────────────────────────────────────┐
│  State Encoder (MLP)                    │
│  Input: concat(task_emb, ctx_emb,       │
│         history_feat, downstream_feat)  │
│  Output: [batch, d_hidden]              │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│Strategy Head │  │ Retry Head   │
│ [5] softmax  │  │ [4] softmax  │
└──────────────┘  └──────────────┘
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Value Head   │  │ (shared)     │
│ V(s) [1]     │  │              │
└──────────────┘  └──────────────┘
```

### 5.3 模块三: 特征提取器 (非RL)

**目标**: 将原始输入转换为RL可用的特征

```
Workflow Feature Extractor:
├── 输入: Workflow对象 (DAG结构)
├── 输出: [d_workflow] 维特征向量
├── 提取内容:
│   ├── 结构特征: 节点数、边数、深度、宽度
│   ├── 并行度: 各层可并行任务数
│   ├── 复杂度: 依赖关系复杂程度
│   └── 任务类型分布: 各类型任务占比
└── 实现: 简单统计 + MLP编码

Task Embedding Cache:
├── 策略: 预计算所有任务描述的embedding
├── 模型: 使用GPT/Claude的embedding API
├── 缓存: 存储在内存/磁盘避免重复计算
└── 优势: 训练时无需实时调用LLM编码

Context Summarizer:
├── 输入: 上游任务结果 (可能很长)
├── 输出: 固定长度摘要向量
├── 方法:
│   ├── 简单: 截断 + 平均池化
│   └── 进阶: 用小LLM摘要后编码
└── 目的: 控制状态维度
```

### 5.4 模块四: 经验收集与回放

**目标**: 收集训练数据，支持高效采样

```
数据结构:

Episode = {
    'task_description': str,
    'selected_workflow_idx': int,
    'workflow_features': List[ndarray],  # N个候选的特征
    'execution_trajectory': List[TaskExecution],
    'final_success_rate': float,
    'total_time': float
}

TaskExecution = {
    'task_id': str,
    'state': ndarray,
    'strategy_action': int,
    'retry_actions': List[int],
    'result': str,
    'success': bool,
    'reward': float,
    'downstream_successes': int
}

存储策略:
├── 内存buffer: 最近1000个episodes
├── 成功案例库: 所有成功的workflow配置
└── 失败案例库: 用于分析失败模式

采样策略:
├── 均匀采样: 基础方法
├── 优先采样: 优先采样reward方差大的
└── 课程采样: 从简单任务逐步到复杂任务
```

---

## 6. 训练流程设计

### 6.1 整体训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Data Collection (Warm-up)                             │
│  ├── 运行原始Flow框架收集baseline数据                             │
│  ├── 记录: 任务、workflow选择、执行结果                            │
│  └── 构建初始训练集 (~100 episodes)                              │
│                                                                 │
│  Phase 2: Pre-training (Imitation Learning)                     │
│  ├── 用收集的成功案例做监督学习                                    │
│  ├── Workflow Selector: 模仿最佳选择                             │
│  ├── Execution Controller: 模仿成功策略                          │
│  └── 目的: 提供合理的初始化                                       │
│                                                                 │
│  Phase 3: RL Fine-tuning                                        │
│  ├── 交替训练两个RL模块                                          │
│  ├── 每个epoch:                                                 │
│  │   ├── 收集K个新episodes (用当前策略)                          │
│  │   ├── 计算returns和advantages                                │
│  │   ├── PPO更新 Workflow Selector                              │
│  │   ├── PPO更新 Execution Controller                           │
│  │   └── 评估并记录metrics                                       │
│  └── 早停: 当性能不再提升时停止                                   │
│                                                                 │
│  Phase 4: Evaluation                                            │
│  ├── 在held-out测试任务上评估                                    │
│  ├── 与baseline对比                                              │
│  └── 消融实验                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 单个Episode的执行流程

```
Episode Execution:

1. 接收任务描述 task_description

2. Workflow生成阶段:
   ├── 调用LLM生成N个候选workflow
   ├── 提取每个候选的特征 → candidate_features
   └── 获取任务embedding → task_embedding

3. Workflow选择阶段:
   ├── state = encode(task_embedding, candidate_features)
   ├── action = workflow_selector.select(state)
   ├── selected_workflow = candidates[action]
   └── 记录: (state, action) → buffer

4. 任务执行阶段 (遍历DAG):
   For each task in topological_order(selected_workflow):
       ├── 等待依赖任务完成
       ├── 获取上游context
       ├── state = encode(task, context, history)
       ├── strategy = exec_controller.select_strategy(state)
       ├── result = llm_execute(task, strategy)
       ├──
       ├── While not validated and retry_count < max_retry:
       │   ├── retry_action = exec_controller.select_retry(state)
       │   ├── if retry_action == "give_up": break
       │   ├── result = llm_retry(task, retry_action)
       │   └── retry_count += 1
       ├──
       └── 记录: (state, action, reward) → buffer

5. Episode结束:
   ├── 计算final_success_rate
   ├── 回填workflow_selector的reward
   └── 存储完整episode到buffer

6. 返回统计信息
```

### 6.3 PPO训练细节

```
PPO Update (for each module):

超参数:
├── learning_rate: 3e-4
├── gamma: 0.99
├── gae_lambda: 0.95
├── clip_ratio: 0.2
├── entropy_coef: 0.01
├── value_coef: 0.5
├── max_grad_norm: 0.5
├── ppo_epochs: 4
└── minibatch_size: 64

单次更新流程:
1. 从buffer采样batch
2. 计算GAE advantages
3. For epoch in range(ppo_epochs):
   ├── 计算新的log_prob和entropy
   ├── 计算ratio = exp(new_log_prob - old_log_prob)
   ├── 计算clipped surrogate loss
   ├── 计算value loss
   ├── total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
   ├── backward and clip gradients
   └── optimizer.step()
4. 清空buffer (on-policy)
```

---

## 7. 实验设计

### 7.1 实验设置

```
Benchmark Tasks:

Tier 1 - 简单任务 (验证基础能力):
├── 简单Python函数实现 (如排序、搜索)
├── 简单网页布局
└── 文本处理任务

Tier 2 - 中等任务 (主要评估):
├── 小型游戏开发 (Tetris, Snake)
├── 完整网站 (前后端)
├── 数据分析pipeline
└── API服务开发

Tier 3 - 复杂任务 (展示上限):
├── 复杂游戏 (多人、网络)
├── 完整应用系统
└── 机器学习pipeline

数据集划分:
├── Training: 70% 任务
├── Validation: 15% 任务
└── Test: 15% 任务 (完全held-out)
```

### 7.2 评估指标

```
主要指标:

1. Success Rate (SR):
   SR = 成功完成的workflow / 总workflow数

2. Task Completion Rate (TCR):
   TCR = 成功完成的task / 总task数

3. Efficiency Score (ES):
   ES = Baseline时间 / 实际时间

4. Retry Efficiency (RE):
   RE = 首次成功率 + 0.5 × 重试成功率

次要指标:

5. Workflow Quality (WQ):
   WQ = 选择的workflow的结构评分

6. Generalization Gap (GG):
   GG = Train SR - Test SR (越小越好)

7. Sample Efficiency (SE):
   SE = 达到目标性能所需的训练episodes
```

### 7.3 对比实验

```
Baselines:

1. Original Flow (启发式):
   ├── Workflow选择: Z-score规则
   └── 执行策略: 固定策略

2. Random Selection:
   ├── Workflow选择: 随机
   └── 执行策略: 随机

3. LLM-only (无RL):
   ├── Workflow选择: 让LLM直接选
   └── 执行策略: 让LLM决定

4. Single-Agent RL:
   ├── 只有Workflow Selector
   └── 执行用固定策略

5. MARL-Flow (Ours):
   ├── RL Workflow Selector
   └── RL Execution Controller

实验矩阵:
┌─────────────────┬────────┬────────┬────────┐
│ Method          │ Tier 1 │ Tier 2 │ Tier 3 │
├─────────────────┼────────┼────────┼────────┤
│ Original Flow   │   ✓    │   ✓    │   ✓    │
│ Random          │   ✓    │   ✓    │   ✓    │
│ LLM-only        │   ✓    │   ✓    │   ✓    │
│ Single-Agent RL │   ✓    │   ✓    │   ✓    │
│ MARL-Flow       │   ✓    │   ✓    │   ✓    │
└─────────────────┴────────┴────────┴────────┘
```

### 7.4 消融实验

```
Ablation Studies:

1. Workflow Selector的作用:
   ├── w/ RL Selector vs w/o (random)
   └── 预期: RL选择显著优于随机

2. Execution Controller的作用:
   ├── w/ RL Controller vs w/o (fixed)
   └── 预期: 自适应策略优于固定策略

3. 候选数量N的影响:
   ├── N ∈ {3, 5, 10, 20}
   └── 预期: 更多候选 → 更好结果，但边际递减

4. 预训练的作用:
   ├── w/ imitation pre-training vs w/o
   └── 预期: 预训练显著加速收敛

5. 下游奖励的作用:
   ├── w/ downstream bonus vs w/o
   └── 预期: 下游奖励提升整体协调性
```

---

## 8. 扩展性设计

### 8.1 为后续创新预留的接口

```
扩展点1: Workflow生成器替换
├── 当前: LLM生成 + RL选择
├── 扩展: RL直接生成DAG结构
└── 接口: WorkflowGenerator抽象类

扩展点2: 多Agent扩展
├── 当前: 单一Execution Controller
├── 扩展: 多个专长化Agent
└── 接口: AgentPool管理多个Agent

扩展点3: 信用分配算法
├── 当前: 简单比例分配
├── 扩展: QMIX、Shapley、Attention
└── 接口: CreditAssigner抽象类

扩展点4: 状态编码器
├── 当前: 预计算embedding + MLP
├── 扩展: 在线Transformer编码
└── 接口: StateEncoder抽象类

扩展点5: 奖励函数
├── 当前: 手工设计
├── 扩展: 学习的奖励函数 (IRL)
└── 接口: RewardCalculator可配置
```

### 8.2 与其他论文结合的潜在方向

```
潜在结合方向:

1. 与Self-Refine类方法结合:
   ├── 思路: RL学习何时触发self-refine
   └── 创新: 自适应refinement策略

2. 与ReAct/CoT类方法结合:
   ├── 思路: RL选择推理策略
   └── 创新: 动态推理路径选择

3. 与Multi-Agent Debate结合:
   ├── 思路: 多个Agent辩论选择workflow
   └── 创新: 辩论式决策的RL优化

4. 与Tool Learning结合:
   ├── 思路: RL学习工具选择和组合
   └── 创新: 工具链的自动优化

5. 与Memory/RAG结合:
   ├── 思路: RL控制记忆检索策略
   └── 创新: 自适应上下文管理

6. 与Planning (如MCTS)结合:
   ├── 思路: RL + 搜索的混合方法
   └── 创新: 学习引导的规划
```

### 8.3 代码结构的扩展性

```
推荐代码结构:

marl_flow/
├── core/                     # 核心抽象
│   ├── base_agent.py         # Agent基类
│   ├── base_encoder.py       # Encoder基类
│   ├── base_reward.py        # Reward基类
│   └── interfaces.py         # 接口定义
│
├── agents/                   # Agent实现
│   ├── workflow_selector.py  # 核心: Workflow选择
│   ├── exec_controller.py    # 核心: 执行控制
│   └── extensions/           # 扩展Agent
│       └── specialized_agent.py
│
├── encoders/                 # 状态编码
│   ├── task_encoder.py
│   ├── workflow_encoder.py
│   └── extensions/
│       └── transformer_encoder.py
│
├── rewards/                  # 奖励计算
│   ├── basic_reward.py
│   └── extensions/
│       ├── learned_reward.py
│       └── credit_assignment.py
│
├── training/                 # 训练相关
│   ├── ppo_trainer.py
│   ├── buffer.py
│   └── curriculum.py
│
├── environments/             # 环境封装
│   ├── flow_env.py           # 将Flow包装为Gym环境
│   └── task_loader.py
│
└── configs/                  # 配置文件
    ├── default.yaml
    └── experiments/
```

---

## 9. 风险与应对

### 9.1 技术风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| RL训练不稳定 | 中 | 高 | 预训练+小学习率+早停 |
| 奖励稀疏 | 高 | 中 | 奖励塑形+中间奖励 |
| 泛化能力差 | 中 | 高 | 多样化训练任务+正则化 |
| LLM调用成本高 | 高 | 中 | 缓存+batch调用+小模型 |

### 9.2 研究风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 创新性不足 | 低 | 高 | 明确定位+充分related work |
| 实验不充分 | 中 | 中 | 预留足够实验时间 |
| 与concurrent work重叠 | 中 | 高 | 持续关注arXiv |

---

## 10. 时间规划

```
Week 1-2: 基础设施
├── 搭建训练框架
├── 实现数据收集pipeline
└── 实现基础Baseline

Week 3-4: 核心模块
├── 实现Workflow Selector
├── 实现Execution Controller
└── 联调测试

Week 5-6: 训练与调优
├── 收集训练数据
├── 预训练+RL训练
└── 超参搜索

Week 7-8: 实验
├── 主实验
├── 消融实验
└── 分析与可视化

Week 9-10: 论文
├── 撰写初稿
├── 补充实验
└── 修改完善

Week 11-12: Buffer
├── 处理reviewer意见
├── 最终修改
└── 准备rebuttal材料
```

---

## 11. 总结

### 核心要点:

1. **资源评估**: 8×H200远超需求，建议用于并行实验
2. **方案精简**: 聚焦两个核心RL模块，避免过度工程化
3. **创新定位**: "Learning to Decompose" + "Adaptive Execution"
4. **扩展性**: 预留清晰接口，支持后续创新结合
5. **可实现性**: 3-4周可完成核心实现，预留充足实验时间

### 与其他论文结合的准备:

当前设计的模块化架构支持：
- 替换/增强任意组件
- 添加新的Agent类型
- 集成新的决策机制
- 引入新的奖励信号

请提供您想结合的论文，我将分析如何融合创新点。
