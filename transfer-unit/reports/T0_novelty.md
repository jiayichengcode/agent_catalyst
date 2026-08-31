# T0 查新报告（2026-08-25）

按 plan §9 词表检索,结论:**核心主张未被占位——"以轨迹/片段为跨任务迁移的归因单元 + 干预式(过滤/掩码)证明"仍是空白**。

## 最近邻(需在论文中显式区分)

| 工作 | 内容 | 与本文的差异 |
|---|---|---|
| Break It Down, Pass It On (arXiv 2608.20274) | 跨任务技能迁移,发现 subtask 级技能优于 task 级 | **提示/记忆层**的显式技能库(文本/代码技能注入),非 RL 梯度通道;无 rollout 过滤、无 span mask。与我们互补:他们证明"显式技能粒度"重要,我们证明"梯度承载单元粒度" |
| GraphGPO (arXiv 2605.26684) | 图式 credit assignment,超越 trajectory 级归因 | **单任务内**步级 credit,目的是改进优势估计;非跨任务迁移归因 |
| Polarity-Entropy / Signed-Capacity (arXiv 2604.11056)、Span-Wasserstein (2604.23318)、DelTA (2605.21467)、ACPO (2607.03126) | RLVR 内 token/span 级 credit:高熵 token 携带主要有效信号 | 全部是**单任务、熵/隐状态定义**的 token 选择;我们是**跨任务、行为事件定义**的 span,且 E7 预置了"高熵等量 token mask"对照专门判别两者(plan §10 预案仍有效,引这批新文即可) |
| Selective Rollout (2605.05802)、StarPO-S/RAGEN、TSR (2602.11767) | rollout 过滤/搜索作为**训练技巧**(效率、稳定性) | 动机是省算力/稳训练,无迁移归因;正好构成我们"惯例性过滤可能丢弃迁移载体"论点的靶子 |
| Stepwise Progress Attribution (2505.20732)、Long-Horizon Trajectory Attribution benchmark (2608.06909) | 步级进度归因/轨迹标注基准 | 单任务 credit 或标注框架,非因果迁移单元定位 |

## 对故事的含义
1. 相关工作已把"trajectory 级归因太粗"炒热(GraphGPO 等)——我们把这条线**转到跨任务维度**,时机好。
2. RLVR token-credit 文献 2026 上半年爆发,审稿人一定拿它压我们 → §10 的预案照旧,但对照实验(高熵 token mask vs 行为 span mask)从"可选"升为"必做"。
3. success-only RFT / 轨迹过滤惯例的文献继续增多 → E5 若证明 F1(仅成功)丢迁移载体,打击面更大。

## 任务级(L1)文献确认拥挤 → 印证做 L3/L4 的战略判断
- MoDoMoDo (2505.24871): 多域 RL 数据配比优化;Omni-Thinker (2507.14783): 任务调度;InternBootcamp (2508.08636): 任务规模化的涌现;Imbalanced Gradients in RL Post-Training of Multi-Task LLMs (2510.19178): 多任务 RL 梯度失衡;Multi-task Code LLMs: Data Mix or Model Merge? (2601.21115)。
- 全部停留在 task/mixture 粒度(plan 预判的 L1 层)。**没有任何一篇把归因下钻到 rollout/span 并做干预验证**——这是本文的差异化空隙,也意味着若结局是 O-task,确实会塌进这片红海(plan §8 风险评估准确)。

检索词已覆盖:trajectory-level attribution cross-task RL / rollout filtering transfer LLM agent / span-level credit multi-task RL / high-entropy token RLVR / multi-task mixture datamodel attribution。未检索到"哪些自生成 rollout 进入梯度才承载跨任务加速"的任何干预式研究。
