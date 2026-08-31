# agent_catalyst

**Tasks, Traces, or Spans?** — 定位多任务 agentic 训练中跨任务加速效应(“催化剂效应”)的最小承载单元。

当任务 C 的存在加速了看似无关的任务 B 的学习,这个效应究竟由什么承载?本仓库围绕四个候选粒度做识别与干预实验:

| 层级 | 假设 | 干预手段 |
|---|---|---|
| **L1 任务级** | 效应是任务族的整体属性 | 移除整个任务 C,中性任务填充 |
| **L2 实例级** | 只有 C 中某类实例携带效应 | 按实例特征二分后替换 |
| **L3 轨迹级** | 只有含特定事件的 rollout 有效 | 梯度前过滤 rollout(advantage-frozen) |
| **L4 片段级** | 效应由 trace 内特定 token span 承载 | span 梯度 mask |

完整设计见 [`transfer_unit_research_plan_v2.md`](transfer_unit_research_plan_v2.md)。

## 主要发现

当前结论汇总在 [`transfer-unit/reports/FINAL_REPORT.md`](transfer-unit/reports/FINAL_REPORT.md)
(B = `sql_query` 多轮工具 agent,Qwen3.5-4B,held-out 48 题 pass@1,n=3 seeds):

- **同任务失败轨迹是毒药** —— 混入目标任务自己的失败 rollout 显著劣于所有其他条件。
- **训练量是第一因子** —— 规模控制后,随机内容的“净催化”消失;欠训练才是主要短板。
- **无关数据不是安全填充** —— 高剂量无关 token 轻度有害,劣于不加。
- **精选错误恢复轨迹是唯一跑赢预算控制的内容** —— 提纯后的 catalyst 有效成分(证据强度中等,待追加 seeds 定案)。
- **RL 侧无催化位点** —— GRPO 优势隔离下混训中性;催化剂是离线蒸馏特有现象。

## 仓库结构

```
transfer_unit_research_plan_v2.md   研究计划 v2(实验设计、判据、任务序列)
transfer-unit/
├── tu/                     核心 Python 包
│   ├── tasks/              任务定义与注册表(sql_query, codegen, pyfix, math_*, neutral, diverse)
│   ├── training/           GRPO 训练循环、rollout、入口
│   ├── offline/            离线蒸馏 / SFT、checkpoint 导出与评测、donor 数据生成
│   ├── tracing/            trace 特征打标与 span 标注器
│   ├── analysis/           datamodel 回归、曲线、加速度量、终版图表
│   └── envs.py             环境封装
├── configs/                实验配置(pilot / hard / v7 / calib 等,31 份)
├── scripts/                实验编排 shell(网格发起、恢复、评测 sweeper、SFT 流水线)
├── tests/                  核心逻辑与 fake-rollout 测试
├── data/                   小规模白名单 / 筛选集(大的 donor_*.jsonl 未入库)
└── reports/                实验报告、预注册预测、文献综述、结果图
```

## 未纳入版本控制的产物

以下目录体积过大或可再生,已在 `.gitignore` 中排除:

| 路径 | 体积 | 说明 |
|---|---|---|
| `runs/` | ~1.5T | 训练 run 输出与 checkpoint |
| `models/` | ~13G | 下载的基座模型权重 |
| `envs/` | ~4.9G | 虚拟环境 |
| `data/*.jsonl`, `transfer-unit/data/*.jsonl` | ~137M | 生成的 donor 数据集,可由 `tu/offline/donate.py` 重新生成 |
