# 论文骨架 × 4 结局（plan §8,按判定门自动选用）

## O-span(G1+G2+G3 全过)——冲奖主线
**标题方向**:Tasks, Traces, or Spans? Locating the Minimal Carrier of Cross-Task Acceleration in Agentic RL
**Abstract 逻辑**:多任务 agentic RL 中任务间加速普遍存在→我们问效应住在哪一层→四层干预式定位(移任务/换实例/滤轨迹/掩片段)→答案:span 级→交付合成片段配方,任务本身可被替代。
**图序**:F1 任务影响热图(Θ)→F2 E5 条件对比曲线→F3 span mask 消融→F4 合成片段兑现(E10)。
**贡献句**:(1)首个跨任务迁移的轨迹/片段级干预式归因框架;(2)advantage-frozen 过滤消融方法学(RL 独有的"同数据不同梯度"对照);(3)行为 span≠高熵 token 的判别实验;(4)任务无关加速数据配方。

## O-trace(G2 过,G3 不过)——强主会论文
**标题方向**:Not the Task, the Trace: Error-Recovery Rollouts Carry Cross-Task Acceleration in Agentic RL
**主张**:移除任务保留其加速轨迹→效应保留;保留任务滤掉这类轨迹→效应消失;载体是轨迹事件(err_recovery),不可再分到 span。
**图序**:F1 加速对存在性(noC vs F6 曲线+剂量)→F2 E5 七条件判别矩阵(核心图,§7 的逻辑表配对比曲线)→F3 E6 离线互证(SFT 平行数据集)→F4 E9 任务无关过滤器兑现 + F1(success-only RFT)的量化代价。
**卖点**:success-only 过滤惯例在系统性丢弃迁移载体——给出多任务场景的修正过滤器。
**审稿人预案**:§10 表全部保留;新增引用(见 T0):GraphGPO/DelTA/polarity-entropy 归入"单任务 credit assignment",Break-It-Down 归入"提示层技能迁移",MoDoMoDo/Omni-Thinker 归入"任务级配比",全部与本文正交。

## O-task(G2 不过)——中档,否定性结论
**标题方向**:Filtering Doesn't Matter: Cross-Task Acceleration in Agentic RL Is a Task-Level Property
**主张**:轨迹内容过滤(成功/失败/错误恢复/随机)不改变跨任务加速→加速由任务整体属性(分布/格式/奖励结构)承载→对"精选轨迹能提升迁移"的普遍直觉是一个被严格控制的否定;success-only RFT 至少无害。
**骨架**:Θ 仪器 + E5 全平结果 + 功效分析(能检测出多大效应)。

## O-null(G1 不过)——方法学论文或终止
RL 噪声下运行级归因的极限:seed 方差 vs 任务效应的方差分解,给出"多任务 RL 归因需要多少 seeds"的功效表。

---
**当前试点映射**:phase 1 的 noC/F6 对 = 压缩版 G1;F3/F5/F6 对比 = 压缩版 G2。试点为单 seed 点估,只用于选择骨架方向与决定 phase 2 加注,不构成论文级证据。
