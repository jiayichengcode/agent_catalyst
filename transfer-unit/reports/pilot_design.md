# E5-lite 试点设计（压缩版执行方案）

## 为什么先跑 E5 而不是按 T4(E1) 顺序
plan 总预算 270 GPU·日;当前可用 8×H100(quota 17)。E1 的 24×60 子集回归(60 GPU·日)在试点阶段不可行也不必要——E5 才是全文最重的 G2 判定门(§6)。策略:**用一对构造的加速对直接进入 E5**,把 E1 留到 pipeline 验证后按需扩容。这是风险倒序:先证明"trace 过滤范式能测出东西",再花大钱找最优任务对。

## 构造的加速对(设计论证)
- **C = pyfix(读报错修一行)**:合成 Python 单行 bug 修复,工具 run/submit。实例生成时保证首次 run 必触发报错 → err_event 率≈100%,err_recovery 供给充足。
- **B = sql_query(多轮 text-to-SQL)**:随机化的表/列名(同义词池)强迫 agent 探索 schema,SqlError 高频出现。
- **共享行为、不同领域**:两任务都需要"读错误观察→定位→修正动作"行为,但领域(Python vs SQL)、工具、答案形式完全不同 → 若 C 加速 B,domain-knowledge 解释被构造性削弱,行为迁移解释被支持。技能双标签(§4)在此为 by-construction。
- **filler = neutral_format**(单轮 JSON 重排,无工具无报错),用于 noC 条件补齐混合。

## 试点条件网格(phase 1, 4 runs × 2 GPU)
| 条件 | 语义 | 判别作用 |
|---|---|---|
| noC | C 从采样池移除,filler 补齐 | 加速是否存在(对照 F6) |
| F6 | C 全保留 | 正对照 |
| F3 | 仅 err_recovery 轨迹进梯度(advantage-frozen) | 载体假设 |
| F5 | 与 F3 同批次等量(条数+token 双匹配)随机轨迹 | 排除"只是多了些 C 梯度" |
phase 2(视 phase 1 结果): F0(全过滤,查经验通道)、F1(仅成功=RFT 惯例)、F4(无报错成功)、F2(仅失败) + 追加 seeds。

## 与 plan 的偏差(诚实清单)
1. **训练器**:自研 minimal GRPO(vllm rollout + HF 训练,advantage-frozen 过滤为一等公民)替代 verl。原因:worker 环境 verl 不可用且 span 级 loss mask 需要改 verl 内核;自研 500 行可审计。GRPO 语义 = 组基线策略梯度(on-policy μ=1, 无 clip 生效, 无 KL, adv 不除 std)。
2. **模型**:Qwen3.5-4B(用户指定模型族;plan 的 1.7B 层直接跳过,4B 是 plan 的 L2-L4 主力档)。
3. **F3/F5 的 C 超采样 ×3**:改变 C 的交互经验量(不改进入梯度的量)。若 F0 显示经验通道无效应,此偏差无害;报告将标注。
4. **novelty 特征、梯度探针(E4/L4 观察通道)暂缺**;E7 span mask 机制已实现,待 G2 结果决定是否开跑。
5. **熵为 top-5 logprob 近似**。
6. 统计:试点 phase 1 单 seed,只做曲线与 time-to-τ 的点估;显著性留给 phase 2 的 3-seed。

## 判读标准(试点级,非论文级)
- noC vs F6 的 B 曲线明显分离 → 加速对成立,进入 phase 2 加 seeds。
- F3 ≈ F6 > F5 ≈ noC → trace 载体信号;F3 ≈ F5 → 数量效应;全部 ≈ → O-task 备胎路线。
- B 基线 pass@1 若 >0.6 或 <0.05 → 任务难度重校准后重跑(记入日志)。
