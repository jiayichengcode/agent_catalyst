# 文献综述:"催化剂数据"这个 idea 别人做过吗?(2026-08-27)

**结论:做过,而且被做了很多遍——分散在至少五个术语群下,从没人叫它"催化剂"。
最接近的一篇(NeurIPS 2025)几乎覆盖了我们正在跑的 math→coding 实验的核心发现。**

## 群 1:跨域推理数据互助(与你的 idea 最直接重合)

- **Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence
  Functions**(arXiv 2505.19949, **NeurIPS 2025**)——**头号最近邻,几乎撞车**。
  用影响函数(EK-FAC 近似 Hessian)在 instance/sequence/token 三个粒度归因,发现:
  ①**math 与 code 数据双向互助**;②**高难度 math 提升 math+code 双域,低难度 code 最利于 code**;
  ③符号化(变量/形式记号)的 math 比对话式 math 更利于迁移到 code;④"探索性行为"(得到正确答案后
  继续找替代解)对两域都有益。模型 LLaMA3-8B / Qwen2.5-7B,SFT + CoT 数据,GSM8k/MBPP/AIME/
  LiveCodeBench。difficulty-flipped 策略把 AIME 从 10%→20%。
  **对我们的冲击**:我们正在跑的"math_hard vs math_easy 催化 codegen"的深度对照,
  其核心结论(高难度数学利于代码)已被这篇以更强的方法(影响函数)在更真实的数据上做过。
- **On Code-Induced Reasoning in LLMs**(2509.21499):code 数据提升通用推理,跨模型族与规模成立。
- **What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code**
  (2605.19762):进一步把"code 提升数学"的载体定位到**结构化推理信号**而非代码本身;
  提出"cognitive scaffold"作为跨域稳定信号(expert-activation 分布保持稳定)——这与我们
  "催化=维持可塑性/防塌缩"的机制假说撞概念。
- **Reinforcement Learning for Tool-Integrated Interleaved Thinking towards Cross-Domain
  Generalization**(2510.11184):RL 侧的跨域泛化。

## 群 2:数据配比优化(工业界主战场,"该掺多少"的定量版)

DomainPilot(2607.22769,按域 loss 动态配比)、DynaMiCS(2605.10770,**用短探针估计跨域效应
斜率矩阵**——本质是"催化强度"的可测量代理)、DynamixSFT(2508.12116)、Data Mixing
Optimization for SFT(2508.11953)、Capacity-Aware Mixture Law(2603.08022)、
MoDoMoDo(2505.24871,多模态 RL 配比)。
共同点:把"辅助数据帮不帮目标"当**配比优化问题**,不问机制;剂量-响应曲线是其副产品。

## 群 3:任务迁移可预测性(2020 年前后的经典线)

Task2Vec(FIM 任务嵌入)、Taskonomy、Exploring and Predicting Transferability across NLP
Tasks(Vu et al., EMNLP 2020)、Task2Box、Grad2Task。
**"哪个源任务能帮目标任务"= 迁移矩阵 + 可预测性**,这正是原 research plan 的 L1 层(Θ 矩阵)。

## 群 4:潜在能力分解(2025 新)

**Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM
Fine-tuning**(2509.13624):训 10 个模型建迁移矩阵 + PCA 抽"潜在能力"(推理/情感/NLU/算术)。
**关键发现与我们的结果一致**:迁移收益**无法用表层数据相似度或源数据质量解释**,
真正起作用的是隐藏统计因子(类别分布、生成长度倾向)与特定语言特征——
这与我们"随机内容的催化=预算效应、真正起作用的是别的东西"高度同构。

## 群 5:辅助目标即正则化 / 课程与数据选择

辅助任务作为 soft regularization(经典多任务学习结论)、课程学习(METIS 2605.11235、
Difficulty Is Not Enough AAAI)、难度定向在线数据选择(2506.05316)。
"掺 1-5% 通用数据防遗忘"是工业 folklore;replay-based continual learning 有严格版本。

## 我们还剩什么空位(诚实评估)

**已被占的**:①辅助数据能帮目标任务(群1-2 全覆盖);②math↔code 双向互助(2505.19949);
③高难度数学更有效(同上);④迁移收益不能用表层相似度解释(2509.13624);⑤配比优化(群2)。

**尚未被占的(我们的实测结果里独有的)**:
1. **预算控制的严格性**:群 1-2 几乎都不做 token-matched 对照(rep×epoch / 更多目标数据)。
   我们的结论"随机跨任务内容的收益 = 训练量效应"是对该文献的一个方法学诘问。
2. **极性不对称的干净证据**:同任务失败=剧毒 vs 跨任务失败=无害;"不在出错,在恢复"。
   influence-function 那篇看的是正向属性,没做失败极性的因子实验。
3. **亚化学计量阴性结果**:严格意义的催化(微量、不被消耗、纯动力学)在 SFT 数据层**不存在**
   ——这个否定性结论没人写过,它给"催化剂"这个隐喻划了边界。
4. **RL vs SFT 的作用位点对比**:同一对任务在 on-policy RL 里无催化位点(优势隔离+数据自多样),
   顺序课程只有点火税。跨范式的受控对比罕见。
5. **方法学伪象清单**(seed 碰撞、端点 ±0.2 方差、归一化混淆、筛选赢家诅咒)+ 逐 checkpoint
   热载评测基建——对做这类消融的人有直接价值。

## 建议的重新定位

不要把论文写成"我们发现辅助数据能催化"(会被 2505.19949 直接压掉),而是写成
**"Auxiliary-data gains in agent distillation are mostly budget: a controlled anatomy"**——
以预算控制为方法核心,主结果是**分离出真正超出预算线的成分(err_recovery / 高难度跨域)与
毒性成分(同任务失败)**,并用亚化学计量阴性结果给"催化"隐喻定界。
这样与 2505.19949 形成互补(他们:影响函数找有益属性;我们:预算控制筛真伪 + agent 任务 +
极性 + 范式对比),而不是竞争。
