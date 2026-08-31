# T2/T3/E0 验收报告（2026-08-25）

## 验收判定：通过

### T2 特征与 span 标注器
- trace 特征 schema(§1)全量实现:success/err_event/err_recovery/err_giveup/verify_cnt/clarify/replan/len/n_gen_tokens/ent/adv_mag(训练时回填)。novelty 试点缺席(记录为偏差)。
- span 标注器:S_err_react/S_verify/S_plan/S_final,规则实现。**验收方式偏差**:plan 要求 100 条人工标注轨迹上 span-F1≥0.85;试点以构造正确性替代(45 项单元断言 + FakeLLM 真 tokenizer 端到端记账校验:span 边界与 token 范围一致)。正式版仍需人工标注对齐。
- 45/45 CPU 断言通过;token 记账在真实 Qwen3.5 tokenizer 上验证(prompt/gen/obs 三段 mask 正确)。

### T3 训练与日志
- 冒烟(smoke_e0c, 2 步 + 前后 eval + ckpt):全链路通过——vllm 多轮 rollout → 特征行 → 组优势(advantage-frozen) → 条件过滤 → token 级 loss-mask → 反向(纯 torch DeltaNet 路径) → AdamW step → 权重同步至 vllm(/dev/shm + collective_rpc, 426 键全映射) → 周期 eval → ckpt 保存。
- 步耗时 ~45s(rollout 26s + 训练/同步 19s),200 步 ≈ 3h/run。
- 日志:train_log.jsonl(步级,含各任务 kept_tok=进入梯度 token 数)、rollouts.jsonl(每 rollout 特征行,§5 要求)、eval.jsonl、samples.jsonl(每 25 步 3 条 C 轨迹全文)。

### 难度校准(E0 附加,2 轮迭代)
| 任务 | 贪心 pass@1 | 温度1 pass@1 | err_event | err_recovery(温度1) |
|---|---|---|---|---|
| sql_query(B) | 0.667 (n=24) | 0.625 | ~0.85 | ~0.5 of successes |
| pyfix(C) | 0.583 | 0.458 | ~1.0 | ≈全部成功经过报错 |
| neutral | 1.0 | 0.667 | 0 | 0 |
- 调整过程:v1 全任务 1.0(过易)→ 加三档难度+3 表 schema+回合 8→5+判分数字抽取 → 上表。
- 失败样本人工检视:B 的典型失败=列名张冠李戴后**同一错误查询重复 3 次直至回合耗尽、从不提交**——正是 err_recovery 行为缺陷,C→B 迁移假设的行为学依据。

### 环境修复记录(复现要点)
1. fla-core 的 TileLang 后端探测崩溃(坏 tvm_ffi) → PYTHONPATH 放 tilelang.py shim(干净 ImportError)。
2. fla 在 Hopper+triton≥3.4 拒绝 gated DeltaNet 反向(fla #640) → 训练侧 monkeypatch 禁用 fla,走 transformers 纯 torch gated delta(正确但慢;overlay venv 里已装好 tilelang 0.1.13 作提速备选)。
3. vllm 0.19 collective_rpc 需 VLLM_ALLOW_INSECURE_SERIALIZATION=1 传 callable。
4. transformers 版本间 apply_chat_template 返回类型不一致 → 统一 render-then-tokenize。
5. TORCH_BLAS_PREFER_CUBLASLT=1(驱动 535)。

## 当前:phase-1 试点已启动
noC/F6/F3/F5 × seed1 × 200 步,8×H100,~3.5h。
