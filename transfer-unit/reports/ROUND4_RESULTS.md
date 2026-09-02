# 第四轮结果:毒药结论在第二个目标任务上复现(2026-09-02)

判读规则在任何 run 启动前落盘于 [`PREREG_ROUND4.md`](PREREG_ROUND4.md),本文件不修改它。

## 1. 2×2(极性 × 来源),四格全部 n=8

B = codegen · 30k 目标 token · eval codegen n=48 · 主指标 = pass@1 曲线 AUC

| 条件 | C 来源 | C 极性 | AUC(原生) | AUC(统一网格) | tt@0.7 † |
|---|---|---|---|---|---|
| `r3_none` | — | — | 0.613 | 0.616 | 11,871 |
| `r3_pyf60` | pyfix(跨任务) | 错误恢复 | **0.719** | 0.718 | 5,681 |
| `r4_pyfail60` | pyfix(跨任务) | **失败** | 0.642 | 0.640 | 11,906 |
| `r4_cgfail60` | codegen(**同任务**) | **失败** | **0.479** | 0.477 | 17,810 |

† `tt@0.7` 只作次指标,见 §4。

## 2. 预注册判读(Mann–Whitney,AUC,双侧)

| | 比较 | p | 判据 | 结果 |
|---|---|---|---|---|
| **P1** 毒药复现 | cgfail60 0.479 vs none 0.613 | **0.0104** | p<0.05 | **满足** |
| **P2** 跨任务失败无害 | pyfail60 0.642 vs none 0.613 | **0.9591** | p≥0.05 | **满足** |
| **P3** 极性是关键 | pyfail60 0.642 vs pyf60 0.719 | 0.1304 | p<0.05 | **不满足** |
| 附加(同极性,只差来源) | cgfail60 0.479 vs pyfail60 0.642 | **0.0148** | — | 显著 |

对照 `PREREG_ROUND4.md` 的四种结果表:`cgfail60 < none` 且 `pyfail60 ≈ none`
→ **主假设成立:毒性 = 同任务 × 失败。**

## 3. 这一轮改写了什么

Round 3 之后留下的说法是「区别不在**是否出错**,而在**是否恢复**」。
**P3 是对这句话的直接检验,它没有通过。** 固定来源为跨任务、只改极性,
错误恢复(0.719)高于失败(0.642),方向对,但 n=8 下 p=0.13,不能宣称。

真正被数据支持的是**来源**这条腿:
- 同任务失败 vs 不掺:**显著更差**(p=0.010)
- 跨任务失败 vs 不掺:**完全无差异**(p=0.959)
- 同任务失败 vs 跨任务失败(同极性):**显著**(p=0.015)

所以结论应表述为「**毒性来自目标任务自己的失败轨迹**」,而不是「失败轨迹本身有毒」,
也不是「关键在于是否恢复」。后两种表述都比数据允许的更强。

## 4. 一个必须标注的采样问题

`r4_cgfail60` 与 `r3_pyf60` 的 `--save-every` 同为 6、C/B 同为 2.0,但前者只跑了
~27 个优化步、后者 ~36 步,B-token 轴上稀疏约 26%(平均间隔 5899 vs 4692)。

原因:填满同样的 60k 辅助预算,codegen 失败轨迹需 55 条(平均 1084 tok/条),
pyfix 只需 88 条(平均 670 tok/条);序列按 4096 padding 打包,**短序列多 → micro-batch 多
→ 优化步多 → 采样更密**。**`save_every` 相同并不意味着采样密度相同**——这是 Round 3
那个 checkpoint 间距偏差换了一种成因,靠调 `save_every` 补偿混合比例的做法在这里无效。

处理方式:`tu/analysis/speed_grid.py` 把所有 run 插值到统一 B-token 网格后重算 AUC。
**四格的原生与统一网格 AUC 差 ≤ 0.003,四项检验的结论逐条一致**(P1 p=0.0104,
P2 p=1.0000,P3 p=0.1049,附加 p=0.0148)。故主指标不由采样密度驱动。

`tt@τ` 对密度敏感(采样越稀,穿越点只会被检得越晚),会**放大** cgfail60 的表观毒性,
因此本轮只作次指标,不作证据。

## 5. 剂量

原始毒药观测在 C=230k,本轮为 **60k**(低约 3.8 倍),效应依然显著。
预注册中那条「null 不算否证、需补 230k 臂」的条款本轮**未被触发**——
效应在更低剂量下就已复现,这加强而非削弱结论。

## 6. 与 Round 3 的关系

Round 3 判定催化剂假设不成立(`pyf60` 打不赢预算对照 `b90k`,p=0.88)。
本轮不改变该结论:`r3_pyf60` 在此仍是四格中最高的(0.719),但它的对照是 `b90k`,不在本设计内。
本轮回答的是另一个问题——**唯一通过全部对照的那个效应(毒药),换一个目标任务还成不成立**。答案是成立。

## 7. 复现

```
# 16 个 run(2 条件 × 8 seeds),每条件一个 8×H100 任务
COND=cgfail60 bash transfer-unit/scripts/r4_fanout.sh
COND=pyfail60 bash transfer-unit/scripts/r4_fanout.sh

# 判读
python -m tu.analysis.speed      --runs-dir ../runs --pattern "r4_cgfail60_s*" --task codegen
python -m tu.analysis.speed_grid --runs-dir ../runs --task codegen \
    --patterns "r3_none_s*,r3_pyf60_s*,r4_cgfail60_s*,r4_pyfail60_s*"
```

环境:Merlin i18n-tt · group `data-oec-magellan-snl2`(723) · cluster 4 (cloudnative-maliva) ·
镜像 `tns_train_sft`(CUDA 12.9;原 `image_912_va` 是 CUDA 11.3,flashinfer 的 GDN kernel
需要 `-std=c++20`,评测会崩)。
