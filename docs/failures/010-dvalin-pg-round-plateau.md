---
title: "010 — dvalin 再走一轮自博弈 PG：仍停在 parity（+0.021@800）"
kind: failure
status: negative
freshness: 2026-11-21
machines: [local, kaggle-t4]
authority: evidence
related:
  - "docs/failures/009-targeted-floor-repair.md"
  - "reports/explore/hist_priv_prereg_20261120.md#11"
---

# 010 — dvalin 自博弈 PG 再一轮：没有越过 parity

## 起点（这是差 0.003 的那一次）

`tg3_bc_e20_s0` 在 40 seeds / 1600 配对局（seeds 9900–9939）：
pooled **0.7725**、floor 在 D25 下干净、配对 Δ **+0.019 [−0.003,+0.041]**。

## 本轮设定

| 项 | 值 |
|---|---|
| 采集 | teacher = `tg3_bc_e20_s0`，`#t=1.0`，seeds 10000–10059 × 20 对手 = **1,200 局**（53,282 train / 5,803 val） |
| 训练 | Kaggle `hist5`：`--loss pg --pg-anchor-kl 0.1 --lr 3e-5 --epochs 1 --no-privileged --init-from tg3_bc_e20_s0.pt`，3 seeds |
| 训练稳定性 | val_acc **0.8454 / 0.8502 / 0.8492**，极差 **0.48pp**（锚确实压住了） |
| 筛选 | 800 配对局，seeds 10100–10119（全新） |

## 结果

```
pooled=0.721  Δ=+0.021  95%=[-0.012,+0.055]  n=800
gates (D25): pooled=True floor=False → 违规 klee_mavuika_mualani
   us 0.575 (low 0.422) vs expert 0.650 (low 0.495)
```

**三个 seed 的训练几乎相同（0.48pp），但整轮没有带来增益**：Δ 从 +0.019（1600 局）
到 +0.021（800 局）在噪声内；pooled 反而从 0.7725 掉到 0.721。

## 归因

1. **KL 锚把更新钉在 teacher 附近**（coef 0.1、1 epoch、53k 样本）——它压住了方差，
   也压住了改进；无锚时则是另一个极端（−0.128…+0.015，`failures/008`）。
   在 1-epoch REINFORCE、batch 256（≈5 局/批）的预算下，这个区间里没有可用的"中间档"。
2. **pooled 本身随种子集漂移**：同一臂 0.7725（seeds 9900）vs 0.721（seeds 10100），
   差 5pp。绝对门槛 ≥0.75 在 800 局粒度上不是稳定判据。
3. **floor 的阻塞 matchup 在轮换**：seeds 9900 是 `double_geo_navia`（D25 后豁免），
   seeds 10100 变成 `klee_mavuika_mualani`（不豁免）。40 局/matchup 的 Wilson low
   标准误 ±8pp，单轮认证的 floor 结论同样不稳。

## 结论

dvalin 在**当前 recipe（1 epoch PG + 每轮 1–2k 局自博弈）**下的天花板是 **专家 parity**，
与方法学上"再走一轮"的期望（+2~3pp）不符。第三个 PASS 未达成：仍为 2 个
（`double_geo_navia` 0.880、`unyielding_geo` 0.909）。

## 重开条件

1. **换更新方式**，不是换种子：PG 的每批样本太少（≈5 局），advantage 方差主导。
   需先做到"每批 ≥50 局"或引入 off-policy 修正（V-trace/Retrace），再谈增益；
2. **每轮数据 ≥2,000 局**且**连续 ≥3 轮**，每轮独立预注册、独立种子认证
   （历史 ladder 在 dgeo 上是 3 轮才跳变，dvalin 已 4 轮不动）；
3. **认证口径固定**：40 seeds / 1600 配对局（80 局/matchup）+ D25 floor，
   任何 800 局以内的 Δ 只作筛选，不写结论。
