---
title: "009 — 定向补采修 floor：单一 matchup 有效，整体认证仍未过"
kind: failure
status: limited
freshness: 2026-11-20
machines: [local, kaggle-t4]
authority: evidence
related:
  - "reports/explore/hist_priv_prereg_20261120.md"
  - "docs/failures/008-history-priv-neutral.md"
---

# 009 — 定向补采 + PG 方差锚

## 假设

1. **定向补采**：dvalin 的 floor 卡在几个 matchup 上，用**脚本专家**当 teacher 补采这些 matchup，
   BC 能学到专家的走法，把 rate 拉上去（用当前模型补采只会再写"输的走法"）。
2. **PG 方差锚**：1 epoch PG 在 seed 间摆动 ±10pp（`failures/008`），加
   `KL(π‖π_warmstart)` 约束可以把它压到可测 2–3pp 效应的水平。

## 结果

### 2.1 floor 的真实结构（n=80/matchup，3 组种子并表）

| matchup | 我方（teacher） | expert | expert Wilson low | 是否豁免 |
|---|---:|---:|---:|---|
| `unyielding_geo` | 0.588（low 0.478） | 0.600 | 0.490 | 否 → 我方过 |
| `shenhe_double_swirl` | 0.562（low 0.453） | 0.613 | 0.503 | 否 → 我方压线过 |
| `dvalin_bonk`（镜像） | 0.438（low 0.334） | 0.475 | 0.369 | **是** |
| `double_geo_navia` | 0.487（low 0.381） | 0.637 | 0.528 | 否 → **唯一硬阻塞** |

**这推翻了 `failures/008` 里"两个 matchup 差 30pp、不可达"的判断**：那是 n=20 的噪声。
真实差距是 `double_geo_navia` 的 **+15pp**，其余 matchup 或已过线、或被 D25 豁免。

### 2.2 定向补采的效果

200 局/对手，expert teacher，seeds 9400–9599，混入 2000 局 base（共 2600–2800 局）：

| 臂 | 屏（seeds 9700, n=400）Δ | `double_geo_navia`（n=20） | `unyielding_geo` |
|---|---|---:|---:|
| teacher（参照） | +0.060 | 0.50 | 0.65 |
| hist2 BC-20（弱 2 matchup） | +0.040 | — | 0.70 |
| hist3 BC-20（+镜像） | +0.040 | 0.50 | 0.75 |
| **hist4 BC-20（+dgeo）** | **+0.068** | **0.65** | 0.60 |
| hist4 BC-30 | +0.052 | 0.65 | 0.80 |

→ 补采**确实**把目标 matchup 的 rate 拉起来（dgeo 0.50 → 0.65），且 pooled Δ 一度到 +0.068。

但**同一模型在不同种子集上不可复现**：hist2 BC-20 的 `unyielding_geo` 在 seeds 9700 是 0.75
（15/20），在 seeds 9600 的 800 配对认证里是 **0.25**（10/40）；整体 Δ 也从 +0.040 掉到 **−0.004**。
40 局/matchup 的标准误约 ±8pp，单组种子的 per-matchup 结论不可用。

### 2.3 PG 方差锚（这条是正结果）

`--pg-anchor-kl 0.1`（`KL(π‖π_init)`）在 3 个 seed 上：

| 配方 | val_acc（3 seeds） | 极差 |
|---|---|---:|
| 无锚（`failures/008`，A1） | 0.556 / 0.625 | 6.9pp |
| 无锚（A0） | 0.749 / 0.820 | 7.1pp |
| **有锚（hist2）** | 0.857 / 0.829 / 0.859 | **3.0pp** |

引擎侧：有锚 arm 在 800 配对局（seeds 9600）上 Δ = **+0.040** [+0.009,+0.071]，
与 teacher 完全一致——锚把它钉在 teacher 附近（这正是方差锚该做的事），也意味着**没有增益**。

## 结论

- 定向补采是**有效但局部**的：能让单个 matchup +15pp，但在 400–800 局的认证粒度上不可复现；
- PG 方差锚**是有效的方法**（3pp vs 7pp），应作为此后所有 1-epoch PG 的默认；
- 第三個 PASS 的判定现在只剩 `double_geo_navia` 一个 matchup（非豁免），需要在 **≥80 局/matchup**
  的功率下认证。

## 重开条件

- 认证一律 **≥40 seeds（1600 配对局，80 局/matchup）**；40 局/matchup 的结论不接受；
- 定向补采只在"差距 ≥10pp 且 expert 不豁免"的 matchup 上做，且必须在新种子上验证，
  单组种子的提升不算；
- 若要继续推 dvalin：先看 `cert_tg3_bc_e20_s9900`（40 seeds）是否同时满足
  `pooled ≥0.75` + 非豁免 matchup Wilson low ≥0.45 + Δ CI_low > 0。
