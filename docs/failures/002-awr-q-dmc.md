---
title: "002 — AWR / Q-DMC / 离线 RL"
kind: failure
status: invalid
freshness: 2026-10-27
machines: [local, a800]
authority: evidence
related: ["reports/train/bc_scale_datafirst.md", "reports/explore/lit/f02_dagger_offline.md"]
---

# 002 — AWR / Q-DMC / 离线 RL

## 假设

用 AWR、Q-learning（DouZero 式 DMC）或离线 RL 在专家语料上超过 BC/专家。

## 设置

- AWR：`--loss awr`，权重 `exp(β(G−V(s)))`；β=0.5/1/2；15–150 ep。
- Q：`--loss q`，只监督被走动作；`--q-target episode|discount`；K̄≈25.5，max 256。
- DMC 参考：DouZero（在线自博弈、γ=1、千万局）。

## 结果

- AWR ≈ BC（15ep），150ep β=1 降到 0.352（−0.156），β=2 更差。
- Q 的 val MSE 最好（0.0609）但策略最差（15ep pooled 0.068 / 150ep 0.275）；与教师一致率 0.187。
- 离线 Q/ DMC 无正结果；胜局筛选 −0.14。
- 证据：`reports/train/kaggle_*_awr_*.json`、`kaggle_h4m_q_disc_e100.json`、`reports/explore/plan_strength_2day_kaggle.md`。

## 归因

- **AWR 定义级问题**：`adv = values - value.detach()`，权重只依赖 `(G,V(s))`，**与动作无关** ⇒ 只是状态重加权 BC，无法改变状态内动作排序；β 越大越退化成胜局筛选。
- **Q 的外推误差**：只监督被走动作（`logits.gather(teacher)`），无 CQL/双 Q/target net/BC 锚；离线 argmax 在未走动作上外推（Fujimoto 2019）。
- **DMC 规模差**：DouZero 是在线自博弈 + 千万局 + γ=1；我们是离线 8k–30k 局、γ^k 缩放，差 3–4 个数量级。
- 因此“AWR/Q/DMC 在本项目全负”不能推出“离线 RL 不行”；AWR 是无效实现，Q/DMC 是 setting 不匹配。

## 重开条件

- **AWR**：不重开；除非有 per-action advantage（同状态多动作标签）。权重置换对照只用于归档证据。
- **Q**：只试一次离散 TD3+BC（`CE + α·BC 锚` 或 CQL 式 `logsumexp` 支撑约束）；先做 C1/C2 诊断（off-support argmax、γ=1）。
- **DMC**：不在本机预算内；需在线自博弈 + 千万局 + 足够 GPU 时间，且先过 parity。

## 不要重演

- 不要扫 AWR β 网格。
- 不要用 `--q-target discount` 直接比较跨深度 Q。
- 不要用 Q 的 val MSE 选模型。
