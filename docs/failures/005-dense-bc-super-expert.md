---
title: "005 — 稠密 BC 超专家"
kind: failure
status: limited
freshness: 2026-10-27
machines: [local, a800]
authority: evidence
related: ["reports/datafirst_bc.md", "reports/train/bc_scale_datafirst.md"]
---

# 005 — 稠密 BC 超专家

## 假设

更多专家语料 + 更大网络/更长训练，能让 BC **超过**同 deck 脚本专家。

## 设置

- 每 deck 独立 BC；20 对手池；compact 语料；d128/d256；6–150 ep；8k–48k 样本。
- 认证：配对种子、双方、纯策略、20 seeds。

## 结果

- BC **从未显著超过**同 deck 专家；最好统计打平（dgeo +0.007 ns）。
- 3/7 组首次过 pooled ≥0.50（strict cluster CI）：dvalin 0.701 / ungeo 0.709 / dgeo 0.689。
- 数据量 8k→48k（同 epoch）不显著；5→15 ep 显著 +0.076。
- `natlan_battleship` 的专家自己仅 0.225–0.285，BC 0.140；floor 不是数据能救的。
- 证据：`reports/datafirst_bc.md`、`reports/train/bc_scale_datafirst.md`。

## 归因

- 目标是**模仿**，上限就是 teacher；BC 没有“超过专家”的机制。
- 语料是 expert-vs-expert，eval 是模型自己的状态分布 → covariate shift。
- 6 ep 预算被低估；数据量轴在 6 ep 下被截断。
- teacher-limited deck 的 floor 由 deck 本身强度决定，不是策略可学。

## 重开条件

- 只把 BC 当**底座**：先到 parity，再接 PG/自博弈（`docs/PLAYBOOK.md` §1）。
- 需要 DAgger 时先做 200 局一致率诊断，gap ≥3pp 才做；DAgger 只修分布，不超专家。
- 需要 floor 时先分层 teacher-limited vs 可修 matchup（`docs/failures/003`）。

## 不要重演

- 不要把“更多专家数据”当超专家手段。
- 不要用 `val_acc` 选 checkpoint。
- 不要用 6 ep 的结论判数据量轴。
