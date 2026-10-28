---
title: "004 — 20-deck 单模型 generalist"
kind: failure
status: negative
freshness: 2026-10-27
machines: [local, a800]
authority: evidence
related: ["reports/explore/plan_strength_2day_kaggle.md", "reports/SPECIALISTS.md"]
---

# 004 — 20-deck 单模型 generalist

## 假设

用更大网络 + 20-deck 混合语料训一个共享 trunk，能同时服务所有 deck。

## 设置

- 多 deck 混合 store（4-deck/20-deck）；d256/d384/d512；40/250 seeds per matchup；150 ep 量级。
- 评测：per-deck 认证，固定 20 脚本池。

## 结果

- 20-deck 单模型：8 个认证 deck 里 **7 个显著变差**（负迁移）。
- 4-deck 250 seeds：val_acc 0.8377；4-deck 40 seeds：0.7281；20-deck 40 seeds：0.6932；加容量 d512 仍 0.6753。
- 约 **3/4 来自每 matchup 密度**，1/4 来自任务数；容量不是瓶颈。
- 证据：`reports/explore/plan_strength_2day_kaggle.md`、`reports/SPECIALISTS.md`。

## 归因

- 共享 trunk 在同一轮里收到方向相反的更新（SA vs dgeo）；deck 间动作/牌池差异大。
- 混合语料稀释每 matchup 密度；每 deck 的状态分布和 teacher 行为不同。
- 评测是按 deck 的强度，单模型没有“deck 路由”以外的收益。

## 重开条件

- 只有在“单模型条件化 population”（如 NeuPL/FiLM）且每 deck 密度足够时才考虑。
- 先做 per-deck specialist；需要 router 时只做 deck 级、公开阵容、100 局/cell（`docs/PLAYBOOK.md`）。
- 不要用 20-deck 单模型做主线。

## 不要重演

- 不要把“加容量”当 20-deck 负迁移的解。
- 不要用 pooled 跨 deck 掩盖单 deck 退化。
