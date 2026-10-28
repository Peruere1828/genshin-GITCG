---
title: "003 — matchup 重加权 / 胜局筛选"
kind: failure
status: negative
freshness: 2026-10-27
machines: [local]
authority: evidence
related: ["reports/explore/plan_strength_2day_kaggle.md", "reports/explore/lit/f06_targeted_data.md"]
---

# 003 — matchup 重加权 / 胜局筛选

## 假设

按 matchup 给样本加权（deficit/balance）或只保留胜局，能修补 floor / 提升强度。

## 设置

- H5：`--matchup-weight deficit`，权重来自 `store.value`（teacher 对局胜负），均值归一化。
- 胜局筛选：只保留 trainee 赢的局做 BC。
- 评测：同 deck 专家基线、配对种子、纯策略。

## 结果

- H5 在 SA 上 −9.1pp，dvalin 上 −3.4pp；多数 matchup 权重只在 0.84–1.22。
- 胜局筛选 −0.14；`%BC`/return-conditioned 同样不适用。
- 证据：`reports/explore/plan_strength_2day_kaggle.md` §Day2b、`reports/train/bc_scale_datafirst.md`。

## 归因

- **权重方向错**：`store.value` 是 teacher 的胜负，不是 student-vs-expert 的可修 gap；deficit 会抬高“老师也输”的 matchup，正好放大最不可学的标签。
- **没有新信息**：只改旧语料的损失权重，不增加目标 matchup 的状态/对手行为覆盖。
- **组内样本太少**：每 matchup 约 20 局，deficit 估计噪声大；再归一化后只微调 0.84–1.22。
- **胜局由对手/种子决定**：同一策略对不同对手的胜负不是策略质量信号。

## 重开条件

- 只在**采集层**做混合：target 25–35% + uniform/anchor 65–75%；单 matchup ≤25%。
- 目标只选 `expert Wilson low ≥45%` 且 `student point <45%` 的可修 matchup；teacher-limited 不补。
- 用 fresh seeds 选 target，另用不重叠 seeds 报分。
- 证据：`docs/PLAYBOOK.md` §1/§2、`reports/explore/lit/f06_targeted_data.md` E1。

## 不要重演

- 不要用 `--matchup-weight deficit` 原实现。
- 不要用整局回报做过滤/条件化。
- 不要只采失败 matchup。
