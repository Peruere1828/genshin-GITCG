---
title: "001 — search-as-teacher / MCCFR 蒸馏"
kind: failure
status: limited
freshness: 2026-10-27
machines: [local]
authority: evidence
related: ["PLAN.md#d23", "PLAN.md#d26", "reports/search/a4_search_budget_ladder.md"]
---

# 001 — search-as-teacher / MCCFR 蒸馏

## 假设

用 MCCFR/搜索 teacher 给根策略和根价值，蒸馏进 CVPN，能让模型超过脚本专家。

## 设置

- 搜索：`envs/cfr_search.py` + `train/cfr_solver.py`；public belief root；depth-1；候选裁剪按 prior/引擎序。
- 蒸馏：`train/distill.py`（soft CE + value MSE + belief loss）。
- 评测：20 对手、配对种子、纯策略；A4 预算阶梯。

## 结果

- 全池配对 Δ 显著为负（约 −0.066/−0.114，不同预算）。
- 搜索覆盖 ~87–89% 专家动作但净负；根价值区分度 ≈0。
- 软目标蒸馏 ≈ 冷启动 BC，无可测正增益；放量只放大有害信号。
- 证据：`reports/search/a4_search_budget_ladder.md`、`reports/train/a3_m3_cpu_screening.md`、`reports/explore/plan_strength_2day_kaggle.md`。

## 归因（2026-10-27 复核）

- **测量假象**：`envs/cfr_probe.py` 把 identity 价值头的负值当概率截断到 `[0,1]`，并把 acting-player 视角当 player-0；这会让所有劣势叶子价值恒为 0，兄弟排序被抹平，足以制造“根价值区分度≈0”。已修（D26）。
- **候选裁剪是任意序**：无 prior 时按引擎选项序取前 C 个，等于对随机叶子取 max，乐观偏差。
- **搜索价值 target 没有独立头**：outcome 头被硬切成 search value，语义混用。
- 因此“搜索在当前实现下无增益”成立；“搜索方法本身有害/不可用”不成立。

## 重开条件（全部满足）

1. parity PASS（`docs/PLAYBOOK.md` §4）。
2. value head 校准：AUC ≥0.6，within-matchup AUC ≥0.55；搜索值用独立头/Huber（或连续值 HL-Gauss）。
3. 候选裁剪非任意：prior/top-k + 100% 合法动作覆盖，不静默删 action。
4. 门控式覆盖：只在 value spread 足够时覆盖专家动作。
5. 小预算 search 直接强度评测先显著 > expert/BC，再谈蒸馏。

## 不要重演

- 不要在旧 `cfr_probe` 语义上重跑；先跑 D26 的修复与重算。
- 不要把 depth-1 + expert 基座的搜索值当成独立 `V^σ`。
- 不要在没有 value resolution 的情况下放量到 A800。
