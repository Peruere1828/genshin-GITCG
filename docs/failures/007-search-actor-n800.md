---
title: "007 — 搜索当 actor（MCCFR 直接选点）"
kind: failure
status: negative
freshness: 2026-11-20
machines: [local]
authority: evidence
related: ["PLAN.md#d30", "PLAN.md#d26", "reports/search/a4_search_budget_ladder.md", "reports/explore/b3_pilot_prereg_20261120.md"]
---

# 007 — 搜索当 actor

## 假设

在决策点上直接用搜索（MCCFR / fork-MC）替换或修正基座策略的动作，能超过同 deck 的脚本专家。
与 001 的区别：001 是"搜索当 teacher 再蒸馏"，本条是"搜索直接出招，不蒸馏"。

## 设置

三套实现，全部纯策略、配对种子、双方各半、固定 20 套池：

1. **fork-MC**（`envs/fork_search.py`）：从 `is_resumable()` 边界快照 fork，注入候选，用固定策略打完求均值（A4）。
2. **MCCFR**（`envs/cfr_search.py` + `train/cfr_solver.py`，外采样 MCCFR，Lanctot 2009）：公共信念粒子为根，
   单步 fork 为转移，语义 key 为动作，终局/前沿为叶值（S-C 预算阶梯）。
3. **belief 根化 fork-MC**（`envs/fork_search.py` + `--belief-root`）：同 1，但根换成 `reps.belief.sample_belief_roots`
   的公开信息确定化（B3）。

## 结果

| 实现 | 预算 | 局数 | expert | search | Δ | 95% CI | 改决策率 | s/搜索 |
|---|---|---:|---:|---:|---:|---|---:|---:|
| MCCFR | 8 iter, **depth 1**, 4 cand, 8 srch | **800** | 0.5613 | 0.4950 | **−6.63pp** | **[−12.5, −0.8]** | 0.866 | 0.44 |
| MCCFR | 8 iter, **depth 2**, 4 cand, 4 srch | **800** | 0.5613 | 0.4475 | **−11.38pp** | **[−16.9, −5.8]** | 0.894 | 5.78 |
| MCCFR | 4/8/16 iter, d1, 小样本阶梯 | 72×5 | 0.4306 | 0.389–0.528 | −0.042…+0.097 | 全部含 0 | 0.72–0.96 | 0.37–6.6 |
| belief fork-MC | 3 roots, 2 rollout, top_k 3 | 32 对 | — | — | −6.25pp | SE 4.28pp | 2/32 局改结果 | 0.44 |
| live fork-MC（偷看） | 同上 | 32 对 | — | — | +6.25pp | SE 4.28pp | 2/32 局改结果 | 0.44 |

来源：`reports/search/ladder_fullB_i8_d1_c4_ms8_20261012T111500.json`、
`ladder_fullD_i8_d2_c4_ms4_20261012T115212.json`、`reports/search/a4_search_budget_ladder.json`、
`reports/search/b3_{belief,live}_r3_*.json`。

## 归因

1. **不是价值头的锅**：depth-1 臂叶子是**终局胜负**，完全不经过价值头，仍然 −6.63pp 且 CI 不含 0。
   D26 修的价值语义只能解释 depth-2 臂更差（−11.38pp）的一部分。
2. **不是"搜索太弱以致不动"**：MCCFR 改了 **87–89%** 的决策，是积极地在改变打法；而 B3 的 fork-MC 只改了
   2/32 局的结果——那一个是"没生效"，这一条是"生效了但更差"。
3. **不是信息泄漏**：MCCFR 用 `reps.belief` 的公开粒子做根，`public_infoset_key` 只看公开字段，
   `tests/test_cfr_solver.py::test_public_infoset_key_ignores_hidden_truth` 钉住。
4. **最可能的机制**：8–16 次外采样 MCCFR 在 ~10³ 动作的博弈上远未收敛，且叶子只给 0/1 终局信号；
   搜索结果是在**对抗自己的 rollout 策略**，而不是真实对手；同时丢掉了专家先验的稳定性。
   文献给出的修法（WEVA 动作抽象、RL-CFR、PCFR+、one-sided GT-CFR）每一条都是独立的工程项，
   且都需要先证明"搜索在该抽象上收敛"。

## 结论

**搜索从决策路径移除**（D30）。保留 `envs/fork_bridge` / `envs/snapshot` 作为引擎能力与离线标注工具；
`envs/fork_search` 的选点逻辑与 `envs/cfr_search` 的 CFR 选点不再作为策略使用。

## 重开条件

同时满足才允许重开：

1. 有**收敛诊断**：在固定子博弈上与精确解（小规模可枚举）对齐，根策略距离随迭代下降；
2. 有**动作抽象**（WEVA / RL-CFR 式），使每节点候选 ≤ ~16 且覆盖率可测；
3. 在 n=800 同口径上先证明**非劣**（Δ CI 下界 ≥ −0.02），再谈增益；
4. 若仍用深度截断，叶值必须先在 `replay_branch` 真值测试上优于常数 0.25。
