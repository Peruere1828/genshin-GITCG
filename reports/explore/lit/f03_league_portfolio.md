# League / population training + specialist portfolio / router

> 调研日期：2026-10-27。只读 `reps/`、`envs/`、`train/`、`eval/`、`scripts/`、`reports/` 与 `refs/` 的必要部分；未改代码、未跑训练/评测、未写 `data/`。
> 口径：纯策略（LLM 不参与），固定 20 套脚本对手，逐 deck 报分，配对种子，信息只来自 `notification.state`。
> 成本口径：采集 ~4,000 局/h（12 核）、评测 ~1,500 局/h；文中 GPU·h 只指 A800 训练，采集/评测是 CPU。
> 本文对应老师问题 3（league / population training），并覆盖 P1 的 specialist portfolio + router。本文把“论文声称”和“我们可迁移的证据”分开写；不能核实的数字标「未核实」。

---

## 0. 一句话结论（可执行）

**结论：固定池上做“约束最优回应”，不做 Nash；每 deck 独立 specialist；router 只用对手阵容、从 100 局/cell 起步。**

**不要为固定 20 套脚本对手去追 Nash 均衡。** 正确目标是“每 deck 的约束最优回应”：主效用是 20 个对手等权的 pooled 胜率，硬约束是每对手 floor（Wilson 下界 ≥ 0.45）；训练 surrogate 用 `pooled − λ·floor_deficit` 或 `CVaR_3`，采样用 PFSP `(1−w)^2` 混合 20% 均匀 + 10% 最差 3 个对手，历史 checkpoint 只占 20–30%。防漂移按“回放混合 + incumbent 非劣门禁 → root-anchor KL → 历史池 + PFSP”的顺序加，别一上来做 EWC。路由只做 **deck 级、按对手阵容的硬路由**，不做逐决策 contextual bandit；每个 `(arm, opponent)` 至少 100 局，经验贝叶斯收缩 + LCB + holdout 种子，收益阈值 +3pp 且最差格不降 >3pp。扩张先扩 deck，不补已 PASS 的 floor：候选是 dvalin / shenhe / electro（expert 0.65–0.74，floor 违规 1–2），SA 与 Skirk 系是结构性劣势，不进阶梯。

---

## 1. 与我们问题的对应关系（Q1–Q5）

### 1.0 先把三个“league”分开

| 层 | 是什么 | 在我们这里的正确用法 | 不该做什么 |
|---|---|---|---|
| **对手池（opponent league）** | 20 套固定脚本 + 历史 checkpoint + （可选）exploiter | 训练采样分布 + 评测锚点 | 不要把脚本对手当成会学习的 Nash 对手 |
| **专家池（specialist league）** | 每个 deck 一个 specialist checkpoint | 独立训练、独立认证、按 deck 组合 | 不要用跨 deck 的 PSRO payoff 矩阵；不同 deck 不是同一游戏的策略 |
| **路由器（router）** | 给定对手阵容，选哪个 specialist/checkpoint | deck 级硬路由；离线选择 + holdout 验证 | 不要逐决策切换；不要在选它的同一批种子上报分 |

关键判断：**我们的部署对手是固定脚本，不是 best responder。** 对固定策略的最优回应是 pure exploitation；Nash/PSRO 的“稳健混合”会把胜率让给不需要稳健的地方。PSRO/Nash 只在两种场景有价值：① 同一 deck 内部的历史 checkpoint / exploiter 池；② 将来做“单模型条件化 population”（NeuPL 类）时。当前 per-deck specialist 路线不需要跨 deck 的 meta-game。

---

### Q1：league 应该优化什么？min-win-rate 还是 pooled？

**结论：pooled 是主效用，min/floor 是约束；不要把 raw min 当训练目标。** 如果必须用一个标量做模型选择，用 `pooled − λ·floor_deficit` 或 `CVaR_3`，不要用单格 min。若部署对手分布不是均匀（例如人类 meta），把 `1/20` 换成 `π_o`；但认证仍按 PLAN 的固定 20 套池，不能混。

记 deck `d` 对对手 `o` 的胜率为 `w_{d,o}`（双向、配对种子；`n_{d,o}` 局）：

```text
Pooled_d      = (1/20) * Σ_o w_{d,o}
Worst_d       = min_o w_{d,o}
CVaR_{d,k}    = (1/k) * Σ_{o ∈ 最差 k 个} w_{d,o}          # k = 3 或 5
FloorDeficit_d= Σ_o max(0, 0.45 − w_{d,o})
J_d           = Pooled_d − λ * FloorDeficit_d              # λ = 1.0 起步
```

为什么不是 raw min：

- `min_o w_o` 是单个 order statistic；40 局/cell 时，真实 0.60 的 matchup 可能样本掉到 0.45，直接优化 min 会过拟合到某一格噪声。
- 纯 min 会牺牲 19 个 matchup 去抬 1 个最差格；pooled 会掉，M4 的 pooled 门槛反而过不了。
- `CVaR_3` 只看最差 3 个，比 min 稳，又比 pooled 更关注长尾；适合做训练 surrogate。
- **门禁仍然用 per-opponent Wilson 下界 ≥ 0.45**；这是约束，不是训练目标。

可执行的“约束最优”写法：

```text
maximize   Pooled_d
subject to Wilson_low(w_{d,o}) >= 0.45   for all o = 1..20
```

工程上不用真解约束优化；用 hinge penalty 或 prioritized sampling 近似：

```text
J_d = Pooled_d − 1.0 * Σ_o max(0, 0.45 − w~_{d,o})
w~_{d,o} = (wins_{d,o} + α) / (n_{d,o} + α + β)     # α = β = 10，Beta 平滑
```

**采样分布（每轮 2,000 局）：**

```text
q_o = 0.70 * PFSP_o + 0.20 * (1/20) + 0.10 * Worst3_o
PFSP_o = (1 − w~_{d,o})^2 / Σ_j (1 − w~_{d,j})^2
Worst3_o = 1/3 if o in 最差 3 个 else 0
# 每对手上限 15%（300 局），下限 2%（40 局），防止饿死已强的 matchup
```

这与 AlphaStar 的 PFSP 同构：补充材料 `multiagent.py` 里 `pfsp(win_rates, weighting="squared")` 就是 `(1−x)^2` 归一化（[AlphaStar supplementary pseudocode](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip)，实际文件在 `pseudocode.zip` 内；正文付费墙，代码已核对）。注意：**PFSP 不是 Lanctot 2017 的论文**；见 §7 的 citation hygiene。

**一个当前代码事实（值得先修）：** `train/data.py::collect_samples` 只把 teacher 放在 player 0（`role="p0"`），`train/collect.py` 的 `CollectTask` 也没有 `side` 字段。也就是说现有 PG 语料只教“先手侧”，而 `scripts/eval_protocol.py` 的认证是双向。建议在 `CollectTask` 加 `side`/`swap`，奇数 seed 交换 deck/opponent 与 policy spec；这是比任何 league 技巧都便宜的潜在修正（**未验证收益，先做 A/B**）。

---

### Q2：specialist 继续 PG 时如何防漂移/遗忘？

四件套的作用不同，不能互相替代：

| 机制 | 防什么 | 最小实现 | 成本 | 失败模式 |
|---|---|---|---|---|
| **回放混合（replay mix）** | 新一轮数据把旧 matchup 冲掉 | 每轮 store 混 20–30% 上一轮 replay；`datafirst_bc_store --replays dir1 dir2` 或 `ConcatStore` | 极低 | 旧数据比例太高 → 学不动新策略；**直接混进 PG 是 off-policy，需 importance weight 或只用于辅助 BC/价值/KL** |
| **incumbent 非劣门禁** | 新 checkpoint 比当前认证版差 | `eval/scoreboard.paired_difference` + `noninferiority_gate`，`CI_low ≥ −0.02`；per-opponent Δ 只作 flag（n<100/cell 不硬判） | 评测 800 局/轮 | 只在 pooled 非劣，长尾已掉 |
| **root-anchor KL** | 多轮累积漂移（chain drift） | 冻结“根 anchor”（第一个 parity+ BC），KL(anchor‖policy) 加权 0.1–0.5；anchor logits 在固定 store 上预计算 | 10–20% 训练开销 | KL 太强 → 学不动；anchor 本身弱 → 锁死在弱策略 |
| **历史池 + PFSP** | 忘记旧策略/旧对手；被旧版本针对 | 20–30% 局对历史 checkpoint；PFSP `(1−w)^2`；只保留最近 4 个 | 20–30% 采集 | 历史池太大 → 目标从固定池偏移 |
| **EWC** | 参数级遗忘 | 对角 Fisher + `Σ F_i(θ_i−θ*_i)^2` | 高（Fisher 估计） | 我们数据非独立同分布，Fisher 噪声大 |

**推荐顺序（按性价比）：**

1. **先做 replay mix + incumbent 非劣门禁**（不改模型，只改数据混合和评测脚本）。这是最小闭环；如果这一步就稳住，先别加 KL。
2. **再加 root-anchor KL**。注意 anchor 选“根 BC”而不是“上一轮”，否则多轮锚点会跟着漂移，等于没锚。`--init-from` 继续用 incumbent，KL 用 root，两个角色分开。
3. **再加历史池 + PFSP**。历史池只放 20–30%，且只在第 2 轮以后启用；第 1 轮没有历史可放。PFSP 权重用 `w~` 平滑，不要用原始 0/1。
4. **最后才考虑 EWC/蒸馏**。`Distral` 的“共享 prior + 每任务 KL”适合将来做共享 trunk；现在 per-deck 独立模型不需要。

**AlphaStar 的可抄细节**（补充材料 `multiagent.py`，已读）：

- `Payoff` 对每对 `(home, away)` 做 0.99 衰减，避免早期对局永久主导。
- 主 agent 每 `2e9` steps 检查一次；只有“对历史池的最小胜率 > 0.7”或 `>4e9` steps 才 checkpoint。
- 15% 的局走 verification branch：先查 main exploiter（若对任一 exploiter 胜率 < 0.3，优先打它），再查 forgetting（对历史 checkpoint 的胜率，去掉单调后缀后若最小 < 0.7，优先打它）。
- PFSP 的 `variance` 权重 `x(1−x)` 用于“对某个主 agent 的历史池”；`squared` 用于全局 PFSP；`linear_capped` 用于 league exploiter。

我们的适配：**把“历史池”换成同 deck 的旧 checkpoint，把“固定池”保持在 70–80%。** 不要照搬 AlphaStar 的 50% PFSP + 35% self-play；它的目标是 league 稳健，我们的目标是固定池 exploitation。

---

### Q3：specialist portfolio + router 用什么信号？要多少数据？如何避免灾难？

**结论：先做 deck 级硬路由，不做逐决策路由。** 我们的对手阵容在开局就是公共信息；`(own_deck, opponent_lineup)` 是合法、稳定、免费的信号。逐决策的公共状态路由需要校准的价值头，当前 value head 还没有标定（搜索分支的负结果就是证据），不值得先做。

| 信号 | 可用时机 | 合法性 | 数据成本 | 建议 |
|---|---|---|---|---|
| 自身 deck id | 开局 | 公开 | 0 | 必用；它就是 specialist 的 key |
| 对手阵容/角色 id | 开局 | 公开（3 个角色可见） | 0 | **主路由信号** |
| 对手已打出的牌/ archetype | 局内前几轮 | 公开 | 中 | 只有对手阵容不可见时才需要；Hearthstone 用第一回合动作做到 78.9% archetype 准确率（[Eger & Sauma Chacón 2020](https://doi.org/10.1145/3402942.3402959)） |
| 公共状态（血量/骰子/场面） | 每轮 | 公开 | 高 | 只用于“局内切换”，需要价值校准 + 迟滞；暂不做 |
| 地图 | — | — | — | 七圣召唤无地图；忽略 |

**数据量：**

- 40 局/cell（20 seeds × 双向）时，单格 SE ≈ 0.079；我们在 `scripts/matchup_router.py` 的 10-seed 探针就是 0.475 vs 0.490，差异是噪声（commit `7929cec`，2026-10-18）。**不要再用 40 局/cell 选路由。**
- 100 局/cell（50 seeds × 双向）时，单格 SE ≈ 0.05；配合配对种子和经验贝叶斯收缩，可以做粗粒度路由（≥10pp 的格间差异）。
- 200 局/cell 时 SE ≈ 0.035；要做 5pp 级路由，至少这个量级。
- 3 个 arm × 20 对手 × 100 局 = 6,000 局/deck；2 个 deck = 12,000 局。按评测 ~1,500 局/h，约 8h；按采集 4,000 局/h，约 3h。

**最小 router（经验贝叶斯 + LCB）：**

```text
for each (arm a, opponent o):
    w~_{a,o} = (wins_{a,o} + κ * pooled_a) / (n_{a,o} + κ)    # κ = 20–50
    lcb_{a,o} = w~_{a,o} − z * sqrt(w~_{a,o}(1−w~_{a,o})/(n_{a,o}+κ))
choose a*(o) = argmax_a lcb_{a,o}
if top-2 lcb 差距 < δ (δ = 0.03): 回退到 pooled 最好的单 arm
```

这比 LinUCB 简单，但更合适：对手 id 是离散且已知的，不需要线性假设；LinUCB 只在“对手身份未知/在线变化”时有优势。若将来要做在线自适应，用 hybrid LinUCB（共享特征 + 每 arm 特征），因为稀疏数据下 hybrid 明显优于 disjoint（[Li et al. 2010](https://arxiv.org/abs/1003.0146)：1% 数据时 hybrid 比 UCB 的 CTR 高 10.3%）。

**避免灾难的六条硬规则：**

1. **只在已认证 non-inferior 的 cell 里路由**；某 arm 在某对手上没打过 100 局，不准选。
2. **收缩到 pooled**：`κ` 让稀疏 cell 退回该 arm 的总体水平，避免“20 局 5 胜 = 0.25”被当成真实差异。
3. **LCB 选择 + 回退**：top-2 重叠时回退到 pooled 最好的单 arm；不要硬 argmax。
4. **holdout 种子**：选择用 seeds A，报分用 seeds B；绝不在同一批种子上选路由再报路由。
5. **worst-cell 约束**：路由后的最差格不得比最佳单 arm 的最差格低 >3pp；否则路由器在长尾上制造灾难。
6. **不做逐决策切换**，除非有一个标定过的 value head 和迟滞机制；否则同一局内来回换策略会破坏 deck 的节奏。

**跨 deck 路由的特殊风险：** deck 在开局后不能换，所以“选错 deck”是灾难性的；只在对手阵容已知时做 deck 级选择。若对手阵容未知（人类随机匹配），默认用 pooled 最好的 generalist/specialist，不要用低置信度的 router。

**低成本 portfolio baseline：** 同一 deck 的多个 checkpoint 可以先试 weight averaging（model soup；[Wortsman et al. 2022](https://arxiv.org/abs/2203.05482)），它不需要 router、推理成本不变；如果 soup 已经≥最佳单 checkpoint，就先别做 router。不同 deck 的模型不能平均（结构/语义不同）。

---

### Q4：如何把 2 个 PASS specialist 扩到更多 deck？先补 floor 还是先扩 deck？

**结论：先扩 deck，不补已 PASS 的 floor。** dgeo/ungeo 的 floor 已经是 0 违规；dgeo 从 r3 0.890 到 r4 0.880，已经饱和。继续在它们身上加轮次的边际收益低于开新 deck。

当前证据（`reports/SPECIALISTS.md`，2026-10-27）：

| deck | expert pooled | specialist pooled | Δ | floor 违规 | 判断 |
|---|---:|---:|---:|---:|---|
| double_geo_navia | 0.7125 | **0.8800** | +0.1675 | 0 | PASS，当锚（r4 已饱和） |
| unyielding_geo | 0.8538 | **0.9087** | +0.0550 | 0 | PASS，当锚（可再 1 轮验证饱和） |
| dvalin_bonk | 0.7425 | 0.7800 | +0.0375 | 1 | **下一批** |
| shenhe_double_swirl | 0.6650 | 0.7025 | +0.0375 | 2 | **下一批** |
| electro_manifestation_nuke | 0.6550 | 0.6850 | +0.0300 | 2 | **下一批** |
| superconduct_aggro | 0.5088 | 0.5050 | −0.0037 | 11 | 结构性劣势，排除 |
| skirk / ice_water / beidou 等 | 0.18–0.38 | 0.09–0.31 | 负 | 15–19 | 结构性劣势，排除 |

**扩张判据（先用同一 20-seed 协议重算 expert ceiling，再决定是否开 PG）：**

```text
expand(deck) iff
    expert_pooled >= 0.60
    and expert_floor_violations <= 3
    and base_pooled >= expert_pooled − 0.03        # parity+ 底座
    and base_clean (fallback/error = 0)
```

理由：

- 如果脚本专家自己对全池都只有 0.19–0.50，且 15–19 个 matchup <0.45，那么任何“模仿/剥削专家”的 specialist 都不可能把 floor 修到 0；这是 deck 的结构上限，不是 PG 能解决的。
- PG 只在 parity+ 底座上有效；底座落后专家时，PG 会放大噪声（`reports/EXPLOIT_LADDER.md`：dvalin 用错底座 −0.06/−0.13；SA 三轮全负）。
- dvalin/shenhe/electro 的 expert 已经在 0.65–0.74，floor 违规只有 1–2 个，是最便宜的扩张对象；它们已经有 BC 底座（如 `dfbc_dvalin_bonk_d256`、`gc_shenhe_bc_d256_e150`、`gc_electro_bc_d256_e150`），不需要重采 BC 语料；`rl_dvalin1` / `rl_shenhe1` / `rl_electro1` 是现有认证 arm。

**预算：** 3 个 deck × 3 轮 × 2,000 局 = 18,000 局新采集（≈4.5h @4,000 局/h）；训练是 9 个 1-epoch PG run，A800 上 <0.5 GPU·h（CPU 档约 1.5h）；评测 20 seeds 筛选 2,400 局（≈1.6h @1,500 局/h），50 seeds 认证最多 6,000 局（≈4h）。合计约 6–10h CPU + 0.5 GPU·h。这个量级可以一轮做完，不需要等大窗口。

**停止规则：** 3 轮后如果 0/3 通过，停止扩张，回去查底座/表示；如果 1/3 通过，只对最接近通过的那 1 个 deck 再加 2 轮（不要三个一起加）；如果 2/3 通过，扩到下一批（优先 nahida/kokomi/raiden、mualani 等 expert 接近 0.6 的 deck）。

---

### Q5：league 的评测口径——固定池 + 历史 checkpoint 池，怎么避免自评偏差？

**结论：固定 20 脚本是“尺子”，历史 checkpoint 是“对手”，两者不能混。** 绝对强度用 fixed-anchor Bradley-Terry；历史池只用于“是否忘记/是否被旧版本针对”；Nash averaging 只作 secondary，不作主门禁。

| 指标 | 估计量 | 代码 | 用途 | 注意 |
|---|---|---|---|---|
| Pooled（等权 20 对手） | `Σ w_o / 20`；Wilson + opponent-cluster CI | `eval/scoreboard.summarize` | 主效用 | cluster CI 才诚实；pooled Wilson 偏乐观 |
| Per-opponent floor | 每对手 Wilson 下界 ≥0.45 | `evaluate_gates(strict_ci=True)` | 硬门禁 | 20 seeds=40 局/cell 时，真实 0.604 才能过；50 seeds=100 局/cell 时，真实 0.55 可过（`reports/explore/eval_protocol.md` §5.2） |
| 配对 Δ | 同 `(opponent, seed, side)` 的 `scoreA − scoreB` | `paired_difference` | 升降级 | 必须用新种子带；`CI_low ≥ −0.02` 才是非劣 |
| Fixed-anchor BT/Elo | 固定锚的 Bradley-Terry MLE + bootstrap | `anchored_bt_rating` | 绝对强度 | 锚本身是 L5.1 在线 Elo；换池要重拟合 |
| Nash average | maxent Nash 权重下的平均 | `open_spiel/python/algorithms/nash_averaging.py` | 次要排序 | 会重加权难/易对手；不等权，不等于部署分布 |
| PFSP-weighted rate | `Σ q_o w_o`, `q_o ∝ (1−w_o)^2` | 自写 10 行 | 难 matchup 诊断 | 会偏向弱格；不能替代 pooled |
| Best-response probe | `1 − min_o w_o` | `best_response_probe` | 脆弱性下界 | 是下界，不是精确 exploitability |
| Router holdout | 选择/计分分种子带 | `scripts/matchup_router.py` | 路由收益 | 绝不在选择种子上报分 |

**固定池 + 历史池的报表结构（建议）：**

```text
A. 绝对强度（每 deck，每 checkpoint）：
   pooled / cluster CI / per-opponent Wilson / floor violations / fallback=0
B. 相对 incumbent：
   paired Δ + CI（新种子带）；per-opponent Δ；是否任一对手机掉 >5pp
C. 历史池：
   对每个历史 checkpoint 的胜率；PFSP-weighted history score；是否 <0.5
D. 评级：
   fixed-anchor BT rating + bootstrap CI；Nash average（secondary）
E. 路由（如果有）：
   routed holdout vs best single；worst-cell；每对手选了哪个 arm；选择/计分种子带
```

**避免自评偏差的清单：**

1. **训练/选择/认证三段种子分离**：训练 seeds 不进评测；选择 seeds 不进最终报分；最终用全新种子带（例如 2000–2049）。
2. **固定锚不是自己**：20 套脚本 expert 是锚；历史 checkpoint 不是锚，不能拿“赢旧自己”代替“赢池”。
3. **双向**：同一 seed 打 side 0/1，抵消先手优势；`eval_protocol` 已实现。
4. **cluster CI 做门禁**：20 个对手是 20 个 cluster；不要用 800 局 pooled Wilson 直接宣称显著。
5. **多臂选择要校正**：20 个对手 × K 个 arm 的 argmax 有 winner's curse；router 必须 holdout + 收缩；checkpoint 选择也要独立种子带。
6. **引擎/卡池锁**：`configs/engine.lock` + 池版本（D11）；跨版本数字不可比。
7. **全量落盘**：所有 arm、所有对手、所有 seed 的 JSONL 保留；只提交聚合，避免只报赢家。
8. **Nash/Elo 只作辅助**：在线 Elo 顺序相关；固定锚 BT 才可跨 run 比较；Nash average 会改变“难度权重”，不能替代部署等权 pooled。

**PSRO/Nash 的正确用法（如果以后做 population）：** 用 `nash_averaging_avt_matrix` 对“checkpoint × 对手”矩阵做 agent-vs-task Nash averaging：先按对手归一化，再解 maxent Nash；得到每个 checkpoint 的 Nash average。它适合回答“这个 checkpoint 在难度加权后是否还强”，不适合替代 floor 门禁。OpenSpiel 的 `nash_averaging.py` 用 cvxpy 解 `max Σ entr(p) s.t. A p ≤ 0, Σp=1`（[代码](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nash_averaging.py)）。如果不想加 cvxpy，用 30 行 entropic mirror descent 也能解 20×N 的小矩阵。

---

## 2. Annotated bibliography

> 核心 15 条（2.1–2.15）+ 4 条补充（2.16–2.19）。可信度：A = 同行评审 + 可复现代码/补充材料；B = 高引用 preprint/技术报告；C = 二手/博客。每条的“证据”只写读到的内容；付费墙正文标「未核实」。

### 2.1 `A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning`（PSRO）
- **问题**：独立 RL 会 overfit 训练对手；需要一种可扩展的 multiagent 算法，兼顾 best response 与 meta-strategy。
- **方法核心**：PSRO = Double Oracle 的策略空间版。维护 empirical payoff tensor `U^Π`；每轮用 meta-solver（regret matching / Hedge / projected replicator dynamics）算 `σ`；对 `σ_{-i}` 采样训练 oracle（近似 best response）；把新策略加入 `Π`。DCH 是并行版：固定层数，每层一个 oracle，在线更新 meta-strategy，避免显式 `K^n` payoff tensor。
- **证据**：Laser Tag small4 上 InRL 的 joint-policy correlation loss `R− = 0.717`，DCH 降到 `0.150`（降低 56.7%）；Leduc 上 DCH/PSRO 早期收敛快于 NFSP，但 NFSP 最终 exploitability 更低；DCH 对固定 bot 的表现更好。来源：[arXiv 1711.00832](https://arxiv.org/abs/1711.00832)。
- **局限**：oracle 训练贵；meta-solver 的选择影响大；非平稳/非零和没有保证；DCH 是异步近似。
- **代码/复现**：[OpenSpiel psro_v2](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/psro_v2/psro_v2.py)、[double_oracle.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/double_oracle.py)。
- **可迁移点**：我们的 20×K 矩阵很小，PSRO 的 meta-solver 可以用来决定“下一轮打哪些对手/哪些历史 checkpoint”；但**不要跨 deck 建 payoff 矩阵**。DCH 的“并行层 + 在线 meta”适合多 deck 并行训练时的预算分配。

### 2.2 `Re-evaluating Evaluation`（Nash averaging）
- **问题**：benchmark 里加入简单任务/弱 agent 会稀释评价；需要一种对冗余不敏感的平均。
- **方法核心**：在 agent-vs-agent 或 agent-vs-task 的 meta-game 上求 **maxent Nash**；用 Nash 平均 `n_A = A p*_A` 作为每个 agent 的分数。maxent 让同分 agent 之间最大熵，自动把冗余 agent 的质量摊开。
- **证据**：加入冗余的 `C1/C2` 后，uniform average 会偏，maxent Nash 给每个 1/6；Atari 上 Nash averaging 后“人类与最强 agent 打平”，说明原先的“超人类”结论被高估。来源：[arXiv 1806.02643](https://arxiv.org/abs/1806.02643)。
- **局限**：AvA 需要反对称 payoff；AvT 需要按 task 归一化并构造块矩阵；Nash 不唯一时用 maxent；计算要解凸优化。
- **代码/复现**：[OpenSpiel nash_averaging.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nash_averaging.py)。
- **可迁移点**：作为 league 报表的 secondary 指标，防止“只打容易对手”的 checkpoint 排名靠前；不要用它替代固定池等权 pooled 和 floor。

### 2.3 `Open-ended Learning in Symmetric Zero-sum Games`（Rectified Nash）
- **问题**：非传递博弈（RPS）里没有“最强 agent”；self-play 会循环，Nash response 会收缩 gamescape。
- **方法核心**：PSRO_rN 的目标是 `E_{w∼p_t}[ max(φ_w(v_t), 0) ]`：只对 Nash 支撑集里、当前 agent 能赢的对手取正收益；“play to your strengths, ignore your weaknesses”。OpenSpiel 的离散版 `rectified_selector` 选“当前 payoff ≥ 平均 payoff”的策略。
- **证据**：Colonel Blotto 和 Differentiable Lotto 上，PSRO_rN 持续优于 self-play、PSRO_U、PSRO_N；gamescape 凸包持续扩大。作者也指出病态模式：如果每个 agent 只有局部 exploit，会生成大量高度特化、不泛化的 agent。来源：[arXiv 1901.08106](https://arxiv.org/abs/1901.08106)。
- **局限**：适合“培养多样性 population”，不适合“修 floor”；忽略弱点会与 M4 的 floor 目标直接冲突；局部 niche 可能过拟合。
- **代码/复现**：[OpenSpiel strategy_selectors.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/psro_v2/strategy_selectors.py)（`rectified_selector`）。
- **可迁移点**：如果我们要训练“同一 deck 的多种风格 specialist”（例如 dgeo 的快攻/控制两套），用 rectified Nash 保持多样性；但 **floor 攻坚要用 PFSP 的“打最差”而不是 rectified Nash 的“忽略最差”**。两者是相反采样，别混。

### 2.4 `Planning in the Presence of Cost Functions Controlled by an Adversary`（Double Oracle）
- **问题**：对手在你看完策略后选 cost function；如何规划。
- **方法核心**：在受限策略集 `Π^t` 上解子博弈均衡 `σ*`；对每个玩家从全空间取 best response `π_i^{t+1} ∈ BR(σ_{-i}*)`；加入 `Π^{t+1}`；重复。两人零和保证收敛到均衡，但最坏情况要枚举全部策略（RPS 是例子）。
- **证据**：PSRO 论文把它作为特例；OpenSpiel 有实现。来源：[McMahan et al. 2003 PDF](http://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf)、[OpenSpiel double_oracle.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/double_oracle.py)。
- **局限**：只给收敛保证，不给样本效率；策略空间大时每轮 best response 很贵。
- **可迁移点**：我们的“加入新 specialist / 新历史 checkpoint”就是 DO 的加策略步骤；但对手固定时，不需要求均衡，只需要 best response。DO 的价值是提醒我们“策略池要显式维护”，不是“必须解 Nash”。

### 2.5 `Deep Reinforcement Learning from Self-Play in Imperfect-Information Games`（NFSP）
- **问题**：不完全信息博弈没有简单 self-play 收敛保证；需要近似 Nash。
- **方法核心**：fictitious self-play + 深度学习。RL 记忆近似“对对手平均策略的 best response”；SL 记忆近似“自己的平均策略”；用 reservoir 采样旧数据；执行时混合平均策略。
- **证据**：Leduc 上接近 Nash；Limit Hold'em 上接近当时 CFR agent 的水平（abstract 级；未逐条核对数值）。来源：[arXiv 1603.01121](https://arxiv.org/abs/1603.01121)。
- **局限**：平均策略在部署时偏保守；大规模深度网络下 reservoir/训练不稳定；不是 league。
- **可迁移点**：NFSP 的“平均策略 + reservoir”是历史池的思想源头之一；但 **PFSP 不在 NFSP 论文里**（见 §7）。我们可以用 reservoir 保存旧 checkpoint，但不需要学平均策略。

### 2.6 `Grandmaster level in StarCraft II using multi-agent reinforcement learning`（AlphaStar）
- **问题**：RTS 的非传递策略空间；单一 self-play 会循环/被针对。
- **方法核心**：league = main agents + main exploiters + league exploiters；PFSP 采样历史对手；主 agent 的 branch 大致是 50% PFSP、35% self-play、15% verification；payoff 0.99 衰减；每 2e9 steps 检查一次，若对历史池最小胜率 >0.7 或已过 4e9 steps 才 checkpoint；最终 agent 取 population 的 Nash distribution。
- **证据**：三个种族 Grandmaster，>99.8% 官方排名玩家；PFSP 在所有指标上优于 FSP；league 的 exploiters 制造了约 3,000,000 个 RPS 循环（≥70% 胜率定义），主 agent 之间只有约 200 个。来源：[Nature 论文](https://www.nature.com/articles/s41586-019-1724-z)、[补充材料 ZIP](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip)。
- **局限**：算力/工程规模巨大；StarCraft 的 z-statistics/race 条件化我们不需要；Nature 正文付费墙，league 细节主要从补充 `pseudocode/multiagent.py` 读，正文措辞「未逐字核实」。
- **代码/复现**：补充材料内 `pseudocode.zip` 的 `multiagent.py`（PFSP、Payoff、MainPlayer、MainExploiter、LeagueExploiter、Historical）。核心 PFSP 就是：
```python
def pfsp(w, weighting="squared"):
    f = {"variance": lambda x: x*(1-x), "linear": lambda x: 1-x,
         "linear_capped": lambda x: min(0.5, 1-x), "squared": lambda x: (1-x)**2}[weighting]
    p = f(np.asarray(w)); return p / p.sum()
```
- **可迁移点**：PFSP 权重、0.99 payoff 衰减、verification branch、checkpoint 规则都可直接缩小规模搬到 `scripts/league_collect.py`；但我们的历史池只占 20–30%，不是 50%。

### 2.7 `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm`（AlphaZero）
- **问题**：能否用单一算法、单一网络、纯 self-play 通吃 chess/shogi/Go。
- **方法核心**：单个网络持续更新；self-play 双方都用 **最新参数** + MCTS；没有历史池、没有 league、没有 best-player gating（AlphaGo Zero 有 55% 替换门槛，AlphaZero 去掉了）。训练 700k steps、batch 4096、5,000 第一代 TPU。
- **证据**：chess/shogi/Go 都达到 superhuman；来源：[arXiv 1712.01815](https://arxiv.org/abs/1712.01815)（Science 版：[DOI 10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)）。
- **局限**：完美信息、确定性、可完美模拟；不完全信息/非传递博弈没有保证；latest-only 容易循环。
- **可迁移点**：作为“最便宜 baseline”：latest-only self-play vs PFSP/history 的 A/B。不要把它当成我们不完全信息卡牌游戏的主方法。

### 2.8 `Dota 2 with Large Scale Deep Reinforcement Learning`（OpenAI Five）
- **问题**：长时程、部分可观测、巨大动作空间的 5v5 游戏；训练中环境和模型不断变。
- **方法核心**：PPO + GAE + 巨大 LSTM；**80% 局打最新参数，20% 打历史版本**；历史对手用 quality score `q_i` 的 softmax 采样；当前 agent 每 10 iterations 加入池，`q_i ← q_i − η/(N p_i)`（η=0.01）当它击败历史对手；surgery 是函数保持的离线变换（`∀o π̂(o)=π(o)`），用于改架构/观测/动作空间。
- **证据**：10 个月训练，20+ 次 surgery；对 OG 2-0，线上 7,257 局胜率 99.4%；Rerun 2 个月/150 PFlops·s-days vs 原 10 个月/770。来源：[arXiv 1912.06680](https://arxiv.org/abs/1912.06680)，Appendix N/B 在 [HTML 版](https://arxiv.org/html/1912.06680v1)。
- **局限**：不是 peer-reviewed；算力巨大；surgery 是工程技巧，不是学习算法；没有 specialist portfolio。
- **可迁移点**：80/20 + quality softmax 可以直接用于我们的历史池；surgery 的思想可以用于“加 router 特征/加 deck id 时保持旧 checkpoint 行为”（零初始化新分支，函数保持）。注意：我们目前把 A2 identity branch 的失败归因于“特征/训练-评测不一致”（首要怀疑，未定论），不是 surgery 本身的问题。

### 2.9 `DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning`
- **问题**：斗地主三人、合作+竞争、巨大动作空间；单一模型很难同时学好地主/农民。
- **方法核心**：DMC（Deep Monte-Carlo）+ 动作编码 + 并行 actor；**三个位置各一个 Q 网络**（landlord / landlord_up / landlord_down），每个位置只用自己的数据、对最新对手模型更新。
- **证据**：单机 4 GPU、数天训练；Botzone 344 个 agent 中排名第一；评测用 100,000 副牌、双向换位。来源：[PMLR v139](https://proceedings.mlr.press/v139/zha21a.html)、[代码](https://github.com/kwai/DouZero)。
- **局限**：位置在开局已知，所以“路由”是免费的；不是 general router；DMC 目标与我们 PG 不同。
- **可迁移点**：**角色/牌组专用模型在卡牌游戏里是被验证的模式**；不要因为“单模型优雅”而强行 20-deck 共享。我们的 per-deck specialist 与 DouZero 的 per-position 模型同构。

### 2.10 `The Algorithm Selection Problem`（Rice 1976）
- **问题**：给定问题实例，如何选最合适的算法。
- **方法核心**：ASP 五元组：问题空间 `P`、特征空间 `F`、算法空间 `A`、性能度量 `p`、选择映射 `S: F → A`；目标是最大化所选算法的性能。后来的 portfolio/selector 都是这个框架的实例。
- **证据**：经典框架，无实验效应量；来源：[DOI](https://doi.org/10.1016/S0065-2458(08)60520-3)、[Purdue 技术报告](https://docs.lib.purdue.edu/cstech/99/)。
- **局限**：1976 年的框架，不涉及在线学习/非平稳；特征工程和选择器要自己定。
- **可迁移点**：router 就是 ASP：`P` = 对手阵容/公共状态，`F` = deck id + 对手 lineup + 公共状态，`A` = specialist checkpoints，`p` = 引擎胜率，`S` = 表/分类器。先把 `F` 限定在“开局可得”的特征，避免信息泄漏。

### 2.11 `Algorithm Selection for Combinatorial Search Problems: A Survey`（Kotthoff 2014）
- **问题**：算法选择/portfolio 的系统和冷启动问题。
- **方法核心**：特征提取 → 性能模型 → 选择器；portfolio 可以并行跑多个算法，selector 只选一个；要处理特征成本、冷启动、概念漂移。
- **证据**：综述，覆盖 SAT/CP/MIP 等；来源：[arXiv 1210.7959](https://arxiv.org/abs/1210.7959)、[AI Magazine](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2460)。
- **局限**：主要不是游戏；没有“对手会适应”的设定。
- **补充**：portfolio 的经典来源还有 Gomes & Selman 2001 `Algorithm portfolios`（Artificial Intelligence 126:43–62，[DOI](https://doi.org/10.1016/S0004-3702(00)00081-3)）与 Huberman et al. 1997 `An economics approach to hard computational problems`（Science 275:51–54，[DOI](https://doi.org/10.1126/science.275.5296.51)）；它们讲“把时间分配给多个算法”，不是“按对手选策略”，但选择器/组合思想同源。
- **可迁移点**：router 的特征要“决策时可得、成本低”；冷启动用 pooled 回退；选择器要在 holdout 实例上评估，不能在训练实例上报分。

### 2.12 `Adaptive Mixtures of Local Experts`（MoE）
- **问题**：多个 expert 如何分工，而不是一个网络学全部。
- **方法核心**：`g_i(x)=softmax(v_i^T x)`，`y=Σ_i g_i(x) f_i(x)`；expert 网络 + gating network 联合训练，把输入空间软划分给 expert。后续 Switch Transformer 用 top-1 routing + load balancing 防止“路由坍缩”。
- **证据**：原始论文是监督学习任务；无游戏效应量；来源：[DOI](https://doi.org/10.1162/neco.1991.3.1.79)、[PDF](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)。
- **局限**：joint training 需要大量数据；router 可能坍缩到少数 expert；可解释性差。
- **可迁移点**：如果将来做“一个模型 + 多个 deck embedding”的 conditional model，MoE 是架构参考；但现在 20 个离散对手用 20×K 表更简单，不需要 MoE。若用 MoE，必须监控每 expert 的负载。

### 2.13 `A Contextual-Bandit Approach to Personalized News Article Recommendation`（LinUCB）
- **问题**：在特征丰富的 arm 集合里做在线选择，平衡探索/利用。
- **方法核心**：disjoint linear model：每个 arm 一个 ridge 估计 `θ̂_a = (D_a^T D_a + I)^{-1} D_a^T c_a`；UCB `x^T θ̂_a + α sqrt(x^T A_a^{-1} x)`；hybrid 版共享部分特征。
- **证据**：Yahoo! Today 新闻推荐；1% 数据时 hybrid LinUCB 的 CTR 比 UCB 高 10.3%（1.493 vs 1.354）；稀疏数据下 hybrid 明显优于 disjoint。来源：[arXiv 1003.0146](https://arxiv.org/abs/1003.0146)。
- **局限**：线性假设；非平稳；reward 延迟；不是游戏。
- **可迁移点**：只在“对手身份未知/在线变化”时用；如果要用，选 hybrid（共享对手/公共状态特征 + 每 arm 特征），因为我们的 cell 数据稀疏。固定 20 套已知对手不需要 LinUCB。

### 2.14 `SATzilla: Portfolio-based Algorithm Selection for SAT`
- **问题**：没有单一 SAT solver 通吃；如何按实例选 solver。
- **方法核心**：用 empirical hardness models（图/结构特征等）在离线训练后，对每个实例预测各 solver 的性能，选最优/最短预期时间；portfolio + selector。
- **证据**：据 JAIR/ADS 摘要，2007 SAT Competition 获 3 金 1 银 1 铜；per-instance selection 显著优于单 solver（具体倍数未逐一核对）。来源：[arXiv 1111.2249](https://arxiv.org/abs/1111.2249)、[JAIR](https://www.jair.org/index.php/jair/article/view/10556/25269)、[项目页](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/)。
- **局限**：需要好特征；训练/测试分布要一致；selector 错误会有代价。
- **可迁移点**：portfolio 的收益来自“不同实例上不同 arm 最强”，不是来自“平均更强”；所以先画 `arm × opponent` 矩阵，确认有可利用的交互，再投入 router。我们的 `scripts/matchup_router.py` 就是干这个的。

### 2.15 `Deck Archetype Prediction in Hearthstone`
- **问题**：能否只从对手第一回合的少量动作预测其 deck archetype。
- **方法核心**：用 hsreplay.net replay；统计分类器 + RNN；输入是 minimal first-turn actions；数据集有 75 个 deck 类型，论文展示最流行 15 类的混淆矩阵。
- **证据**：RNN 在 holdout 上 78.9%（±0.4%）accuracy，weighted F1 0.73（75 类整体）；最 common-class baseline 57.7%；统计分类器 78.3%；"Quest Shaman" precision 0.97。来源：[DOI](https://doi.org/10.1145/3402942.3402959)、[PDF](https://slothlab.info/assets/pdf/eger2020fdg.pdf)。
- **局限**：Hearthstone 的 deck 在开局不完全可见；我们的七圣召唤 3 个角色开局可见，识别更容易；但“识别”≠“知道怎么打”。
- **可迁移点**：如果对手阵容不可见，router 可以用前几轮公开动作做 archetype 推断；我们当前不需要，直接用 lineup。

### 2.16 `SelectiveNet: A Deep Neural Network with an Integrated Reject Option`
- **问题**：模型何时应该拒答/回退，而不是强行预测。
- **方法核心**：联合优化预测和 reject；用 coverage-risk 曲线选择阈值；目标是“在给定 coverage 下最小风险”。
- **证据**：图像分类/回归上的 selective prediction；来源：[arXiv 1901.09192](https://arxiv.org/abs/1901.09192)。
- **局限**：不是游戏；reject 需要可用的 fallback。
- **可迁移点**：router 应该有“拒答”选项：当 top-2 LCB 重叠/数据不足时回退到 pooled 最好的单 specialist。这比强行路由安全。

### 2.17 `Overcoming catastrophic forgetting in neural networks`（EWC）
- **问题**：顺序多任务训练会忘记旧任务。
- **方法核心**：用对角 Fisher 信息衡量参数重要性，损失加 `Σ F_i (θ_i − θ*_i)^2`；只对重要参数强约束。
- **证据**：多个监督/RL 任务上减少遗忘；来源：[arXiv 1612.00796](https://arxiv.org/abs/1612.00796)。
- **局限**：Fisher 估计贵；非独立同分布/非平稳数据下不稳定；超参难调。
- **可迁移点**：作为 KL anchor 的备选；如果 KL 不够再上，不建议第一版就用。

### 2.18 `Distral: Robust Multitask Reinforcement Learning`（补充阅读）
- **问题**：多任务 RL 中任务间干扰。
- **方法核心**：学一个共享“distilled”策略，每个任务策略对共享策略做 KL 正则；共享策略吸收共性，任务策略保留差异。
- **证据**：多任务 RL 基准；来源：[arXiv 1707.04175](https://arxiv.org/abs/1707.04175)。
- **局限**：需要任务间有共享结构；我们的 20-deck 负迁移说明共享 trunk 风险高。
- **可迁移点**：如果将来做“单模型多 deck”，用共享 prior + per-deck KL，而不是硬共享全部参数；但先证明 per-deck specialist 的上限。

### 2.19 `NeuPL: Neural Population Learning`（补充阅读）
- **问题**：PSRO 每轮重新学基本技能，样本效率低。
- **方法核心**：用一个 conditional network 表示整个 population；按 best-response 目标训练不同 conditioning；共享基本技能。
- **证据**：在对称零和游戏上展示了比逐轮 PSRO 更高的样本效率（abstract 级，具体数值未逐条核对）；来源：[arXiv 2202.07415](https://arxiv.org/abs/2202.07415)。
- **局限**：对称零和假设；conditioning/capacity 设计难；不是固定池 exploitation。
- **可迁移点**：如果将来 20 个 specialist 的维护成本太高，可以试“一个 conditional model + deck/对手 embedding”；但 A2 的教训说明 id 分支容易 store→engine 不一致（首要怀疑），先做小规模 A/B。

---

## 3. 最小可复现 recipe（按性价比排序）

### R1. Floor-aware PFSP 采样 + per-opponent 非劣门禁
- **输入**：当前 deck 的认证 checkpoint、20 套脚本对手、上一轮 `w_{d,o}`。
- **输出**：2,000 局新 replay、下一轮 checkpoint、per-opponent 报表。
- **关键超参**：`q = 0.7*PFSP + 0.2*uniform + 0.1*worst3`；`(1−w~)^2`；每对手 cap 15%、floor 2%；`α=β=10`；2,000 局/轮；fresh seed band。
- **伪代码**：
```text
w~ = beta_smooth(wins, n, α=10, β=10)
q  = normalize(0.7*(1-w~)^2 + 0.2/20 + 0.1*worst3_mask)
for o in sample(opponents, q, n=2000):
    collect(deck, o, fresh_seed, teacher=ckpt#t=1.0)
train PG 1 epoch from ckpt
eval on fresh band: pooled + per-opponent Wilson + paired Δ vs incumbent
promote iff pooled Δ CI_low >= -0.02 and floor violations <= incumbent; flag if any opponent Δ < -0.10 (n>=100/cell 才硬判)
```
- **需要改的仓库文件**：`train/collect.py`（`--opponent-weights`、`CollectTask.side`/swap）、`scripts/league_collect.py`（新，混合 expert/history）、`eval/scoreboard.py`（per-opponent non-inferiority helper）、`scripts/eval_protocol.py`（调用）。
- **成本**：2,000 局 ≈ 0.5h（4,000 局/h）；评测 800 局 ≈ 0.5h；无 GPU。
- **预期信号**：floor 违规数下降，pooled 不降 >1pp；最差 matchup 净涨。
- **失败判据**：pooled 掉 >2pp，或最差 matchup 掉 >5pp，或 floor 违规增加；说明采样过偏，回到 0.5/0.3/0.2 或加 uniform。

### R2. Root-anchor KL + replay mix（防多轮漂移）
- **输入**：当前 incumbent、root anchor（第一个 parity+ BC）、本轮 store、上一轮 replay。
- **输出**：带 KL 约束的 PG checkpoint。
- **关键超参**：`kl_coef ∈ {0.1, 0.3, 0.5}`；anchor = root，不是上一轮；replay mix 20–30%；PG 1 epoch、lr 3e-5、entropy 0.01、grad clip 1.0。
- **伪代码**：
```text
store = concat(current_replay, 0.25 * previous_replay)
anchor_logits = frozen_anchor.forward(store)          # 预计算一次
# 安全用法：旧 replay 只进辅助 BC/value/KL，不直接进 PG policy loss
loss = PG_loss(on-policy part) + bc_coef * BC(old part) + kl_coef * mean(KL(anchor || policy))
train 1 epoch from incumbent
gate: paired Δ CI_low >= -0.02; floor violations <= incumbent; clean; per-opponent Δ < -0.10 只 flag
```
- **注意**：如果一定要把旧 replay 混进 PG，必须用 behavior policy 的 log-prob 做 clipped importance weight；当前 `Store` 没存 old log-prob，先别这么干。
- **需要改的仓库文件**：`scripts/train_bc_scale.py`（`Config.kl_anchor/kl_coef`；`run_batch` 加 KL）、`scripts/datafirst_bc_train.py`（CLI）。
- **成本**：训练 +10–20%；2,000 局采集 0.5h；GPU <0.05h/run。
- **预期信号**：per-opponent 方差下降，历史池胜率不降；pooled 不降。
- **失败判据**：KL >0.1 且 pooled 不涨；或 entropy 塌缩/训练不稳；降 `kl_coef` 或换 anchor。

### R3. 历史 checkpoint 池 + AlphaStar 式 verification
- **输入**：同 deck 的最近 3–4 个 checkpoint、当前 incumbent、固定池。
- **输出**：混合 replay（80% 固定池 + 20% 历史）、对历史池的胜率报表。
- **关键超参**：历史比例 20–30%；PFSP `(1−w)^2`；只保留 4 个；checkpoint 条件“对历史池最小胜率 >0.7”或每 2 轮；payoff 0.99 衰减。
- **伪代码**：
```text
for round:
    collect 80% vs 20 scripted (PFSP)
    collect 20% vs sample(history_ckpts, PFSP)
    PG 1 epoch
    if min_win_rate_vs_history < 0.7: add worst history to next round's pool
    checkpoint if min_win_rate_vs_history > 0.7 or rounds_since_last >= 2
```
- **需要改的仓库文件**：`scripts/league_collect.py`（history list + mixed task grid）、`scripts/eval_history.py`（新，candidate vs history ckpts）、`train/collect.py`（manifest 加 history 指纹）。
- **成本**：20% extra games；4 个历史 checkpoint × 800 局 = 3,200 局评测 ≈ 2h；GPU 忽略。
- **预期信号**：对历史池的 PFSP-weighted score 不降；固定池 pooled 不降；旧版本不再能 >50% 击败新版本。
- **失败判据**：固定池 pooled 掉 >1pp，或对历史池最小胜率 <0.5；说明历史池比例太大，降回 10–15%。

### R4. Matchup router：经验贝叶斯 + LCB + honest split
- **输入**：K 个 arm 的 `data/eval/*.jsonl`（同一 deck、同一对手池、不同 checkpoint）。
- **输出**：每对手的 arm 选择表、routed holdout 胜率、worst-cell 报表。
- **关键超参**：`κ = 20–50`；`z = 1.96`；`δ = 0.03`；每 cell ≥100 局；选择/计分种子带分离（例如 seeds 0–24 / 25–49）。
- **伪代码**：
```text
for a, o:
    w~ = (wins + κ*pooled_a) / (n + κ)
    lcb = w~ - z*sqrt(w~(1-w~)/(n+κ))
pick a*(o) = argmax lcb; if top2 gap < δ: fallback to best pooled arm
score on holdout: routed = mean_o score(a*(o), o)
report routed vs best single vs teacher; per-cell picks
```
- **需要改的仓库文件**：`scripts/matchup_router.py`（`--min-games`、`--kappa`、`--seed-band`、LCB/fallback；已有 selection/holdout 切分）。
- **成本**：3 arm × 20 对手 × 100 局 = 6,000 局/deck；2 deck = 12,000 局 ≈ 8h 评测（1,500/h）或 3h 采集（4,000/h）；GPU 0。
- **预期信号**：routed holdout ≥ best single +3pp，且 worst-cell 不降 >3pp；至少 3/20 个对手选了非 pooled-best arm。
- **失败判据**：routed < best single +1pp，或 worst-cell 掉 >3pp，或 selection/holdout 的 per-cell 排名相关 <0.5；放弃 router，保留单 specialist，并先试同 deck 的 model soup。

### R5. Nash-averaged league 报表（secondary）
- **输入**：`checkpoint × opponent` 矩阵（同一 deck 的历史 checkpoint + 固定 20 对手）。
- **输出**：每个 checkpoint 的 Nash average、uniform pooled、BT rating、floor。
- **关键超参**：按对手归一化到 [0,1]；maxent Nash；`eps=0`；bootstrap 2000。
- **伪代码**：
```text
M_norm[:, o] = (M[:, o] - min) / (max - min)
solve max Σ entr(p) s.t. M_norm @ p <= 0, sum p = 1, p >= 0
score_i = (M_norm @ p)[i]
```
- **需要改的仓库文件**：`eval/nash.py`（新，30–60 行；或移植 OpenSpiel `nash_averaging_avt_matrix`）、`scripts/league_report.py`（新）。
- **成本**：复用评测日志；求解 <1 min；无 GPU。
- **预期信号**：Nash 排序与 uniform pooled 的 Spearman >0.8；若差异大，说明有“只打容易对手”的 checkpoint。
- **失败判据**：Nash 排序在不同种子带间不稳定（Spearman <0.6）；此时只报告 uniform pooled + floor，不用 Nash 做决策。

---

## 4. 预注册实验（只给 2 个）

> 预注册要求：在采集前把假设、arm、种子带、阈值、停止条件写进 `reports/explore/lit/f03_prereg.md` 或 PLAN §0；事后不得改阈值。下面所有局数按“新增采集/评测”计，不含已有语料。

### E1：把 3 个 near-parity deck 扩到 PASS（带防漂移）

**假设**：在 dvalin / shenhe / electro 上，用“PFSP + 20% 历史池 + root-anchor KL + incumbent 非劣门禁”的 3 轮 PG，能在不牺牲 pooled 的前提下把至少 2/3 个 deck 推到现有 specialist 门禁 PASS（pooled ≥0.50、floor Wilson 下界 ≥0.45、clean）。

**对照**：每个 deck 的现有认证 arm（dvalin `rl_dvalin1` 0.780；shenhe `rl_shenhe1` 0.7025；electro `rl_electro1` 0.685）。这不是同种子随机对照，而是“新方法 vs 已认证 incumbent”的历史对照；报告时明确标注。

**arm/流程**：
- 每 deck 3 轮；每轮 2,000 局；第 1 轮 100% 固定池，第 2–3 轮 80% 固定池 + 20% 历史 checkpoint（同 deck 的上一轮/上上轮）。
- 每轮再混 20% 上一轮 replay，但只进辅助 BC/value/KL，不直接进 PG policy loss（避免 off-policy PG 偏差）。
- 固定池采样：`0.7*PFSP + 0.2*uniform + 0.1*worst3`；`(1−w~)^2`；cap 15%/floor 2%。
- PG：1 epoch、lr 3e-5、entropy 0.01、grad clip 1.0、`--init-from` incumbent；`kl_coef=0.1` 对 root BC。
- 每轮换 fresh seed band（训练 3000+；评测 2000+，互不重叠）。

**新采集局数**：3 deck × 3 轮 × 2,000 = **18,000 局**（其中 20% 是历史池，仍计入总采集）。
**训练 GPU·h**：9 个 1-epoch PG，A800 上 **<0.5 GPU·h**（CPU 档约 1.5h；数据平面是瓶颈）。
**评测局数**：20 seeds 筛选 3 × 800 = **2,400 局**；通过筛选的 deck 用 50 seeds 认证，最多 3 × 2,000 = **6,000 局**；合计 **≤8,400 局**（~5.6h @1,500 局/h）。
**主要指标**：通过现有 specialist 门禁的 deck 数（0–3）。
**判定阈值**：
- 成功：**≥2/3** deck PASS；且每个 deck 的 pooled 不比 incumbent 低 >2pp；floor 违规 ≤1（筛选）/ 0（认证）。
- 失败：0/3 PASS；或任一 deck pooled 掉 >2pp；或出现 fallback/error。
**停止条件**：3 轮后 0/3 → 停止扩张，查底座/表示；1/3 → 只对最接近通过的那 1 个 deck 加 2 轮；2/3 → 扩到下一批。
**预期**：dvalin 最可能先过（已 0.780、1 违规）；shenhe/electro 需要 floor 从 2 违规降到 0，难度更高；2/3 是合理门槛。

### E2：2 个 PASS specialist 的 checkpoint portfolio + router（honest split）

**假设**：在 dgeo / ungeo 上，历史 checkpoint 之间存在可利用的 matchup 互补；按对手选 checkpoint 的 router，在 holdout 种子上能比最佳单 checkpoint 高 ≥3pp，且不制造 worst-cell 灾难。

**arm**：每个 deck 3 个 arm：BC 底座（A0）、r2（中期）、final（r3/r4）。选择这三个是因为它们代表“弱但多样 / 中期 / 最强”，且都是已有 checkpoint。
**数据**：每个 arm × 20 对手 × 100 局（50 seeds × 双向）= 2,000 局/deck/arm；2 deck × 3 arm = **12,000 局**全新评测（不混旧日志，避免种子带不一致）。
**切分**：seeds 0–24 选择（50 局/cell），seeds 25–49 计分（50 局/cell）；选择/计分完全隔离。
**router**：经验贝叶斯 `κ=30` + LCB `z=1.96`；top-2 LCB 差 <0.03 时回退到 pooled-best arm。
**训练 GPU·h**：**0**（查表/逻辑回归在 CPU）；若加一个小 logistic router，<0.01 GPU·h。
**主要指标**：routed holdout pooled vs best single arm（同 holdout）。
**判定阈值**：
- 成功：routed ≥ best single + **0.03**；paired bootstrap 95% CI 下界 >0；worst-cell ≥ best single worst-cell −0.03；≥3/20 个对手选了非 pooled-best arm；无 cell <0.45。
- 失败：routed < best single +0.01；或 worst-cell 掉 >3pp；或选择/计分 per-cell 排名相关 <0.5。
**停止条件**：失败即放弃 router，保留单 specialist（可先试 model soup）；成功则只在 dgeo 上线，ungeo 若单独不显著不强行推广。
**成本**：12,000 局 ≈ 8h 评测（1,500/h）或 3h 采集（4,000/h）；建议夜间跑；无 GPU。
**风险**：100 局/cell 只能做粗路由；若真实差异 2–3pp，本实验会判失败——这是故意的保守设计，避免重演 10-seed 路由的假阳性（0.475 vs 0.490）。

---

## 5. 与现有负结果的关系（为什么这次不会重演）

| 现有负结果 | 本文为什么不重演 | 硬护栏 |
|---|---|---|
| 20-deck 单模型，8 个 deck 里 7 个显著变差（`reports/EXPLOIT_LADDER.md`） | 不共享 trunk；每 deck 独立 specialist；router 只在部署时选，不联合训练 | 不做 20-deck 联合 BC；不共用梯度 |
| A2 identity embeddings：store val_acc 0.866，引擎 dgeo 0.266 / ungeo 0.358（`reports/explore/a2_id_screening.md`） | router 用 deck id 作为硬 key，不学 id embedding；不用 val_acc 做选择 | 只用引擎胜率；id 特征不做可学习分支 |
| search-as-teacher 全池 Δ 显著负（−0.066/−0.114；`reports/explore/plan_strength_2day_kaggle.md`） | league 只用 PG，不引入搜索 teacher；不依赖根价值 | 搜索分支保持关闭；除非价值头标定 |
| AWR/Q/DMC 负结果（`reports/explore/plan_strength_2day_kaggle.md`） | 更新仍是 REINFORCE + baseline；不换成 Q/AWR | 不在 league 里混离线 RL 目标 |
| 10-seed matchup router 0.475 vs 0.490（commit `7929cec`） | 100 局/cell + 收缩 + LCB + holdout；门槛 +3pp | 低于 100 局/cell 不选路由；选择/计分种子分离 |
| SA 三轮全负；Skirk 系 expert <0.5（`reports/EXPLOIT_LADDER.md`、`reports/SPECIALISTS.md`） | 扩张前先算 expert ceiling；SA/Skirk 排除 | `expert_pooled ≥0.60` 且 `floor 违规 ≤3` 才开 PG |
| PG 只在 parity+ deck 有效；dvalin 错底座 −0.06/−0.13（`reports/EXPLOIT_LADDER.md`） | 扩张候选都是 parity+；底座不够先补 BC | `base ≥ expert −0.03` 才开 PG |
| 10 seeds 下 floor 实际是“每 matchup ≥67%”（`reports/explore/eval_protocol.md` §5.2） | 20 seeds 筛选、50 seeds 认证；用 Wilson 下界 | 不用 10 seeds 宣称 floor PASS |
| dgeo r3 0.890 → r4 0.880 饱和（`reports/EXPLOIT_LADDER.md`） | 不再给已 PASS 的 deck 加轮次；转做 router/扩张 | 每 deck 最多 3 轮，除非预注册新假设 |

---

## 6. 不要做什么（negative guidance）

1. **不要**把 20 套脚本对手当成 Nash 对手去解 equilibrium；它们不会 best-respond，Nash 会牺牲你不需要的 exploitability。
2. **不要**跨 deck 建 PSRO/DO payoff 矩阵；不同 deck 不是同一游戏的策略，矩阵没有意义。PSRO 只用于同 deck 的历史 checkpoint/exploiter。
3. **不要**用 raw `min_o w_o` 作为训练目标；用 `CVaR_3` 或 `pooled − λ·floor_deficit`，门禁再用 Wilson 下界。
4. **不要**在 40 局/cell 上选 router；这是 10-seed 探针失败的原因。至少 100 局/cell，最好 200。
5. **不要**在选择 router 的同一批种子上报路由收益；必须 holdout。
6. **不要**用 `val_acc` 选 checkpoint/router；A2 已经证明 store val_acc 与引擎胜率脱钩。
7. **不要**对 expert ceiling <0.45 的 deck 开 PG；这是 deck 的结构上限，不是策略问题。
8. **不要**只打最差对手；保留 20% uniform + 10% worst3，避免 pooled 崩。
9. **不要**用 KL 代替非劣门禁；KL 可以锁在一个弱 anchor 上。anchor 用 root，评测用 incumbent。
10. **不要**用在线 Elo 做认证；用 fixed-anchor BT + cluster CI。Nash average 只作 secondary。
11. **不要**在 league 里混 search teacher / AWR / Q / DMC；这些在我们这里都是负结果。
12. **不要**逐决策切换 specialist；没有标定价值头之前，只在 deck 级路由。
13. **不要**跨 engine/卡池版本比较数字；`configs/engine.lock` + 池版本必须一致。
14. **不要**只报赢家；所有 arm、所有对手、所有 seed 的 JSONL 保留，只提交聚合。

---

## 7. 参考（链接、版本/日期、可信度）

> 访问日期均为 2026-10-27。可信度：A = 同行评审 + 代码/补充材料；B = 高引用 preprint/技术报告；C = 二手/博客。

### 7.1 League / game theory

1. **Lanctot et al. 2017, `A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning`** — NeurIPS 2017；[arXiv 1711.00832](https://arxiv.org/abs/1711.00832)。A。
2. **Balduzzi et al. 2018, `Re-evaluating Evaluation`** — NeurIPS 2018；[arXiv 1806.02643](https://arxiv.org/abs/1806.02643)。A。
3. **Balduzzi et al. 2019, `Open-ended Learning in Symmetric Zero-sum Games`** — ICML 2019；[arXiv 1901.08106](https://arxiv.org/abs/1901.08106)。A。
4. **McMahan, Gordon & Blum 2003, `Planning in the Presence of Cost Functions Controlled by an Adversary`** — ICML 2003；[PDF](http://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf)。A。
5. **Heinrich & Silver 2016, `Deep Reinforcement Learning from Self-Play in Imperfect-Information Games`** — [arXiv 1603.01121](https://arxiv.org/abs/1603.01121)。B（preprint；NFSP 的正式发表信息未核实）。
6. **Vinyals et al. 2019, `Grandmaster level in StarCraft II using multi-agent reinforcement learning`** — Nature 575:350–354；[DOI](https://doi.org/10.1038/s41586-019-1724-z)；[补充材料 ZIP](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip)（`pseudocode/multiagent.py` 已读）。A（正文付费墙，league 细节以补充代码为准）。
7. **Silver et al. 2017, `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm`** — [arXiv 1712.01815](https://arxiv.org/abs/1712.01815)；Science 版 [DOI 10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)。A。
8. **OpenAI et al. 2019, `Dota 2 with Large Scale Deep Reinforcement Learning`** — [arXiv 1912.06680](https://arxiv.org/abs/1912.06680)；[HTML 版](https://arxiv.org/html/1912.06680v1)（Appendix N/B 已读）。B。
9. **Zha et al. 2021, `DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning`** — ICML 2021，PMLR 139:12333–12344；[PMLR](https://proceedings.mlr.press/v139/zha21a.html)；[代码](https://github.com/kwai/DouZero)。A。

### 7.2 Portfolio / algorithm selection / routing

10. **Rice 1976, `The Algorithm Selection Problem`** — Advances in Computers 15:65–118；[DOI](https://doi.org/10.1016/S0065-2458(08)60520-3)；[Purdue 技术报告](https://docs.lib.purdue.edu/cstech/99/)。A。
11. **Kotthoff 2014, `Algorithm Selection for Combinatorial Search Problems: A Survey`** — AI Magazine 35(3):48–60；[arXiv 1210.7959](https://arxiv.org/abs/1210.7959)；[AAAI](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2460)。A。
12. **Jacobs, Jordan, Nowlan & Hinton 1991, `Adaptive Mixtures of Local Experts`** — Neural Computation 3(1):79–87；[DOI](https://doi.org/10.1162/neco.1991.3.1.79)；[PDF](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)。A。
13. **Li, Chu, Langford & Schapire 2010, `A Contextual-Bandit Approach to Personalized News Article Recommendation`** — WWW 2010；[arXiv 1003.0146](https://arxiv.org/abs/1003.0146)。A。
14. **Xu, Hutter, Hoos & Leyton-Brown 2008, `SATzilla: Portfolio-based Algorithm Selection for SAT`** — JAIR 32:565–606；[arXiv 1111.2249](https://arxiv.org/abs/1111.2249)；[JAIR](https://www.jair.org/index.php/jair/article/view/10556/25269)；[项目页](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/)。A。
15. **Eger & Sauma Chacón 2020, `Deck Archetype Prediction in Hearthstone`** — FDG 2020；[DOI](https://doi.org/10.1145/3402942.3402959)；[PDF](https://slothlab.info/assets/pdf/eger2020fdg.pdf)。A。

### 7.3 防遗忘 / 补充阅读

16. **Geifman & El-Yaniv 2019, `SelectiveNet: A Deep Neural Network with an Integrated Reject Option`** — [arXiv 1901.09192](https://arxiv.org/abs/1901.09192)。A。
17. **Kirkpatrick et al. 2017, `Overcoming catastrophic forgetting in neural networks`** — PNAS；[arXiv 1612.00796](https://arxiv.org/abs/1612.00796)。A。
18. **Teh et al. 2017, `Distral: Robust Multitask Reinforcement Learning`** — [arXiv 1707.04175](https://arxiv.org/abs/1707.04175)。A。
19. **Liu et al. 2022, `NeuPL: Neural Population Learning`** — 2022；[arXiv 2202.07415](https://arxiv.org/abs/2202.07415)；ICLR 2022 的 venue 信息未核实。B。

### 7.4 Citation hygiene（重要）

- **1709.04326 不是 PFSP，也不是 PSRO。** 它是 `Learning with Opponent-Learning Awareness`（LOLA，Foerster et al. 2018；发表 venue 未核实）：[arXiv 1709.04326](https://arxiv.org/abs/1709.04326)。PFSP 的可执行定义在 AlphaStar 补充材料 `pseudocode/multiagent.py` 的 `pfsp()`；PSRO 在 1711.00832。
- **1902.01724 不是 AlphaStar 主论文**，而是 `AlphaStar: An Evolutionary Computation Perspective`（GECCO '19 Companion）：[arXiv 1902.01724](https://arxiv.org/abs/1902.01724)。主论文是 Nature 575:350–354，[DOI 10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z)。本文的 league/PFSP 细节以 Nature 补充代码为准；1902.01724 只用于“最终 agent 是 Nash distribution”这一句。
- **PFSP 的权重函数**（AlphaStar 补充代码已核对）：`variance = x(1−x)`、`linear = 1−x`、`linear_capped = min(0.5, 1−x)`、`squared = (1−x)^2`；主 PFSP 分支用 `squared`。不要把它写成“Lanctot 2017 的 PFSP”。

---

## 8. 落地顺序（一页版）

| 顺序 | 动作 | 产物 | 成本 | 判定 |
|---|---|---|---|---|
| 1 | `CollectTask.side`/swap + `--opponent-weights` | 双向 PG 语料 + PFSP 采样 | 0.5 天代码 + 2,000 局 | pooled 不降、floor 不增 |
| 2 | replay mix + per-opponent 非劣门禁 | 稳定的 3 轮 PG | 1 天代码 | 每轮不劣于 incumbent |
| 3 | root-anchor KL | 防 chain drift | 0.5 天代码 | KL<0.1 且 pooled 不降 |
| 4 | 历史池 + verification | 防忘记 | 1 天代码 + 20% 采集 | 历史池最小胜率 >0.5 |
| 5 | E1：扩 dvalin/shenhe/electro | 2–3 个新 PASS | 18,000 局采集 + ≤8,400 局评测 + <0.5 GPU·h | ≥2/3 PASS |
| 6 | E2：dgeo/ungeo router | 路由表或“不做”结论 | 12,000 局 | ≥+3pp 且 worst-cell 不崩 |
| 7 | R5：Nash 报表 | league 报表 | 复用日志 | 排序稳定 |

**最终建议**：先把第 1–2 步做成默认采集/门禁（它们不依赖任何新论文，纯粹是把已有 `eval/scoreboard.py` 和 `train/collect.py` 用对），再开 E1。Router 放到 E1 之后：没有 3–5 个可路由的强 arm，router 的实验设计本身没有统计功效。
