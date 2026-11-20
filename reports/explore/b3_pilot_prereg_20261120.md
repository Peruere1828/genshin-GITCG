# 预注册：B3 pilot —— belief 根化的 actor 侧搜索（2026-11-20）

> **status: prereg（跑之前写定，跑完不改判定条件）**
> 决策：D29（主线 = 突破教师天花板）。依据：`reports/explore/imperfect_info_lit_20261120.md` §0.5–0.7。
> 口径：纯策略、CPU、固定 20 套脚本池、配对种子（D11 / EVAL_PROTOCOL）。

---

## 1. 问题

**Q**：把 fork 搜索从「活的快照（等于偷看对手手牌）」换成「从公开信念确定化的根」之后，
它还能不能超过同 deck 的脚本专家？

这是 D23 之后唯一还没被否掉的"搜索"用法：
- A4/D23 否掉的是**搜索当 teacher 再蒸馏给策略网**（软标签劣于硬标签，daybreak 独立复现：搜索 +20pp，蒸馏只带走 ~10%）。
- 本预注册测的是**搜索直接当 actor**，且叶子评估器不用价值网，用现成的脚本专家（Obscuro 的 ablation 支持这一点：
  换掉 Stockfish 的"傻瓜评估器"仍有 55.0% 胜前 SOTA）。

## 2. 已确认的前提（有证据，不是假设）

| 前提 | 证据 |
|---|---|
| belief 根化搜索**已经实现**且有测试 | `envs/fork_search.SearchContext.belief_root` + `belief_roots_for()`；`tests/test_belief_root.py` |
| 它在引擎里**跑得通** | 2026-11-20 smoke：`belief_root=True, belief_samples=2, top_k=2, rollouts=1, search_every=20, max_searches=2` → 2 次搜索、每根 2 个 belief root、整局 11.86s（同种子无搜索基线 4.94s） |
| 确定化不泄漏隐藏真值 | `reps/belief.sample_belief_roots` 只吃 masked state；`tests/test_belief_root.py` 钉住"隐藏真值不影响根" |
| CLI 已支持 | `python -m scripts.run_search_agent --belief-root --belief-samples N`，且默认跑 EXPERT/SEARCH **成对**两模式 |

## 3. 设计与臂

单臂规格（配对）：同一 `(deck, opponent, seed)` 下跑 `EXPERT`（脚本专家，无搜索）与 `SEARCH`（同专家 + fork 搜索）各一局。

| 配置 | 用途 | CLI |
|---|---|---|
| `b3_belief_r3` | **主臂**：公平搜索 | `--belief-root --belief-samples 3` |
| `b3_live_r3` | **上界臂**：活在真实快照上搜索（偷看） | 不加 `--belief-root` |

上界臂不是要给结论，而是给**"公平化代价"**：`(live − expert) − (belief − expert)` 就是 belief 不确定性的价格。
如果上界臂都不超过专家，问题不在 belief，在搜索本身。

**固定参数**（不进扫描，避免事后调）：

```
--deck superconduct_aggro      # 池内专家 ≈0.51，接近 50%，增益可测
--rollouts 2
--top-k 3
--search-every 10
--max-searches 6
--workers 8
```

**对手（4 个，先验写定，不许换）**：`superconduct_aggro`、`superconduct_mika_attack`、
`natlan_battleship`、`dvalin_bonk`。

**种子**：`0..7`（8 个）。→ 每臂 `4 × 8 × 2 = 64` 局；两臂共 128 局。

## 4. 判定（跑之前定死）

**主指标**：`SEARCH − EXPERT` 的配对差（按 `(opponent, seed)` 配对，只取双方都干净的局），
在 4 个对手上做聚合，报点估计 + 对手聚类 bootstrap CI。

| 结果 | 判定 |
|---|---|
| 点估计 ≥ +5pp 且聚类 CI 下界 > 0 | **PASS**：扩大 n 到 400 对（1,400–1,600 局）做认证级复验 |
| 点估计 在 (−5, +5)pp 且 CI 含 0 | **INCONCLUSIVE**：只在"上界臂同时 PASS"时，才把 n 加到 200 对重判一次；否则归档 |
| 点估计 ≤ −5pp 且 CI 上界 < 0 | **FAIL**：actor 侧搜索在本预算下有害，搜索线整体冻结（含上界臂） |

**次要指标**（只记录，不参与判定）：
- 每次搜索的候选均值分布（`means`/`counts`）——若候选均值几乎全 1.0/0.0，说明 `rollouts=2` 分辨率不足（A4 的病灶）。
- `searched_decisions` 与实际 `wall_seconds`。
- `belief_roots` 字段（确认根数生效）。

**预注册停止规则**：
- 任一臂出现 `error` 或 `truncated` > 5% → 停，先修再跑；
- 主臂与上界臂的 `searched_decisions` 差 > 20% → 视为配置漂移，停。

## 5. 成本（实测外推）

- 基线局 4.94s；2 roots × 2 candidates × 1 rollout × 2 searches = 11.86s（+7s）。
- 主臂每次搜索 = 3 roots × 4 candidates（3 + 必含 base）× 2 rollouts = 24 fork 任务，
  6 次搜索/局 → 144 任务/局；按 smoke 的有效速率（~0.6s/任务/8 workers）≈ +11s/局。
- 64 局（32 搜索 + 32 专家）≈ **10–15 分钟**墙钟；128 局（两臂）≈ 25 分钟。
- 落盘：`data/search/<tag>.jsonl`（逐局，可断点续跑）+ `reports/search/<tag>.json`（聚合）。

## 6. 本实验**不**测什么

- 不测 belief 头的质量（那是 B1，另跑，0 新对局）；
- 不测特权 critic / PTIE（B2）；
- 不测搜索蒸馏（D23 已永久关闭）；
- 不测"搜索能否上 A800 放量"——本臂在 12 核 CPU 上跑不完就没有放量的意义。

## 7. 结果落点

- PASS → `docs/PLAYBOOK.md` 追加一条"actor 侧 belief 根化搜索"，并把 B2 提为下一个前置；
- FAIL → `docs/failures/007-actor-search-belief-root.md`，并把 D29 的杠杆排序改成 B1→B2（不再提搜索）；
- INCONCLUSIVE → 只更新 `reports/search/`，不改决策。

---

## 8. 结果（2026-11-20，主臂 `b3_belief_r3`）

规格：4 对手 × 8 seeds × 2 模式 = 64 局，主臂全员完成，`errors=0`、`truncated=0`。
落盘：`data/search/b3_belief_r3.jsonl`、`reports/search/b3_belief_r3_20261120T143304.json`。

| 指标 | 值 |
|---|---|
| 可用配对 | 32 |
| `SEARCH − EXPERT` 均值 | **−0.0625** |
| 胜/负/平（按配对） | **0 / 2 / 30** |
| search 局均墙钟 | 13.6s（expert 3.1s，≈4.4×） |
| 每局实际搜索次数 | 2–6（`max_searches=6`，`search_every=10`） |

**判定**：预注册三分支均未严格触发（点估计 ≤ −5pp，但 n=32 的 CI 上界不可能 < 0）。
按"不事后改判据"处理：**记为 INCONCLUSIVE-负向**——点估计为负、**零上行信号**（没有任何一局是
搜索赢而专家输的），且 30/32 局结果完全相同。

**诊断（比结论更重要）**：搜索几乎不改变决策。原因与 A4/D23 同构——
`rollouts=2` 下候选均值只有 0/0.5/1 三档（smoke 里出现过三个候选全 1.0），
根本分不开差异；`top_k=3` 里通常包含 base 本身。也就是说**这一版测的是"搜索噪声"，不是"搜索强度"**。

**下一步（不改判据）**：跑上界臂 `b3_live_r3`（同规格、不加 `--belief-root`，即偷看真实快照）。
- 若上界臂也 ≈0 → 瓶颈是搜索预算（rollouts/深度），不是 belief 的公平化代价；
  应先做"预算阶梯"（rollouts 2→8、depth 1→2），再谈 belief 质量；
- 若上界臂显著为正 → 公平化代价真实存在，B1（belief 校准）成为前置。

### 8.1 上界臂 `b3_live_r3`（同日完成，同规格、不加 `--belief-root`）

| 臂 | n | `SEARCH−EXPERT` | SE | 胜/负/平 |
|---|---:|---:|---:|---|
| `b3_belief_r3`（公平） | 32 | −0.0625 | 0.043 | 0 / 2 / 30 |
| `b3_live_r3`（偷看快照） | 32 | +0.0625 | 0.043 | 2 / 0 / 30 |

**结论（修正上节的两分支预期）**：两臂都在 ±1.5 SE 内、方向对称、各只翻转 2/32 局。
即**"公平化代价"没有被建立**，也没有被否定——这个预算下根本无法分辨。
真正稳定的观察是：**搜索几乎不改变决策**（64 局里 60 局结果与专家相同）。

因此下一杠杆是**搜索预算**（`rollouts` 2→8、深度 1→2、候选集从"含 base 的 top-k"改为"确实不同类的候选"），
而不是 belief 质量。B1 校准暂不前置；等预算阶梯出现正信号后再接。
