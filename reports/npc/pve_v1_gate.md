# PvE-v1 门禁报告（decklab 组牌 × 16 关 × 多种子）

> 里程碑：PLAN §4.4 `PvE-v1`（D17/D19）。口径：**纯策略**，contestant = decklab 池 C 组牌，NPC = 关卡 lineup + greedy `NpcPolicy`。
> 生成：2026-10-12T22:36:31；tag=`pve_v1_gate`。

## 门禁判定：**PASS** — 三星达成率 74.7% (门槛 ≥ 60.0%)，Wilson95 [70.2%, 78.8%]，cluster95 [63.0%, 85.7%]

- 胜率 99.7%（Wilson95 [98.5%, 100.0%]）
- 样本：384 局 = 16 关 × 24 种子；error 0 / truncated 0；平均星 2.61

## 逐关（primary gate 臂）

| 关卡 | 局数 | 胜 | 胜率 | 三星 | 三星率 | 平均星 | 平均回合 |
|---|---|---|---|---|---|---|---|
| blanche | 24 | 23 | 95.8% | 5 | 20.8% | 2.17 | 7.2 |
| buyun | 24 | 24 | 100.0% | 18 | 75.0% | 2.75 | 4.8 |
| chefmao | 24 | 24 | 100.0% | 10 | 41.7% | 2.42 | 5.8 |
| ellin | 24 | 24 | 100.0% | 16 | 66.7% | 2.67 | 5.2 |
| irontonguetian | 24 | 24 | 100.0% | 15 | 62.5% | 2.58 | 5.2 |
| jifang | 24 | 24 | 100.0% | 24 | 100.0% | 3.00 | 2.4 |
| lan | 24 | 24 | 100.0% | 23 | 95.8% | 2.96 | 4.0 |
| liusu | 24 | 24 | 100.0% | 16 | 66.7% | 2.67 | 5.3 |
| marjorie | 24 | 24 | 100.0% | 23 | 95.8% | 2.96 | 3.5 |
| patton | 24 | 24 | 100.0% | 21 | 87.5% | 2.83 | 4.2 |
| shitou | 24 | 24 | 100.0% | 14 | 58.3% | 2.58 | 4.5 |
| swan | 24 | 24 | 100.0% | 24 | 100.0% | 1.00 | 3.5 |
| timaeus | 24 | 24 | 100.0% | 24 | 100.0% | 3.00 | 4.1 |
| timmie | 24 | 24 | 100.0% | 22 | 91.7% | 2.92 | 4.5 |
| uncletian | 24 | 24 | 100.0% | 21 | 87.5% | 2.88 | 4.6 |
| wagner | 24 | 24 | 100.0% | 11 | 45.8% | 2.46 | 2.5 |

## 逐目标（primary gate 臂）

| 目标 | 出现局数 | 达成 | 达成率 |
|---|---|---|---|
| defeated_less_than | 120 | 120 | 100.0% |
| play_equipment_card | 48 | 35 | 72.9% |
| play_event_cards | 24 | 24 | 100.0% |
| trigger_elemental_reactions | 96 | 84 | 87.5% |
| use_elemental_burst | 72 | 68 | 94.4% |
| win | 384 | 383 | 99.7% |
| win_within_rounds | 360 | 290 | 80.6% |

## 组牌质量 vs 打牌质量（消融）

| 臂 | 牌组 | 策略 | 局数 | 胜率 | 三星率 | Wilson95 | cluster95 |
|---|---|---|---|---|---|---|---|
| gate | poolc_superconduct | contestant | 384 | 99.7% | 74.7% | [70.2%, 78.8%] | [63.0%, 85.7%] |
| play_plain | poolc_superconduct | contestant_plain | 384 | 99.7% | 61.5% | [56.5%, 66.2%] | [45.3%, 76.8%] |
| play_heuristic | poolc_superconduct | heuristic | 384 | 68.8% | 33.9% | [29.3%, 38.7%] | [13.8%, 56.2%] |
| deck_defensive | poolc_superconduct_defensive | contestant | 384 | 100.0% | 72.4% | [67.7%, 76.6%] | [58.9%, 84.6%] |
| deck_meta | superconduct_aggro | contestant | 384 | 97.9% | 74.2% | [69.6%, 78.3%] | [58.6%, 88.0%] |
| smoke_ref | superconduct_aggro | expert:superconduct_aggro | 384 | 97.9% | 62.0% | [57.0%, 66.7%] | [42.4%, 80.5%] |

## decklab 组牌口径（D19）

decklab 当前**没有**学习/Agent 式自动组牌（`decklab/README.md` 待办，`decklab.sensitivity` 只做固定模板内单卡换卡评测）。本门禁用 `decklab/builder.py` 的**确定性规则组牌**近似 D19：以固定 meta 牌组（`superconduct_aggro`）的**池 C 可用核心**（含张数）为骨架，再按人工写定的池 C 填充序（节奏/aggro + 目标覆盖，含 1 张武器保证 `play_equipment_card` 可达成）补到 30 张（每卡 ≤2 张、全部来自池 C）。这是「固定 meta deck + decklab 变体」口径，**不是自动组牌 Agent**；偏差已在此记录。角色：甘雨(1101,冰)+刻晴(1403,雷)+北斗(1405,雷)。

## 与 B2/B3 冒烟的差异

B2/B3 冒烟（`reports/npc/npc_pve_smoke_v1.json (PvE-B2/B3)`）：48 局 = 16 关 × 3 种子，contestant = 脚本专家 superconduct_aggro（自带 deck 专属规则，不打装备牌），win 0.979 / 三星 0.625。本门禁把同一配置在 24 种子下复跑（`smoke_ref` 臂）：win 0.979 / 三星 0.620。primary 门禁臂换成池 C 组牌 + contestant 策略；两臂差异来自① 种子数（3→24）② deck（`superconduct_aggro` → 池 C 组牌）③ 策略（脚本专家 → contestant：装备/事件目标由 0% 提升，见逐目标表）。

## 未决问题

- `swan` 的 `objectives_verified=false`（eurogamer 只列 1 条目标）→ 其「三星」等于「赢下」，会把总三星率略微抬高；门禁口径未按 3 条目标标准化。
- `shitou` 的 `win_within_rounds.n=44` 未核实（疑为 4/14 笔误）→ 该目标实际从不失败，对该关三星率无约束。
- 关卡特殊规则未在引擎内模拟：`artifact_cards_free`（蒂玛乌斯）、`chef_mao_support_present` + `pyro_application_on_action_phase_start`（卯师傅）；对局按普通规则跑，蒂玛乌斯/卯师傅偏难/偏易需另立项。
- decklab 无 Agent 式自动组牌：本门禁用 `decklab/builder.py` 规则组牌近似（偏差见上）。后续若实现学习/Agent 组牌，需重跑本门禁。
- PvE 机制近似（D17）：NPC 无手牌/无骰子/不打出行动牌；引擎只有全局手牌/骰子配置，NPC 仍持有被 `NpcPolicy` 闲置的手牌与骰子。
- `win_within_rounds`（多数 n=5）是主要失分项：打装备/事件牌会牺牲节奏，「目标达成」与「速胜」存在张力，是当前三星率的主要上限。

## 代码评审后的复核（2026-10-14）

评审 M4 指出 `envs/match.py::_collect_events` 漏记**非伤害路径**的元素反应（引擎在纯元素附着 `applyAura` 上也计反应）。已修（新增 `apply_aura` 记录分支 + 扩展实体 owner 登记）。

修复后按同一配置复跑（`--tag pve_v1_gate2`，384 局），与本报告**逐臂、逐目标、逐关、逐局完全一致**：primary 三星 287/384 = 0.747396、win 383/384；`trigger_elemental_reactions` 84/96；384 行的 `winner/rounds/stars/objectives/decisions/errors` **0 处不同**。原因：门禁配置中 `applyAura` 反应仅出现 **5 次，全部在 `irontonguetian` 且全部为 NPC（player=1）侧**，contestant 侧 0 次；更宽扫描（1152 局）12 次亦全为 NPC。故 M4 对本门禁结果**零影响**，评审设想的 `patton/buyun/shitou/lan` 掉星**未能复现**。

同批复核另修 `npc/eval.py`（`base_seed` 折进 seed、同类型 objectives 去重键、流式落盘降内存）、`npc/contestant.py`（seed 透传）、`npc/policy.py`（死子句）、`decklab/builder.py`（补牌）——均**不改变**门禁数字。

再把 `apply_aura` 反应归因从「批次**最后一个** `skill_used`」改为「**最近前驱** `skill_used`」（修正跨玩家批次的误归因）后，按同配置再复跑（`--tag pve_v1_gate3`）：6 臂聚合**完全一致**，逐局 384 行中**仅 `wall_seconds`（计时）不同**，`winner/rounds/stars/objectives/decisions/fallbacks` 全部相同；primary 三星率 0.747396 不变。

## 复现

```bash
python -m scripts.run_npc_gate --stages all --seed-count 24 --workers 8 --tag pve_v1_gate
```

原始逐局记录：`data/npc/<tag>__<arm>.jsonl`（gitignore，可 `--tag` 断点续跑）。
