# PvE-B2/B3 交付：`NpcPolicy` + `NpcSpec`/三星判定 + 评测入口

> **status: superseded（2026-10-27）**：被 `reports/npc/pve_v1_gate.md`（PvE-v1）取代；
> 本文保留为 B2/B3 历史证据。

> 里程碑：PLAN §4.4 `PvE-B2`（策略）+ `PvE-B3`（规格与判定 + 评测入口），D17。
> 定位：PvE 调试/训练辅助（确定性假想敌池 + 三星门禁），纯策略口径。
> 日期：2026-10-12。引擎：`gitcg` 0.21.0 + B1 overlay（17 敌人 `.gts`）。

## 1. 改动 / 新增文件

**新增包 `npc/`**

| 文件 | 作用 |
|---|---|
| `npc/spec.py` | `NpcSpec`（关卡）+ 敌人 typed dataclass；中文/英文名 → 引擎角色 id 映射（`appears_in` 交叉校验）；A/E/Q 技能 id + 伤害表 |
| `npc/policy.py` | `NpcPolicy`（确定性 greedy + rotation），`npc_policy_for_stage` 装配 |
| `npc/objectives.py` | 7 类挑战目标的达成判定 + 三星判定 |
| `npc/eval.py` | 评测入口：关卡 × 种子批量、流式落盘 + 断点续跑 + 指纹护栏、单/多进程 |
| `npc/eval_worker.py` | 多进程 worker（复用 `envs.rollout.run_task_shard` 的 JSON-lines 子进程协议） |
| `npc/__init__.py` | 公共导出 |

**`envs/match.py`（additive，不破坏既有字段）**

- `DecisionRecord` 新增 `subject_definition_id`、`card_kind`（打出牌的类别：weapon/artifact/event/support）。
- 新增 `MatchEvent`（`kind="skill"/"reaction"`，含 `player`/`skill_type`/`reaction_type`）。
- `PolicyPlayer` 新增可选 `event_sink`：把引擎 `Notification.mutation` 里的 `skill_used` / `damage(reaction_type)` 投影成 `MatchEvent`。
- `run_match` 新增 `record_events: bool = False`；`MatchRecord` 新增 `events`；`as_dict(include_events=...)`，默认行为不变。

**其他**

- `pyproject.toml`：packages 增加 `"npc"`。
- `tests/test_npc_spec.py`、`tests/test_npc_policy.py`、`tests/test_npc_objectives.py`、`tests/test_npc_eval.py`。
- `reports/npc/`（本报告 + 冒烟聚合）。

## 2. `NpcPolicy` 设计

- **永不打出行动牌**（PvE 近似，D17）：`action_play_card`、`action_elemental_tuning` 一律不选；无可用技能时直接 `declare_end`。
- **不主动切人**：`action_switch_active` 不选；切人/被动（如冰箭丘丘人的 `Hide`、战斗开始时附属的护盾/磨刃）全部由引擎驱动，不计行动。
- **greedy**：在合法 `action_use_skill` 里选「当前可造成最大伤害」者。伤害来自 `npc.spec` 的 A/E/Q 伤害表（技能 definition id → 伤害，与 `.gts` 逐一交叉校验，见 §4）。
- **平局固定顺序**：`(角色槽位, 技能 definition id 升序)`。技能 id 按 `角色id*10 + k`（k 依 A→E→Q 升序）编号，故升序即 A→E→Q。
- **rotation（节奏约束，可选，多套）**：`ROTATIONS` 提供 `greedy`（默认，无约束）/`cycle_a_e_q`/`cycle_e_q_a`/`cycle_q_e_a`/`cycle_e_a`。rotation 模式下对「当前出战角色」按固定技能种类循环取优先技能（仅当该种类有造成伤害的合法技能，否则回退 greedy），指针仅在实际出招时前进。
- **非行动请求**：`choose_active` 取最小槽位、`reroll_dice`/`switch_hands` 取「不变更」分支（确定性）。
- **确定性**：`choose` 是合法动作集（+ rotation 指针）的纯函数；同种子逐决策复现（`tests/test_npc_policy.py`，含真实对局双跑与 rotation 双跑）。

## 3. `NpcSpec` 与关卡数据

- `load_stages()` 解析 `configs/npc/npc_stages_mond_liyue.json` 的 16 关（`lineup`/`reserve`/`special_rules`/`objectives`），并用 `npc_enemies_all.json` 的 `appears_in` 交叉校验每一条出战/预备阵容（不对应即报错）。
- `NpcStage.deck_characters` = lineup 按 `count` 展开的引擎角色 id，如提米 `(3302, 3302)`、蒂玛乌斯 `(3102, 3101, 3302)`、玛乔丽 4 角色。`reserve_characters` 单独暴露（当前不并入对局阵容）。
- 敌人 → 引擎卡 id 映射写在 `ENEMY_ENGINE_IDS`（19 类，官方 GCG 敌人卡 id；与 B1 一致），`load_enemies()` 校验每个 slug 都能解析。
- 伤害表 `enemy_damage_table()`：按 `角色卡id*10 + k` 依 A→E→Q 编号取 `damage`；E 缺失时 Q 顺延为 k=2（如冰箭丘丘人 A=1/Q=4）；被动占下一槽；额外准备技（愚人众风拳前锋军 35025）接在被动之后。

## 4. objectives 判定覆盖（7 类 → 依据的轨迹字段）

| 目标类型 | 依据 | 需要新增事件采集？ |
|---|---|---|
| `win` | `MatchRecord.winner == 0` | 否 |
| `win_within_rounds` | `winner == 0` 且 `MatchRecord.rounds <= n` | 否 |
| `defeated_less_than` | `winner == 0` 且**终局快照**里 player 0 被击倒角色数 `< k`（需 `record_final_state=True`） | 否（用 `final_state_json`） |
| `use_elemental_burst` | `MatchEvent(kind="skill", skill_type="burst", player=0)` 计数 ≥ count | **是**（`skill_used` mutation） |
| `trigger_elemental_reactions` | `MatchEvent(kind="reaction", player=0)` 计数 ≥ count | **是**（`damage.reaction_type != 0`，按伤害来源 id 归属玩家） |
| `play_equipment_card` | `DecisionRecord(chosen_kind="action_play_card", card_kind=kind)` 计数 ≥ count | 半新增：`DecisionRecord.card_kind`（在决策时从手牌 `EntityState.type/equipment` 解析） |
| `play_event_cards` | `DecisionRecord(..., card_kind="event")` 计数 ≥ count | 同上 |

- **三星 = 该关列出的全部目标达成**（`three_star = all(achieved)`；`stars = 达成数`）。数据里斯万（swan）只列 1 条目标（`objectives_verified=false`），按此定义其「三星」等于「赢下」——B5 汇总时应结合 `objectives_verified` 说明。
- 反应归属：`damage.source_id`（伤害来源实例 id，角色/召唤均可）经「实例 id → 玩家」映射归属；映射在每个 notification 的公开状态上刷新。
- 典型的 `envs.match.run_match` 调用：`record_decisions=True, record_events=True, record_final_state=True`（`npc.eval` 已默认如此）。

## 5. 评测入口与冒烟结果

`npc.eval` 输出**胜率 + 三星达成率**；逐关 × 逐种子批量；逐行流式落盘 `data/npc/<run>.jsonl` + `task_index` 断点续跑 + spec 指纹护栏；`workers>1` 走 `npc.eval_worker` 子进程（避免 fork 引擎死锁）。

**冒烟**（全 16 关 × 种子 {1,2,3}；contestant = `superconduct_aggro`，脚本专家策略；NPC = 各关 lineup + greedy `NpcPolicy`）：

- 48 局，**win_rate 0.979**（Wilson95 [0.891, 0.996]），**三星率 0.625**（[0.484, 0.748]），平均星数 2.42，0 错误 / 0 截断。
- 逐目标达成率：`win` 0.98、`win_within_rounds` 0.78、`defeated_less_than` 1.00、`trigger_elemental_reactions` 1.00、`use_elemental_burst` 0.67、`play_event_cards` 0.33、**`play_equipment_card` 0.00**。
- 单关样例（可复现）：`timmie` seed=1 → 胜，4 回合，三星；`timaeus` seed=1 → 胜，2 星（缺「打出 1 张圣遗物牌」）；`wagner` seed=1 → 胜，2 星（缺「打出 1 张武器牌」）。

冒烟要点：**失分集中在「打牌质量」而非「组牌/对局质量」**——脚本专家 deck 几乎不打装备牌，导致所有 `play_equipment_card` 目标恒失败（timaeus/wagner 直接掉一星）。这正是 B5「报表拆分组牌质量 vs 打牌质量」的实例：B5 不能只换牌，还要选/造一个会打装备牌的 contestant 策略，否则 `play_equipment_card` 类目标会被策略卡死。

## 6. 与 B5（decklab 组牌 × 16 关 × 多种子，三星 ≥ 60%）的衔接

- B5 直接调用 `npc.eval.run_npc_eval(NpcEvalSpec(stage_names=..., contestant=<decklab DeckSpec>, seeds=(...), policy0=<spec>, ...))`：`contestant` 接受任意 `DeckSpec`（decklab 产物即可），`policy0` 用 `envs.policy.build_policy` 能解析的 spec。
- 门禁统计用 `NpcEvalResult.summary()` 的 `overall.three_star_rate`（附 Wilson CI）+ `stages` 逐关 + `objectives` 逐目标，天然支持「组牌质量（胜率/被击倒）vs 打牌质量（装备/事件/爆发/反应）」拆分。
- 规模化：`--workers N` 多进程；`--run-id` 生成 `data/npc/<run>.jsonl` 可断点续跑。B5 可按关卡/分片多次调用同一 tag 累积。
- 建议：汇总门禁时标注 `objectives_verified`（斯万目标不全，会虚高三星率），并对 `play_equipment_card` 类先确认 contestant 会打装备牌。

## 7. 未决问题

1. **胜利归属随座位**：`npc.eval` 固定 contestant = player 0、NPC = player 1（玩家恒先手）。引擎无 per-player 手牌/骰子配置，NPC 仍会拿到（并被 `NpcPolicy` 闲置的）手牌与骰子——按 D17 属可接受的近似。
2. **三星定义与不完整数据**：`swan` 仅 1 条已验证目标；是否把「3 条目标」标准化为门禁口径需 B5/主线确认。
3. **`play_equipment_card` 现状**：冒烟里脚本专家不改打牌策略时该目标 0%；B5 需要能打装备牌的 contestant 策略（或在门禁口径中显式区分）。
4. **关卡特殊规则未实现**：`artifact_cards_free`（蒂玛乌斯）、`chef_mao_support_present` / `pyro_application_on_action_phase_start`（卯师傅）当前都不是引擎内特殊规则，评测未模拟；若门禁需要，需在引擎 overlay 侧另立项。
5. **rotation 语义**：现为「按角色技能种类循环的节奏约束」，与原作逐回合序列无关（D17 明确不逆向）；是否需要更多 rotation（如按能量/爆发优先）待 B5 反馈。
