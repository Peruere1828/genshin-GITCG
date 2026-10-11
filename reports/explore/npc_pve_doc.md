# PvE NPC 复刻文档（路线 2）

> 状态：**现行**（取代 `npc_pve_research.md`、`npc_route2_plan.md`）。
> 原则：只保留正确做法，用"应该…"式肯定表述；已删除错误/过时/踩坑内容。

---

## 0. 决策

- **敌人数据应该用爬虫获取**：NPC 的出战阵容、以及每个敌人角色的技能/被动/HP，应从公开源抓取（见 §3）。
- **出牌策略应该自己生成**：NPC 策略由我们实现（**greedy + 多种 rotation 组合**），不做逐关实测逆向。
- **不应追求完整行动序列**：完整序列的获取成本不现实，改用确定性规则。
- **引擎应该用 overlay 重建**：PvE 敌人角色需要移植进引擎数据并重建 `libgitcg.so`（见 §4），`refs/` 保持只读。

---

## 1. PvE 机制（应该知道的规则）

- 「冒险挑战」类 NPC **没有手牌、没有元素骰、不打出行动牌**，每回合按**固定技能序列**行动；玩家可用「行动意图」预览。
- **玩家恒定先手**。→ 复刻时 AI/玩家应为 player 0，NPC 为 player 1。
- 敌人**转换形态后重新生成行动意图**。
- 「三星」= 游戏内「挑战目标」(PvE only)：通常 **3 条**，第 1 条恒为「赢得胜利」。
  - 已核实类型：`N 回合内胜利`、`胜利时被击倒角色数小于 2`、`没有角色被击倒`、`使用 1 次元素爆发`、`触发 2 次元素反应`。
  - 「认真胜负」通常**只有 1 条**（赢得胜利）；「胜冠之试」是 PvP，不属于 NPC。
- PvE 模式：大世界冒险挑战 / 决斗对局 / 每周来客挑战 / 角色邀约 / 酒馆挑战（友好对局 + 认真胜负）/ 铸境研炼（活动）。

## 2. 概念区分

- **怪物角色**：可作为卡牌被玩家使用（deck-legal），如丘丘岩盔王、愚人众等。
- **NPC 敌人**：PvE 关卡专属的敌方角色（史莱姆 / 丘丘人射手 / 萨满 / 深渊法师 / 野伏 / 蕈兽…），**不在引擎的 PvP 卡池里**，需要移植。

## 3. 数据源与抓取（应该用这些）

- **GameWith**（本环境可达）：NPC 挑战者列表（每关的 Opposing / Reserve Lineup）+ 每张卡的技能/HP/被动页。
- **Fandom** `Genius Invokation TCG/Stage/*`：Active Lineup + Action Cards + Special Rules + Required Player Level；可用 MediaWiki API 枚举（`Category:Genius Invokation TCG Stages`）。
- **HoYoWiki**（entry 页）：关卡信息与特殊规则。
- **观测枢 / 米哈游百科**：挑战目标原文。
- 应抓取的字段：
  - 关卡 → `{出战阵容, 预备阵容, 难度档, 特殊规则, 挑战目标(3 条)}`
  - 敌人角色 → `{HP, 元素, 充能上限, A/E/Q 技能(名称/费用/效果), 被动}`

## 4. 引擎与构建（事实 + 做法）

- `gitcg` 是 cffi 绑定到 `libgitcg.so`；**卡牌数据编死在 .so 内**（V8 + 打包的 JS 数据）。
- 引擎只实现 **PvP 卡池**；PvE 敌人不存在于其中。
- 角色/技能用 **`.gts` DSL** 定义，位于 `packages/data/src/characters/<element>/*.gts`，由 `scripts/generators/imports.ts` 自动登记导入表。
- 重建 `libgitcg.so` 需要工具链：**cmake ≥ 3.25、node 26、pnpm 12、ninja**，并链接 V8（`piovium/v8-cmake-barbara`）。
- `refs/genius-invokation` **只读**（PLAN §6.5）：应该在**构建 overlay**（复制到 `build/engine/` 后再加我们的文件/补丁）里新增敌人定义，不修改 refs。
- 引擎支持 **全局** `initialHandsCount / initialDiceCount` 配置（**无 per-player 版本**）→ NPC 的"无手牌无骰子"应该通过**策略近似**实现（NPC 永不打出行动牌）。

## 5. 出牌策略（应该自己生成）

`NpcPolicy` 提供确定性规则，默认组合使用：

- **greedy**：每次行动选"当前可造成最大伤害"的技能；平局按固定顺序（角色 → 技能）打破。
- **rotation**：每角色一条固定轮转 + 固定出手顺序；可配置多套。
- **script**：可选模式；若某关有实测序列，则照抄。
- **组合**：以 greedy 为主，rotation 作为节奏约束。

该方案确定性、可复现、难度可调；"角色被击倒后如何处理"由我们自己的规则定义，不需要逆向原作。

## 6. 已获取示例（蒂玛乌斯）

- 阵容：冰箭丘丘人 / 冰盾丘丘暴徒 / 炽热斧头丘丘暴徒，**HP 3/8/8**。
- 技能/被动：见 `npc_enemies_timaeus.json`（冰箭丘丘人**无 E 战技**；被动 `Hide` = 攻击后切下一角色；冰盾丘丘暴徒被动 `寒霜大盾`；炽热斧头丘丘暴徒被动 `燃火之刃`）。
- 充能机制：**A/E 各 +1，Q 耗 2，上限 2**。
- 切人：由角色被动（如 `Hide`）/ 脚本强制切人驱动，**不计行动**。

## 7. 里程碑

| | 交付 | 验收 |
|---|---|---|
| M0 引擎可重建 | overlay 重建 `libgitcg.so` | 现有 `pytest -m "not slow"` 全绿 + 跑通一局 |
| M1 敌人移植 | 敌人 `.gts`（技能/被动/HP） | 可组牌、跑满、无错误 |
| M2 策略层 | `NpcPolicy`（greedy + rotation） | 同种子可复现 |
| M3 目标 + 评测 | `NpcSpec` + 目标判定 + 评测入口 | 输出胜率 + 目标达成率 |
| M4 接入组牌 | AI 在 C 池内组牌 → 对 NPC 池评测 | 闭环可跟踪 |

## 8. 待办

- 抓取蒙德/璃月各 NPC 的**挑战目标（3 条原文）**。
- 移植敌人角色 `.gts`（三只丘丘人起）。
- 实现 `NpcPolicy`（greedy + rotation）与离线单测。
- 接入评测（胜率 + 目标达成）。

## 9. 相关文件

- `configs/npc/npc_enemies_timaeus.json` — 蒂玛乌斯三只敌人的技能数据（权威）。
- `configs/npc/npc_spec_example_timaeus.json` — 蒂玛乌斯关卡规格（阵容/目标/参考日志）。
- `reports/explore/npc_lineups_mond_liyue.md` — 蒙德/璃月 NPC 出战阵容。
- `configs/pools/pool_C_v34_mondliyue.json` — AI 侧卡池（角色 19 + 牌 107）。
