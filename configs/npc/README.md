# configs/npc — PvE NPC 规格（入库）

PvE「冒险挑战」NPC 的**关卡规格**与**敌人数据**定义。做法、机制与路线见
[`reports/explore/npc_pve_doc.md`](../../reports/explore/npc_pve_doc.md)。

## 文件

| 文件 | 内容 | 权威性 |
|---|---|---|
| `npc_enemies_timaeus.json` | 蒂玛乌斯三只敌人的 HP / 元素 / 充能 / A-E-Q / 被动 | **权威**（逐条对照公开源，文件内 `source`） |
| `npc_spec_example_timaeus.json` | 蒂玛乌斯关卡规格：阵容、特殊规则、挑战目标、参考日志 | 规格权威；`reference_log` **非权威** |

## 约定（应该这样做）

- **敌人数据应忠实**：HP / 元素 / 充能 / A-E-Q / 被动 从公开源抓取（来源见各文件 `source`）。
- **出牌策略应由我们自己生成**（`NpcPolicy`：greedy + rotation），**不**照抄原作逐回合序列。
  - `npc_spec_example_timaeus.json` 里的 `reference_log` 只是实测记录，**仅供参考、不用于实现**。
- NPC 侧：无手牌 / 无骰子 / 不打行动牌；**玩家恒定先手**（复刻时 NPC = player 1）。
- 充能：**A/E 各 +1，Q 耗 2，上限 2**；切人由角色被动（如 `Hide`）/ 脚本强制切人驱动，**不计行动**。

## 来源

- 阵容：GameWith《NPC Challengers Locations & Map》（`article/show/37530`）；
  交叉源 HoYoWiki 条目、Fandom `Genius Invokation TCG/Stage/*`。
- 敌人技能/HP/被动：GameWith 各卡页（链接见 `npc_enemies_timaeus.json` 内 `gamewith`）。
- 挑战目标：观测枢 / Fandom Stage 页（**待抓取逐条原文**）。
