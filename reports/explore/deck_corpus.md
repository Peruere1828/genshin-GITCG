# 七圣召唤公开卡组语料 (deck corpus) — 探索任务 1

> 生成：2026-10-11（UTC）；由 `scripts/explore/render_deck_report.py`
> 从验证后的 `data/deck_corpus/decks.json` 渲染。卡组数据/机器可读副本见
> [`reports/explore/deck_corpus.json`](./deck_corpus.json)，采集脚本
> `scripts/explore/fetch_deck_corpus.py`，加载器 `decklab/corpus.py`。

## 0. 摘要 (TL;DR)

- 共 **73** 套公开卡组，**73** 套通过分享码解码为合法
  **3 角色 + 30 行动牌**；其中 **73** 套又各跑通 1 局脚本对局（无引擎错误/无截断）。
- 来源：miHoYo 官方「卡牌广场」API（`api-takumi.mihoyo.com/event/cardsquare/`，无需登录）为主，
  外加手工从公开攻略/视频抄录并逐条验证的当前版本卡组。
- 官方积分赛池（49 套）为 **3.7–4.3 赛事归档**（2023-07 → 2024-01）；官方 `index_recommend` 会轮换，
  本次含 2024-07 / 2025-09 条目；全语料时间跨度 2023-05 → 2025-09。
- 一切以**引擎分享码解码**为准：解码失败 / 非 3+30 / 含未知 id 的码一律剔除，未收录未验证卡组。

## 1. 方法与验证 (method)

**数据获取**：页面 `webstatic.mihoyo.com/ys/event/bbs-lineup-qskp/index.html#/pc/home` 是 JS SPA，
真正的数据在 JSON API `api-takumi.mihoyo.com/event/cardsquare/`。本次调用的端点：

| 端点 | 含义 | 条数 |
|---|---|---|
| `match_pool_list?page_size=50` | 燃斗无双积分赛池 | 49 |
| `index_recommend` | 首页推荐 | 4 |
| `honor_match_pool` | 荣誉 + 随机视频精选 | 3 |
| `labels` / `match_pool_filter` | 标签/赛事筛选（仅上下文） | — |

每条记录自带官方 `card_code`（分享码）、`role_cards`（3 角色）、`action_cards`（30 牌）与作者攻略 `desc`。
跨端点重复分享码已去重（`also_seen_in` 记录别名）。

**验证口径**（`scripts/explore/fetch_deck_corpus.py`）：

1. 用仓库移植的引擎编解码 `agents.scripted.codec.decode_share_code` 解码分享码 → 定义 id；
2. 断言 **3 角色 + 30 行动牌** 且所有 id 在资产目录（`data/assets`）中存在；
3. 用 `envs.match.run_match` 跑 **1 局**（己方策略 `heuristic`、对手策略 `legal_random`，对手牌组
   固定为脚本专家 `superconduct_aggro`，种子 7），确认无引擎异常、无截断。

## 2. 规模与分布 (overview)

- 验证：codec **73/73** 通过；engine 再验 **73** 套全过。
- 体系分布（全部 73 套）：

| 体系 Archetype | 套数 |
|---|---|
| 扩散/风 | 7 |
| 水债/召唤-水 | 6 |
| 激化/召唤 | 6 |
| 莫皇草/召唤穿透 | 6 |
| 超导/雷鹤 | 6 |
| 双岩/结晶 | 5 |
| 火刻/超载 | 4 |
| 冻结/控制 | 4 |
| 快攻 | 3 |
| 超导/值伤 | 3 |
| 水皇草/召唤-水 | 3 |
| 感电/超载 | 3 |
| 超载/双火 | 3 |
| 召唤流 | 2 |
| 神宵/冻结融化 | 2 |
| 三色/值伤 | 2 |
| 控制/中速 | 2 |
| 爆发速攻 | 1 |
| 双水/中速 | 1 |
| 燃烧/激化 | 1 |
| 接力/蒸发 | 1 |
| 风扩散/三色 | 1 |
| 那维/水龙 | 1 |

## 3. 卡组总表 — 官方卡牌广场 (official corpus, 53 套, grouped by archetype)

图例：验证 = `✅ codec`（分享码解码为 3+30 合法牌组）/ `✅ codec+engine`（另跑通 1 局）。
核心/package 为「满编装备/天赋 + 主要支援」的紧凑摘要（非完整列表）。打法要点取自作者攻略原文（截断），
克制为该体系的通用应对（体系级建议，非逐张对位）。网络来源的补充卡组见 §4。


### 扩散/风（3 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `H0HBSt8YF3EAh50VCaDR9IEPGEEQA7YQCzFgBLoQC0GhijUYE6FR6IUeGIFQ3NQdDcAA` | ✅ codec+engine | 蚀灭的源焰之主/恰斯卡/吞星之鲸 | 扩散/风 | 千岩牢固×2; 逐影猎人×2; 晨曦酒庄×2; 鸣神大社×2; 湖中垂柳×2; 中央实验室遗址×2 | 大家好，今天版本更新了七圣召唤的新角色——蚀灭的源焰之主（后续简称为火龙） 火龙在拥有12血硬身板的同时，元素战绩也是非常标准的打3火伤模板，可以进行快速出伤，但这并不是我们这次… | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [miHoYo·推荐] |
| 2 | `AABw4AQOAuCA7mQPBhBA8ZcPCVBw9b0PDGCQ9skPDZCA/dkPDdGQAtsQDaHwC98QDrAA` | ✅ codec+engine | 申鹤/重云/枫原万叶 | 扩散/风 | 风物之诗咏×2; 赌徒的耳环×2; 立本×2; 寻宝仙灵×2; 桓那兰那×1; 红羽团扇×1 | 开局找过牌和蓄爆牌，第一回合如果有仙灵打出来然后开出重云的元素战技，得到两回合的冰附魔效果，切到申鹤，此时还剩三骰，骰子够好可以开出申鹤的元素战技，不好可以选择立本或者岩与契约蓄… | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [miHoYo·积分赛] |
| 3 | `AjGh9SUQBEGB9nUQCFFB95gQCmFx+b4QDJHR/coRDbKRDO4RD8LRD/IREOIREPMREPEB` | ✅ codec+engine | 菲谢尔/砂糖/深渊咏者·渊火 | 扩散/风 | 烬火重燃×2; 赌徒的耳环×2; 桓那兰那×2; 立本×2 | 这是一套自己瞎琢磨的卡组，强度不算高因此并不推荐大家使用，作为参考即可。先前在3.7版本的米游社“燃斗无双”预选赛中，我凭此卡组拿了五连胜，属于运气很不错了。因剪辑工具有问题，只… | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [miHoYo·积分赛] |

### 水债/召唤-水（6 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AxBQ0RoNA1Bw3qMPE1AB9TAPE2BB+DQPCoGACMIQDIEgCsQQDLFgC8YQDOGQD8kRDRAA` | ✅ codec+engine | 纯水精灵·洛蒂娅/菲谢尔/愚人众·火之债务处理人 | 水债/召唤-水 | 饰金之梦×2; 花海甘露之光×2; 海染砗磲×1; 沉沦之心×1; 派蒙×2; 卯师傅×2 | 写在开头：4.2的版本调整将纯水精灵以及愚人众火之债务处理人双双削弱，这导致纯水精灵从一个超模卡变成亏模卡，同时讨债人的血量上限削弱导致后续单挑能力变弱，使得在当前环境对水债并不… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |
| 2 | `AXCgyTcMA5BQ9HMPB0Aw9ZUPCVBQ9rMPC4Aw/bgPC9GQCrkQC6GwC7wQC7HAD8cQDPAA` | ✅ codec+engine | 菲谢尔/愚人众·火之债务处理人/纯水精灵·洛蒂娅 | 水债/召唤-水 | 悉数讨回×2; 幸运儿银冠×2; 璃月港口×2; 天守阁×2; 须弥城×2; 阿圆×2 | 水债皇在3.8版本环境中，可以说是独踞一方的环境定义者卡组，随着环境的演进，也出现了各种各样的环境定制构筑。这套牌便是我个人为比赛环境定制的一套水债皇构筑，取得了包括燃斗无双12… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |
| 3 | `ApDA2jUOA5Cw6YwOCKDA6p0PCTDQ9bMPC2Aw/bgPC/CA/7wQC6HAC70QDLFwDskQDPAA` | ✅ codec+engine | 钟离/纯水精灵·洛蒂娅/丘丘岩盔王 | 水债/召唤-水 | 贯虹之槊×2; 千岩牢固×2; 璃月港口×2; 西风大教堂×2; 须弥城×2; 立本×2 | 对付快攻卡组和一些克制卡组（如神宵）可以依赖纯水岩丘王进行一定程度的拖延和对攻，岩丘王配合纯水的召唤物或者深渊召唤前期可以打出非常强大的压制力。而对于后期的卡组就回归普通双岩的思… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |
| 4 | `A5BQ9CEPA0Bw9XEPB1Aw9pUPCWBQ+LMPC9EwCrgQC6GAC7sQC7HADrwQDOFwD8cQDPAA` | ✅ codec+engine | 纯水精灵·洛蒂娅/八重神子/愚人众·火之债务处理人 | 水债/召唤-水 | 幸运儿银冠×2; 百川奔流×1; 悉数讨回×1; 璃月港口×2; 西风大教堂×2; 须弥城×2 | 大家好，我是瓜皮儿，3.8版本燃斗无双16强和火星杯冠军，在这里对我比赛常用的一套卡组做一下简单的介绍。 起手留牌：阿圆、璃月港、须弥城，立本，其他视情况而定。 如面对快攻卡组，… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |
| 5 | `AyBw8jUPAUCg9XMPCVBQ9pUPCYBw+b0PDNAg/cgQDKGACskQDLHQC9kQD+EQDvEQD/AA` | ✅ codec+engine | 愚人众·火之债务处理人/纯水精灵·洛蒂娅/菲谢尔 | 水债/召唤-水 | 幸运儿银冠×2; 悉数讨回×1; 赌徒的耳环×1; 提米×2; 桓那兰那×1; 派蒙×1 | 这套水债皇是我在第二届米游社燃斗无双比赛中使用过的卡组，水债皇在之前官方统计中是胜率很高的卡组，可见卡组的稳定性和强势。但在这两天体验下来，新版本水皇体系并不吃香，所以请谨慎使用… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |
| 6 | `AyBw9TUPAVCg9nMPCWBQ+LMPC4Aw/cIPDNEgAsgQDKGACskQDLGQC/EQD+EQDvIQD/AA` | ✅ codec+engine | 愚人众·火之债务处理人/纯水精灵·洛蒂娅/菲谢尔 | 水债/召唤-水 | 悉数讨回×1; 幸运儿银冠×1; 璃月港口×2; 派蒙×2; 提米×2; 立本×2 | 这是我平时很喜欢玩的一套由纯水精灵，菲谢尔，愚人众·火之债务处理人组成的卡组，江湖人称"水债皇"，构筑上携带了幸运儿银冠和债务处理人的天赋卡，以保证债务人的后期收割能力，思路也相… | 召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。 | [miHoYo·积分赛] |

### 激化/召唤（4 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AsAQ7DIPAxAQ8ZsPDEAg9cIPDFBg9sYPDNGQCskQDbEQC9EQDdEgD94QDvGwEesRDhAA` | ✅ codec+engine | 八重神子/瑶瑶/纳西妲 | 激化/召唤 | 绝缘之旗印×1; 派蒙×2; 卯师傅×2; 立本×2; 迪娜泽黛×2 | 起手留牌：如果有其它伙伴就留缇娜泽黛，没有其它伙伴可以全换掉找大派以及立本。 主要思路：t1瑶瑶e，t2神子两段e+草神e或者继续蓄爆，t3根据局势选择开瑶瑶eq或者蓄蕴种印，在… | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [miHoYo·积分赛] |
| 2 | `ARBg8hoPAyAQ85cPCUBw9L0PDFCA9cgPDWAg+NIPDZGQCtkQDaGwC+0QDrHQDvERDzAA` | ✅ codec+engine | 可莉/菲谢尔/纳西妲 | 激化/召唤 | 赌徒的耳环×2; 提米×2; 拉娜×2; 寻宝仙灵×2; 桓那兰那×1 | 这是我这个版本海选赛一直在用的卡组，也是成功给我带来了5次海选4-1出线的好成绩。作为一套优秀的三色套牌，我觉得很适合分享出来。 先讲一下思路，大体有两张思路，菲谢尔奥兹开切草神… | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [miHoYo·积分赛] |
| 3 | `AgAQ8S8PAxAA9IEPCUBw9ZcPDFCA9sgPDICQ/ckPDdGQAusQDqGwCuwQDrHAC/AQD+AA` | ✅ codec+engine | 八重神子/柯莱/提纳里 | 激化/召唤 | 赌徒的耳环×2; 阿莫斯之弓×1; 提米×2; 立本×2; 寻宝仙灵×1 | 打法思路 开局找仙灵或者心海，最好的伙伴，提米。 第一回合：首发八重神子，有过牌先过牌，然后E技能。之后切换提纳里，然后开E技能。 第二回合：提纳里重击，然后根据手牌和对手实际情… | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [miHoYo·积分赛] |
| 4 | `AiAQ8jAPAkDw9JcPCVBw9cgPDYCQ/NkPDsCw/esPDtHACuwQD6EAC/AQD7EQDPEQD8AA` | ✅ codec+engine | 八重神子/提纳里/柯莱 | 激化/召唤 | 赌徒的耳环×2; 寻宝仙灵×2; 提米×1 | 打法思路 开局找仙灵,最好的伙伴,快快缝补术 第一回合有仙灵先用仙灵，然后八重e,切，提那里e,条件允许的情况下，并且能打出激化，就使用快快缝补术这是最理想的状态，一般情况下只能… | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [miHoYo·积分赛] |

### 莫皇草/召唤穿透（6 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AhAQ8gsPAyAQ9JcPCUBw9cIPDFAg9skPDGCQ+NEPDYEQCtIQDaEhCzcQE7FwDt0QD+AA` | ✅ codec+engine | 八重神子/莫娜/纳西妲 | 莫皇草/召唤穿透 | 赌徒的耳环×2; 派蒙×2; 立本×2; 迪娜泽黛×2; 拉娜×2 | 公式就是开局神子E+蓄爆，然后莫娜E草神大E（或者看手牌和骰子情况双小E，小E+蓄），T3 草神尽量再动一下，然后配合风自赌徒本大爷仙跳墙等关键牌斩杀，神子比较怕被胡桃而打不出二… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |
| 2 | `ARAA4gsOAyAQ8EwPBBDA9MIPDEAg9cYPDGBg/MkPDMGQC9EQDbEQDtIQDeHgEeERDjAA` | ✅ codec+engine | 妮露/莫娜/纳西妲 | 莫皇草/召唤穿透 | 星天的花雨×2; 派蒙×2; 卯师傅×2; 立本×2; 迪娜泽黛×2 | 起手无脑留：妮露天赋，灶火，迪娜泽黛。其余牌看情况，对中慢速卡组找大派蒙，立本，拉娜等过牌和蓄爆。面对雷狼火拳类卡组找交给我吧，缝补，烤鸡等爆发或抢动数牌。 有蓄爆时一般选择t1… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |
| 3 | `ASCg8wsPA0AQ9JcPCVBw9acPCmBw+L0PDJCA+cgPDdEgCtIQDqHQC/EQD7EQDvIRDzAA` | ✅ codec+engine | 菲谢尔/莫娜/纳西妲 | 莫皇草/召唤穿透 | 赌徒的耳环×2; 酒渍船帽×2; 提米×2; 拉娜×2; 桓那兰那×1 | 构筑方面，酒渍船帽可换为无常之面或须弥城，送你一程，白垩之术可换一掷乾坤，磐岩盟契等 打法方面，主要有感电轴和激化轴 感电轴：t1菲谢尔e 莫娜e，t2莫娜e 纳西妲5e，也就是… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |
| 4 | `AbBg7TEOANCw8VIPBRAg8noPB0Cg9HwPB1DA9X0PB2HQCpcQCbFwC9MQDeEwDtkRDTAA` | ✅ codec+engine | 可莉/纳西妲/莫娜 | 莫皇草/召唤穿透 | 砰砰礼物×2; 祭礼残章×2; 千夜浮梦×2; 盈满之实×2; 赌徒的耳环×2; 老章×2 | 酱酱！可莉登场！ 由可莉莫娜小草神组成的三法器速攻队！这套卡组的思路与构筑来自我的好朋友红颜~。是一套十分强力的卡组！ 秘传让可莉吃到了极大的版本红利，独特的重击机制可以让可莉迅… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |
| 5 | `ARCg8gsPAyAQ9JcPCUBw9b0PC1DQ9sgPDICA+dIPDZEgCtkQDaGQC9sQDuHQDvERDzAA` | ✅ codec+engine | 菲谢尔/莫娜/纳西妲 | 莫皇草/召唤穿透 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 拉娜×2; 寻宝仙灵×2 | 这套卡组打了两个海选全部5-0，晋级赛第一场也4-1，强度有保障。 起手留牌： 通用：提米，最好的伙伴 快攻内战：本大爷，白垩，赌徒 填费用：拉娜，仙灵，心海，桓纳兰那（如有多个… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |
| 6 | `ACCw8hoPA0AQ9ZcPCVBw9r0PC2DQ+MgPDJCA+ckPDdAg/dkQDaGQC+0QD7EQDvEQD+AA` | ✅ codec+engine | 莫娜/菲谢尔/纳西妲 | 莫皇草/召唤穿透 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 寻宝仙灵×2; 立本×1 | 第二届米游社燃斗无双使用过的牌组。 比利君直播间偷师的牌组，帮我拿了个海选赛五胜，一半靠运气一半靠牌组强度。 封面那四场莫皇草是海选小组赛的后四场比赛。 构筑方面:上面的牌组是已… | 召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。 | [miHoYo·积分赛] |

### 超导/雷鹤（5 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AQFB8iEQARFx9ZgQClFx9r4QDGHR98oQDYGR/s4QDdLRA9oRDqKxC+ARDrLxDOERD+EB` | ✅ codec+engine | 重云/雷电将军/申鹤 | 超导/雷鹤 | 赌徒的耳环×2; 桓那兰那×2; 立本×2; 刘苏×2; 寻宝仙灵×1 | 该卡组为璃月雷鹤（又称冰影）的一种构筑，低蓄爆，高增伤。 该卡组主要来说有三种开局方式：重云开拔刀轴，拔刀轴，手动轴 1、重云拔刀轴： 该轴是面对双岩等慢速卡组时最适合使用的流程… | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [miHoYo·积分赛] |
| 2 | `AABg7RoOANBw8ZcPCRBw870PCzDQ9MgPDECQ9ckPDFDQ9s0PDdGwAt8QDbHwC+ARDjAA` | ✅ codec+engine | 优菈/菲谢尔/申鹤 | 超导/雷鹤 | 赌徒的耳环×2; 桓那兰那×2; 立本×2; 刘苏×2; 提米×1 | 起手有流苏留流苏，没有流苏可以找运筹帷幄过牌，或是桓那兰那/立本+秘传续爆 开局最理想的展开一般是皇女e+流苏+申鹤e,这样可以保证t2申鹤直接开出元素爆发 若没有流苏可以视骰子… | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [miHoYo·积分赛] |
| 3 | `AeFB8iEQARFx85gQCkFx9b4QDFHR9skQDWGB+coQDdKRA9oRDqLxC+ARD7IBDOERD+EB` | ✅ codec+engine | 重云/雷电将军/申鹤 | 超导/雷鹤 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 立本×2; 寻宝仙灵×1 | 双冰雷神卡组基本是在T3回合靠重云和[雷神申鹤]召唤物收尾打出极高的爆发获得比赛胜利。 开局留立本和桓那兰那这样的蓄爆牌，仙跳墙也可以留一手 在第一回合根据骰子的情况和敌方牌组可… | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [miHoYo·积分赛] |
| 4 | `AABA4CAOAOBw8ZcPCRBw9L0PC0DQ9cgPDFCA9skPDICQ/c0QDKHQC9sQDbHwDt8QDuAA` | ✅ codec+engine | 重云/雷电将军/申鹤 | 超导/雷鹤 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 立本×2; 刘苏×2 | 开局留牌：提米、流苏（拔刀轴不留）、立本、小派（面对极速快攻可留药、面对火刻可留本大爷） 后期关键牌：莲花酥、土豆饼、本大爷、赌徒、双冰共鸣、仙跳墙、星天、送你一程、速切 从开局… | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [miHoYo·积分赛] |
| 5 | `AQFB8iEQARFx9ZgQCkFx9r4QDFHR98kQDYGB/soQDdKRA84RDqKRC+ARDrLxDOERD+EB` | ✅ codec+engine | 重云/雷电将军/申鹤 | 超导/雷鹤 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 立本×2; 刘苏×1 | 构筑方面，是否应该携带刘苏仍有争议，但岩与契约已经不再值得携带。两张仙跳墙是更优秀的选择，送你一程作为针对卡，可以换成刘苏/土豆饼/交给我吧等值得携带的单卡。打法方面，常见的公式… | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [miHoYo·积分赛] |

### 双岩/结晶（3 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `ARCQwSUTEWCgx2EMEnAwyZAMCZAQ3p0OCXDR5zQOE4BA6LMPCxEwC7wQC7HAD70QDPAA` | ✅ codec+engine | 迪希雅/琴/琳妮特 | 双岩/结晶 | 千岩牢固×2; 花海甘露之光×2; 蒲公英的国土×1; 如影流露的冷刃×1; 祭礼剑×1; 璃月港口×2 | 在4.3共鸣海选中，我使用大古老师的赤风卡组两次溃败于下头水凝之手后，我选择科研了下这套更慢速的场地流赤风。 构筑：首先因为携带了琳妮特天赋，所以携带风龙废墟就有较高的收益，与此… | 盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。 | [miHoYo·积分赛] |
| 2 | `AqCw6QkOApDg6ogOCKCA7Z0OCdDQ9LMPC0Aw9rgPC5CA/bwQCyHAC8cQDLFwD8kQDfAA` | ✅ codec+engine | 诺艾尔/芭芭拉/荒泷一斗 | 双岩/结晶 | 钟剑×2; 千岩牢固×2; 璃月港口×2; 西风大教堂×2; 须弥城×2; 阿圆×2 | 这套双岩一斗算是当前版本中相对主流和强势的一套牌了，首先一斗一般作为主C，超模的数值配合上头盔重剑让一斗宛如战神下凡，配合风与自由和重攻击更是能做到三连斩，终结比赛。其次芭芭拉的… | 盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。 | [miHoYo·积分赛] |
| 3 | `AHCQxy4MApCw24gOCJCA6Z0OCaDQ6rMPC0Aw9LgPC9CA/rkQC7GQC7sQC+HADrwQDPAA` | ✅ codec+engine | 芭芭拉/荒泷一斗/诺艾尔 | 双岩/结晶 | 钟剑×2; 千岩牢固×2; 璃月港口×2; 西风大教堂×2; 天守阁×2; 须弥城×2 | 构筑：胡桃，魈，珊瑚宫灵活卡位，可以换成风与自由/速切/食物 起手优先找阿园，璃月港，心海，立本，千岩头盔。天守阁也可以留，这套主要靠资源碾压，蓄爆续航牌非常关键。 卡组思路：打… | 盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。 | [miHoYo·积分赛] |

### 火刻/超载（4 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AeAw4xYOAzCQ5E8OBEDw9JcPCUBw9bMPC1Aw9rwPC2HAC8EQDLEQDMcQDOFwDskRDRAA` | ✅ codec+engine | 班尼特/可莉/无相之雷 | 火刻/超载 | 冒险憧憬×2; 赌徒的耳环×2; 璃月港口×2; 须弥城×2; 风龙废墟×2; 阿圆×2 | 4.2版本的环境水债一家独大，火拳是少数能够与水债五五开甚至小优的卡组。火拳分为债拳，宵宫拳以及可莉拳，各有优劣。这里推荐一套可莉拳。可莉拳的优势是能够利用可莉多动为对方的角色挂… | 双火超载 otk：怕高血与治疗，注意站位防超载换人；用冻结/眩晕断连。 | [miHoYo·积分赛] |
| 2 | `AZAwyRYNAdDA408OCTAA5JAOCUBw9JcPC1Aw9rMPC2DA/bwQDCEQC8EQDLFwEccRDDAA` | ✅ codec+engine | 班尼特/可莉/刻晴 | 火刻/超载 | 祭礼剑×2; 赌徒的耳环×2; 冒险憧憬×1; 璃月港口×2; 须弥城×2; 风龙废墟×2 | 128强比赛用的传统阿园火刻。 可惜对上了天胡莫优散，在本就劣势对局的情况下仅仅只差一个班天赋就能完成绝地翻盘。 个人觉得非常可惜，于是发此视频纪念一下。 | 双火超载 otk：怕高血与治疗，注意站位防超载换人；用冻结/眩晕断连。 | [miHoYo·积分赛] |
| 3 | `ATAw5BYOAUDA7U8OBNDw8ZAPCRBw9JcPC1DA9bwPDGCQ9skPDdGQC9kQDbGwEeMRDjAA` | ✅ codec+engine | 班尼特/可莉/刻晴 | 火刻/超载 | 冒险憧憬×2; 赌徒的耳环×2; 祭礼剑×1; 须弥城×2; 立本×2; 寻宝仙灵×2 | 喵神的须弥城构筑的火刻，加上了一手提瓦特煎蛋。这个煎蛋曾在比赛中发挥奇效，先后靠煎蛋赢了雨中邮差的璃月雷鹤、雪国的水债皇，雪国还为此发了视频（https://www.bilibi… | 双火超载 otk：怕高血与治疗，注意站位防超载换人；用冻结/眩晕断连。 | [miHoYo·积分赛] |
| 4 | `ATAw5BYOAUDA7U8OBNDw8ZAPCRAA9JcPCUBw9cIPDFAg9skPDGGQC9IQDeEgDuMRDjAA` | ✅ codec+engine | 班尼特/可莉/刻晴 | 火刻/超载 | 冒险憧憬×2; 祭礼剑×2; 赌徒的耳环×2; 派蒙×2; 立本×2; 拉娜×2 | 开局保留璃月港口、大派蒙、拉娜（和祭礼剑） 班尼特开局：打出祭礼剑和拉娜，进行两次元素战技，而后下大派蒙或璃月港结束回合。之后再利用大招冒险憧憬（有天赋最好，没有可以先切人等待天… | 双火超载 otk：怕高血与治疗，注意站位防超载换人；用冻结/眩晕断连。 | [miHoYo·积分赛] |

### 冻结/控制（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `APCg3wQOAABg4EAPBEAA9JcPCWBw9sIQDCEgA8YQDDFgDMkQDMGQDdEQDdEQEN4RDQAA` | ✅ codec+engine | 行秋/重云/优菈 | 冻结/控制 | 吐纳真定×2; 赌徒的耳环×2; 派蒙×2; 卯师傅×2; 立本×2; 迪娜泽黛×2 | 下面介绍的是我在4.2版本比赛中使用的一套克制otk和火拳的冻结卡组。 开局留牌：迪娜泽黛、大派、立本、重云天赋、（如果对手是快攻，有卯师傅有立本大派可以留烟熏鸡） 卡组思路： … | 冻结控场：带解冻/速切与单核高血，避免主C被冻；抢先击杀冰核。 | [miHoYo·积分赛] |
| 2 | `AABA7RYOANBg8UAPCRBw85cPCzDQ9L0PDEAw9cMPDFDw9tkPDWGQCd8QDbHwDuARDjAA` | ✅ codec+engine | 重云/可莉/优菈 | 冻结/控制 | 赌徒的耳环×2; 吐纳真定×1; 桓那兰那×2; 凯瑟琳×2; 寻宝仙灵×2; 鲸井小弟×1 | 牌组思路主打一个点杀，优菈冷酷之心加上融化刚好-7，正好在可莉e/q的斩杀线，凯瑟琳于交给我吧提高快切上手率主打威慑，让对手主c不敢站在前台输出。绝云锅巴可以在第一回合刚好斩杀对… | 冻结控场：带解冻/速切与单核高血，避免主C被冻；抢先击杀冰核。 | [miHoYo·积分赛] |

### 快攻（3 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `ATCg9DEPAVBg9ZcPCWBw9s0PDJDQ/dIPDdEgAtsQDqHQCvEQD7EQDvIQD+EgE/MRDzAA` | ✅ codec+engine | 菲谢尔/纳西妲/可莉 | 快攻 | 赌徒的耳环×2; 刘苏×2; 拉娜×2 | 可爱草是一套出伤迅速，综合能力很强的三色反应队。卡组由菲谢尔（小艾咪），可莉和纳西妲三张角色卡组成。 由于其具有很快的出伤速度，我倾向于发挥这一优势，放弃蓄爆，高效输出。构筑方面… | 快攻：加血+护盾+清场，拖入中后期。 | [miHoYo·积分赛] |
| 2 | `ACCw8xwPATDQ9I8PCEDw9ZcPCVBw9r0PC2HQCuUQDqFQC+YQDrFgDvEQD+EQE/IRDzAA` | ✅ codec+engine | 莫娜/刻晴/赛诺 | 快攻 | 旅行剑×2; 赌徒的耳环×2; 桓那兰那×2 | 前言：我玩七圣召唤最初的三套卡组之一，另外两套因为行秋削了加上玩的少给删了。3.3版本拿它打的神宵，当时最优成绩3/7开，我3。算是冷门卡组，一千六七百把见过两次。玩这种eee的… | 快攻：加血+护盾+清场，拖入中后期。 | [miHoYo·积分赛] |
| 3 | `AkAA9AsPAlBw9ZcPCWBw9r0PDNGAAsgQDCGQA8kQDDHwCtsQD6EQC/EQD8EgDPIQD+AA` | ✅ codec+engine | 雷电将军/莫娜/魈 | 快攻 | 赌徒的耳环×2; 提米×2; 立本×2; 桓那兰那×1; 鲸井小弟×1 | 此卡组为高爆发速攻PVP卡组，旨在三回合内打出超高爆发结束战斗，能与水草皇、双冰雷神一战！ 打法思路：第一回合 雷神AE+续爆；第二回合 雷神开大，之后切莫娜E或者切莫娜后续爆结… | 快攻：加血+护盾+清场，拖入中后期。 | [miHoYo·积分赛] |

### 超导/值伤（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AlEQNwkTAHEgOj4PA1Dg9kUPE2EBCDAQE4FBCjQQC6HQC8YQDOFgDskQDfEQD9ERDRAA` | ✅ codec+engine | 八重神子/芭芭拉/迪奥娜 | 超导/值伤 | 猫爪冰摇×2; 饰金之梦×2; 花海甘露之光×2; 光辉的季节×1; 卯师傅×2; 迪娜泽黛×2 | 开局找花海甘露或者迪娜泽黛和卯师傅，T1八重招杀生樱然后可以切迪奥娜或者芭芭拉元素战技，或者切人铺牌，前期来花海甘露可以快速刷牌。 T2迪奥娜元素战技，(T1芭芭拉站场T2也切迪… | 超导值伤：怕加血/免疫，抢先或用冻结/眩晕打断核心行动。 | [miHoYo·推荐] |
| 2 | `AVDg5QQOAWCw85cPCUBw9MgPDFCA9ckPDGGQAsoQDDGgA9YQDcFgDNkQDeGQENsRDgAA` | ✅ codec+engine | 北斗/重云/雷泽 | 超导/值伤 | 赌徒的耳环×2; 提米×2; 立本×2; 常九爷×2; 参量质变仪×2 | 遇到快攻争取t2让北斗开大，遇到水债等中慢速，可以t2续爆，t3一波带走，适时可以勾引对面强杀满能北斗，借助本大爷和强雷启动雷泽 | 超导值伤：怕加血/免疫，抢先或用冻结/眩晕打断核心行动。 | [miHoYo·积分赛] |

### 水皇草/召唤-水（3 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AzAQ9DUPAUCg9ZUPCVBw9r0PC2DQ+MkPDJCQ+dIPDdEgCvEQD6EQC/IQD7EgDvMQD+AA` | ✅ codec+engine | 纳西妲/纯水精灵·洛蒂娅/菲谢尔 | 水皇草/召唤-水 | 幸运儿银冠×1; 赌徒的耳环×1; 桓那兰那×2; 立本×2; 拉娜×2 | 水皇草，一套中速偏快的卡组，主要进攻思路是草神打出的蕴种印配合回合结束的召唤物反应来打出伤害。 有3种开局一种是草开，一般是草神开5e，给对面全体挂上蕴种印，接蓄爆，然后纯水5e… | 召唤+激化混伤：清召唤物、穿透打后排；避免被拖入长线资源战。 | [miHoYo·积分赛] |
| 2 | `AyBQ8jEPAVCg9ZcPCWBw9r0PC5DQ+cIPDNCA/cgQDKGQCskQDbGQC9kQD+EQDvEQD/AA` | ✅ codec+engine | 纯水精灵·洛蒂娅/纳西妲/菲谢尔 | 水皇草/召唤-水 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 立本×2; 寻宝仙灵×2 | 燃斗无双竞赛主人的任务，晋级赛当时我录了几局像好多都删了，这局有点戏剧性还留着。打比赛脑子发昏渣操特别多，但是双方都有失误，这才是有看点的比赛嘛。 3.8把老卡组都删了，凭印象还… | 召唤+激化混伤：清召唤物、穿透打后排；避免被拖入长线资源战。 | [miHoYo·积分赛] |
| 3 | `AzAQ9DUPAUCg9ZcPCVBw9r0PC2DQ+MIPDJAg+cgPDNCA/ckQDKGQCvEQD7EQC/MQD+AA` | ✅ codec+engine | 纳西妲/纯水精灵·洛蒂娅/菲谢尔 | 水皇草/召唤-水 | 赌徒的耳环×2; 桓那兰那×2; 派蒙×2; 提米×2; 立本×2 | 开局我方考虑，内战2方均无T2内伤害压制能力，走草神轴出伤更快，同时t1一动后手可以反先切人不被草神打到反应。T3赌徒白垩皇女大后操作空间更大，选择草神开。卡密老师选择纯水开。 … | 召唤+激化混伤：清召唤物、穿透打后排；避免被拖入长线资源战。 | [miHoYo·积分赛] |

### 感电/超载（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AdDA8QsPAhBg9ZcPCVBw9r0PC5DQ+cgPDNCA/ckQDKGQCtIQDbEgC9kQDeGQDu0RDjAA` | ✅ codec+engine | 刻晴/莫娜/温迪 | 感电/超载 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 立本×2; 拉娜×2 | 卡组单卡选择上：如果打双岩卡组可以加入祭礼剑；打水皇卡组可以选择带魈（护法之誓）；打快攻卡组则可以考虑去掉拉娜，带鲸井小弟；也可以带温妮莎传奇+凝光的组合。食物不建议少带，毕竟这… | 感电连锁：单体高血/治疗可稀释连锁伤害；拆掉触发器。 | [miHoYo·积分赛] |
| 2 | `AWCg5g8OAdCw7ZcPCRBw8b4PCyDg9cgPDFCQ9skPDGHQAs0QDaFgC9sQDrFQDuURDgAA` | ✅ codec+engine | 菲谢尔/坎蒂丝/雷泽 | 感电/超载 | 赌徒的耳环×2; 镇守之森×2; 立本×2; 刘苏×2; 提米×1 | 核心牌:流苏 强能之雷（双雷共鸣）赌徒 立本 出其不意制胜牌:风与自由 本大爷还没有输！重攻击 对战思路:#我方先手情况 坎蒂丝“圣仪·仓鹫庇卫”e起手抵挡伤害并充能为敌方挂水，… | 感电连锁：单体高血/治疗可稀释连锁伤害；拆掉触发器。 | [miHoYo·积分赛] |

### 超载/双火（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AVDQ5hUOEWDg9JcPCUBx9TEPE2Ax9jMQEzFRAzUTDLGQCskQDbGQDNkQDcHQDuURDjAA` | ✅ codec+engine | 赛诺/宵宫/雷音权现 | 超载/双火 | 赌徒的耳环×2; 灵光明烁之心×2; 浮溯之珏×1; 湖中垂柳×2; 立本×2; 寻宝仙灵×2 | 哈喽，大家好，这里是你们的千辰(也是亓)，今天给大家带来一套雷宵的卡组构筑，那么老规矩，先干正事，给大家说说这套怎么玩。 1.打法:概括:利用雷共鸣烟熏鸡去尽快打满宵宫充能，后续… | 双火超载：加血/减伤，注意站位与速切。 | [miHoYo·推荐] |
| 2 | `FUDB3YsOAVDQ5VkOGWAh5pIPCUBw9JcPE2Ax9zMTFbFxO5MVC7HQlskZDGGRC2cQGbAA` | ✅ codec+engine | 久岐忍/夏沃蕾/赛诺 | 超载/双火 | 勘探钻机×2; 赌徒的耳环×2; 灵光明烁之心×2; 落羽的裁择×1; 黄金剧团的奖赏×1; 立本×2 | 哈喽大家好，我是蓝雨，4.8更新了一张名字叫勘探钻机枪，集增伤，抗伤，过牌于一体，相信大家都能看得出来这把武器的强度，围绕着勘探钻机这把武器，以及新出的角色牌夏沃蕾，以此研发出了… | 双火超载：加血/减伤，注意站位与速切。 | [miHoYo·推荐] |

### 召唤流（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `A4Cg8ScPAxCA8pcPCTBw87MPC1Aw9bQPC2BA+b0PDPCQ/8kQDRGwCucQDqFwC+gQDrAA` | ✅ codec+engine | 魔偶剑鬼/魈/深渊咏者·渊火 | 召唤流 | 赌徒的耳环×2; 璃月港口×2; 骑士团图书馆×2; 立本×2; 桓那兰那×1 | 开局剑鬼三召唤物，来水雷火都赚，但脸黑抽到冰就只能说乐不思蜀天过了，如果想再赌一点可以带星天之照，但现在版本需要躲避伤害剑鬼t2容易被秒我就带了白垩，下落斩，对魈也很有用，渊火可… | 召唤流：AOE 清召唤物、抢先杀召唤核心。 | [miHoYo·积分赛] |
| 2 | `A5AQ7RQOAdCg8YEPCBAQ8pcPCTBw9bkPC1CQ9r0PC2DQ/cgQDKGACskQDLGQC9kQDeAA` | ✅ codec+engine | 纳西妲/安柏/菲谢尔 | 召唤流 | 阿莫斯之弓×2; 赌徒的耳环×2; 天守阁×2; 桓那兰那×2; 提米×2; 立本×2 | 本卡组由3.7版本伊始有过一些登场率的安柏、甘雨、菲谢尔卡组演化而来。 卡组的核心是安柏，次核心是菲谢尔。将原本的甘雨替换成纳西妲是由于当前环境中双水、双岩队较多切较为强力，甘雨… | 召唤流：AOE 清召唤物、抢先杀召唤核心。 | [miHoYo·积分赛] |

### 神宵/冻结融化（2 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `APBQ4BUOAABw75cOCfBw8r0PDCCA9cgPDFCg9soQDTFgA9YQDbGQC9kQDcGwDN8RDQAA` | ✅ codec+engine | 神里绫华/宵宫/申鹤 | 神宵/冻结融化 | 赌徒的耳环×2; 提米×2; 常九爷×2; 参量质变仪×2; 寻宝仙灵×2 | 经过3.8宵宫q的增强、雷与永恒和磐岩这两张超模单卡的加入，神宵卡组一下子提高了很多强度，成功在3.8七圣召唤的环境中打下了一片天。 *当前环境下的神宵由于宵宫3充的尴尬还没有得… | 冻结+融化爆发：带解冻与高血，抢先杀输出。 | [miHoYo·积分赛] |
| 2 | `AABQ4BUOAPBw75cPCSBw9L0PC0DQ9cgPDFCA9soQDDGgA9kQDbGQC9sQDcHwDt8RDgAA` | ✅ codec+engine | 神里绫华/宵宫/申鹤 | 神宵/冻结融化 | 赌徒的耳环×2; 桓那兰那×2; 提米×2; 常九爷×2; 寻宝仙灵×2 | 和实力在自己之上的大佬去打水债皇或者双岩内战，可能只有两三成胜率。但如果选择神宵，就可以做到只看自己的牌运决定胜负，胡了就赢，不胡就寄。 首先解说一下视频，在燃斗无双晋级赛最后一… | 冻结+融化爆发：带解冻与高血，抢先杀输出。 | [miHoYo·积分赛] |

### 三色/值伤（1 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AABQ7wcOAfBQ9JcPCUBw9b0PDFAg9skPDGCQ980QDSFgA9YQDTGwC98QDbHwEOARDgAA` | ✅ codec+engine | 神里绫华/申鹤/宵宫 | 三色/值伤 | 赌徒的耳环×2; 立本×2; 参量质变仪×2; 桓那兰那×1; 派蒙×1 | 【前言】：本人菜鸡，构筑和思路并非完全合理，旨在与各位交流学习，欢迎驻足指点。 【角色】：神里绫华+宵宫+任意（其他冰火角色均可，个人喜欢全女阵容所以选择申鹤。申鹤可以在宵宫暴毙… | 三色值伤快攻：抢先/加血；拆属性共鸣与饰金之梦。 | [miHoYo·积分赛] |

### 爆发速攻（1 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AnDQ3iAOA5Cw6VwPCUBw9JcPCVCw9r0PDIAg+MIPDJCQ/ckQDKHQCs0QDbEQC9EQE8AA` | ✅ codec+engine | 阿贝多/雷电将军/丘丘岩盔王 | 爆发速攻 | 赌徒的耳环×2; 万千的愿望×1; 绝缘之旗印×1; 派蒙×2; 立本×2; 刘苏×2 | 推荐起手留：遇上雷狼，神宵，夜兰等otk卡组推荐开局留流苏 如果没有就找其他伙伴牌和迪娜泽戴 第一回合雷神e接流苏阿贝多e 没有流苏就把剩下的费用抽牌或者蓄爆用光 如果骰子情况不… | 爆发速攻：加血+护盾+清场，拖入中后期。 | [miHoYo·积分赛] |

### 双水/中速（1 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `ABCQ4SoOACDQ4mYPB0Cg9XoPCWDQ+J0PC5Aw/bkPC9GQCrwQC7HQC8kQDOGQD9oQDvAA` | ✅ codec+engine | 芭芭拉/凝光/珊瑚宫心海 | 双水/中速 | 祭礼残章×2; 千岩牢固×2; 储之千日，用之一刻×1; 天守阁×2; 立本×2; 璃月港口×1 | 大家好，这里是米游社燃斗无双128强选手赤瞳，我给大家带来了我参加比赛时用的最多的一套牌，同时也是我胜率最高的一套牌组，这个版本更新了秘传牌后我也是用秘传牌踢掉了一张运筹帷幄，很… | 双水续航：抢先压制、拆支援区；避免被拖入长线资源战。 | [miHoYo·积分赛] |

### 燃烧/激化（1 套）

| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `A9AQ0RINAhDw604OBLDg7JcOCcBw8LMPCwAw8bwPCxDA870PDDAg9MIPDEBw9c0PDFAA` | ✅ codec+engine | 纳西妲/香菱/柯莱 | 燃烧/激化 | 交叉火力×2; 赌徒的耳环×2; 璃月港口×2; 须弥城×2; 派蒙×2; 刘苏×2 | 主打燃烧反应的牌组 利用高频的燃烧反应多次触发纳西妲e对群，召唤物回合结束对单打点收割。 正常打法，如果开局有刘苏就切香菱，e或天赋e，第二回合开大再纳西妲战场，第三回合尽量滤抽… | 燃烧激化：清召唤物、注意燃烧持续伤害；抢先杀香菱/柯莱。 | [miHoYo·积分赛] |

## 4. 当前版本补充 (web-sourced, 20 套)

这些分享码来自公开攻略/视频，已用引擎解码验证。来源可信度见末列；`52pk` 为 SEO 聚合站，
其攻略文字无法交叉验证，仅分享码本身通过解码。


| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |
|---|---|---|---|---|---|---|---|---|
| 1 | `AtDw6xoOEbDA7IMOCcBw8JcPCgBB8TUPExBQ9cYPE2FxAjcQDZGQCdkQE7GBCzgRDTAA` | ✅ codec+engine | 柯莱/菲谢尔/艾尔海森 | 激化/召唤 | 赌徒的耳环×2; 王下近侍×1; 沙王的投影×1; 湖中垂柳×2; 玛梅赫×2; 寻宝仙灵×2 | 经典双草激化体系：艾尔海森二段挂草吃满激化增伤，柯莱收尾穿透王下近侍；融合赌徒/垂柳/化种/仙灵 1 费大过牌。绝云锅巴用于艾尔海森战技接奥兹后下回合普攻带走 10 血角色。 | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [gamersky] |
| 2 | `AoHi3B0PAlHR5pgPCmFy8TYQFAFR9scQDWJhBMsRDTKhDdIRDsIRDtoRDtKSETkSFAEB` | ✅ codec+engine | 北斗/艾尔海森/赛诺 | 激化/召唤 | 赌徒的耳环×2; 湖中垂柳×2; 卯师傅×2; 常九爷×2; 迪娜泽黛×2 | 艾尔海森强普二段挂草瞬间吃满激化增伤，与北斗配合伤害很高，PVE 体验好；PVP 下艾尔海森生存差，作者建议改玩雷狼。 | 激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。 | [gamersky] |
| 3 | `AhCw0xkNEeCx6SQOEpBA6oMOCKAw9IgPCGCA/ZcQCSFwC8IQDLEgDsYQDOFgD9EQDfAA` | ✅ codec+engine | 诺艾尔/迪希雅/五郎 | 双岩/结晶 | 犬奔·疾如风×2; 王下近侍×2; 钟剑×2; 赌徒的耳环×2; 派蒙×2; 卯师傅×2 | 五郎增益多，王下近侍多动伤害高，后续接诺艾尔元素爆发，名为双岩；用老章武器流+赌徒走进攻端。 | 盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。 | [gamersky] |
| 4 | `AjCg3iAOEZCx6SQOEqBB7ysOCPAw9IMPCVBw9pcQDKEgC8IQDLFgDsYQDeEQD9ERDRAA` | ✅ codec+engine | 凝光/雷电将军/五郎 | 双岩/结晶 | 犬奔·疾如风×2; 王下近侍×2; 赌徒的耳环×2; 四风原典×1; 派蒙×2; 卯师傅×2 | 五郎/凝光替代申鹤/重云，王下近侍替代贯月矢，新时代雷鹤；作者自评构筑待调。 | 盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。 | [gamersky] |
| 5 | `AJGhORcNAOCB3yANBPBg4JEPCUAQ9J0PCZDQ/bwQCyHAC8YQDLFgDtEQE+FxDzcRExAA` | ✅ codec+engine | 行秋/莱依拉/七七 | 控制/中速 | 风鹰剑×2; 千岩牢固×2; 归芒携信×1; 重帘留香×1; 须弥城×2; 卯师傅×2 | 给莱伊拉找的队伍，目标是养神装七七/行秋；作者自评生存较差。 | 控制中速：用 OTK 在其成型前斩杀，拆支援区。 | [gamersky] |
| 6 | `AhBRxxkMEXGgOWETEpAw3pEOCXAQ550OCYDR6DQPC9EwALMQCwGAC7gQC7HAD7wQDPAA` | ✅ codec+engine | 琴/林尼/琳妮特 | 扩散/风 | 风鹰剑×2; 千岩牢固×2; 蒲公英的国土×1; 如影流露的冷刃×1; 花海甘露之光×1; 璃月港口×2 | 风元素养大哥，林尼负责挂火；生存与伤害较均衡的养大个卡组。 | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [gamersky] |
| 7 | `EhKBOAwUAXFR3kIPBRES4jAPEyHx45gQCkFx9cMQDVEh98cRDbJhDcoRDcKREdISDgEB` | ✅ codec+engine | 夜兰/莫娜/神里绫华 | 冻结/控制 | 寒天宣命祝词×2; 裁叶萃光×2; 赌徒的耳环×2; 派蒙×2; 卯师傅×2; 立本×2 | 夜兰高数值连携水伤配合绫华普攻；莫娜提供变轴速冻，灵活的永冻快攻。 | 冻结控场：带解冻/速切与单核高血，避免主C被冻；抢先击杀冰核。 | [gamersky] |
| 8 | `AYHQOCMNEbCA5ZcOCVBx5jQOE2BA8MYPDABg9c0PDWEQAtEQE7FxCzcQDcGQENkREwAA` | ✅ codec+engine | 赛诺/多莉/夜兰 | 感电/超载 | 赌徒的耳环×2; 花海甘露之光×2; 卯师傅×2; 迪娜泽黛×2; 玛梅赫×2; 寻宝仙灵×2 | 多莉+夜兰打感电，第三回合赛诺出山，雷普攻配合夜兰水连携刀刀感电；构筑待改进。 | 感电连锁：单体高血/治疗可稀释连锁伤害；拆掉触发器。 | [gamersky] |
| 9 | `EZCA2xUOABDA4VEOBSAQ4oMPCFAw9ZcQCSFwA8YQDDFgDM0QDMHQDdEQDdEQENkRDQAA` | ✅ codec+engine | 夜兰/宵宫/达达利亚 | 接力/蒸发 | 长野原龙势流星群×2; 王下近侍×2; 赌徒的耳环×2; 卯师傅×2; 刘苏×2; 迪娜泽黛×2 | 夜兰开大，宵宫天赋战技点火，普攻蒸发同时攒能量，再用宵宫大招给公子普攻带火连携，上限极高的接力套牌。 | 接力蒸发：打断接力链（冻结/眩晕）、加血压其斩杀线。 | [gamersky] |
| 10 | `EhGB0hgOATGh34QPCRGh4osPEyHh45gQClJxA7QRDGIxB8MRDbJRDMYSDQKREcoSDhEB` | ✅ codec+engine | 夜兰/胡桃/行秋 | 控制/中速 | 千岩长枪×2; 赌徒的耳环×2; 王下近侍×1; 和璞鸢×1; 璃月港口×2; 瓦格纳×2 | 胡桃/行秋/夜兰，夜兰与行秋谁先开大仍在摸索，作者自评有改进空间。 | 控制中速：用 OTK 在其成型前斩杀，拆支援区。 | [gamersky] |
| 11 | `EYGQOBoTA6Bx4yIOEjAg5IMPCFAw9ZcPCWFxADQQEwFBAjUTE7FRCjYQDbGQC9kQE+AA` | ✅ codec+engine | 林尼/菲谢尔/愚人众·火之债务处理人 | 超载/双火 | 完场喝彩×2; 王下近侍×2; 赌徒的耳环×2; 花海甘露之光×2; 湖中垂柳×2; 寻宝仙灵×2 | 皇女挂雷召奥兹，林尼驻场输出打超载；王下近侍+林尼天赋提高伤害，残局讨债人单挑，愚人众伏兵续元素，前期伤害足、中期顺滑的长线快攻。 | 双火超载：加血/减伤，注意站位与速切。 | [gamersky] |
| 12 | `EYFQOCANANBw344NCPDg4JcOCQBx8TUPExBQ9c0PDFDQ9tEQE6FxCjcQDbGQC9kQE+AA` | ✅ codec+engine | 「女士」/雷电将军/申鹤 | 超导/雷鹤 | 贯月矢×2; 赌徒的耳环×2; 湖中垂柳×2; 刘苏×2; 玛梅赫×2; 寻宝仙灵×2 | 雷鹤中女士平替重云，引入高伤害火元素爆发，箓灵配合严寒增伤，成熟配队融合新角色。 | 物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。 | [gamersky] |
| 13 | `EVFQNwsTEXHhOigPEhCA8ZcPCVBw9aQPE2EBCDAQE4FBOzQTDLFAC8QQDbEQDtEQDeAA` | ✅ codec+engine | 「女士」/莫娜/雷音权现 | 三色/值伤 | 悲号回唱×2; 赌徒的耳环×2; 饰金之梦×2; 花海甘露之光×2; 沙王的投影×1; 蒂玛乌斯×2 | 雷音权现每回合增 1 伤 + 天赋抽牌，消化女士严寒，残局火形态女士；三色快攻用饰金之梦体系，首日表现优秀。 | 三色值伤快攻：抢先/加血；拆属性共鸣与饰金之梦。 | [gamersky] |
| 14 | `AUGy0iAOARFi1ioOCeFh8pgQChFy9jEQFGICCTURFIJBPMIUDbIRPcMUDcIhDMURDbEB` | ✅ codec+engine | 莫娜/特瓦林/优菈 | 扩散/风 | 赌徒的耳环×2; 饰金之梦×2; 花海甘露之光×2; 毁裂风涡×1; 狼的末路×1; 风龙废墟×2 | 经典莫娜蓄爆强风扩、优菈收割的风三色；坍陷与契机放大特瓦林 5 骰元素战技收益。 | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [gamersky] |
| 15 | `ACGxOh8OEXCh5yMOEoCQ6JcPCRBw8cEPDEAQ9MIPDFAg9ckTDMGQPNEQDbEQC9IRDRAA` | ✅ codec+engine | 莫娜/特瓦林/琳妮特 | 扩散/风 | 赌徒的耳环×2; 如影流露的冷刃×1; 毁裂风涡×1; 风龙废墟×2; 派蒙×2; 立本×2 | 莫优龙变体，收割位换琳妮特；作者自评特瓦林最优解仍待研究。 | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [gamersky] |
| 16 | `EZHwOAcTAoFgOpcOCXBx5zQOE4BA6MIPDFEgPMYTDMFgCtEQDbERCzcQE+FwDtkRDRAA` | ✅ codec+engine | 特瓦林/申鹤/温迪 | 扩散/风 | 赌徒的耳环×2; 花海甘露之光×2; 派蒙×2; 卯师傅×2; 迪娜泽黛×2; 玛梅赫×2 | 申鹤双风，开箓灵再让特瓦林扩散更合适，但很怕送你一程，思路偏极端。 | 风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。 | [gamersky] |
| 17 | `AlGh3wwPEtFi7isQE0Gh9ZgQClFy9zERFIICCTURFLJBDMcRDdJhDtIRDuIREtYUDuEB` | ✅ codec+engine | 菲谢尔/莫娜/若陀龙王 | 风扩散/三色 | 晦朔千引×2; 赌徒的耳环×2; 饰金之梦×2; 花海甘露之光×2; 卯师傅×2; 迪娜泽黛×2 | 若陀龙王三色卡组，变身魔术大量 0 骰变 3 骰，轻松叠 2 层被动，作为快攻也能持续施压。 | 风扩散三色 otk：抢先、带治疗抗爆发；注意其蓄爆回合。 | [gamersky] |
| 18 | `ANDRvWwVHYDRw+AVHqAB4nUOEiDB9oAPCWHRaEgdFDGBEYEWGJERabcdC0Fg1LYdC8AA` | ✅ codec+engine | 珊瑚宫心海/芙宁娜/茜特菈莉 | 冻结/控制 | 五重天的寒雨×2; 老兵的容颜×2; 逐影猎人×2; 「诸君听我颂，共举爱之杯！」×1; 图莱杜拉的回忆×1; 晨曦酒庄×2 | 去除蒂玛乌斯，打法芙芙战绩配合茜特菈莉冻结；加入更多速切。回合末停茜特菈莉抢先，但遇极快攻慎用。 | 冻结控场：带解冻/速切与单核高血，避免主C被冻；抢先击杀冰核。 | [bilibili] |
| 19 | `AmHB5yEQARER8pgQCiFx85oQCjGR9LUQDEFB9b4QDVHB9s0QDWHR984QD4FR+eYRD8EB` | ✅ codec+engine | 刻晴/雷电将军/甘雨 | 超导/值伤 | 赌徒的耳环×2; 流放者头冠×2; 骑士团图书馆×2; 田铁嘴×2; 刘苏×2; 桓那兰那×1 | 刻晴/雷电将军/甘雨，主打 PVE 无脑速推公式，视频称 5.0 依然最强。适合作为环境无关的脚本对局基线。 | 超导值伤：怕加血/免疫，抢先或用冻结/眩晕打断核心行动。 | [bilibili] |
| 20 | `AYAx3E8OATBg408OBEDw5JcPCVBw9cEPDGAR9jUVE7FRCjYQE7FgC9kQDeGRETgREzAA` | ✅ codec+engine | 班尼特/那维莱特/可莉 | 那维/水龙 | 冒险憧憬×2; 赌徒的耳环×2; 风龙废墟×2; 湖中垂柳×2; 欧庇克莱歌剧院×2; 寻宝仙灵×2 | 分享码经引擎解码合法（班尼特/那维莱特/可莉），但来源为 SEO 聚合站、攻略细节无法交叉验证，可信度低，仅作 那维莱特 原型样例。 | 水龙成长：速攻抢在其装备成型前击杀，或用冻结/控制打断。 | [52pk·低可信] |

## 5. 如何反制主流体系 (how to counter)

- **超导/雷鹤、火刻/超载、超载/双火、爆发速攻**（otk 快攻）：怕加血与免疫（莲花酥、仙跳墙、
  治疗料理）；抢先或用冻结/眩晕打断主 C 行动，注意站位防超载换人。
- **水债/水皇、水皇草、莫皇草、召唤流**（召唤+穿透）：用扩散/超载 AOE 清召唤物（送你一程、
  砂糖/温迪/风与自由），抢先压制纯水精灵/召唤核心，避免拖入长线资源战。
- **冻结/控制**（永冻）：带解冻/速切与单核高血，避免主 C 被冻；抢先击杀冰核（绫华/重云/优菈）。
- **双岩/结晶**：盾与结晶肉，用穿透伤害（扩散/超载）绕盾或直接点杀未成型核心。
- **激化/召唤、感电/超载**：清召唤物、AOE 消耗；单体高血/治疗可稀释感电连锁。
- **扩散/风、风扩散三色/双风**：群伤与养大哥，分散站位、少带可被扩散的附着；点杀风核或拆装备。
- **那维/水龙**：成长型，速攻抢在装备成型前击杀，或用冻结/控制打断。

## 6. 局限与未获取 (limitations)

- 官方「卡牌广场」的 `match_pool_list` 是 **3.7–4.3 赛事归档**（最新条目 2024-01）；`index_recommend`
  会随版本轮换（本次抓到 2024-07 与 2025-09 的推荐）。其 `game_cardgroup`/`creator_cardgroup`/
  `my_cardgroup` 端点需要登录，未使用。
- 当前版本（5.x/6.x）公开分享码多藏在视频简介/评论区与图片攻略里，**未被结构化 API 暴露**；
  本次仅手工抄录到少量可验证的码（§4），未收录任何无法解码的字符串。
- 一些 SEO 聚合站（如 `777game`、`modernwarshipsdl`）给出「T0 卡组」但**不含可复制分享码或明显编造**，
  已排除，不作为来源。
- `archetype` 标签与 `core_cards` 为启发式摘要，仅供检索/分组，非权威分类。

## 7. 复用 (how to use)

```bash
# 采集 + 验证（写 data/deck_corpus/，gitignore）
python -m scripts.explore.fetch_deck_corpus
# 采集 + 分享码解码 + 每套跑 1 局引擎验证 + 导出提交副本
python -m scripts.explore.fetch_deck_corpus --no-fetch --engine --export-reports
# 加载为 DeckSpec（无网络）
python -c "from decklab.corpus import corpus_deck_specs; print(len(corpus_deck_specs()))"
# 快速测试（JSON 缺失则 skip）
python -m pytest tests/test_deck_corpus.py -q
```

## 8. 来源 (references)


- miHoYo 卡牌广场 (SPA)：<https://webstatic.mihoyo.com/ys/event/bbs-lineup-qskp/index.html#/pc/home>
- miHoYo cardsquare API：<https://api-takumi.mihoyo.com/event/cardsquare/>
- gamersky 4.3 卡组推荐 (17 套)：<https://www.gamersky.com/handbook/202312/1688434.shtml>
- bilibili 5.7 双水茜：<https://www.bilibili.com/video/BV1As3DzbEAM/>
- bilibili 刻雷甘 PVE：<https://www.bilibili.com/video/BV1eo4y1u7s9/>
- 52pk 那维双火 (低可信)：<https://www.52pk.net/news/cidh463.html>
