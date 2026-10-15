"""C3（WS8/D18）：把 NPC 敌人阵容变成**人可以拿去打**的牌组。

D18 说「人可持 NPC 敌人卡组」。NPC「冒险挑战」的阵容有三个真实特征，
本模块逐条对应处理（不做「凑 3 人」这种想当然的假设）：

1. **人数不固定**（关卡数据里 1 / 2 / 4 人都有，蒙德+璃月 16 关从 1 人到 6 人不等，
   含同一敌人的多份）→ **原样保留、不跳过、不去重、不凑数**。引擎的
   `initPlayerState` 只按牌组里的角色列表建局、不校验人数（PvE 线已经用
   2–4 人的 NPC 阵容跑通门禁），所以 2 人/4 人阵容一样能开局。
2. **支援区（预备）角色**：出战全灭后才上场 → 数据里是 `reserve`。引擎**没有**
   支援区机制，只能把它们接在出战角色后面一起进牌组（= 后台角色），靠
   「出战没死就不主动切」来近似原版语义；`n_support` 会显式标出来。
3. **多轮挑战**（打完一波换新战场、玩家状态保留）→ 单局表达不了，关卡数据目前也
   没有 `waves` 字段；本接口按**单局**给阵容，多轮是 PvE 线（B）的事，
   见 `semantics.waves`。

NPC 本身不打行动牌（PvE 近似），但**人**要正常出牌，所以行动牌统一用
`decklab` 池 C 的固定 core/fill 填满 30 张（牌严格来自池内，角色数不限——
用 `decklab.builder.fill_cards`，绕开 `build_deck` 的「恰好 3 个不同角色」校验）。

浏览器侧拿到 `{characters, cards}` 后**直接当牌组对象传给引擎**，不走分享码：
NPC 角色 id 不在官方 `share_id.json` 里，`DEFAULT_ASSETS_MANAGER.encode` 会抛错。
"""

from __future__ import annotations

from typing import Any

from common.paths import REPO_ROOT

DEFAULT_STAGES_PATH = REPO_ROOT / "configs" / "npc" / "npc_stages_mond_liyue.json"

# 给人用的 NPC 牌组要说清三件事的近似程度（前端展示 + 回给测试用）。
SEMANTICS: dict[str, str] = {
    "lineup_size": (
        "出战人数按关卡原样（含同一敌人的多份），不凑 3 人；引擎按牌组角色列表"
        "建局、不校验人数，PvE 线已用 2–4 人阵容跑通。"
    ),
    "support": (
        "预备（支援区）角色接在出战角色之后进牌组，进局即为后台角色：原版语义是"
        "「出战全灭后上场」，引擎没有支援区机制，只能靠不主动切换来近似。"
    ),
    "waves": (
        "多轮挑战（新战场接续、玩家角色血量/卡牌保留）不是一局能表达的，关卡数据"
        "当前也没有 waves 字段——本接口按单局给阵容，多轮归 PvE 线（PLAN §4.4）。"
    ),
    "cards": (
        "NPC 不打行动牌；这里统一用 decklab 池 C 固定规则填 30 张，"
        "人才能以正常玩家身份出牌。"
    ),
    "share_code": (
        "NPC 角色 id 不在官方 share_id 表里，编不出分享码——前端按牌组对象直传。"
    ),
}


def npc_decks(stages_path: str | None = None) -> dict[str, Any]:
    """返回**全部** NPC 关卡阵容（出战 + 预备）当玩家牌组用。

    形状：
    ``{"decks": [{slug, display, region, enemies, characters,
                  n_on_field, n_support, total, cards}],
       "cards_source": <池名>, "semantics": {...}}``
    ``characters`` = 出战（按 count 展开，保留重复）+ 预备（同样展开），顺序即上场
    优先级。
    """
    from decklab.builder import (
        POOL_C_FILL,
        POOL_C_META_CORE,
        load_pool,
        fill_cards,
    )
    from npc.spec import load_stages

    stages = load_stages(stages_path)
    pool = load_pool()
    cards = fill_cards(pool, core=POOL_C_META_CORE, fill=POOL_C_FILL)

    decks: list[dict[str, Any]] = []
    for slug in sorted(stages):
        stage = stages[slug]
        on_field = [int(c) for c in stage.deck_characters]
        support = [int(c) for c in stage.reserve_characters]
        characters = on_field + support
        decks.append(
            {
                "slug": slug,
                "display": stage.display,
                "region": stage.region,
                "enemies": [entry.name for entry in stage.lineup],
                "reserve_enemies": [entry.name for entry in stage.reserve],
                "characters": characters,
                "n_on_field": len(on_field),
                "n_support": len(support),
                "total": len(characters),
                "cards": list(cards),
            }
        )
    return {
        "decks": decks,
        "cards_source": pool.name,
        "semantics": dict(SEMANTICS),
    }
