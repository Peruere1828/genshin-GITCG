"""Build ``configs/npc/npc_stages_mond_liyue.json`` — the 16 Mondstadt/Liyue
NPC "Adventure Challenge" stage specs.

Inputs (all curated + cited, no scraping at build time):

* lineups          — ``reports/explore/npc_lineups_mond_liyue.md`` (GameWith
                     "NPC Challengers Locations & Map", article 37530); the
                     adventure-challenge lineups are the fixed enemy parties.
* HP               — ``configs/npc/npc_enemies_all.json`` (genshin-db game data).
* objectives       — eurogamer's adventure-challenge reward table
                     ("Every Adventure Challenge NPC in Genshin Impact");
                     Marjorie + Chef Mao are cross-checked verbatim against the
                     Chinese BWIKI (``wiki.biligame.com/ys``) NPC pages.
* special rules    — Chef Mao: BWIKI 卯师傅 page (verified); Timaeus:
                     ``reports/explore/npc_pve_doc.md`` §6 (HoYoWiki entry/2547,
                     not re-fetched this round -> ``verified: false``).

Run::

    python -m scripts.explore.build_npc_stages
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENEMIES = REPO_ROOT / "configs" / "npc" / "npc_enemies_all.json"
OUT = REPO_ROOT / "configs" / "npc" / "npc_stages_mond_liyue.json"

STRATEGY = "self-generated（greedy + rotation）；不使用逐关实测序列"

# Objective helpers -> {type, params, zh text, verified}
def win() -> dict:
    return {"type": "win", "text": "赢得胜利", "verified": True}


def win_within(n: int, verified: bool = True, reason: str | None = None) -> dict:
    obj = {"type": "win_within_rounds", "n": n, "text": f"{n}回合内胜利", "verified": verified}
    if reason:
        obj["unverified_reason"] = reason
    return obj


def defeated_less_than(k: int = 2) -> dict:
    return {"type": "defeated_less_than", "k": k, "text": "胜利时，角色被击倒次数小于2", "verified": True}


def use_burst(n: int = 1) -> dict:
    return {"type": "use_elemental_burst", "count": n, "text": f"使用{n}次元素爆发", "verified": True}


def reactions(n: int) -> dict:
    return {"type": "trigger_elemental_reactions", "count": n, "text": f"触发{n}次元素反应", "verified": True}


def equipment(kind: str) -> dict:
    zh = {"weapon": "武器牌", "artifact": "圣遗物牌"}[kind]
    return {
        "type": "play_equipment_card",
        "kind": kind,
        "count": 1,
        "text": f"打出1张{zh}",
        "verified": True,
    }


def events(n: int) -> dict:
    return {"type": "play_event_cards", "count": n, "text": f"打出{n}张事件牌", "verified": True}


EURO = "eurogamer.net/genius-invokation-card-adventure-challenge-npcs"
BWIKI = "wiki.biligame.com/ys"

# Stage table. lineups/reserves are (enemy_id, count) in slot order.
STAGES: list[dict] = [
    # ----------------------------- Mondstadt -----------------------------
    {
        "name": "timmie",
        "display": "提米",
        "display_en": "Timmie",
        "region": "mondstadt",
        "lineup": [("blazingaxemitachurl", 2)],
        "reserve": [],
        "objectives": [win(), win_within(5), defeated_less_than(2)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": {"text": "A Tale of Two Hilichurls", "verified": True, "source": "HoYoWiki entry/2492"},
    },
    {
        "name": "swan",
        "display": "斯万",
        "display_en": "Swan",
        "region": "mondstadt",
        "lineup": [("hilichurlshooter", 2), ("hilichurlfighter", 1)],
        "reserve": [],
        # eurogamer lists a single reward tier for Swan -> one objective only.
        "objectives": [win()],
        "objectives_verified": False,
        "objectives_source": EURO,
        "objectives_unverified_reason": "eurogamer 仅列 1 条目标（其余 NPC 均 3 条），疑为缺失，待核实",
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "wagner",
        "display": "瓦格纳",
        "display_en": "Wagner",
        "region": "mondstadt",
        "lineup": [("hydroslime", 2), ("anemosamachurl", 1)],
        "reserve": [("electrohilichurlshooter", 2)],
        "objectives": [win(), equipment("weapon"), win_within(5)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "marjorie",
        "display": "玛乔丽",
        "display_en": "Marjorie",
        "region": "mondstadt",
        "lineup": [("cryohilichurlshooter", 2), ("hilichurlshooter", 1), ("electrohilichurlshooter", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), defeated_less_than(2)],
        "objectives_verified": True,
        "objectives_source": f"{EURO}; cross-checked {BWIKI}/玛乔丽",
        "special_rules": [],
        "stage_title": {"text": "都在射程范围内", "verified": True, "source": f"{BWIKI}/玛乔丽"},
    },
    {
        "name": "blanche",
        "display": "布兰琪",
        "display_en": "Blanche",
        "region": "mondstadt",
        "lineup": [("blazingaxemitachurl", 2), ("iceshieldwallmitachurl", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), use_burst(1)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "timaeus",
        "display": "蒂玛乌斯",
        "display_en": "Timaeus",
        "region": "mondstadt",
        "lineup": [("cryohilichurlshooter", 1), ("iceshieldwallmitachurl", 1), ("blazingaxemitachurl", 1)],
        "reserve": [],
        "objectives": [win(), equipment("artifact"), win_within(5)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": ["artifact_cards_free"],
        "special_rules_verified": False,
        "special_rules_source": "reports/explore/npc_pve_doc.md §6 (HoYoWiki entry/2547)",
        "special_rules_unverified_reason": "HoYoWiki 本轮不可直连核实；沿用 npc_pve_doc.md 记录",
        "stage_title": None,
        "notes": {
            "differs_from_npc_spec_example_timaeus": (
                "npc_spec_example_timaeus.json 的 objectives 为占位（defeated_less_than 2）；"
                "经 eurogamer 核实实际目标第 2 条为「打出1张圣遗物牌」、第 3 条为「5回合内胜利」。"
                "本文件为准，原文件保留不动。"
            )
        },
    },
    {
        "name": "patton",
        "display": "巴顿",
        "display_en": "Patton",
        "region": "mondstadt",
        "lineup": [("anemosamachurl", 1), ("blazingaxemitachurl", 1), ("dendrosamachurl", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), reactions(1)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "ellin",
        "display": "艾琳",
        "display_en": "Ellin",
        "region": "mondstadt",
        "lineup": [("cryohilichurlshooter", 1), ("blazingaxemitachurl", 1), ("electroslime", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), defeated_less_than(2)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    # ------------------------------- Liyue -------------------------------
    {
        "name": "jifang",
        "display": "纪芳",
        "display_en": "Jifang",
        "region": "liyue",
        "lineup": [("hydroslime", 1), ("dendrosamachurl", 1), ("electrohilichurlshooter", 1)],
        "reserve": [("hydroslime", 1), ("anemosamachurl", 1), ("cryohilichurlshooter", 1)],
        "objectives": [win(), win_within(5), defeated_less_than(2)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "buyun",
        "display": "步云",
        "display_en": "Bu'yun",
        "region": "liyue",
        "lineup": [("anemosamachurl", 1), ("cryoabyssmage", 1), ("electroslime", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), reactions(2)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "liusu",
        "display": "刘苏",
        "display_en": "Tea Master Liu Su",
        "region": "liyue",
        "lineup": [("fatuipyroagent", 1), ("fatuiskirmisheranemoboxervanguard", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), use_burst(1)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "shitou",
        "display": "石头",
        "display_en": "Shitou",
        "region": "liyue",
        "lineup": [("cracklingaxemitachurl", 1), ("cryohilichurlshooter", 1), ("blazingaxemitachurl", 1)],
        "reserve": [],
        "objectives": [
            win(),
            win_within(44, verified=False, reason="eurogamer 原文为 'within 44 Rounds'，疑为笔误（4/14?），待游戏内核实"),
            reactions(2),
        ],
        "objectives_verified": False,
        "objectives_source": EURO,
        "objectives_unverified_reason": "第 2 条回合数 44 疑似笔误",
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "irontonguetian",
        "display": "田铁嘴",
        "display_en": "Iron Tongue Tian",
        "region": "liyue",
        "lineup": [("fatuipyroagent", 1), ("fatuiskirmisherhydrogunnerlegionnaire", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), use_burst(1)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "chefmao",
        "display": "卯师傅",
        "display_en": "Chef Mao",
        "region": "liyue",
        "lineup": [("fatuipyroagent", 1), ("mirrormaiden", 1)],
        "reserve": [],
        "objectives": [win(), win_within(5), defeated_less_than(2)],
        "objectives_verified": True,
        "objectives_source": f"{EURO}; cross-checked {BWIKI}/卯师傅",
        "special_rules": ["chef_mao_support_present", "pyro_application_on_action_phase_start"],
        "special_rules_verified": True,
        "special_rules_source": f"{BWIKI}/卯师傅",
        "special_rules_text": "我方支援区初始存在一张「卯师傅」；行动阶段开始时，对双方的出战角色附着火元素。",
        "stage_title": {"text": "烈火烹油", "verified": True, "source": f"{BWIKI}/卯师傅"},
    },
    {
        "name": "uncletian",
        "display": "天叔",
        "display_en": "Uncle Tian",
        "region": "liyue",
        "lineup": [("cryoabyssmage", 1), ("pyroabyssmage", 1)],
        "reserve": [],
        "objectives": [win(), events(2), win_within(5)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
    {
        "name": "lan",
        "display": "岚姐",
        "display_en": "Lan",
        "region": "liyue",
        "lineup": [("dendrosamachurl", 1), ("pyroabyssmage", 1), ("electrohilichurlshooter", 1)],
        "reserve": [],
        "objectives": [win(), win_within(6), reactions(2)],
        "objectives_verified": True,
        "objectives_source": EURO,
        "special_rules": [],
        "stage_title": None,
    },
]


def build() -> dict:
    enemies = {e["id"]: e for e in json.loads(ENEMIES.read_text(encoding="utf-8"))["enemies"]}

    def party(pairs):
        out = []
        for slot, (eid, count) in enumerate(pairs, start=1):
            e = enemies[eid]
            out.append(
                {"slot": slot, "id": eid, "name": e["name"], "en": e["en"], "hp": e["hp_card"], "count": count}
            )
        return out

    stages = []
    for spec in STAGES:
        stage = {
            "name": spec["name"],
            "display": spec["display"],
            "display_en": spec["display_en"],
            "region": spec["region"],
            "lineup": party(spec["lineup"]),
            "reserve": party(spec["reserve"]),
            "special_rules": spec["special_rules"],
            "objectives": spec["objectives"],
            "strategy": STRATEGY,
            "sources": {
                "lineup": "gamewith/37530 (Adventure Challenge) via reports/explore/npc_lineups_mond_liyue.md",
                "objectives": spec.get("objectives_source"),
                "special_rules": spec.get("special_rules_source"),
            },
            "objectives_verified": spec.get("objectives_verified", False),
        }
        for key in ("stage_title", "notes", "special_rules_verified", "special_rules_text",
                    "special_rules_unverified_reason", "objectives_unverified_reason"):
            if spec.get(key) is not None:
                stage[key] = spec[key]
        stages.append(stage)

    return {
        "schema_version": "1.0",
        "generated": date.today().isoformat(),
        "generator": "scripts/explore/build_npc_stages.py",
        "scope": "16 Mondstadt (8) + Liyue (8) NPC Adventure Challenge stages",
        "facts": [
            "NPC 无手牌 / 无骰子 / 不打出行动牌；玩家恒定先手（复刻时 NPC = player 1）",
            "敌人 HP/技能见 npc_enemies_all.json（genshin-db 游戏数据）",
            "充能：A/E 各 +1，Q 耗净充能，上限 = energy；切人由被动/脚本驱动，不计行动",
        ],
        "objective_types": {
            "win": "赢得胜利",
            "win_within_rounds": "N 回合内胜利（n）",
            "defeated_less_than": "胜利时被击倒次数 < k",
            "use_elemental_burst": "使用 count 次元素爆发",
            "trigger_elemental_reactions": "触发 count 次元素反应",
            "play_equipment_card": "打出 count 张 kind（weapon/artifact）装备牌",
            "play_event_cards": "打出 count 张事件牌",
        },
        "sources": {
            "lineups": "GameWith NPC Challengers (article/show/37530)",
            "objectives": "eurogamer.net adventure-challenge reward table; BWIKI cross-check for Marjorie & Chef Mao",
            "enemy_stats": "genshin-db (theBowja/genshin-db) — see configs/npc/npc_enemies_all.json",
            "special_rules": "BWIKI 卯师傅 (Chef Mao); npc_pve_doc.md §6 (Timaeus, HoYoWiki/2547)",
        },
        "discrepancies_with_npc_spec_example": {
            "note": "npc_spec_example_timaeus.json 保持不变（占位/参考）；下列以本文件为准。",
            "items": [
                {
                    "stage": "timaeus",
                    "field": "objectives",
                    "npc_spec_example": ["win", "win_within_rounds(null)", "defeated_less_than(2)"],
                    "npc_stages_mond_liyue": ["win", "play_equipment_card(artifact)", "win_within_rounds(5)"],
                    "source": EURO,
                }
            ],
        },
        "stages": stages,
    }


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(data['stages'])} stages)")


if __name__ == "__main__":
    main()
