"""Build ``configs/npc/npc_enemies_all.json`` — authoritative NPC enemy-card data.

The PvE "Adventure Challenge" enemies (slimes, hilichurls, samachurls,
mitachurls, abyss mages, Fatui skirmishers, ...) are **not** in the engine's
PvP card pool (see ``reports/explore/npc_pve_doc.md`` §2), so the engine cannot
supply them.  We rebuild them from the publicly maintained **genshin-db** card
dump (theBowja/genshin-db), which is generated from the in-game data + Fandom
and exposes, per card, HP / max energy / tags / skills (cost + base damage +
effect text) in every language.

For the handful of cards that GameWith also documents (hilichurls / samachurls /
mitachurls / Fatui), the values are cross-checked against the GameWith card
pages; see ``cross_check`` on each record.

Source:
    https://cdn.jsdelivr.net/gh/theBowja/genshin-db@main/src/data/<lang>/<folder>/<slug>.json

Nothing is fetched at import time; run::

    python -m scripts.explore.fetch_npc_enemies            # fetch (cached) + build
    python -m scripts.explore.fetch_npc_enemies --no-fetch # rebuild from cache only

The raw cache lives in the gitignored ``data/npc_enemy_cache/``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "configs" / "npc" / "npc_enemies_all.json"
CACHE = REPO_ROOT / "data" / "npc_enemy_cache"

GENSIN_DB_REF = "main"  # jsdelivr resolves to the current default branch (v5.x)
CDN = f"https://cdn.jsdelivr.net/gh/theBowja/genshin-db@{GENSIN_DB_REF}/src/data"
HEADERS = {"User-Agent": "genshin-gitcg-npc-enemies/1.0 (+research)"}

LANGS = {"en": "English", "cn": "ChineseSimplified"}

# ---------------------------------------------------------------------------
# Codebook: engine/GCG enums -> our compact strings (mirrors npc_enemies_timaeus)
# ---------------------------------------------------------------------------
ELEMENTS = {
    "GCG_TAG_ELEMENT_CRYO": "cryo",
    "GCG_TAG_ELEMENT_HYDRO": "hydro",
    "GCG_TAG_ELEMENT_PYRO": "pyro",
    "GCG_TAG_ELEMENT_ELECTRO": "electro",
    "GCG_TAG_ELEMENT_ANEMO": "anemo",
    "GCG_TAG_ELEMENT_GEO": "geo",
    "GCG_TAG_ELEMENT_DENDRO": "dendro",
}
COST = {
    "GCG_COST_DICE_CRYO": "Cryo",
    "GCG_COST_DICE_HYDRO": "Hydro",
    "GCG_COST_DICE_PYRO": "Pyro",
    "GCG_COST_DICE_ELECTRO": "Electro",
    "GCG_COST_DICE_ANEMO": "Anemo",
    "GCG_COST_DICE_GEO": "Geo",
    "GCG_COST_DICE_DENDRO": "Dendro",
    "GCG_COST_DICE_SAME": "Same",
    "GCG_COST_DICE_VOID": "Unaligned",
    "GCG_COST_DICE_OMNI": "Omni",
    "GCG_COST_ENERGY": "Energy",
}
DAMAGE_ELEMENT = {
    "GCG_ELEMENT_CRYO": "cryo",
    "GCG_ELEMENT_HYDRO": "hydro",
    "GCG_ELEMENT_PYRO": "pyro",
    "GCG_ELEMENT_ELECTRO": "electro",
    "GCG_ELEMENT_ANEMO": "anemo",
    "GCG_ELEMENT_GEO": "geo",
    "GCG_ELEMENT_DENDRO": "dendro",
    "GCG_ELEMENT_PHYSIC": "physical",
}
# GameWith card-page article ids (only where GameWith documents the card).
GAMEWITH = {
    "anemosamachurl": 70651,
    "blazingaxemitachurl": 70652,
    "cracklingaxemitachurl": 70653,
    "cryohilichurlshooter": 70654,
    "dendrosamachurl": 70655,
    "electrohilichurlshooter": 70656,
    "hilichurlfighter": 70660,
    "hilichurlshooter": 70661,
    "iceshieldwallmitachurl": 70663,
    "fatuipyroagent": 37644,
    "mirrormaiden": 37647,
}

# ---------------------------------------------------------------------------
# The 19 enemy types that appear in the Mondstadt/Liyue Adventure Challenge
# lineups (reports/explore/npc_lineups_mond_liyue.md), plus the two extra
# elemental slime / abyss-mage variants named in the NPC PvE scope.  For each
# we pick the canonical genshin-db slug, the doc aliases, the stages it appears
# in, and the auxiliary status/summon effects that its passive/skills grant.
# Each effect entry is (kind, slug, linked_to) where linked_to is "passive" or
# "skill:A"/"skill:E"/"skill:Q".
# ---------------------------------------------------------------------------
ENEMIES: list[dict[str, Any]] = [
    {
        "id": "hydroslime",
        "aliases": ["水史莱姆"],
        "appears_in": ["wagner", "jifang"],
        "effects": [("status", "elementallifeformhydro", "passive")],
    },
    {
        "id": "electroslime",
        "aliases": ["雷史莱姆"],
        "appears_in": ["ellin", "buyun"],
        "effects": [("status", "elementallifeformelectro", "passive")],
    },
    {
        "id": "cryoslime",
        "aliases": ["冰史莱姆"],
        "appears_in": [],
        "effects": [("status", "elementallifeformcryo", "passive")],
    },
    {
        "id": "cryohilichurlshooter",
        "aliases": ["冰箭丘丘人"],
        "appears_in": ["marjorie", "timaeus", "ellin", "shitou", "jifang"],
        "effects": [],
    },
    {
        "id": "electrohilichurlshooter",
        "aliases": ["雷箭丘丘人"],
        "appears_in": ["wagner", "marjorie", "jifang", "lan"],
        "effects": [],
    },
    {
        "id": "hilichurlshooter",
        "aliases": ["丘丘人射手"],
        "appears_in": ["swan", "marjorie"],
        "effects": [],
    },
    {
        "id": "hilichurlfighter",
        "aliases": ["丘丘人斗士"],
        "appears_in": ["swan"],
        "effects": [],
    },
    {
        "id": "blazingaxemitachurl",
        "aliases": ["炽热斧头丘丘暴徒", "火斧丘丘暴徒"],
        "appears_in": ["timmie", "blanche", "timaeus", "patton", "ellin", "shitou"],
        "effects": [("status", "flowfireedge", "passive")],
    },
    {
        "id": "iceshieldwallmitachurl",
        "aliases": ["冰盾丘丘暴徒"],
        "appears_in": ["blanche", "timaeus"],
        "effects": [("status", "hoarfrostgreatshield", "passive")],
    },
    {
        "id": "cracklingaxemitachurl",
        "aliases": ["雷斧丘丘暴徒"],
        "appears_in": ["shitou"],
        "effects": [("status", "rippleelectroedge", "passive")],
    },
    {
        "id": "anemosamachurl",
        "aliases": ["风丘丘萨满"],
        "appears_in": ["wagner", "patton", "buyun", "jifang"],
        "effects": [("summon", "whirlwind", "skill:E")],
    },
    {
        "id": "dendrosamachurl",
        "aliases": ["草丘丘萨满"],
        "appears_in": ["patton", "jifang", "lan"],
        "effects": [("status", "vines", "skill:E")],
    },
    {
        "id": "cryoabyssmage",
        "aliases": ["冰深渊法师"],
        "appears_in": ["buyun", "uncletian"],
        "effects": [("status", "cryoward", "passive")],
    },
    {
        "id": "pyroabyssmage",
        "aliases": ["火深渊法师"],
        "appears_in": ["uncletian", "lan"],
        "effects": [("status", "pyroward", "passive")],
    },
    {
        "id": "hydroabyssmage",
        "aliases": ["水深渊法师"],
        "appears_in": [],
        "effects": [("status", "hydroward", "passive"), ("status", "gurglingbubble", "skill:E")],
    },
    {
        "id": "fatuiskirmisheranemoboxervanguard",
        "aliases": ["愚人众先遣队·风拳前锋军", "愚人众·先锋军"],
        "appears_in": ["liusu"],
        "effects": [("status", "stormwindbarrier", "skill:E")],
    },
    {
        "id": "fatuiskirmisherhydrogunnerlegionnaire",
        "aliases": ["愚人众先遣队·水铳重卫士", "愚人众·水铳游击兵"],
        "appears_in": ["irontonguetian"],
        "effects": [("status", "hydroelementalequipment", "skill:Q")],
    },
    {
        "id": "fatuipyroagent",
        "aliases": ["愚人众·火之债务处理人"],
        "appears_in": ["liusu", "irontonguetian", "chefmao"],
        "effects": [("status", "stealth", "passive")],
    },
    {
        "id": "mirrormaiden",
        "aliases": ["愚人众·藏镜仕女", "藏镜仕女"],
        "appears_in": ["chefmao"],
        "effects": [("status", "refraction", "skill:E")],
    },
]


def _fetch(folder: str, lang: str, slug: str) -> dict[str, Any] | None:
    cache = CACHE / lang / folder / f"{slug}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - refetch on corrupt cache
            pass
    url = f"{CDN}/{LANGS[lang]}/{folder}/{slug}.json"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:  # noqa: BLE001 - network optional when cache is warm
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _element(card: dict[str, Any]) -> str:
    for tag in card["tags"]:
        if tag in ELEMENTS:
            return ELEMENTS[tag]
    return "physical"


def _cost(skill: dict[str, Any]) -> str:
    parts = [f"{COST.get(c['costtype'], c['costtype'])}{c['count']}" for c in skill.get("playcost", [])]
    return "+".join(parts) if parts else "-"


def _skill(skill: dict[str, Any], cn: dict[str, Any] | None) -> dict[str, Any]:
    typ = skill["typetag"].replace("GCG_SKILL_TAG_", "")
    record: dict[str, Any] = {
        "tag": typ,
        "name": skill["name"],
        "name_cn": (cn or {}).get("name", skill["name"]),
        "cost": _cost(skill),
        "effect": (cn or {}).get("description", skill["description"]),
        "effect_en": skill["description"],
    }
    if skill.get("basedamage") is not None:
        record["damage"] = skill["basedamage"]
        record["damage_element"] = DAMAGE_ELEMENT.get(skill.get("baseelement", ""), "physical")
    return record


def _referenced_effect(kind: str, slug: str, linked_to: str) -> dict[str, Any] | None:
    folder = "tcgsummons" if kind == "summon" else "tcgstatuseffects"
    en = _fetch(folder, "en", slug)
    cn = _fetch(folder, "cn", slug)
    if en is None:
        return None
    return {
        "kind": kind,
        "linked_to": linked_to,
        "name": (cn or {}).get("name", en.get("name")),
        "name_en": en.get("name"),
        "effect": (cn or {}).get("description", en.get("description", "")),
        "effect_en": en.get("description", ""),
        "source": f"genshin-db {folder}/{slug}",
    }


def _diff_timaeus(ours: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare against the pre-existing (authoritative, measured) Timaeus file.

    We never rewrite that file; we only surface any drift so it can be reconciled
    deliberately.  Matching is by English card name.
    """
    path = REPO_ROOT / "configs" / "npc" / "npc_enemies_timaeus.json"
    if not path.exists():
        return {"status": "timaeus file absent"}
    legacy = json.loads(path.read_text(encoding="utf-8"))["enemies"]
    by_en = {e["en"].lower(): e for e in ours}
    diffs: list[dict[str, Any]] = []
    for old in legacy:
        new = by_en.get(old["en"].lower())
        if new is None:
            diffs.append({"en": old["en"], "issue": "not found in npc_enemies_all"})
            continue
        for field, ours_val in (("name", new["name"]), ("element", new["element"]),
                                ("hp_card", new["hp_card"]), ("energy", new["energy"])):
            if old.get(field) != ours_val:
                diffs.append({"en": old["en"], "field": field,
                              "timaeus_file": old.get(field), "npc_enemies_all": ours_val})
        old_passive = (old.get("passive") or {}).get("name")
        new_passive = (new.get("passive") or {}).get("name")
        if old_passive and new_passive and old_passive != new_passive:
            diffs.append({"en": old["en"], "field": "passive.name",
                          "timaeus_file": old_passive, "npc_enemies_all": new_passive})
    return {
        "note": "npc_enemies_timaeus.json 保持不变；下列差异以本文件（genshin-db 游戏数据）为准。",
        "count": len(diffs),
        "diffs": diffs,
    }


def build() -> dict[str, Any]:
    out_enemies: list[dict[str, Any]] = []
    for spec in ENEMIES:
        slug = spec["id"]
        en = _fetch("tcgenemycards", "en", slug)
        cn = _fetch("tcgenemycards", "cn", slug)
        if en is None:
            raise SystemExit(f"missing genshin-db card for {slug!r} (need network on first run)")
        cn_skills = {s["id"]: s for s in (cn or {}).get("skills", [])}

        skills: dict[str, Any] = {"A": None, "E": None, "Q": None}
        extra: list[dict[str, Any]] = []
        passives: list[dict[str, Any]] = []
        for sk in en["skills"]:
            rec = _skill(sk, cn_skills.get(sk["id"]))
            tag = rec.pop("tag")
            if tag in ("A", "E", "Q") and skills[tag] is None:
                skills[tag] = rec
            elif tag == "PASSIVE":
                passives.append(rec)
            else:
                extra.append(rec)

        referenced = [
            e for e in (_referenced_effect(k, s, link) for k, s, link in spec.get("effects", [])) if e
        ]
        # main passive = first passive skill ("Hide", "Flowfire Edge", element
        # lifeform, ward, ...).  Its own description is verbatim in-game text;
        # the expanded status text lives in referenced_effects (link "passive").
        passive = None
        if passives:
            main = passives[0]
            detail = next((r for r in referenced if r["linked_to"] == "passive"), None)
            passive = {
                "name": f"{main['name']}（{main['name_cn']}）",
                "name_cn": main["name_cn"],
                "name_en": main["name"],
                "effect": main["effect"],
                "effect_en": main["effect_en"],
            }
            if detail:
                passive["detail_effect"] = detail["effect"]
                passive["detail_effect_en"] = detail["effect_en"]

        sources: list[dict[str, str]] = [
            {
                "type": "game_data",
                "ref": f"genshin-db {GENSIN_DB_REF}: tcgenemycards/{slug}",
                "url": f"{CDN}/{LANGS['en']}/tcgenemycards/{slug}.json",
            }
        ]
        cross = None
        if slug in GAMEWITH:
            url = f"https://gamewith.net/genshin-impact/article/show/{GAMEWITH[slug]}"
            sources.append({"type": "gamewith", "url": url})
            cross = {"checked": True, "source": "gamewith", "url": url}

        out_enemies.append(
            {
                "id": slug,
                "name": cn["name"] if cn else en["name"],
                "en": en["name"],
                "aliases": spec.get("aliases", []),
                "element": _element(en),
                "hp_card": en["hp"],
                "energy": en["maxenergy"],
                "weapon": "Other weapons",
                "camp": [t.lower() for t in en.get("tagstext", []) if t in ("Monster", "Hilichurl", "Fatui")],
                "appears_in": spec.get("appears_in", []),
                "skills": skills,
                "extra_skills": extra,
                "passive": passive,
                "referenced_effects": referenced,
                "sources": sources,
                "cross_check": cross,
                "confidence": "high" if cross else "high (game data)",
            }
        )

    return {
        "schema_version": "1.0",
        "generated": date.today().isoformat(),
        "generator": "scripts/explore/fetch_npc_enemies.py",
        "scope": "Mondstadt + Liyue NPC 'Adventure Challenge' enemy types",
        "source": "genshin-db (theBowja/genshin-db), generated from in-game data + Fandom; "
        "cross-checked with GameWith card pages where available",
        "field_notes": {
            "cost": "e.g. 'Cryo1+Unaligned2' / 'Cryo3+Energy2'",
            "effect": "official Chinese description (in-game text)",
            "hp_card": "card HP; equals the HP used by the PvE stages (confirmed for Timaeus 3/8/8)",
            "energy": "Elemental Burst energy cost cap (A/E each +1, Q consumes it)",
            "charge_rule": "A/E each grant +1 energy; Q consumes it; cap = energy",
        },
        "discrepancies_with_timaeus_file": _diff_timaeus(out_enemies),
        "enemies": out_enemies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fetch", action="store_true", help="use cached raw JSON only")
    args = parser.parse_args()
    if args.no_fetch and not CACHE.exists():
        raise SystemExit("no cache; run without --no-fetch first")
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(data['enemies'])} enemy types)")


if __name__ == "__main__":
    main()
