"""Fetch + validate a corpus of PUBLIC 七圣召唤 (Genius Invokation TCG) decks.

Exploration task 1: assemble publicly available decks + deck guides for later use
as an expanded evaluation pool / deck-lab input.

Two sources:

1. **miHoYo 卡牌广场** (official, no login) — the JS SPA at
   https://webstatic.mihoyo.com/ys/event/bbs-lineup-qskp/index.html#/pc/home
   is driven by the JSON API base ``https://api-takumi.mihoyo.com/event/cardsquare/``.
   Each deck record already carries the official ``card_code`` share code plus the
   character/action card lists and the author's own guide text (``desc``).
   Endpoints used (see ``data/deck_corpus/README.md`` for the full provenance):
   - ``match_pool_list``  : 燃斗无双积分赛 pool (competitive, 3.7-4.3 era)
   - ``index_recommend``  : home-page recommendations
   - ``honor_match_pool`` : honour + random video pick
2. **Web guides** — hand-curated share codes found via web search, each tied to
   its source URL. Only codes that decode to a legal 3-character + 30-card deck
   are kept (see ``_WEB_DECKS`` below).

Everything is written to the gitignored ``data/deck_corpus/`` (raw API JSON +
normalised ``decks.json``). Nothing here is required for the fast test; the
test skips when the JSON is absent.

Run::

    python -m scripts.explore.fetch_deck_corpus            # fetch + validate codes
    python -m scripts.explore.fetch_deck_corpus --engine   # + 1 scripted game per deck
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "deck_corpus"
RAW_DIR = DATA_DIR / "raw"
WEB_SOURCES_FILE = REPO_ROOT / "scripts" / "explore" / "web_decks.json"

API_BASE = "https://api-takumi.mihoyo.com/event/cardsquare"
SPA_URL = "https://webstatic.mihoyo.com/ys/event/bbs-lineup-qskp/index.html#/pc/home"
HEADERS = {"User-Agent": "genshin-gitcg-deck-corpus/1.0 (+research)"}

# Endpoint -> (raw filename, list key inside data)
API_ENDPOINTS = {
    "match_pool_list": ("match_pool_list.json", "list", {"page_size": 50}),
    "index_recommend": ("index_recommend.json", "recommend_card_group", {}),
    "honor_match_pool": ("honor_match_pool.json", "honor_match_card_group", {}),
}

# ---------------------------------------------------------------------------
# Archetype labelling (heuristic, from characters + author tags/title).
# Kept deliberately simple and transparent; labels are advisory, not ground truth.
# ---------------------------------------------------------------------------
# Specific archetype keywords (title + author tags) win over generic plan tags.
_SPECIFIC_RULES: list[tuple[str, str]] = [
    ("莫皇陀", "风扩散/三色"),
    ("雷音", "三色/值伤"),
    ("四色", "三色/值伤"),
    ("莫皇草", "莫皇草/召唤穿透"),
    ("水债", "水债/召唤-水"),
    ("水皇", "水债/召唤-水"),
    ("水八", "水债/召唤-水"),
    ("妮绽放", "绽放/召唤"),
    ("雷鹤", "超导/雷鹤"),
    ("超导", "超导/值伤"),
    ("冻结", "冻结/控制"),
    ("永冻", "冻结/控制"),
    ("双岩", "双岩/结晶"),
    ("激化", "激化/召唤"),
    ("感电", "感电/超载"),
    ("扩散", "扩散/风"),
    ("超载", "超载/双火"),
    ("火刻", "火刻/超载"),
    ("神宵", "神宵/冻结融化"),
    ("莫优", "风扩散/三色"),
    ("莫风", "风扩散/三色"),
    ("鹤风", "风扩散/三色"),
    ("双风", "风扩散/双风"),
    ("林债", "超载/召唤"),
    ("女士", "三色/值伤"),
    ("绽放", "绽放/召唤"),
    ("接力", "接力/蒸发"),
    ("燃烧", "燃烧/激化"),
    ("双水", "双水/中速"),
    ("召唤流", "召唤流"),
]
_GENERIC_RULES: list[tuple[str, str]] = [
    ("otk", "OTK"),
    ("快攻", "快攻"),
    ("速攻", "快攻"),
    ("控制", "控制/中速"),
    ("中速", "控制/中速"),
    ("中慢速", "控制/中速"),
    ("慢速", "控制/中速"),
    ("运营", "控制/中速"),
    ("爆发", "爆发速攻"),
]


def label_archetype(char_names: list[str], tags: list[str], title: str) -> str:
    text = " ".join([title, *tags])
    lowered = text.lower()
    names = set(char_names)

    def any_of(*words: str) -> bool:
        return any(w in names for w in words)

    def all_of(*words: str) -> bool:
        return all(w in names for w in words)

    # 1) Character composition (most reliable; a title may name the *opponent*).
    if "那维莱特" in names:
        return "那维/水龙"
    if all_of("纯水精灵·洛蒂娅", "纳西妲"):
        return "水皇草/召唤-水"
    if "纯水精灵·洛蒂娅" in names:
        return "水债/召唤-水"
    if all_of("雷电将军", "申鹤") or all_of("雷电将军", "重云") or "米卡" in names:
        return "超导/雷鹤"
    if all_of("可莉", "班尼特") or all_of("可莉", "刻晴"):
        return "火刻/超载"
    if all_of("莫娜", "纳西妲") or all_of("八重神子", "莫娜"):
        return "莫皇草/召唤穿透"
    if all_of("诺艾尔", "五郎") or all_of("诺艾尔", "荒泷一斗") or all_of("凝光", "五郎"):
        return "双岩/结晶"
    if all_of("八重神子", "提纳里") or all_of("柯莱", "提纳里") or "艾尔海森" in names:
        return "激化/召唤"
    if all_of("菲谢尔", "雷泽") or "坎蒂丝" in names:
        return "感电/超载"
    if all_of("重云", "优菈") or all_of("神里绫华", "莫娜") or all_of("行秋", "重云"):
        return "冻结/控制"
    if all_of("宵宫", "达达利亚"):
        return "接力/蒸发"
    if all_of("香菱", "柯莱") or all_of("香菱", "纳西妲"):
        return "燃烧/激化"
    if "特瓦林" in names or all_of("枫原万叶", "申鹤") or "砂糖" in names:
        return "扩散/风"
    # 2) Explicit archetype keywords in the title/tags.
    for needle, label in _SPECIFIC_RULES:
        if needle in text or needle in lowered:
            return label
    # 3) Generic plan tags.
    for needle, label in _GENERIC_RULES:
        if needle in text or needle in lowered:
            return label
    return "其他/未分类"


# Archetype counter hints (generic; honest archetype-level advice).
COUNTER_HINTS = {
    "水债/召唤-水": "召唤物多、续航强：靠扩散/超载 AOE 清召唤物（送你一程、砂糖/温迪），抢先压制纯水精灵。",
    "水皇草/召唤-水": "召唤+激化混伤：清召唤物、穿透打后排；避免被拖入长线资源战。",
    "莫皇草/召唤穿透": "召唤穿透伤：用护盾/治疗扛穿透、抢先杀掉趁手角色，风与自由清召唤区。",
    "超导/雷鹤": "物伤/超导 otk：怕加血与免疫（莲花酥、治疗），抢先或用冻结卡主C行动。",
    "超导/值伤": "超导值伤：怕加血/免疫，抢先或用冻结/眩晕打断核心行动。",
    "双水/中速": "双水续航：抢先压制、拆支援区；避免被拖入长线资源战。",
    "燃烧/激化": "燃烧激化：清召唤物、注意燃烧持续伤害；抢先杀香菱/柯莱。",
    "接力/蒸发": "接力蒸发：打断接力链（冻结/眩晕）、加血压其斩杀线。",
    "召唤流": "召唤流：AOE 清召唤物、抢先杀召唤核心。",
    "爆发速攻": "爆发速攻：加血+护盾+清场，拖入中后期。",
    "火刻/超载": "双火超载 otk：怕高血与治疗，注意站位防超载换人；用冻结/眩晕断连。",
    "冻结/控制": "冻结控场：带解冻/速切与单核高血，避免主C被冻；抢先击杀冰核。",
    "双岩/结晶": "盾/结晶肉：用穿透伤害（扩散/超载）绕盾，或直接点杀未成型核心。",
    "激化/召唤": "激化增伤+召唤：清召唤物、AOE 消耗；抢先杀草/雷输出。",
    "感电/超载": "感电连锁：单体高血/治疗可稀释连锁伤害；拆掉触发器。",
    "扩散/风": "风扩散群伤：分散站位、少带可被扩散的附着；抢先击杀风核。",
    "风扩散/三色": "风扩散三色 otk：抢先、带治疗抗爆发；注意其蓄爆回合。",
    "风扩散/双风": "双风养大哥：点杀主C或拆装备；避免被无伤运营拖死。",
    "超载/召唤": "超载+召唤：清召唤物、防换人；用加血压其斩杀线。",
    "超载/双火": "双火超载：加血/减伤，注意站位与速切。",
    "神宵/冻结融化": "冻结+融化爆发：带解冻与高血，抢先杀输出。",
    "绽放/召唤": "绽放召唤：清召唤物、抢先压制妮露/草原核。",
    "那维/水龙": "水龙成长：速攻抢在其装备成型前击杀，或用冻结/控制打断。",
    "三色/值伤": "三色值伤快攻：抢先/加血；拆属性共鸣与饰金之梦。",
    "OTK": "OTK：抢先、留解牌/加血，注意其组件上手时机。",
    "快攻": "快攻：加血+护盾+清场，拖入中后期。",
    "控制/中速": "控制中速：用 OTK 在其成型前斩杀，拆支援区。",
    "其他/未分类": "—",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_BASE}/{path}{query}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.load(resp)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_raw() -> dict[str, list[dict[str, Any]]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, (filename, key, params) in API_ENDPOINTS.items():
        payload = _api_get(name, params)
        (RAW_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        out[name] = list(payload.get("data", {}).get(key, []) or [])
    # Extra context (not decks, but useful provenance).
    labels = _api_get("labels")
    (RAW_DIR / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    filter_payload = _api_get("match_pool_filter")
    (RAW_DIR / "match_pool_filter.json").write_text(
        json.dumps(filter_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def _assets():
    from agents.scripted.assets import load_assets

    return load_assets()


def _decode(share_code: str, assets) -> tuple[tuple[int, ...], tuple[int, ...]]:
    from agents.scripted.codec import decode_share_code

    return decode_share_code(share_code, assets.share_to_id)


def _card_summary(cards: tuple[int, ...], assets) -> list[str]:
    """Compact 'package' summary: full (2x) equipment/talents + notable supports."""
    from collections import Counter

    counts = Counter(cards)
    equipment: list[str] = []
    supports: list[str] = []
    for cid, n in counts.items():
        card = assets.cards.get(cid)
        if card is None:
            continue
        tags = set(card.tags)
        label = f"{card.name}×{n}"
        if {"GCG_TAG_WEAPON", "GCG_TAG_ARTIFACT", "GCG_TAG_TALENT"} & tags:
            equipment.append(label)
        elif {"GCG_TAG_ALLY", "GCG_TAG_PLACE", "GCG_TAG_ITEM"} & tags:
            supports.append(label)
    equipment.sort(key=lambda s: -int(s.rsplit("×", 1)[1]))
    supports.sort(key=lambda s: -int(s.rsplit("×", 1)[1]))
    return equipment[:5] + supports[:4]


def _slug_sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def build_entries(
    api_data: dict[str, list[dict[str, Any]]], assets
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_codes: dict[str, str] = {}

    def add(
        *,
        slug: str,
        title: str,
        share_code: str,
        source: str,
        source_url: str,
        author: str = "",
        created_at: int | None = None,
        tags: list[str] | None = None,
        guide: str = "",
        stats: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not share_code:
            return
        if share_code in seen_codes:
            # Duplicate share code across endpoints: keep the first, record alias.
            entries[[e["slug"] for e in entries].index(seen_codes[share_code])].setdefault(
                "also_seen_in", []
            ).append(slug)
            return
        record: dict[str, Any] = {
            "slug": slug,
            "title": title,
            "source": source,
            "source_url": source_url,
            "author": author,
            "created_at": created_at,
            "tags": tags or [],
            "share_code": share_code,
            "guide": guide,
            "stats": stats or {},
        }
        if extra:
            record.update(extra)
        try:
            chars, cards = _decode(share_code, assets)
            unknown = [
                c for c in chars if c not in assets.characters
            ] + [c for c in cards if c not in assets.cards]
            char_names = [assets.character(c).name for c in chars]
            record.update(
                {
                    "characters": list(chars),
                    "cards": list(cards),
                    "character_names": char_names,
                    "card_names": [assets.card(c).name for c in cards],
                    "core_cards": _card_summary(cards, assets),
                    "archetype": label_archetype(char_names, record["tags"], title),
                }
            )
            record["validation"] = {
                "share_code_decode": True,
                "deck_shape": f"{len(chars)}+{len(cards)}",
                "deck_shape_ok": len(chars) == 3 and len(cards) == 30,
                "ids_known": not unknown,
                "unknown_ids": unknown,
            }
            if not (len(chars) == 3 and len(cards) == 30 and not unknown):
                record["validation"]["error"] = "illegal deck shape/ids"
        except Exception as exc:  # noqa: BLE001
            record["validation"] = {
                "share_code_decode": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            record.update(
                {"characters": [], "cards": [], "character_names": [], "archetype": "invalid"}
            )
        seen_codes[share_code] = slug
        entries.append(record)

    # 1) miHoYo API decks.
    for idx, d in enumerate(api_data.get("match_pool_list", []), start=1):
        add(
            slug=f"cs-mp-{idx:02d}",
            title=str(d.get("title", "")),
            share_code=str(d.get("card_code", "")),
            source="mihoyo_cardsquare/match_pool_list",
            source_url=SPA_URL,
            author=str(d.get("nickname", "")),
            created_at=d.get("created_at"),
            tags=list(d.get("tags", []) or []),
            guide=str(d.get("desc", "")),
            stats={
                "like": d.get("like_cnt"),
                "favour": d.get("favour_cnt"),
                "view": d.get("view_cnt"),
                "miyoushe_id": d.get("id"),
                "label_ids": d.get("label_ids"),
            },
            extra={"raw_endpoint_id": d.get("id")},
        )
    for idx, d in enumerate(api_data.get("index_recommend", []), start=1):
        add(
            slug=f"cs-rec-{idx}",
            title=str(d.get("title", "")),
            share_code=str(d.get("card_code", "")),
            source="mihoyo_cardsquare/index_recommend",
            source_url=SPA_URL,
            author=str(d.get("nickname", "")),
            created_at=d.get("created_at"),
            tags=list(d.get("tags", []) or []),
            guide=str(d.get("desc", "")),
            stats={
                "like": d.get("like_cnt"),
                "favour": d.get("favour_cnt"),
                "view": d.get("view_cnt"),
                "miyoushe_id": d.get("id"),
                "label_ids": d.get("label_ids"),
            },
            extra={"raw_endpoint_id": d.get("id")},
        )
    for idx, d in enumerate(api_data.get("honor_match_pool", []), start=1):
        add(
            slug=f"cs-honor-{idx}",
            title=str(d.get("title", "")),
            share_code=str(d.get("card_code", "")),
            source="mihoyo_cardsquare/honor_match_pool",
            source_url=SPA_URL,
            author=str(d.get("nickname", "")),
            created_at=d.get("created_at"),
            tags=list(d.get("tags", []) or []),
            guide=str(d.get("desc", "")),
            stats={"miyoushe_id": d.get("id"), "label_ids": d.get("label_ids")},
            extra={"raw_endpoint_id": d.get("id")},
        )

    # 2) Hand-curated web decks.
    if WEB_SOURCES_FILE.exists():
        web = json.loads(WEB_SOURCES_FILE.read_text(encoding="utf-8"))
        for idx, d in enumerate(web, start=1):
            slug = d.get("slug") or f"web-{idx:02d}"
            add(
                slug=slug,
                title=str(d.get("title", "")),
                share_code=str(d.get("share_code", "")).strip(),
                source=str(d.get("source", "web")),
                source_url=str(d.get("source_url", "")),
                author=str(d.get("author", "")),
                created_at=d.get("created_at"),
                tags=list(d.get("tags", []) or []),
                guide=str(d.get("note", "")),
                stats={"source_confidence": d.get("confidence", "medium")},
                extra={"source_title": d.get("source_title", "")},
            )
    return entries


def run_engine_validation(
    entries: list[dict[str, Any]], *, opponent_slug: str = "superconduct_aggro", seed: int = 7
) -> None:
    """Play one scripted game per deck (deck-under-test = heuristic, opponent = random)."""
    from agents.scripted.assets import load_assets  # noqa: F401
    from envs.decks import deck_spec
    from envs.match import run_match
    from envs.policy import heuristic_policy, legal_random_policy
    from reps.schema import DeckSpec

    assets = _assets()
    del assets
    opponent = deck_spec(opponent_slug)
    for i, entry in enumerate(entries, start=1):
        if not entry.get("characters"):
            continue
        spec = DeckSpec(
            name=entry["slug"],
            characters=tuple(entry["characters"]),
            cards=tuple(entry["cards"]),
        )
        start = time.perf_counter()
        try:
            record = run_match(
                spec,
                opponent,
                heuristic_policy(),
                legal_random_policy(seed),
                seed=seed,
                max_steps=4000,
            )
            entry.setdefault("validation", {})["engine_game"] = {
                "ok": record.error is None and not record.truncated,
                "winner": record.winner,
                "rounds": record.rounds,
                "error": record.error,
                "truncated": record.truncated,
                "opponent": opponent_slug,
                "seed": seed,
                "wall_seconds": round(time.perf_counter() - start, 2),
                "policy_errors0": list(record.io_errors0),
            }
        except Exception as exc:  # noqa: BLE001
            entry.setdefault("validation", {})["engine_game"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "opponent": opponent_slug,
                "seed": seed,
            }
        status = entry["validation"]["engine_game"].get("ok")
        print(f"  [{i}/{len(entries)}] {entry['slug']:12s} engine_ok={status}")


def _compact_corpus(corpus: dict[str, Any]) -> dict[str, Any]:
    """Small committed copy: share codes + labels + validation, no long guides."""
    keep = (
        "slug",
        "title",
        "source",
        "source_url",
        "source_title",
        "author",
        "created_at",
        "tags",
        "share_code",
        "characters",
        "cards",
        "character_names",
        "archetype",
        "core_cards",
        "confidence",
    )
    decks = []
    for entry in corpus["decks"]:
        compact = {k: entry[k] for k in keep if k in entry}
        validation = entry.get("validation", {})
        compact["validation"] = {
            "share_code_decode": validation.get("share_code_decode"),
            "deck_shape": validation.get("deck_shape"),
            "deck_shape_ok": validation.get("deck_shape_ok"),
            "ids_known": validation.get("ids_known"),
        }
        game = validation.get("engine_game")
        if game:
            compact["validation"]["engine_game_ok"] = game.get("ok")
            compact["validation"]["engine_game_error"] = game.get("error")
        guide = entry.get("guide") or ""
        if guide:
            compact["guide_excerpt"] = guide[:140]
        decks.append(compact)
    return {**{k: v for k, v in corpus.items() if k != "decks"}, "decks": decks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="store_true", help="run 1 scripted game per deck")
    parser.add_argument("--no-fetch", action="store_true", help="use cached raw JSON only")
    parser.add_argument(
        "--export-reports",
        action="store_true",
        help="also write a compact committed copy to reports/explore/deck_corpus.json",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="only re-export the existing data/deck_corpus/decks.json to reports/",
    )
    parser.add_argument(
        "--relabel-only",
        action="store_true",
        help="recompute archetype labels in-place (keeps engine validation results)",
    )
    args = parser.parse_args()

    if args.relabel_only:
        corpus = json.loads((DATA_DIR / "decks.json").read_text(encoding="utf-8"))
        for entry in corpus["decks"]:
            if entry.get("character_names"):
                entry["archetype"] = label_archetype(
                    entry["character_names"], entry.get("tags", []), entry["title"]
                )
        (DATA_DIR / "decks.json").write_text(
            json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("relabeled data/deck_corpus/decks.json")
        return

    if args.export_only:
        corpus = json.loads((DATA_DIR / "decks.json").read_text(encoding="utf-8"))
        reports_path = REPO_ROOT / "reports" / "explore" / "deck_corpus.json"
        reports_path.parent.mkdir(parents=True, exist_ok=True)
        reports_path.write_text(
            json.dumps(_compact_corpus(corpus), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"wrote {reports_path} ({reports_path.stat().st_size} bytes)")
        return

    if args.no_fetch:
        api_data = {}
        for name, (filename, key, _) in API_ENDPOINTS.items():
            payload = json.loads((RAW_DIR / filename).read_text(encoding="utf-8"))
            api_data[name] = list(payload.get("data", {}).get(key, []) or [])
    else:
        api_data = fetch_raw()

    assets = _assets()
    entries = build_entries(api_data, assets)
    valid = [e for e in entries if e.get("validation", {}).get("deck_shape_ok")]
    invalid = [e for e in entries if e not in valid]

    if args.engine:
        print(f"running engine validation for {len(valid)} decks ...")
        run_engine_validation(valid)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = {
        "generated_at": _now_iso(),
        "generator": "scripts/explore/fetch_deck_corpus.py",
        "sources": {
            "mihoyo_cardsquare": {
                "spa": SPA_URL,
                "api_base": API_BASE,
                "endpoints": {k: v[0] for k, v in API_ENDPOINTS.items()},
            },
            "web_guides": str(WEB_SOURCES_FILE.relative_to(REPO_ROOT)),
        },
        "counts": {
            "total": len(entries),
            "validated": len(valid),
            "invalid": len(invalid),
            "engine_validated": sum(
                1 for e in valid if e.get("validation", {}).get("engine_game")
            ),
            "engine_ok": sum(
                1
                for e in valid
                if e.get("validation", {}).get("engine_game", {}).get("ok")
            ),
        },
        "decks": entries,
    }
    out = DATA_DIR / "decks.json"
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(entries)} decks; {len(valid)} validated)")

    if args.export_reports:
        reports_path = REPO_ROOT / "reports" / "explore" / "deck_corpus.json"
        reports_path.parent.mkdir(parents=True, exist_ok=True)
        reports_path.write_text(
            json.dumps(_compact_corpus(corpus), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"wrote {reports_path} ({reports_path.stat().st_size} bytes)")

    for e in invalid:
        print("  INVALID:", e["slug"], e.get("validation"))


if __name__ == "__main__":
    main()
