"""Render reports/explore/deck_corpus.md from the validated corpus JSON.

Reads ``data/deck_corpus/decks.json`` (preferred) or
``reports/explore/deck_corpus.json`` and writes the committed human summary.

Run::

    python -m scripts.explore.render_deck_report
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decklab import corpus as corpus_mod
from scripts.explore.fetch_deck_corpus import COUNTER_HINTS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "reports" / "explore" / "deck_corpus.md"

SOURCE_TAG = {
    "mihoyo_cardsquare/match_pool_list": "[miHoYo·积分赛]",
    "mihoyo_cardsquare/index_recommend": "[miHoYo·推荐]",
    "mihoyo_cardsquare/honor_match_pool": "[miHoYo·荣誉]",
    "gamersky/handbook": "[gamersky]",
    "bilibili/video-description": "[bilibili]",
    "52pk/aggregator": "[52pk·低可信]",
}

REFERENCE_URLS = {
    "miHoYo 卡牌广场 (SPA)": "https://webstatic.mihoyo.com/ys/event/bbs-lineup-qskp/index.html#/pc/home",
    "miHoYo cardsquare API": "https://api-takumi.mihoyo.com/event/cardsquare/",
    "gamersky 4.3 卡组推荐 (17 套)": "https://www.gamersky.com/handbook/202312/1688434.shtml",
    "bilibili 5.7 双水茜": "https://www.bilibili.com/video/BV1As3DzbEAM/",
    "bilibili 刻雷甘 PVE": "https://www.bilibili.com/video/BV1eo4y1u7s9/",
    "52pk 那维双火 (低可信)": "https://www.52pk.net/news/cidh463.html",
}


def _cell(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "/")


def _validation(entry: dict[str, Any]) -> str:
    v = entry.get("validation", {})
    if not v.get("deck_shape_ok"):
        return "❌ codec"
    game = v.get("engine_game")
    if game is None:
        return "✅ codec"
    return "✅ codec+engine" if game.get("ok") else "✅ codec / ⚠️ engine"


def _source_tag(entry: dict[str, Any]) -> str:
    return SOURCE_TAG.get(entry.get("source", ""), f"[{entry.get('source','?')}]")


def _table(entries: list[dict[str, Any]]) -> str:
    lines = [
        "| # | 分享码 Share code | 验证 | 角色 Characters | 体系 Archetype | 核心/package | 打法要点 | 克制 Counter | 来源 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(entries, start=1):
        chars = "/".join(e.get("character_names", []))
        core = "; ".join(e.get("core_cards", [])[:6])
        plan = _cell(e.get("guide_excerpt") or e.get("guide", ""), 90)
        counter = _cell(COUNTER_HINTS.get(e["archetype"], "—"), 70)
        lines.append(
            f"| {i} | `{e['share_code']}` | {_validation(e)} | {chars} | {e['archetype']} | "
            f"{core} | {plan} | {counter} | {_source_tag(e)} |"
        )
    return "\n".join(lines)


def main() -> None:
    path = corpus_mod.corpus_path()
    if path is None:
        raise SystemExit("no corpus JSON; run scripts.explore.fetch_deck_corpus first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    decks: list[dict[str, Any]] = payload["decks"]
    counts = payload.get("counts", {})

    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    official_decks = [e for e in decks if e.get("source", "").startswith("mihoyo")]
    web_decks = [e for e in decks if not e.get("source", "").startswith("mihoyo")]
    for e in official_decks:
        by_arch[e["archetype"]].append(e)
    for group in by_arch.values():
        group.sort(key=lambda e: (e.get("source", ""), e["slug"]))

    engine_ok = sum(
        1 for e in decks if e.get("validation", {}).get("engine_game", {}).get("ok")
    )

    def _range(entries: list[dict[str, Any]]) -> str:
        stamps = [e["created_at"] for e in entries if e.get("created_at")]
        if not stamps:
            return "n/a"
        return (
            f"{datetime.fromtimestamp(min(stamps), tz=timezone.utc):%Y-%m} → "
            f"{datetime.fromtimestamp(max(stamps), tz=timezone.utc):%Y-%m}"
        )

    pool_decks = [
        e for e in decks if e.get("source", "").endswith("match_pool_list")
    ]
    date_range = _range(pool_decks)
    all_range = _range(decks)
    arch_counts = Counter(e["archetype"] for e in decks)

    parts: list[str] = []
    parts.append(
        f"""# 七圣召唤公开卡组语料 (deck corpus) — 探索任务 1

> 生成：{datetime.now(timezone.utc):%Y-%m-%d}（UTC）；由 `scripts/explore/render_deck_report.py`
> 从验证后的 `data/deck_corpus/decks.json` 渲染。卡组数据/机器可读副本见
> [`reports/explore/deck_corpus.json`](./deck_corpus.json)，采集脚本
> `scripts/explore/fetch_deck_corpus.py`，加载器 `decklab/corpus.py`。

## 0. 摘要 (TL;DR)

- 共 **{len(decks)}** 套公开卡组，**{counts.get('validated', len(decks))}** 套通过分享码解码为合法
  **3 角色 + 30 行动牌**；其中 **{engine_ok}** 套又各跑通 1 局脚本对局（无引擎错误/无截断）。
- 来源：miHoYo 官方「卡牌广场」API（`api-takumi.mihoyo.com/event/cardsquare/`，无需登录）为主，
  外加手工从公开攻略/视频抄录并逐条验证的当前版本卡组。
- 官方积分赛池（49 套）为 **3.7–4.3 赛事归档**（{date_range}）；官方 `index_recommend` 会轮换，
  本次含 2024-07 / 2025-09 条目；全语料时间跨度 {all_range}。
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

- 验证：codec **{counts.get('validated', 0)}/{counts.get('total', 0)}** 通过；engine 再验 **{engine_ok}** 套全过。
- 体系分布（全部 {len(decks)} 套）：

| 体系 Archetype | 套数 |
|---|---|"""
    )
    for arch, n in arch_counts.most_common():
        parts.append(f"| {arch} | {n} |")

    parts.append(
        f"""
## 3. 卡组总表 — 官方卡牌广场 (official corpus, {len(official_decks)} 套, grouped by archetype)

图例：验证 = `✅ codec`（分享码解码为 3+30 合法牌组）/ `✅ codec+engine`（另跑通 1 局）。
核心/package 为「满编装备/天赋 + 主要支援」的紧凑摘要（非完整列表）。打法要点取自作者攻略原文（截断），
克制为该体系的通用应对（体系级建议，非逐张对位）。网络来源的补充卡组见 §4。
"""
    )
    for arch, _n in arch_counts.most_common():
        name = arch
        group = by_arch.get(name)
        if not group:
            continue
        parts.append(f"\n### {name}（{len(group)} 套）\n")
        parts.append(_table(group))

    parts.append(
        f"""
## 4. 当前版本补充 (web-sourced, {len(web_decks)} 套)

这些分享码来自公开攻略/视频，已用引擎解码验证。来源可信度见末列；`52pk` 为 SEO 聚合站，
其攻略文字无法交叉验证，仅分享码本身通过解码。

"""
    )
    parts.append(_table(web_decks))

    parts.append(
        """
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

"""
    )
    for label, url in REFERENCE_URLS.items():
        parts.append(f"- {label}：<{url}>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
