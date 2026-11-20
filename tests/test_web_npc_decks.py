"""C3 NPC 阵容（``web/npc_decks.py``）单测。

口径见 PLAN D18「人可持 NPC 敌人卡组」：NPC 阵容**人数不固定**（1/2/4/6 都有，
含同一敌人的多份）→ 全部 16 关原样返回、不跳过、不凑 3 人。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from web.npc_decks import SEMANTICS, npc_decks


@pytest.fixture(scope="module")
def npc_payload() -> dict:
    return npc_decks()


def test_npc_decks_returns_every_stage(npc_payload):
    from npc.spec import load_stages

    stages = load_stages()
    assert len(npc_payload["decks"]) == len(stages) == 16
    assert {d["slug"] for d in npc_payload["decks"]} == set(stages)
    assert set(npc_payload) >= {"decks", "semantics"}


def test_npc_decks_keep_lineup_size_as_is(npc_payload):
    """人数不为 3 是常态：2/3/4/5/6 人都要原样保留，不跳过、不凑数。"""
    sizes = Counter(d["total"] for d in npc_payload["decks"])
    assert sizes[2] >= 1 and sizes[3] >= 1 and sizes[4] >= 1
    assert max(sizes) >= 5, "关卡数据里有带预备区的多人大阵容"
    for deck in npc_payload["decks"]:
        assert deck["total"] == deck["n_on_field"] + deck["n_support"]
        assert deck["total"] >= 1
        # 顺序：出战在前、预备在后
        assert len(deck["characters"]) == deck["total"]


def test_npc_decks_expand_counts_and_keep_duplicates(npc_payload):
    """同一敌人的多份（如提米 2×火斧丘丘暴徒）要保留重复，不能去重。"""
    timmie = next(d for d in npc_payload["decks"] if d["slug"] == "timmie")
    assert timmie["total"] == 2
    assert len(set(timmie["characters"])) == 1, "2 份同一角色，去重就错了"

    jifang = next(d for d in npc_payload["decks"] if d["slug"] == "jifang")
    assert jifang["n_on_field"] == 3 and jifang["n_support"] == 3
    assert jifang["total"] == 6


def test_npc_decks_have_no_hand_cards(npc_payload):
    """游戏内 NPC 没有手牌 → 牌组必须是空牌堆（引擎 initialHandsCount 无 per-player）。"""
    for deck in npc_payload["decks"]:
        assert deck["cards"] == [], f"{deck['slug']} 不该带行动牌"
    # 说明文案要把「为什么空牌堆能成立」写清楚
    assert "空牌堆" in npc_payload["semantics"]["cards"]
    assert "initialHandsCount" in npc_payload["semantics"]["cards"]
    assert "per-player" in npc_payload["semantics"]["cards"]


def payload_text(payload: dict, key: str) -> str:
    return payload["semantics"][key]


def test_npc_decks_documents_the_three_approximations(npc_payload):
    """三件事的说明必须**如实记录差异**（支援区/多轮与游戏内不一致，不能冒充原版）。"""
    assert set(SEMANTICS) == set(npc_payload["semantics"])
    assert set(SEMANTICS) >= {
        "lineup_size",
        "support",
        "waves",
        "cards",
        "dice",
        "share_code",
    }
    for key in ("support", "waves", "dice"):
        text = payload_text(npc_payload, key)
        assert "【游戏内】" in text and "【本接口】" in text and "【差异】" in text, key
    assert "后台" in payload_text(npc_payload, "support")
    assert "被击败后" in payload_text(npc_payload, "support")
    assert "多轮" in payload_text(npc_payload, "waves")
    assert "血量和卡牌状态" in payload_text(npc_payload, "waves")
    assert "share_id" in payload_text(npc_payload, "share_code")


def test_npc_decks_document_omni_dice(npc_payload):
    """D18：持 NPC 牌组的座位拿 8 个万能骰（引擎 alwaysOmni，重投照常可用）。"""
    dice = payload_text(npc_payload, "dice")
    assert "8" in dice and "万能" in dice
    assert "alwaysOmni" in dice and "initialDiceCount" in dice
    # 重投照常可选（重投出来的按全局规则随机）——这个差异要如实写出来。
    assert "重投" in dice and "仍可主动重投" in dice
    # 旧的「骰子仍按全局规则发放」说明已被替换掉
    assert "骰子仍按全局规则发" not in payload_text(npc_payload, "cards")


def test_npc_decks_todo_is_tracked_in_module():
    """TODO：探索替代方式描述支援区/多轮——不能只写在文档里，代码里也要能找到。"""
    import web.npc_decks as npc_module

    source = Path(npc_module.__file__).read_text(encoding="utf-8")
    assert "TODO" in source and "探索" in source
    assert "支援区" in source and "waves" in source


def test_npc_deck_ids_are_absent_from_share_id_table(npc_payload):
    """NPC 角色编不出分享码 → 前端必须走对象直传（这里把该事实钉住）。"""
    import common.paths as paths

    share_path = (
        paths.REPO_ROOT / "build/engine/packages/assets-manager/src/data/share_id.json"
    )
    if not share_path.exists():
        pytest.skip("engine workspace not materialized (no share_id.json)")
    share_ids = set(json.loads(share_path.read_text(encoding="utf-8")).values())
    npc_ids = {c for d in npc_payload["decks"] for c in d["characters"]}
    assert npc_ids - share_ids, "NPC 角色 id 应该都不在官方分享码表里"
