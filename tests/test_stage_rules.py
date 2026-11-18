"""Stage special-rule payload + CreateParam plumbing (engine attr 2002).

Engine-free: the CreateParam test swaps in a fake so no game is built. The
engine-side behaviour (initial supports, start-of-battle effects) is covered by
``tests/test_npc_enemies.py`` and the real-game smoke in
``docs/howto/npc-pve.md``.
"""

from __future__ import annotations

import json

from reps.schema import DeckSpec

import envs.match as match_module
from envs.match import ATTR_CREATEPARAM_STAGE_RULES, _build_create_param
from npc.stage_rules import (
    build_stage_rules,
    load_rules,
    stage_rules_json,
    verify_rules,
)

SUPPORT_STAGES = {
    "timmie",
    "wagner",
    "timaeus",
    "ellin",
    "buyun",
    "liusu",
    "irontonguetian",
    "chefmao",
    "uncletian",
    "lan",
}
EFFECT_STAGES = {"blanche", "patton", "liusu", "shitou", "chefmao"}
EMPTY_STAGES = {"swan", "marjorie", "jifang"}


def _deck(name: str) -> DeckSpec:
    return DeckSpec(name=name, characters=(1, 2, 3), cards=tuple(range(1, 31)))


class _FakeCreateParam:
    def __init__(self, version: str | None = None) -> None:
        self.version = version
        self.characters: dict[int, list[int]] = {}
        self.cards: dict[int, list[int]] = {}
        self.attrs: dict[int, int | str] = {}

    def set_characters(self, who: int, characters: list[int]) -> None:
        self.characters[int(who)] = [int(c) for c in characters]

    def set_cards(self, who: int, cards: list[int]) -> None:
        self.cards[int(who)] = [int(c) for c in cards]

    def set_attr(self, key: int, value: int | str) -> None:
        self.attrs[int(key)] = value


def test_all_16_stages_load_and_match_the_stage_specs():
    assert len(load_rules()) == 16
    report = verify_rules()
    assert report["stage_specs"] == report["rules"] == 16
    assert report["errors"] == []


def test_payload_coverage_matches_the_authoritative_data():
    rules = load_rules()
    assert {name for name, rule in rules.items() if rule.supports} == SUPPORT_STAGES
    assert {name for name, rule in rules.items() if rule.effects} == EFFECT_STAGES
    assert {name for name, rule in rules.items() if rule.is_empty} == EMPTY_STAGES
    assert {name for name, rule in rules.items() if rule.via_support_effects} == {
        "wagner",
        "timaeus",
    }


def test_timmie_payload_has_the_starting_support():
    assert build_stage_rules("timmie") == {
        "player": 0,
        "supports": [{"card_id": 322007, "removable": True}],
        "effects": [],
    }


def test_blanche_payload_has_initial_energy():
    assert build_stage_rules("blanche") == {
        "player": 0,
        "supports": [],
        "effects": [{"kind": "initial_energy", "value": 2, "target": "friendly_all"}],
    }


def test_chefmao_payload_combines_support_and_element_application():
    assert build_stage_rules("chefmao") == {
        "player": 0,
        "supports": [{"card_id": 322005, "removable": True}],
        "effects": [
            {"kind": "element_application", "element": "PYRO", "target": "both_active"}
        ],
    }


def test_empty_stages_produce_no_payload_or_attribute():
    for stage in EMPTY_STAGES:
        assert build_stage_rules(stage) is None
        assert stage_rules_json(stage) is None


def test_stage_rules_json_round_trips():
    for stage, rule in load_rules().items():
        text = stage_rules_json(stage)
        if rule.is_empty:
            assert text is None
            continue
        assert text is not None
        payload = json.loads(text)
        assert set(payload) == {"player", "supports", "effects"}
        assert payload == build_stage_rules(stage)


def test_build_create_param_sets_the_stage_rules_string(monkeypatch):
    monkeypatch.setattr(match_module, "CreateParam", _FakeCreateParam)
    payload = stage_rules_json("chefmao")
    assert payload is not None
    create_param = _build_create_param(
        _deck("contestant"),
        _deck("npc"),
        seed=7,
        stage_rules=payload,
    )
    assert ATTR_CREATEPARAM_STAGE_RULES == 2002
    assert create_param.attrs[ATTR_CREATEPARAM_STAGE_RULES] == payload


def test_build_create_param_default_is_unchanged(monkeypatch):
    monkeypatch.setattr(match_module, "CreateParam", _FakeCreateParam)
    create_param = _build_create_param(_deck("contestant"), _deck("npc"), seed=7)
    # The old attribute set is intact and the new one is absent.
    assert set(create_param.attrs) == {2000, 2001, 102}
    assert ATTR_CREATEPARAM_STAGE_RULES not in create_param.attrs


def test_npc_eval_fingerprint_tracks_stage_rules():
    from npc.eval import NpcEvalSpec, _spec_fingerprint

    base = dict(
        stage_names=("timmie",),
        contestant=_deck("d"),
        seeds=(1,),
        policy0="heuristic",
    )
    enabled = NpcEvalSpec(**base)
    disabled = NpcEvalSpec(**base, stage_rules=False)
    assert enabled.stage_rules is True
    assert _spec_fingerprint(enabled) != _spec_fingerprint(disabled)
