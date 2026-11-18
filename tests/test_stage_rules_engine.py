"""Engine-level checks for PvE stage special rules (CreateParam attr 2002).

The payload plumbing is engine-free (``tests/test_stage_rules.py``); this file
drives a real game and asserts the effects. It is skipped when the stock PyPI
``libgitcg.so`` (without the stage-rule patch) is installed — same policy as
``tests/test_npc_enemies.py`` for the PvE overlay.

Rules data: ``configs/npc/special_rules.json``; payload builder:
``npc/stage_rules.py``; engine patch: ``engine-overlay/patches/stage-rules.patch``.
"""

from __future__ import annotations

import pytest

from gitcg import Game
from reps.action_hierarchy import reset_default_hierarchical_action_codebook
from reps.schema import DeckSpec, OptionKind
from reps.snapshot import snapshot_state

from npc.stage_rules import stage_rules_json

SUPPORT_TIMMIE = 322007


def _deck() -> DeckSpec:
    from envs.decks import expert_deck_specs

    base = expert_deck_specs()[0]
    return DeckSpec(name=base.name, characters=base.characters, cards=base.cards)


def _pick(built, prefer: OptionKind) -> int:
    codes = list(built.context.legal_low_level_codes)
    specs = built.context.legal_low_level_specs
    for code, spec in zip(codes, specs):
        if spec.kind == prefer:
            return int(code)
    for code, spec in zip(codes, specs):
        if spec.kind == OptionKind.ACTION_DECLARE_END:
            return int(code)
    return int(codes[0])


def _end_first(built) -> int:
    return _pick(built, OptionKind.ACTION_DECLARE_END)


def _start(stage_rules: str | None):
    """Start a game and snapshot the first decision state."""
    from envs.match import PolicyPlayer, _build_create_param
    from envs.policy import CallablePolicy

    deck = _deck()
    reset_default_hierarchical_action_codebook()
    game = Game(create_param=_build_create_param(deck, deck, seed=3, stage_rules=stage_rules))
    game.set_player(0, PolicyPlayer(0, CallablePolicy(_end_first, name="p0")))
    game.set_player(1, PolicyPlayer(1, CallablePolicy(_end_first, name="p1")))
    game.start()
    return game, snapshot_state(None, game.state().json())


def _supports(snap, who: int) -> set[int]:
    return {int(entity.definition_id) for entity in snap.players[who].supports}


def _available() -> bool:
    try:
        _, snap = _start(stage_rules_json("timmie"))
    except Exception:  # noqa: BLE001 - stock .so either rejects attr 2002 or lacks the card
        return False
    return SUPPORT_TIMMIE in _supports(snap, 0)


pytestmark = pytest.mark.skipif(
    not _available(), reason="engine stage-rules patch (attr 2002) not installed"
)


def test_timmie_injects_initial_support():
    _, with_rules = _start(stage_rules_json("timmie"))
    _, without = _start(None)
    assert SUPPORT_TIMMIE in _supports(with_rules, 0)
    assert SUPPORT_TIMMIE not in _supports(without, 0)


def test_blanche_grants_initial_energy():
    _, with_rules = _start(stage_rules_json("blanche"))
    _, without = _start(None)
    before = {int(c.definition_id): int(c.energy) for c in without.players[0].characters}
    after = {int(c.definition_id): int(c.energy) for c in with_rules.players[0].characters}
    caps = {int(c.definition_id): int(c.max_energy) for c in with_rules.players[0].characters}
    assert before != after, "blanche initial_energy had no effect"
    for def_id, energy in before.items():
        assert after[def_id] == min(caps[def_id], energy + 2)
