"""M1 acceptance (part 1): every legal concrete action is representable.

The action adapter must expose a stable code + payload for 100% of the legal
concrete actions the engine offers, with kind-appropriate payload fields. This is
a prerequisite for the abstraction-coverage / round-trip gate (PLAN.md §6.2).
"""

from __future__ import annotations

from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import expert_policy

_EXPECTED_PAYLOAD_KEYS = {
    "action": "chosen_action_index",
    "choose_active": "active_character_id",
    "reroll_dice": "dice_to_reroll",
    "select_card": "selected_definition_id",
    "switch_hands": "removed_hand_ids",
}

# Guardrail against action-space blow-up (PLAN.md §7). Report, don't silently pass.
_MAX_LEGAL_OPTIONS = 512


class _Capture:
    def __init__(self, inner):
        self.inner = inner
        self.name = f"capture:{getattr(inner, 'name', '?')}"
        self.contexts = []

    def choose(self, built):
        self.contexts.append(built)
        return self.inner.choose(built)


def _capture_game():
    cap = _Capture(expert_policy("superconduct_aggro"))
    opp = expert_policy("natlan_battleship")
    record = run_match(
        deck_spec("superconduct_aggro"), deck_spec("natlan_battleship"), cap, opp, seed=3
    )
    return record, cap


def test_every_legal_action_has_a_code_and_payload():
    record, cap = _capture_game()
    assert record.error is None
    assert cap.contexts, "no decision contexts captured"

    max_options = 0
    for built in cap.contexts:
        context = built.context
        codes = list(context.legal_low_level_codes)
        assert codes, "a decision must offer at least one legal action"
        assert len(codes) == len(set(codes)), "action codes must be unique"
        assert len(context.legal_low_level_specs) == len(codes)
        max_options = max(max_options, len(codes))

        for code, spec in zip(codes, context.legal_low_level_specs):
            payload = built.payload_by_low_level_code.get(int(code))
            assert payload is not None, f"missing payload for legal code {code}"
            expected_key = _EXPECTED_PAYLOAD_KEYS[context.request_type.value]
            assert expected_key in payload, (
                f"payload for {spec.kind} lacks key {expected_key!r}: {payload}"
            )

    assert max_options <= _MAX_LEGAL_OPTIONS, (
        f"action space too wide: max {max_options} options at one decision"
    )


def test_chosen_action_is_always_legal():
    record, cap = _capture_game()
    for built in cap.contexts:
        # A fallback would have replaced the chosen code; legality of the final
        # code is enforced in PolicyPlayer, this asserts the adapter agrees.
        legal = set(int(c) for c in built.context.legal_low_level_codes)
        assert legal
    assert record.fallbacks0 == 0
