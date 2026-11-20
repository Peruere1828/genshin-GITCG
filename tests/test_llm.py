"""LLM layer: client plumbing, JSON parsing, assist fallback.

These use a ``MockLLMClient`` (no network) except the ``contexts`` fixture, which
plays one local game to obtain real decision contexts.
"""

from __future__ import annotations

import pytest

from common.llm import MockLLMClient, extract_json_object, redact
from agents.llm_assist import AssistConfig, LLMAssistPolicy
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import expert_policy, _asset_catalog


def test_extract_json_object_variants():
    assert extract_json_object('{"choice": 1, "reason": "x"}') == {"choice": 1, "reason": "x"}
    assert extract_json_object('```json\n{"choice": 2}\n```') == {"choice": 2}
    assert extract_json_object('Sure! {"choice": 3} done') == {"choice": 3}
    assert extract_json_object("no json here") is None
    assert extract_json_object(None) is None


def test_redact_removes_api_keys():
    text = "Authorization: Bearer sk-abcdef1234567890 and sk-x"
    assert "sk-abcdef1234567890" not in redact(text)
    assert "sk-***" in redact(text)


class _Capture:
    def __init__(self, inner):
        self.inner = inner
        self.name = "capture"
        self.contexts = []

    def choose(self, built):
        self.contexts.append(built)
        return self.inner.choose(built)


@pytest.fixture(scope="module")
def contexts():
    cap = _Capture(expert_policy("superconduct_aggro"))
    run_match(
        deck_spec("superconduct_aggro"),
        deck_spec("natlan_battleship"),
        cap,
        expert_policy("natlan_battleship"),
        seed=5,
    )
    action_ctx = [b for b in cap.contexts if b.context.request_type.value == "action"]
    assert action_ctx
    return action_ctx


def _assist(client, *, budget=4, base=None):
    base = base or expert_policy("superconduct_aggro")
    return LLMAssistPolicy(
        base,
        client,
        assets=_asset_catalog(),
        opponent_name="natlan_battleship",
        config=AssistConfig(budget_per_game=budget, min_legal_options=1),
    )


def test_assist_falls_back_on_unparseable_response(contexts):
    client = MockLLMClient(["not json"] * 10)
    policy = _assist(client, budget=2)
    built = contexts[0]
    base_code = policy.base.choose(built)
    code = policy.choose(built)
    assert code == base_code
    assert policy.interventions[-1].error == "unparseable"
    assert policy.interventions[-1].called


def test_assist_accepts_valid_choice_and_respects_budget(contexts):
    client = MockLLMClient(['{"choice": 1, "reason": "alt"}'] * 10)
    policy = _assist(client, budget=2)
    calls = 0
    for built in contexts:
        before = len(policy.interventions)
        policy.choose(built)
        if len(policy.interventions) > before:
            calls += 1
    assert calls == 2  # budget respected
    assert any(iv.changed for iv in policy.interventions)


def test_assist_prompt_has_no_hidden_opponent_cards(contexts):
    client = MockLLMClient(['{"choice": 0}'] * 5)
    policy = _assist(client, budget=1)
    policy.choose(contexts[0])
    prompt = client.prompts[0][1]["content"]
    # Opponent hand is described only by size, never by card names.
    assert "opponent hand size=" in prompt
