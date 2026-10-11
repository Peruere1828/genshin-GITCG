"""Human policy (terminal play): option parsing and quit handling."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.human import HumanPolicy


@dataclass
class _Ctx:
    legal_low_level_codes: tuple[int, ...]
    legal_low_level_specs: tuple = ()
    request_type: str = "action"
    player_view: object = None


@dataclass
class _Built:
    context: _Ctx


class _Inputs:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self, _prompt):
        return self.values.pop(0)


def _policy(values):
    return HumanPolicy(player=1, input_fn=_Inputs(values), output_fn=lambda *_a: None)


def test_human_policy_returns_chosen_code():
    policy = _policy(["1"])
    built = _Built(_Ctx((10, 20, 30)))
    assert policy.choose(built) == 20


def test_human_policy_retries_on_bad_input():
    policy = _policy(["x", "9", "2"])
    built = _Built(_Ctx((10, 20, 30)))
    assert policy.choose(built) == 30


def test_human_policy_quit_raises():
    policy = _policy(["q"])
    with pytest.raises(KeyboardInterrupt):
        policy.choose(_Built(_Ctx((1, 2))))


def test_human_policy_empty_options():
    policy = _policy([])
    assert policy.choose(_Built(_Ctx(()))) == -1
