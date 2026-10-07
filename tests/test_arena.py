"""Arena end-to-end, including the subprocess rollout path."""

from __future__ import annotations

import pytest

from eval.arena import ArenaSpec, run_arena
from eval.opponents import opponent_by_name, scripted_opponents


def _tiny_spec(workers: int) -> ArenaSpec:
    pool = (opponent_by_name("superconduct_aggro"), opponent_by_name("natlan_battleship"))
    return ArenaSpec(contestants=pool, pool=pool, seeds=(0, 1), workers=workers, tag="test")


def test_scripted_registry_is_populated():
    opponents = scripted_opponents()
    assert len(opponents) >= 20
    assert all(opp.policy.startswith("expert:") for opp in opponents)


@pytest.mark.slow
def test_arena_sequential_summary():
    result = run_arena(_tiny_spec(workers=1))
    assert len(result.matches) == 8  # 2 contestants x 2 pool x 2 seeds
    assert all(m["error"] is None for m in result.matches)
    summary = result.summary()
    assert summary["n_matches"] == 8
    assert summary["contestants"]
    assert set(summary["elo"]) >= {"superconduct_aggro", "natlan_battleship"}


@pytest.mark.slow
def test_arena_multiprocess_matches_sequential():
    sequential = run_arena(_tiny_spec(workers=1))
    parallel = run_arena(_tiny_spec(workers=2))
    assert len(parallel.matches) == len(sequential.matches)
    # Deterministic tasks: same (deck, seed) yields the same winner regardless of
    # which process ran it.
    key = lambda m: (m["deck0"], m["deck1"], m["seed"])  # noqa: E731
    seq = {key(m): m["winner"] for m in sequential.matches}
    par = {key(m): m["winner"] for m in parallel.matches}
    assert seq == par
