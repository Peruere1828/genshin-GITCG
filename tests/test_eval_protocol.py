"""Statistics/aggregation for the strength-evaluation protocol (no engine).

Covers the engine-free half of ``reports/explore/eval_protocol.md``: Wilson and
cluster CIs, per-opponent stratification + floor, go/no-go gates, paired-seed
differences, the fixed-anchor Bradley-Terry rating, the best-response probe,
sample-size planning, and the task-enrichment mapping used by the runner.
"""

from __future__ import annotations

import math

import pytest

from eval.scoreboard import (
    MatchOutcome,
    anchored_bt_rating,
    best_response_probe,
    cluster_bootstrap_ci,
    detection_sample_size,
    evaluate_gates,
    floor_sample_size,
    noninferiority_gate,
    paired_difference,
    sample_size_for_halfwidth,
    sample_size_table,
    summarize,
    win_probability,
)
from scripts.eval_protocol import (
    ProtocolSpec,
    ProtocolTask,
    _enrich,
    outcomes_from_rows,
)


def outcome(
    opponent: str,
    seed: int,
    side: int,
    score: float,
    *,
    rounds: int = 6,
    decisions: int = 10,
    fallbacks: int = 0,
    error: str | None = None,
    truncated: bool = False,
) -> MatchOutcome:
    return MatchOutcome(
        opponent=opponent,
        seed=seed,
        side=side,
        score=score,
        rounds=rounds,
        decisions=decisions,
        fallbacks=fallbacks,
        error=error,
        truncated=truncated,
    )


def _block(opponent: str, wins: int, losses: int, *, side: int = 0) -> list[MatchOutcome]:
    rows = [outcome(opponent, i, side, 1.0) for i in range(wins)]
    rows += [outcome(opponent, 100 + i, side, 0.0) for i in range(losses)]
    return rows


# --------------------------------------------------------------------------- #
# summary / stratification / gates
# --------------------------------------------------------------------------- #
def test_summarize_pooled_and_per_opponent():
    rows = _block("a", 9, 1) + _block("b", 5, 5)
    summary = summarize(rows)
    assert summary["games"] == 20
    assert math.isclose(summary["pooled_rate"], 14 / 20)
    assert set(summary["per_opponent"]) == {"a", "b"}
    assert math.isclose(summary["per_opponent"]["a"]["rate"], 0.9)
    assert math.isclose(summary["per_opponent"]["b"]["rate"], 0.5)
    # Cluster mean is the *unweighted* mean of per-opponent rates.
    assert math.isclose(summary["cluster_mean"], 0.7)
    # Pooled Wilson interval brackets the point estimate.
    assert summary["pooled_ci_low"] <= summary["pooled_rate"] <= summary["pooled_ci_high"]


def test_summarize_excludes_errors_and_truncated():
    rows = _block("a", 3, 1) + [
        outcome("a", 50, 0, 0.0, error="boom"),
        outcome("a", 51, 0, 1.0, truncated=True),
    ]
    summary = summarize(rows)
    assert summary["games"] == 4  # only clean games count
    assert summary["quality"]["errors"] == 1
    assert summary["quality"]["truncated"] == 1
    assert summary["quality"]["invalid"] == 2
    assert math.isclose(summary["per_opponent"]["a"]["rate"], 0.75)


def test_floor_violations_and_quality_proxies():
    rows = _block("strong", 9, 1) + _block("weak", 2, 8)
    rows[0] = outcome("strong", 0, 0, 1.0, decisions=10, fallbacks=1)
    summary = summarize(rows, floor=0.45)
    assert summary["floor_violations"] == ["weak"]
    quality = summary["quality"]
    assert quality["fallbacks"] == 1
    assert quality["decisions"] == 200
    assert math.isclose(quality["fallback_rate"], 1 / 200)
    assert math.isclose(quality["illegal_rate"], 1 / 200)


def test_evaluate_gates_screening_and_strict():
    strong = summarize(_block("a", 8, 2) + _block("b", 7, 3), floor=0.45)
    gates = evaluate_gates(strong, pooled_threshold=0.5)
    assert gates["pooled_pass"] and gates["floor_pass"] and gates["pass"]
    # Strict CI cannot certify on this tiny sample.
    strict = evaluate_gates(strong, pooled_threshold=0.5, strict_ci=True)
    assert strict["pooled_pass"] is False
    weak = summarize(_block("a", 4, 6), floor=0.45)
    assert evaluate_gates(weak, pooled_threshold=0.5)["pooled_pass"] is False


# --------------------------------------------------------------------------- #
# paired comparison
# --------------------------------------------------------------------------- #
def test_paired_difference_uses_shared_keys_only():
    a = _block("a", 6, 4) + [outcome("b", 0, 0, 1.0)]
    b = _block("a", 6, 4) + [outcome("b", 7, 0, 0.0)]  # disjoint key for b
    paired = paired_difference(a, b)
    assert paired is not None
    assert paired["n_pairs"] == 10
    assert math.isclose(paired["delta"], 0.0)
    assert paired["per_opponent"] == {"a": {"n": 10, "delta": 0.0}}


def test_paired_difference_detects_consistent_improvement():
    b = _block("a", 5, 5)  # wins seeds 0-4, loses seeds 100-104
    keys = [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
    a = [outcome("a", i, 0, 1.0) for i in keys]  # A wins every shared seed
    paired = paired_difference(a, b)
    assert paired["n_pairs"] == 10
    assert math.isclose(paired["delta"], 0.5)  # flips the five B losses
    assert paired["ci_low"] > 0.0
    assert paired["significant_positive"] is True


def test_paired_difference_none_when_disjoint():
    a = _block("a", 1, 1)
    b = _block("b", 1, 1)
    assert paired_difference(a, b) is None


def test_noninferiority_gate():
    assert noninferiority_gate({"ci_low": 0.01}, tolerance=0.02)["accept"] is True
    assert noninferiority_gate({"ci_low": -0.05}, tolerance=0.02)["accept"] is False
    # A tiny tolerated regression is still accepted (noise guard).
    assert noninferiority_gate({"ci_low": -0.01}, tolerance=0.02)["accept"] is True


# --------------------------------------------------------------------------- #
# fixed-anchor rating + best-response probe
# --------------------------------------------------------------------------- #
def test_win_probability_symmetry():
    assert math.isclose(win_probability(1500, 1500), 0.5)
    assert win_probability(1700, 1500) > 0.5
    assert math.isclose(win_probability(1700, 1500), 1 - win_probability(1500, 1700))


def test_anchored_rating_reacts_to_wins():
    anchors = {"strong": 1800.0, "weak": 1200.0}
    winners = [outcome("strong", i, 0, 1.0) for i in range(10)] + [
        outcome("weak", i, 0, 1.0) for i in range(10)
    ]
    losers = [outcome("strong", i, 0, 0.0) for i in range(10)] + [
        outcome("weak", i, 0, 0.0) for i in range(10)
    ]
    good = anchored_bt_rating(winners, anchors, n_boot=200)
    bad = anchored_bt_rating(losers, anchors, n_boot=200)
    assert good["rating"] > bad["rating"]
    assert good["ci_low"] <= good["rating"] <= good["ci_high"]
    assert good["mean_expected_score"] > 0.5


def test_anchored_rating_none_without_anchors():
    rows = _block("a", 2, 2)
    assert anchored_bt_rating(rows, {}) is None


def test_best_response_probe_reports_worst():
    summary = summarize(_block("strong", 9, 1) + _block("weak", 3, 7), floor=0.45)
    probe = best_response_probe(summary["per_opponent"], floor=0.45)
    assert probe["worst_opponent"] == "weak"
    assert math.isclose(probe["exploitability_lower_bound"], 0.7)
    assert math.isclose(probe["matchup_gap"], 0.3)
    assert probe["worst_clear_floor"] is False


# --------------------------------------------------------------------------- #
# sample-size planning
# --------------------------------------------------------------------------- #
def test_sample_size_for_halfwidth_matches_textbook():
    assert sample_size_for_halfwidth(0.10) == 97  # ceil(1.96^2 * 0.25 / 0.01)
    assert sample_size_for_halfwidth(0.05) == 385
    # Smaller target half-width needs more games.
    assert sample_size_for_halfwidth(0.02) > sample_size_for_halfwidth(0.05)


def test_sample_size_table_shape():
    table = sample_size_table(pool_size=20)
    assert len(table) == 7
    assert all(row["games_per_opponent"] >= 1 for row in table)
    assert table[0]["total_games"] < table[-1]["total_games"]


def test_floor_and_detection_sample_sizes_monotone():
    assert floor_sample_size(0.60) < floor_sample_size(0.50)
    assert detection_sample_size(0.10) < detection_sample_size(0.02)


def test_cluster_bootstrap_brackets_mean():
    values = [0.9, 0.5, 0.7, 0.4, 0.6]
    mean, low, high = cluster_bootstrap_ci(values, n_boot=500, seed=7)
    assert math.isclose(mean, sum(values) / len(values))
    assert low <= mean <= high


# --------------------------------------------------------------------------- #
# runner plumbing (no engine)
# --------------------------------------------------------------------------- #
def _spec(**overrides) -> ProtocolSpec:
    base = dict(
        contestant="expert:superconduct_aggro",
        contestant_deck="superconduct_aggro",
        opponents=("natlan_battleship",),
        seeds=(0, 1),
        sides=(0, 1),
    )
    base.update(overrides)
    return ProtocolSpec(**base)


def test_protocol_task_side_assignment():
    spec = _spec()
    tasks = {t.side: t for t in spec.tasks()}
    side0 = tasks[0].to_match(spec)
    assert side0.deck0 == "superconduct_aggro" and side0.deck1 == "natlan_battleship"
    assert side0.policy0 == "expert:superconduct_aggro"
    assert side0.policy1 == "expert:natlan_battleship"
    side1 = tasks[1].to_match(spec)
    assert side1.deck0 == "natlan_battleship" and side1.deck1 == "superconduct_aggro"
    assert side1.policy0 == "expert:natlan_battleship"
    assert side1.policy1 == "expert:superconduct_aggro"


def test_fingerprint_changes_with_spec():
    assert _spec().fingerprint() != _spec(seeds=(0, 1, 2)).fingerprint()


def test_enrich_score_and_outcomes_roundtrip():
    spec = _spec()
    task = ProtocolTask(index=0, opponent="natlan_battleship", seed=3, side=0)
    won = _enrich(
        {"task_index": 0, "winner": 0, "rounds": 7, "decisions0": 12, "fallbacks0": 0,
         "decisions1": 11, "fallbacks1": 0},
        task,
        spec,
    )
    assert won["score"] == 1.0
    assert won["decisions"] == 12
    lost_side1 = ProtocolTask(index=1, opponent="natlan_battleship", seed=4, side=1)
    lost = _enrich({"task_index": 1, "winner": 0, "rounds": 8,
                    "decisions0": 9, "fallbacks0": 0, "decisions1": 13, "fallbacks1": 0},
                   lost_side1, spec)
    assert lost["score"] == 0.0
    assert lost["decisions"] == 13

    outcomes = outcomes_from_rows([won, lost])
    assert outcomes[0].key == ("natlan_battleship", 3, 0)
    assert outcomes[0].score == 1.0
    assert outcomes[1].score == 0.0
