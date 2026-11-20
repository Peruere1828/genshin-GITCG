"""D25 teacher-limited floor exemption in ``eval.scoreboard.evaluate_gates``."""

from __future__ import annotations

from eval.scoreboard import evaluate_gates


def _summary(per_opponent):
    return {
        "pooled_rate": 0.80,
        "cluster_ci_low": 0.72,
        "floor": 0.45,
        "per_opponent": per_opponent,
        "quality": {"fallback_rate": 0.0, "error_rate": 0.0},
        "floor_violations": [],
    }


def _row(rate: float, ci_low: float, games: int = 40):
    return {"games": games, "rate": rate, "ci_low": ci_low}


def test_raw_gate_flags_a_weak_matchup():
    summary = _summary({"weak": _row(0.35, 0.25), "ok": _row(0.70, 0.55)})
    gates = evaluate_gates(summary, strict_ci=True)
    assert gates["floor_pass"] is False
    assert gates["floor_violations"] == ["weak"]
    assert gates["floor_mode"] == "wilson_low"
    assert gates["floor_exempt"] == []


def test_expert_failing_the_same_matchup_exempts_it():
    """The expert cannot clear the floor either -> teacher-limited -> exempt."""
    summary = _summary({"weak": _row(0.35, 0.25), "ok": _row(0.70, 0.55)})
    expert = {"weak": _row(0.39, 0.29), "ok": _row(0.75, 0.60)}
    gates = evaluate_gates(summary, strict_ci=True, expert_per_opponent=expert)
    assert gates["floor_pass"] is True
    assert gates["floor_violations"] == []
    assert gates["floor_exempt"] == ["weak"]
    assert gates["floor_mode"] == "wilson_low_d25"


def test_expert_clearing_the_matchup_keeps_the_violation():
    """A matchup the expert clears is a real deficit, never exempt."""
    summary = _summary({"weak": _row(0.35, 0.25)})
    expert = {"weak": _row(0.65, 0.50)}
    gates = evaluate_gates(summary, strict_ci=True, expert_per_opponent=expert)
    assert gates["floor_pass"] is False
    assert gates["floor_violations"] == ["weak"]
    assert gates["floor_exempt"] == []


def test_exemption_does_not_rescue_a_failing_pooled_gate():
    summary = _summary({"weak": _row(0.35, 0.25)})
    summary["pooled_rate"] = 0.60          # below the M4 absolute bar
    summary["cluster_ci_low"] = 0.55
    expert = {"weak": _row(0.40, 0.30)}
    gates = evaluate_gates(
        summary, pooled_threshold=0.75, strict_ci=True, expert_per_opponent=expert
    )
    assert gates["floor_pass"] is True
    assert gates["pooled_pass"] is False
    assert gates["pass"] is False
