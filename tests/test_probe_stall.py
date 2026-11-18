"""``scripts.probe_stall``: raw-row selection + a real stage trace.

The row-selection helpers are pure (no engine). The trace test plays one real
NPC stage match, so it needs the ``gitcg`` wheel and the NPC config data.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from scripts.probe_stall import analyse, check_round_cap, read_rows, select_raw_games


def _row(*, rounds: int, winner: int | None, stage: str = "chefmao", seed: int = 1) -> dict:
    return {"stage": stage, "seed": seed, "rounds": rounds, "winner": winner}


def test_select_raw_games_keeps_draws_and_long_rows():
    rows = [
        _row(rounds=5, winner=0, seed=1),
        _row(rounds=12, winner=1, seed=2),
        _row(rounds=15, winner=None, seed=3),
        _row(rounds=3, winner=None, seed=4),
    ]
    selected = select_raw_games(rows, min_rounds=10)
    assert [int(row["seed"]) for row in selected] == [2, 3, 4]


def test_select_raw_games_without_min_rounds_keeps_only_draws():
    rows = [_row(rounds=15, winner=0), _row(rounds=15, winner=None)]
    assert select_raw_games(rows, min_rounds=None) == [rows[1]]


def test_read_rows_skips_blank_and_torn_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"stage": "chefmao", "seed": 1}) + "\n"
        + "\n"
        + '{"stage": "chefmao", "seed": 2, "rounds": ',
        encoding="utf-8",
    )
    assert read_rows(path) == [{"stage": "chefmao", "seed": 1}]


def test_check_round_cap_reports_attribute_and_default():
    info = check_round_cap()
    if "error" in info:
        pytest.skip(f"engine unavailable: {info['error']}")
    assert info["attribute_exists"] is True
    assert info["attribute_value"] == 105
    assert info["default_config"]["maxRoundsCount"] == 15
    # Pinning the finding: run_match never sends the max-rounds attribute.
    assert info["run_match_sets_it"] is False


@pytest.mark.slow
def test_analyse_traces_a_real_stage_match():
    pytest.importorskip("gitcg")
    try:
        from npc.spec import get_stage

        get_stage("chefmao")
    except Exception as exc:  # NPC configs / engine data missing
        pytest.skip(f"stage data unavailable: {exc}")

    with redirect_stdout(io.StringIO()):
        trace = analyse("chefmao", 1, policy_spec="heuristic")

    assert trace["stage"] == "chefmao"
    assert trace["rounds_table"], "per-round HP table must not be empty"
    assert trace["mutations"], "public mutation stream must not be empty"
    assert trace["damage_events"] > 0
    kinds = {mutation["kind"] for mutation in trace["mutations"]}
    assert "damage" in kinds and "skill_used" in kinds and "set_winner" in kinds
    # Every round must expose both sides' character HP.
    for entry in trace["rounds_table"]:
        assert set(entry["hp_end"]) == {0, 1}
    assert set(trace["decisions"]["declare_end_decisions"]) == {0, 1}
    final = trace["final"]
    assert final["phase"] == "game_end"
    assert len(final["players"]) == 2
    assert final["winner"] == trace["winner"]
