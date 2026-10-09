"""Cluster data plane (L5/M3): sample serialization + resumable collection.

Fast tests cover the on-disk format (round-trip, torn-line tolerance, fingerprint
stability); the slow test drives the real parallel collector and its resume path
on a tiny grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reps.observation_encoder import BeliefTarget, EncodedObservation
from train.collect import CollectSpec, collect
from train.data import (
    Sample,
    iter_task_records,
    load_replay_samples,
    observation_from_payload,
    observation_to_payload,
    sample_from_payload,
    sample_to_payload,
)


def _synthetic_observation() -> EncodedObservation:
    return EncodedObservation(
        token_features=((1.0, 2.0), (3.0, 4.0)),
        token_mask=(True, False),
        opponent_token_mask=(True, True),
        option_features=((0.5, 0.6), (0.7, 0.8)),
        option_mask=(True, False),
        low_level_action_codes=(1, 2),
        low_to_high_codes=(2, 3),
        high_level_mask=(True, False),
        privileged_state=(0.0, 1.0),
        belief_target=BeliefTarget(
            hand_histogram=(0.1, 0.2),
            hand_size=2.0,
            remaining_deck_histogram=(0.3,),
            burst_ready_count=1.0,
            group_hand_presence=(1.0,),
            group_remaining_deck_presence=(0.5,),
        ),
        opponent_tag_id=3,
        opponent_entity_id=4,
        opponent_deck_id=5,
    )


def test_observation_roundtrip_is_lossless():
    obs = _synthetic_observation()
    payload = observation_to_payload(obs)
    assert json.loads(json.dumps(payload))  # JSON-serializable
    assert observation_from_payload(json.loads(json.dumps(payload))) == obs


def test_sample_roundtrip_is_lossless():
    sample = Sample(
        observation=_synthetic_observation(),
        teacher_index=1,
        value_target=1.0,
        request_type="action",
        step_index=7,
        deck="a",
        opponent="b",
    )
    payload = json.loads(json.dumps(sample_to_payload(sample)))
    assert sample_from_payload(payload) == sample


def test_collect_spec_tasks_and_fingerprint_are_stable():
    spec = CollectSpec(deck="a", opponents=("b", "c"), seeds=(0, 1, 2))
    tasks = spec.tasks()
    assert [t.index for t in tasks] == list(range(6))
    assert [(t.opponent, t.seed) for t in tasks[:4]] == [
        ("b", 0), ("b", 1), ("b", 2), ("c", 0)
    ]
    assert spec.fingerprint() == spec.fingerprint()


def test_tolerant_reader_skips_torn_line(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"index": 0, "samples": []}) + "\n"
        + '{"index": 1, "samples": [',  # torn
        encoding="utf-8",
    )
    records = list(iter_task_records(path))
    assert [r["index"] for r in records] == [0]


def test_load_replay_samples_flattens(tmp_path: Path):
    sample = Sample(
        observation=_synthetic_observation(),
        teacher_index=0,
        value_target=0.0,
        request_type="action",
        step_index=0,
        deck="a",
        opponent="b",
    )
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"index": 0, "error": None, "samples": [sample_to_payload(sample)]}) + "\n",
        encoding="utf-8",
    )
    loaded = load_replay_samples(tmp_path)
    assert loaded == [sample]


@pytest.mark.slow
def test_collect_is_resumable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GITCG_DATA_DIR", str(tmp_path))
    spec = CollectSpec(deck="superconduct_aggro", opponents=("natlan_battleship",), seeds=(0, 1))
    first = collect(spec, tag="t", workers=2, progress=False)
    assert first["games"] == 2
    assert first["samples"] > 0
    assert first["ran_this_call"] == 2

    second = collect(spec, tag="t", workers=2, progress=False)
    assert second["ran_this_call"] == 0  # nothing left to do
    assert second["games"] == first["games"]

    from train.data import load_samples
    samples = load_samples(tmp_path / "replays" / "t")
    assert len(samples) == first["samples"]
