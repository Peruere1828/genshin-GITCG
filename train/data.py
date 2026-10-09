"""Replay data collection for CVPN training (PLAN.md WS3/WS6).

Collects decision samples by playing games with a *teacher* policy (scripted
expert, or later a search resolver) and encoding each decision context with
``reps.observation_encoder``. The teacher's chosen action is stored as an option
**index** (positional), not a low-level code, so samples are unaffected by the
process-global action-code assignment (AGENTS.md "重放确定性坑").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import Policy, build_policy
from eval.stats import outcome_score
from reps.observation_encoder import (
    BeliefTarget,
    EncodedObservation,
    TokenObservationEncoder,
)


@dataclass
class Sample:
    observation: EncodedObservation
    teacher_index: int
    value_target: float
    request_type: str
    step_index: int
    deck: str
    opponent: str

    @property
    def n_options(self) -> int:
        return len(self.observation.option_features)


class _CaptureTeacher:
    """Wrap a policy and record (built_context, chosen_code) for each decision."""

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self.name = f"capture:{getattr(inner, 'name', '?')}"
        self.items: list[tuple[object, int]] = []

    def choose(self, built) -> int:
        code = int(self.inner.choose(built))
        self.items.append((built, code))
        return code


def collect_samples(
    deck: str,
    opponent: str,
    seeds: Sequence[int],
    *,
    encoder: TokenObservationEncoder,
    teacher_spec: str | None = None,
    opponent_spec: str | None = None,
) -> list[Sample]:
    """Collect teacher-labelled samples for ``deck`` (player 0) vs ``opponent``."""
    samples: list[Sample] = []
    for seed in seeds:
        teacher = build_policy(
            teacher_spec or f"expert:{deck}", seed=derive_seed(seed, "p0", deck), role="p0"
        )
        capture = _CaptureTeacher(teacher)
        opp = build_policy(
            opponent_spec or f"expert:{opponent}",
            seed=derive_seed(seed, "p1", opponent),
            role="p1",
        )
        record = run_match(deck_spec(deck), deck_spec(opponent), capture, opp, seed=int(seed))
        if record.error:
            continue
        value = outcome_score(record.winner, 0)
        for built, code in capture.items:
            context = built.context
            codes = list(int(c) for c in context.legal_low_level_codes)
            if int(code) not in codes:
                continue
            index = codes.index(int(code))
            observation = encoder.encode_context(
                context, history=(), include_training_targets=False
            )
            if not observation.option_features:
                continue
            samples.append(
                Sample(
                    observation=observation,
                    teacher_index=index,
                    value_target=value,
                    request_type=context.request_type.value,
                    step_index=context.step_index,
                    deck=deck,
                    opponent=opponent,
                )
            )
    return samples


def sample_stats(samples: Sequence[Sample]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0}
    sizes = sorted(s.n_options for s in samples)
    wins = sum(1 for s in samples if s.value_target == 1.0)
    return {
        "n": n,
        "mean_options": sum(sizes) / n,
        "max_options": sizes[-1],
        "distinct_request_types": sorted({s.request_type for s in samples}),
        "win_fraction": wins / n,
    }


# --------------------------------------------------------------------------- #
# serialization (cluster data plane: CPU nodes collect -> storage -> GPU nodes train)
# --------------------------------------------------------------------------- #
def observation_to_payload(observation: EncodedObservation) -> dict[str, Any]:
    """JSON-serializable payload for an ``EncodedObservation`` (lossy-free)."""
    belief = observation.belief_target
    return {
        "token_features": [list(row) for row in observation.token_features],
        "token_mask": list(observation.token_mask),
        "opponent_token_mask": list(observation.opponent_token_mask),
        "option_features": [list(row) for row in observation.option_features],
        "option_mask": list(observation.option_mask),
        "low_level_action_codes": list(observation.low_level_action_codes),
        "low_to_high_codes": list(observation.low_to_high_codes),
        "high_level_mask": list(observation.high_level_mask),
        "privileged_state": list(observation.privileged_state),
        "belief_target": {
            "hand_histogram": list(belief.hand_histogram),
            "hand_size": float(belief.hand_size),
            "remaining_deck_histogram": list(belief.remaining_deck_histogram),
            "burst_ready_count": float(belief.burst_ready_count),
            "group_hand_presence": list(belief.group_hand_presence),
            "group_remaining_deck_presence": list(belief.group_remaining_deck_presence),
        },
        "opponent_tag_id": int(observation.opponent_tag_id),
        "opponent_entity_id": int(observation.opponent_entity_id),
        "opponent_deck_id": int(observation.opponent_deck_id),
    }


def observation_from_payload(payload: dict[str, Any]) -> EncodedObservation:
    belief = payload["belief_target"]
    return EncodedObservation(
        token_features=tuple(tuple(float(v) for v in row) for row in payload["token_features"]),
        token_mask=tuple(bool(v) for v in payload["token_mask"]),
        opponent_token_mask=tuple(bool(v) for v in payload["opponent_token_mask"]),
        option_features=tuple(tuple(float(v) for v in row) for row in payload["option_features"]),
        option_mask=tuple(bool(v) for v in payload["option_mask"]),
        low_level_action_codes=tuple(int(v) for v in payload["low_level_action_codes"]),
        low_to_high_codes=tuple(int(v) for v in payload["low_to_high_codes"]),
        high_level_mask=tuple(bool(v) for v in payload["high_level_mask"]),
        privileged_state=tuple(float(v) for v in payload["privileged_state"]),
        belief_target=BeliefTarget(
            hand_histogram=tuple(float(v) for v in belief["hand_histogram"]),
            hand_size=float(belief["hand_size"]),
            remaining_deck_histogram=tuple(float(v) for v in belief["remaining_deck_histogram"]),
            burst_ready_count=float(belief["burst_ready_count"]),
            group_hand_presence=tuple(float(v) for v in belief["group_hand_presence"]),
            group_remaining_deck_presence=tuple(
                float(v) for v in belief["group_remaining_deck_presence"]
            ),
        ),
        opponent_tag_id=int(payload["opponent_tag_id"]),
        opponent_entity_id=int(payload["opponent_entity_id"]),
        opponent_deck_id=int(payload["opponent_deck_id"]),
    )


def sample_to_payload(sample: Sample) -> dict[str, Any]:
    return {
        "observation": observation_to_payload(sample.observation),
        "teacher_index": int(sample.teacher_index),
        "value_target": float(sample.value_target),
        "request_type": sample.request_type,
        "step_index": int(sample.step_index),
        "deck": sample.deck,
        "opponent": sample.opponent,
    }


def sample_from_payload(payload: dict[str, Any]) -> Sample:
    return Sample(
        observation=observation_from_payload(payload["observation"]),
        teacher_index=int(payload["teacher_index"]),
        value_target=float(payload["value_target"]),
        request_type=payload["request_type"],
        step_index=int(payload["step_index"]),
        deck=payload["deck"],
        opponent=payload["opponent"],
    )


def write_samples_jsonl(path: str | Path, samples: Sequence[Sample]) -> int:
    """Append flat per-sample JSONL records; return the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_payload(sample), ensure_ascii=False) + "\n")
    return len(samples)


def iter_samples_jsonl(path: str | Path) -> Iterator[Sample]:
    """Yield samples from a flat per-sample JSONL file (ignores torn lines)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield sample_from_payload(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue


def load_samples(path: str | Path) -> list[Sample]:
    """Load samples from a flat JSONL file *or* a replay dir of task records."""
    path = Path(path)
    if path.is_dir():
        return load_replay_samples(path)
    return list(iter_samples_jsonl(path))


# --------------------------------------------------------------------------- #
# task-record format used by the resumable collector (one line = one game)
# --------------------------------------------------------------------------- #
def iter_task_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield complete task records, skipping a torn trailing line and blanks."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn/partial line -> treat that task as not done
            yield record


def load_replay_samples(replay_dir: str | Path) -> list[Sample]:
    """Flatten a collector replay dir into samples (samples of every complete game)."""
    replay_dir = Path(replay_dir)
    samples: list[Sample] = []
    for record in iter_task_records(replay_dir / "samples.jsonl"):
        for payload in record.get("samples", []):
            samples.append(sample_from_payload(payload))
    return samples
