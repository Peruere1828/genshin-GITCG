"""Replay data collection for CVPN training (PLAN.md WS3/WS6).

Collects decision samples by playing games with a *teacher* policy (scripted
expert, or later a search resolver) and encoding each decision context with
``reps.observation_encoder``. The teacher's chosen action is stored as an option
**index** (positional), not a low-level code, so samples are unaffected by the
process-global action-code assignment (AGENTS.md "重放确定性坑").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import Policy, build_policy
from eval.stats import outcome_score
from reps.observation_encoder import EncodedObservation, TokenObservationEncoder


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
