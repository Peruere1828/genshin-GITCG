"""Replay-branch Monte-Carlo teacher (PLAN.md D12 fallback / L5.4 / WS3).

Why replay branching instead of live-game forking
-------------------------------------------------
Continual resolving needs to evaluate several candidate actions from one decision
point. The pybinding cannot *fork a live game* (see
``envs/snapshot.py`` / ``envs.snapshot.FORK_LIMITATION``), but a full game is
deterministic given ``(decks, engine seed, action sequence)``. So we reproduce any
decision point by replaying from the initial state and injecting one candidate
action there, then finishing the game with a rollout policy. Averaging the outcome
over rollout seeds gives a Monte-Carlo action-value estimate that can be used as a
search/distillation target.

Cost is ``(1 + option_count * rollouts)`` full games per evaluated decision, so
this is an *offline* teacher for M3, not an in-game search. When a true engine
bridge lands, it replaces the replay loop without changing the call sites.

Determinism notes (AGENTS.md "重放确定性坑")
-------------------------------------------
* ``run_match`` resets the process-global action codebook per game, and choices are
  recorded as **option indices** (positional), never raw low-level codes.
* A branch whose injected option equals the base policy's own choice and whose
  rollout is the base policy reproduces the base game exactly. This is the L5.4
  branch-determinism acceptance and is asserted in ``tests/test_engine_bridge.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import MatchRecord, run_match
from envs.policy import Policy, build_policy
from eval.stats import estimate_rate, outcome_score


@dataclass(frozen=True)
class DecisionPoint:
    """One decision made by the branching player during a base trajectory."""

    player: int
    ordinal: int
    request_type: str
    option_count: int
    chosen_option: int
    step_index: int


class RecordingPolicy:
    """Wrap a policy and record the (option-index) decisions of its player."""

    def __init__(self, inner: Policy, player: int) -> None:
        self.inner = inner
        self.player = int(player)
        self.name = f"record:{getattr(inner, 'name', '?')}"
        self.points: list[DecisionPoint] = []

    def choose(self, built) -> int:
        code = int(self.inner.choose(built))
        codes = [int(c) for c in built.context.legal_low_level_codes]
        index = codes.index(code) if code in codes else -1
        self.points.append(
            DecisionPoint(
                player=self.player,
                ordinal=len(self.points) + 1,
                request_type=built.context.request_type.value,
                option_count=len(codes),
                chosen_option=index,
                step_index=int(built.context.step_index),
            )
        )
        return code


@dataclass
class InjectionPolicy:
    """Play ``prefix`` up to ``ordinal``, force ``option_index`` there, then ``rollout``.

    ``prefix`` reproduces the deterministic base trajectory; ``rollout`` plays the
    rest (the same object as ``prefix`` gives a deterministic one-step deviation,
    a stochastic policy gives a Monte-Carlo estimate).
    """

    prefix: Policy
    rollout: Policy
    ordinal: int
    option_index: int
    name: str = "inject"
    _n: int = field(default=0, init=False, repr=False)

    def choose(self, built) -> int:
        self._n += 1
        if self._n == self.ordinal:
            codes = [int(c) for c in built.context.legal_low_level_codes]
            if codes:
                return int(codes[min(self.option_index, len(codes) - 1)])
            return int(self.rollout.choose(built))
        inner = self.prefix if self._n < self.ordinal else self.rollout
        return int(inner.choose(built))


@dataclass(frozen=True)
class OptionValue:
    option_index: int
    mean: float
    n: int
    scores: tuple[float, ...]
    is_base_choice: bool

    def as_dict(self) -> dict:
        return {
            "option_index": self.option_index,
            "mean": self.mean,
            "n": self.n,
            "scores": list(self.scores),
            "is_base_choice": self.is_base_choice,
        }


@dataclass
class DecisionEvaluation:
    deck: str
    opponent: str
    seed: int
    player: int
    ordinal: int
    request_type: str
    base_choice: int
    base_outcome: float
    options: dict[int, OptionValue]

    @property
    def best_option(self) -> int:
        """The option with the highest mean rollout value (ties -> lower index)."""
        return min(self.options, key=lambda i: (-self.options[i].mean, i))

    @property
    def base_choice_mean(self) -> float | None:
        value = self.options.get(self.base_choice)
        return value.mean if value is not None else None

    def as_dict(self) -> dict:
        best = self.best_option
        return {
            "deck": self.deck,
            "opponent": self.opponent,
            "seed": self.seed,
            "player": self.player,
            "ordinal": self.ordinal,
            "request_type": self.request_type,
            "base_choice": self.base_choice,
            "base_outcome": self.base_outcome,
            "best_option": best,
            "best_option_mean": self.options[best].mean,
            "options": {str(k): v.as_dict() for k, v in sorted(self.options.items())},
        }


def _build_policies(
    deck: str,
    opponent: str,
    seed: int,
    *,
    player: int,
    base_spec: str | None,
    opp_spec: str | None,
    rollout_spec: str | None,
    rollout_seed: int,
) -> tuple[Policy, Policy, Policy]:
    role = "p0" if player == 0 else "p1"
    opp_role = "p1" if player == 0 else "p0"
    base = build_policy(
        base_spec or f"expert:{deck}",
        seed=derive_seed(seed, role, "branch-base"),
        role=role,
    )
    opp = build_policy(
        opp_spec or f"expert:{opponent}",
        seed=derive_seed(seed, opp_role, opponent),
        role=opp_role,
    )
    rollout = build_policy(
        rollout_spec or base_spec or f"expert:{deck}",
        seed=derive_seed(seed, role, "branch-rollout", rollout_seed),
        role=role,
    )
    return base, opp, rollout


def capture_trajectory(
    deck: str,
    opponent: str,
    seed: int,
    *,
    player: int = 0,
    base_spec: str | None = None,
    opp_spec: str | None = None,
) -> tuple[MatchRecord, list[DecisionPoint]]:
    """Play the base game and return its record plus the player's decision points."""
    base, opp, _ = _build_policies(
        deck,
        opponent,
        seed,
        player=player,
        base_spec=base_spec,
        opp_spec=opp_spec,
        rollout_spec=None,
        rollout_seed=0,
    )
    recorder = RecordingPolicy(base, player)
    policy0, policy1 = (recorder, opp) if player == 0 else (opp, recorder)
    record = run_match(deck_spec(deck), deck_spec(opponent), policy0, policy1, seed=int(seed))
    return record, recorder.points


def replay_branch(
    deck: str,
    opponent: str,
    seed: int,
    *,
    player: int = 0,
    ordinal: int | None = None,
    option_index: int | None = None,
    base_spec: str | None = None,
    opp_spec: str | None = None,
    rollout_spec: str | None = None,
    rollout_seed: int = 0,
) -> MatchRecord:
    """Replay from ``seed``; if ``ordinal`` is given, inject ``option_index`` there.

    With ``ordinal is None`` this is an ordinary base-policy game. Otherwise the
    branching player follows the base policy for earlier decisions, takes
    ``option_index`` at ``ordinal``, then hands over to the rollout policy.
    """
    base, opp, rollout = _build_policies(
        deck,
        opponent,
        seed,
        player=player,
        base_spec=base_spec,
        opp_spec=opp_spec,
        rollout_spec=rollout_spec,
        rollout_seed=rollout_seed,
    )
    if ordinal is None:
        role_policy: Policy = base
    else:
        role_policy = InjectionPolicy(
            prefix=base,
            rollout=rollout,
            ordinal=int(ordinal),
            option_index=int(option_index or 0),
        )
    policy0, policy1 = (role_policy, opp) if player == 0 else (opp, role_policy)
    return run_match(deck_spec(deck), deck_spec(opponent), policy0, policy1, seed=int(seed))


def evaluate_decision(
    deck: str,
    opponent: str,
    seed: int,
    *,
    player: int = 0,
    ordinal: int,
    base_spec: str | None = None,
    opp_spec: str | None = None,
    rollout_spec: str | None = None,
    rollouts: int = 1,
    top_k: int | None = None,
) -> DecisionEvaluation:
    """Monte-Carlo evaluate the candidate actions at (``player``, ``ordinal``).

    ``rollouts`` games are averaged per option; each rollout uses a derived seed so
    a stochastic ``rollout_spec`` (e.g. ``"legal_random"``) yields an MC estimate.
    ``top_k`` restricts the evaluated options to the base choice plus the
    ``top_k - 1`` lowest-index alternatives (Pareto of cost vs coverage).
    """
    if rollouts < 1:
        raise ValueError("rollouts must be >= 1")
    record, points = capture_trajectory(
        deck, opponent, seed, player=player, base_spec=base_spec, opp_spec=opp_spec
    )
    base_outcome = outcome_score(record.winner, player)
    point = next((p for p in points if p.ordinal == ordinal), None)
    if point is None:
        raise KeyError(
            f"no decision ordinal {ordinal} for player {player} "
            f"(game had {len(points)} decisions)"
        )

    option_indices = list(range(point.option_count))
    if top_k is not None and top_k < len(option_indices):
        keep = [point.chosen_option]
        for i in option_indices:
            if len(keep) >= top_k:
                break
            if i not in keep:
                keep.append(i)
        option_indices = sorted(keep)

    options: dict[int, OptionValue] = {}
    for option_index in option_indices:
        scores: list[float] = []
        for r in range(rollouts):
            rollout_seed = derive_seed(seed, "branch", player, ordinal, option_index, r)
            rec = replay_branch(
                deck,
                opponent,
                seed,
                player=player,
                ordinal=ordinal,
                option_index=option_index,
                base_spec=base_spec,
                opp_spec=opp_spec,
                rollout_spec=rollout_spec,
                rollout_seed=rollout_seed,
            )
            if rec.error:
                continue
            scores.append(outcome_score(rec.winner, player))
        mean = sum(scores) / len(scores) if scores else 0.0
        options[option_index] = OptionValue(
            option_index=option_index,
            mean=mean,
            n=len(scores),
            scores=tuple(scores),
            is_base_choice=(option_index == point.chosen_option),
        )

    return DecisionEvaluation(
        deck=deck,
        opponent=opponent,
        seed=int(seed),
        player=int(player),
        ordinal=int(ordinal),
        request_type=point.request_type,
        base_choice=point.chosen_option,
        base_outcome=base_outcome,
        options=options,
    )


def aggregate_option_values(
    evaluations: Sequence[DecisionEvaluation],
) -> dict[int, dict]:
    """Pool per-option rollout scores across many evaluations (teacher summary)."""
    pooled: dict[int, list[float]] = {}
    for evaluation in evaluations:
        for option_index, value in evaluation.options.items():
            pooled.setdefault(option_index, []).extend(value.scores)
    summary: dict[int, dict] = {}
    for option_index, scores in pooled.items():
        rate = estimate_rate(sum(scores), len(scores))
        summary[option_index] = {"mean": rate.rate, "n": len(scores), **rate.as_dict()}
    return summary
