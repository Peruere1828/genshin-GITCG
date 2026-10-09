"""Fork-based search policy + driver (PLAN.md WS3, M3 "实用版 continual resolving").

This is the practical resolver: at a decision point it evaluates candidate actions
by forking the last ``is_resumable()`` boundary snapshot (``envs.fork_bridge``),
injecting each candidate, rolling out with a fixed policy, and averaging the
outcome. It is the online counterpart of ``train.replay_branch``'s offline
replay-branch MC teacher -- same idea, O(1) fork instead of O(depth) replay.

Anchoring the search to a boundary
----------------------------------
A decision is delivered from *inside* ``game.step()``, so no boundary snapshot
exists at that moment. The driver therefore snapshots after every step while the
game reports ``is_resumable()`` and remembers how many decisions each player had
made at that point. A candidate task then forks that snapshot and replays the
decisions since it (the *prefix*) before injecting -- the record-replay contract
verified in ``reports/engine/probe_boundary_fork_*`` (I10). The initial state is
used as the first boundary: it is a complete state, so a fork from it reproduces
the live game (verified; unlike mid-phase ``canResume:false`` pauses).

Safety of the search
--------------------
The base policy is consulted once per decision regardless of whether the search
runs, so the expert's own RNG advances exactly as it would without search. Its
choice is always one of the evaluated candidates, so a search with enough
rollouts cannot be surprised into ignoring the expert action.

Cost is ``candidates x rollouts`` full games per searched decision; ``top_k`` and
``search_every`` bound it. Every decision is logged (``SearchDecision``) for the
coach/teacher and for distillation features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from gitcg import Game, GameStatus

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.fork_bridge import ForkBridge, ForkTask
from envs.match import MatchRecord, PolicyPlayer, _build_create_param
from envs.policy import Policy, build_policy
from envs.snapshot import capture_snapshot


@dataclass
class SearchDecision:
    """One searched decision's outcome (also feeds the coach / distillation)."""

    player: int
    ordinal: int
    request_type: str
    option_count: int
    base_option: int
    candidates: tuple[int, ...]
    means: dict[int, float]
    counts: dict[int, int]
    chosen_option: int
    searched: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "ordinal": self.ordinal,
            "request_type": self.request_type,
            "option_count": self.option_count,
            "base_option": self.base_option,
            "candidates": list(self.candidates),
            "means": {str(k): v for k, v in self.means.items()},
            "counts": {str(k): v for k, v in self.counts.items()},
            "chosen_option": self.chosen_option,
            "searched": self.searched,
        }


@dataclass
class SearchContext:
    """Mutable state shared between the driver loop and the search policies."""

    bridge: ForkBridge
    game_attrs: Mapping[int, int]
    rollout_specs: tuple[str | None, str | None]
    base_seed: int
    records: list[list[int]] = field(default_factory=lambda: [[], []])
    boundary: tuple[str, int, int] | None = None
    searches: list[SearchDecision] = field(default_factory=list)

    def prefix(self, player: int) -> tuple[int, ...]:
        if self.boundary is None:
            return ()
        base = self.boundary[1] if player == 0 else self.boundary[2]
        return tuple(self.records[player][base:])

    def update_boundary(self, snapshot: str) -> None:
        self.boundary = (snapshot, len(self.records[0]), len(self.records[1]))


def _select_candidates(option_count: int, top_k: int | None) -> list[int]:
    if option_count <= 0:
        return []
    if top_k is None or top_k >= option_count:
        return list(range(option_count))
    if top_k <= 1:
        return [0]
    return sorted({round(i * (option_count - 1) / (top_k - 1)) for i in range(top_k)})


@dataclass
class ForkSearchPolicy:
    """A ``Policy`` that refines the base choice with fork-based MC evaluation."""

    player: int
    context: SearchContext
    deck0: str = ""
    deck1: str = ""
    base: Policy | None = None
    rollouts: int = 1
    top_k: int | None = 4
    search_every: int = 1
    max_searches: int | None = None
    max_steps: int = 5000
    name: str = field(default="fork_search")
    _calls: int = field(default=0, init=False, repr=False)
    _n_searched: int = field(default=0, init=False, repr=False)

    @property
    def _default_rollout(self) -> str:
        return "legal_random"

    def _rollout_spec(self) -> str:
        spec = self.context.rollout_specs[self.player]
        return spec or self._default_rollout

    def _opp_rollout_spec(self) -> str:
        opp = 1 - self.player
        spec = self.context.rollout_specs[opp]
        return spec or self._default_rollout

    def choose(self, built) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        if not codes:
            return -1
        self._calls += 1
        base_code = int(self.base.choose(built)) if self.base is not None else codes[0]
        base_idx = codes.index(base_code) if base_code in codes else 0

        if (
            self.context.boundary is None
            or len(codes) == 1
            or self.rollouts < 1
            or (self.search_every > 1 and self._calls % self.search_every != 0)
            or (self.max_searches is not None and self._n_searched >= self.max_searches)
        ):
            return base_code

        self._n_searched += 1
        return self._search(built, codes, base_code, base_idx)

    def _search(self, built, codes: list[int], base_code: int, base_idx: int) -> int:
        context = self.context
        snapshot = context.boundary[0]  # type: ignore[index]
        acting = self.player
        ordinal = len(context.records[acting])  # decisions before this one
        prefix_self = context.prefix(acting)
        prefix_opp = context.prefix(1 - acting)

        candidates = _select_candidates(len(codes), self.top_k)
        if base_idx not in candidates:
            candidates = sorted(set(candidates) | {base_idx})

        roll_spec = self._rollout_spec()
        opp_spec = self._opp_rollout_spec()
        rollout0, rollout1 = (roll_spec, opp_spec) if acting == 0 else (opp_spec, roll_spec)

        ordinal_inject = len(prefix_self) + 1
        tasks: list[ForkTask] = []
        index_map: dict[int, int] = {}
        order: list[int] = []
        for option in candidates:
            for r in range(self.rollouts):
                task_index = len(order)
                order.append(option)
                index_map[task_index] = option
                rollout_seed = derive_seed(
                    context.base_seed, "fork-search", acting, ordinal, r
                )
                tasks.append(
                    ForkTask(
                        index=task_index,
                        snapshot=snapshot,
                        game_attrs=dict(context.game_attrs),
                        deck0=self.deck0,
                        deck1=self.deck1,
                        prefix0=prefix_self if acting == 0 else prefix_opp,
                        prefix1=prefix_self if acting == 1 else prefix_opp,
                        inject_player=acting,
                        inject_ordinal=ordinal_inject,
                        inject_option=option,
                        rollout0=rollout0,
                        rollout1=rollout1,
                        rollout_seed=rollout_seed,
                        max_steps=self.max_steps,
                    )
                )

        results = context.bridge.evaluate(tasks)
        scores: dict[int, list[float]] = {option: [] for option in candidates}
        for result in results:
            option = index_map.get(result.index)
            if option is None:
                continue
            scores[option].append(result.outcome(acting))

        means = {o: (sum(v) / len(v) if v else 0.0) for o, v in scores.items()}
        counts = {o: len(v) for o, v in scores.items()}
        if means:
            best = min(means, key=lambda o: (-means[o], 0 if o == base_idx else 1, o))
        else:
            best = base_idx
        chosen_code = int(codes[best])

        context.searches.append(
            SearchDecision(
                player=acting,
                ordinal=ordinal,
                request_type=getattr(built.context.request_type, "value", str(built.context.request_type)),
                option_count=len(codes),
                base_option=base_idx,
                candidates=tuple(candidates),
                means=means,
                counts=counts,
                chosen_option=best,
                searched=True,
            )
        )
        return chosen_code


class _RecordingPolicy:
    """Outer policy wrapper recording each decision's positional option index."""

    def __init__(self, inner: Policy, player: int, records: list[list[int]]) -> None:
        self.inner = inner
        self.player = int(player)
        self.records = records
        self.name = f"record:{getattr(inner, 'name', '?')}"

    def choose(self, built) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        code = int(self.inner.choose(built))
        idx = codes.index(code) if code in codes else (-1 if codes else 0)
        self.records[self.player].append(idx)
        return code


@dataclass
class SearchMatchResult:
    """A search match's record plus the per-decision search log."""

    record: MatchRecord
    searches: list[SearchDecision]

    @property
    def winner(self) -> int | None:
        return self.record.winner

    def as_dict(self, *, include_searches: bool = False) -> dict[str, Any]:
        payload = self.record.as_dict()
        payload["searched_decisions"] = len(self.searches)
        if include_searches:
            payload["searches"] = [s.as_dict() for s in self.searches]
        return payload


def run_search_match(
    deck0: str,
    deck1: str,
    *,
    bridge: ForkBridge,
    seed: int,
    search_players: Sequence[int] = (0,),
    rollouts: int = 1,
    top_k: int | None = 4,
    search_every: int = 1,
    max_searches: int | None = None,
    rollout_specs: tuple[str | None, str | None] | None = None,
    game_attrs: Mapping[int, int] | None = None,
    max_steps: int = 5000,
    version: str | None = None,
    base_policies: tuple[Policy | None, Policy | None] | None = None,
    record_final_state: bool = False,
) -> SearchMatchResult:
    """Play one game in which ``search_players`` refine their moves with fork search.

    Non-search players use an expert policy for their own deck; search players use
    their base policy plus fork search. Decisions are recorded as positional option
    indices so the fork tasks replay exactly.
    """
    from reps.action_hierarchy import reset_default_hierarchical_action_codebook

    if bridge.workers == 0:
        raise ValueError(
            "run_search_match needs a subprocess fork bridge (workers >= 1): "
            "forking inside a live game callback corrupts the engine's JS runtime. "
            "ForkBridge(workers=0) is only for offline fork tasks."
        )
    reset_default_hierarchical_action_codebook()
    spec0, spec1 = deck_spec(deck0), deck_spec(deck1)
    create_param = _build_create_param(spec0, spec1, seed=seed, version=version)
    game = Game(create_param=create_param)
    attrs = dict(game_attrs or {})
    for attr, value in attrs.items():
        game.set_attr(int(attr), int(value))

    overrides = base_policies or (None, None)
    search_set = set(int(p) for p in search_players)
    rollout_specs = rollout_specs or (None, None)
    context = SearchContext(
        bridge=bridge,
        game_attrs=attrs,
        rollout_specs=_normalize_rollout_specs(rollout_specs, deck0, deck1),
        base_seed=int(seed),
    )

    policies: list[Policy] = []
    for player, deck_slug in ((0, deck0), (1, deck1)):
        base = overrides[player]
        if base is None:
            base = build_policy(f"expert:{deck_slug}", seed=derive_seed(seed, f"p{player}", "search-base"), role=f"p{player}")
        if player in search_set:
            policy: Policy = ForkSearchPolicy(
                player=player,
                context=context,
                deck0=deck0,
                deck1=deck1,
                base=base,
                rollouts=rollouts,
                top_k=top_k,
                search_every=search_every,
                max_searches=max_searches,
                max_steps=max_steps,
            )
        else:
            policy = base
        policies.append(_RecordingPolicy(policy, player, context.records))

    players = [PolicyPlayer(p, policies[p]) for p in (0, 1)]
    game.set_player(0, players[0])
    game.set_player(1, players[1])

    # The initial state is a valid fork boundary (see module docstring).
    context.update_boundary(capture_snapshot(game))

    error: str | None = None
    truncated = False
    start = time.perf_counter()
    try:
        game.start()
        steps = 0
        while game.is_running():
            if steps >= max_steps:
                truncated = True
                break
            if game.is_resumable():
                context.update_boundary(capture_snapshot(game))
            game.step()
            steps += 1
    except Exception as exc:  # engine/IO failures are otherwise swallowed by cffi
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - start

    winner: int | None = None
    rounds = 0
    final_json = None
    try:
        winner = game.winner()
        rounds = int(game.round_number())
        if record_final_state:
            final_json = capture_snapshot(game)
    except Exception:
        pass
    if error is None and game.status() == GameStatus.ABORTED:
        try:
            error = f"aborted: {game.error()}"
        except Exception:
            error = "aborted"

    record = MatchRecord(
        deck0=spec0.name,
        deck1=spec1.name,
        seed=int(seed),
        winner=winner,
        rounds=rounds,
        decisions0=len(context.records[0]),
        decisions1=len(context.records[1]),
        fallbacks0=players[0].fallbacks,
        fallbacks1=players[1].fallbacks,
        io_errors0=tuple(players[0].io_errors),
        io_errors1=tuple(players[1].io_errors),
        error=error,
        truncated=truncated,
        wall_seconds=wall,
        final_state_json=final_json,
    )
    return SearchMatchResult(record=record, searches=list(context.searches))


def _normalize_rollout_specs(
    rollout_specs: tuple[str | None, str | None], deck0: str, deck1: str
) -> tuple[str | None, str | None]:
    return (
        rollout_specs[0] or f"expert:{deck0}",
        rollout_specs[1] or f"expert:{deck1}",
    )
