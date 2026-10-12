"""Engine adapter for the external-sampling MCCFR solver (PLAN.md M3-CPU S-C).

This module is the S-C **first step**: it plugs the engine-free solver
(``train/cfr_solver.py``, S-B) into the real game and produces a runnable CFR
search policy (``CFRSearchPolicy``) plus a driver (``run_cfr_search_match``).

Design (how each ``ParticleGame`` operation reaches the engine)
----------------------------------------------------------------
A search node is the engine **decision** reached by replaying, from a fixed
belief-root boundary snapshot, the per-player positional option prefixes
(``(root_snapshot, prefix0, prefix1)``). Every node is therefore a deterministic
replay of an ``is_resumable()`` boundary -- the same anchoring the practical fork
resolver uses (``envs/fork_search``), extended to a tree:

* ``root_particles`` -- public-belief determinizations
  (``train.cfr_solver.belief_root_particles`` over ``reps.belief``, S-A); no
  hidden truth enters the root (charter §5.2);
* ``transition`` -- one single-step fork (``envs.cfr_probe``: fork boundary in a
  subprocess, replay prefixes, capture the next decision). Transitions extend a
  player's prefix; they never snapshot a non-boundary state;
* ``is_terminal`` / ``acting_player`` / ``legal_actions`` -- the captured node's
  public info; actions carry **semantic keys**
  (``reps.action_hierarchy.semantic_action_key_for_spec``), never raw
  process-local codes; a prior only ranks (it is ``SearchAction.prior``), it
  never deletes a legal action (§5.4);
* ``infoset_key`` -- ``train.cfr_solver.public_infoset_key`` over the masked
  public view (no hidden truth);
* ``terminal_utility`` -- engine winner, zero-sum (draw = 0);
* ``frontier_value`` -- a pluggable value oracle: a CVPN checkpoint value head
  when ``value_spec`` is set, else the documented public heuristic
  (``envs.cfr_probe.heuristic_frontier_probability``). Returned player-0 relative
  and converted to the traverser's perspective (zero-sum).

Discipline (violating it is a rework)
-------------------------------------
* forks run only in subprocesses (``ForkBridge(workers>=1)``); the driver refuses
  ``workers == 0`` just like ``run_search_match`` (D13 C4/C5);
* snapshots are taken only at ``is_resumable()`` boundaries;
* the search never reads hidden truth (roots/infosets/frontier use public fields
  only -- charter §5.2);
* strength口径 is the pure-policy mode: this module calls no LLM.

S-C衔接点 (teacher export)
--------------------------
``CFRSearchDecision`` (logged per searched decision) carries the solver's **root
average strategy** keyed by semantic action, the **root value**, per-action
values and visit counts -- exactly the ``search_teacher_policy`` /
``search_teacher_value`` targets the S-C distillation stage
(``train.collect`` policy CE + value regression + belief scoring) consumes. The
belief target itself comes from ``reps.belief`` (S-A) and is unchanged here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

from gitcg import Game, GameStatus

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.fork_bridge import ForkBridge, ForkResult, ForkTask
from envs.fork_search import (
    SearchContext,
    _RecordingPolicy,
    _observe_public_record,
)
from envs.match import MatchRecord, PolicyPlayer, _build_create_param
from envs.policy import Policy, build_policy
from envs.snapshot import capture_snapshot
from train.cfr_solver import (
    RootParticle,
    RootSolution,
    SearchAction,
    belief_root_particles,
    solve as cfr_solve,
)

__all__ = [
    "EngineParticleGame",
    "CFRSearchPolicy",
    "CFRSearchDecision",
    "CfrMatchResult",
    "run_cfr_search_match",
]


# --------------------------------------------------------------------------- #
# engine-backed ParticleGame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _CfrNode:
    """One decision node: a belief-root particle + per-player option prefixes."""

    root_index: int
    prefix0: tuple[int, ...]
    prefix1: tuple[int, ...]
    probe: ForkResult


class EngineParticleGame:
    """``train.cfr_solver.ParticleGame`` over the real engine via the fork bridge.

    ``probe_fn(snapshot, prefix0, prefix1) -> ForkResult`` is the single-step fork;
    it defaults to a subprocess ``ForkBridge`` probe (``envs.cfr_probe``) and is
    injectable so unit tests can drive the adapter with a fake engine.
    """

    def __init__(
        self,
        *,
        bridge: ForkBridge | None,
        deck0: str,
        deck1: str,
        game_attrs: Mapping[int, int],
        root_fn: Any,
        root_player: int,
        prior_by_key: Mapping[Any, float] | None = None,
        required_key: Any = None,
        candidate_limit: int | None = None,
        value_spec: str = "",
        max_steps: int = 5000,
        probe_fn: Any = None,
    ) -> None:
        self.bridge = bridge
        self.deck0 = deck0
        self.deck1 = deck1
        self.game_attrs = dict(game_attrs or {})
        self._root_fn = root_fn
        self.root_player = int(root_player)
        self._prior = dict(prior_by_key or {})
        self._required_key = required_key
        self._candidate_limit = candidate_limit
        self._value_spec = str(value_spec)
        self._max_steps = int(max_steps)
        self._probe_fn = probe_fn
        self._cache: dict[tuple, ForkResult] = {}
        self._root_snapshots: list[str] = []
        # Unrestricted root options/keys, so the policy can map the solved root
        # strategy (semantic keys) back to a live option index/code.
        self.root_options: tuple[int, ...] = ()
        self.root_keys: tuple[Any, ...] = ()

    # -- ParticleGame protocol --------------------------------------------- #
    def root_particles(self) -> Sequence[RootParticle]:
        particles = list(self._root_fn())
        self._root_snapshots = [str(p.state) for p in particles]
        out: list[RootParticle] = []
        for index, particle in enumerate(particles):
            probe = self._probe(index, (), ())
            if probe.node_terminal:
                continue
            node = _CfrNode(index, (), (), probe)
            if not out:
                self.root_options = tuple(int(o) for o in probe.legal_options)
                self.root_keys = tuple(probe.legal_keys)
            out.append(RootParticle(state=node, weight=particle.weight, label=particle.label))
        return out

    def is_terminal(self, node: _CfrNode) -> bool:
        return bool(node.probe.node_terminal)

    def acting_player(self, node: _CfrNode) -> int:
        if node.probe.acting_player is None:
            raise RuntimeError(f"probe node has no acting player: {node.probe.error!r}")
        return int(node.probe.acting_player)

    def legal_actions(self, node: _CfrNode) -> Sequence[SearchAction]:
        if node.probe.node_terminal:
            return []
        keys = tuple(node.probe.legal_keys)
        priors = [float(self._prior.get(key, 0.0)) for key in keys]
        indices = list(range(len(keys)))
        if (
            self._candidate_limit is not None
            and not node.prefix0
            and not node.prefix1
            and len(indices) > int(self._candidate_limit)
        ):
            keep = sorted(indices, key=lambda i: (-priors[i], i))[: int(self._candidate_limit)]
            if self._required_key is not None and self._required_key in keys:
                keep.append(keys.index(self._required_key))
            indices = sorted(set(keep))
        return [SearchAction(key=keys[i], prior=priors[i]) for i in indices]

    def infoset_key(self, node: _CfrNode) -> Hashable:
        return node.probe.infoset_key

    def transition(self, node: _CfrNode, action_key: str) -> _CfrNode | None:
        keys = tuple(node.probe.legal_keys)
        try:
            option = keys.index(action_key)
        except ValueError:
            return None
        acting = int(node.probe.acting_player) if node.probe.acting_player is not None else 0
        if acting == 0:
            return self._node(node.root_index, (*node.prefix0, option), node.prefix1)
        return self._node(node.root_index, node.prefix0, (*node.prefix1, option))

    def terminal_utility(self, node: _CfrNode, traverser: int) -> float:
        winner = node.probe.winner
        if winner is None:
            return 0.0
        return 1.0 if int(winner) == int(traverser) else -1.0

    def frontier_value(self, node: _CfrNode, traverser: int) -> float:
        # probe.frontier_p0 is player-0 win probability; centre to [-1, 1] and flip
        # for the other traverser (zero-sum).
        probability = node.probe.frontier_p0
        if probability is None:
            probability = 0.5
        centred = 2.0 * float(probability) - 1.0
        return centred if int(traverser) == 0 else -centred

    # -- root strategy -> option mapping ----------------------------------- #
    def root_option_for_key(self, key: Any) -> int | None:
        try:
            return int(self.root_keys.index(key))
        except ValueError:
            return None

    # -- internals --------------------------------------------------------- #
    def _probe(self, root_index: int, prefix0: tuple[int, ...], prefix1: tuple[int, ...]) -> ForkResult:
        cache_key = (int(root_index), tuple(prefix0), tuple(prefix1))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        snapshot = self._root_snapshots[root_index]
        if self._probe_fn is not None:
            result = self._probe_fn(snapshot, prefix0, prefix1)
        else:
            if self.bridge is None or getattr(self.bridge, "workers", 0) == 0:
                raise ValueError(
                    "EngineParticleGame needs a subprocess ForkBridge (workers >= 1) "
                    "or an injected probe_fn: forking inside a live game callback "
                    "corrupts the engine's JS runtime."
                )
            task = ForkTask(
                index=0,
                snapshot=snapshot,
                game_attrs=self.game_attrs,
                deck0=self.deck0,
                deck1=self.deck1,
                prefix0=tuple(prefix0),
                prefix1=tuple(prefix1),
                kind="probe",
                value_spec=self._value_spec,
                max_steps=self._max_steps,
            )
            result = self.bridge.evaluate([task])[0]
        if result.error is not None:
            raise RuntimeError(f"CFR probe failed: {result.error}")
        self._cache[cache_key] = result
        return result

    def _node(self, root_index: int, prefix0: tuple[int, ...], prefix1: tuple[int, ...]) -> _CfrNode:
        return _CfrNode(root_index, prefix0, prefix1, self._probe(root_index, prefix0, prefix1))


# --------------------------------------------------------------------------- #
# decision log / teacher export
# --------------------------------------------------------------------------- #
@dataclass
class CFRSearchDecision:
    """One searched decision: the root solution (S-C teacher targets) + context."""

    player: int
    ordinal: int
    request_type: str
    option_count: int
    base_option: int
    chosen_option: int
    iterations: int
    depth: int
    particles: int
    value: float
    strategy: dict[Any, float]
    action_values: dict[Any, float]
    visits: dict[Any, int]
    prior: dict[Any, float] | None = None
    searched: bool = True

    def teacher_policy(self) -> dict[Any, float]:
        """Root average strategy (semantic key -> prob): the S-C policy target."""
        return {key: float(p) for key, p in self.strategy.items()}

    def teacher_value(self) -> float:
        """Root value from the searcher's perspective: the S-C value target."""
        return float(self.value)

    def as_dict(self, *, include_keys: bool = True) -> dict[str, Any]:
        def _keys(mapping: Mapping[Any, Any]) -> dict[str, Any]:
            return {str(key): value for key, value in mapping.items()}

        payload: dict[str, Any] = {
            "player": self.player,
            "ordinal": self.ordinal,
            "request_type": self.request_type,
            "option_count": self.option_count,
            "base_option": self.base_option,
            "chosen_option": self.chosen_option,
            "iterations": self.iterations,
            "depth": self.depth,
            "particles": self.particles,
            "value": self.value,
            "searched": self.searched,
        }
        if include_keys:
            payload["strategy"] = _keys(self.strategy)
            payload["action_values"] = _keys(self.action_values)
            payload["visits"] = {str(k): int(v) for k, v in self.visits.items()}
            payload["prior"] = None if self.prior is None else _keys(self.prior)
        return payload


@dataclass
class CfrMatchResult:
    """A CFR search match: the game record plus per-decision root solutions."""

    record: MatchRecord
    searches: list[CFRSearchDecision]

    @property
    def winner(self) -> int | None:
        return self.record.winner

    def as_dict(self, *, include_searches: bool = False) -> dict[str, Any]:
        payload = self.record.as_dict()
        payload["searched_decisions"] = len(self.searches)
        if include_searches:
            payload["searches"] = [s.as_dict() for s in self.searches]
        return payload


# --------------------------------------------------------------------------- #
# CFR search policy
# --------------------------------------------------------------------------- #
@dataclass
class CFRSearchPolicy:
    """A ``Policy`` that refines each decision with engine MCCFR root solving.

    Mirrors ``envs.fork_search.ForkSearchPolicy``'s budget knobs (``search_every``,
    ``max_searches``) and always consults the base policy once per decision so its
    RNG advances identically with or without search.
    """

    player: int
    context: SearchContext
    deck0: str = ""
    deck1: str = ""
    base: Policy | None = None
    prior: Any = None
    iterations: int = 8
    depth: int = 1
    candidate_limit: int | None = None
    value_spec: str = ""
    search_every: int = 1
    max_searches: int | None = None
    max_steps: int = 5000
    belief_root: bool = True
    belief_samples: int = 2
    request_types: tuple[str, ...] | None = None
    probe_fn: Any = None
    name: str = field(default="cfr_search")
    _calls: int = field(default=0, init=False, repr=False)
    _n_searched: int = field(default=0, init=False, repr=False)

    def choose(self, built: Any) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        if not codes:
            return -1
        self._calls += 1
        base_code = int(self.base.choose(built)) if self.base is not None else codes[0]
        base_idx = codes.index(base_code) if base_code in codes else 0

        request_type = str(getattr(built.context.request_type, "value", built.context.request_type))
        if (
            self.context.boundary is None
            or len(codes) == 1
            or self.iterations < 1
            or (self.request_types is not None and request_type not in self.request_types)
            or (self.search_every > 1 and self._calls % self.search_every != 0)
            or (self.max_searches is not None and self._n_searched >= self.max_searches)
        ):
            return base_code

        self._n_searched += 1
        chosen = self._search(built, codes, base_idx)
        if chosen is None:
            # Root unavailable (e.g. an early ``initHands`` boundary where hands
            # are not dealt, so a public-belief determinization cannot be built):
            # skip without consuming the search budget. Never fall back to the
            # live hidden-truth snapshot (charter §5.2).
            self._n_searched -= 1
            return base_code
        return chosen

    def _root_particles(self, snapshot: str, acting: int) -> list[RootParticle] | None:
        context = self.context
        if self.belief_root and context.deck_cards[1 - acting]:
            try:
                return belief_root_particles(
                    snapshot,
                    root_player=acting,
                    opponent_deck_cards=context.deck_cards[1 - acting],
                    samples=max(1, int(self.belief_samples)),
                    tracker=context.trackers[acting],
                )
            except Exception:
                return None
        return [RootParticle(state=snapshot, weight=1.0, label="live")]

    def _search(self, built: Any, codes: list[int], base_idx: int) -> int | None:
        from reps.action_hierarchy import semantic_action_key_for_spec

        context = self.context
        acting = int(self.player)
        ordinal = len(context.records[acting])
        snapshot = context.boundary[0]  # type: ignore[index]

        roots = self._root_particles(snapshot, acting)
        if roots is None:
            return None

        specs = tuple(built.context.legal_low_level_specs)
        keys: list[Any] = []
        for index, code in enumerate(codes):
            if index < len(specs):
                keys.append(semantic_action_key_for_spec(specs[index]))
            else:
                from reps.action_hierarchy import try_semantic_action_key_for_code

                keys.append(try_semantic_action_key_for_code(int(code)) or ("unknown", int(code)))

        prior_by_key: dict[Any, float] = {}
        prior_log: dict[Any, float] | None = None
        if self.prior is not None and hasattr(self.prior, "distribution"):
            distribution = list(self.prior.distribution(built))
            if len(distribution) == len(codes):
                prior_by_key = {keys[i]: float(distribution[i]) for i in range(len(keys))}
                prior_log = dict(prior_by_key)
        required_key = keys[base_idx]

        game = EngineParticleGame(
            bridge=context.bridge,
            deck0=self.deck0,
            deck1=self.deck1,
            game_attrs=context.game_attrs,
            root_fn=lambda: roots,
            root_player=acting,
            prior_by_key=prior_by_key,
            required_key=required_key,
            candidate_limit=self.candidate_limit,
            value_spec=self.value_spec,
            max_steps=self.max_steps,
            probe_fn=self.probe_fn,
        )
        seed = derive_seed(context.base_seed, "cfr-search", acting, ordinal)
        solution: RootSolution = cfr_solve(
            game,
            iterations=max(0, int(self.iterations)),
            depth=max(0, int(self.depth)),
            root_player=acting,
            seed=seed,
        )

        chosen_key = solution.best_action()
        option = game.root_option_for_key(chosen_key) if chosen_key is not None else None
        chosen_idx = option if option is not None else base_idx

        context.searches.append(
            CFRSearchDecision(
                player=acting,
                ordinal=ordinal,
                request_type=str(getattr(built.context.request_type, "value", built.context.request_type)),
                option_count=len(codes),
                base_option=base_idx,
                chosen_option=chosen_idx,
                iterations=solution.iterations,
                depth=solution.depth,
                particles=solution.particles,
                value=float(solution.value),
                strategy=dict(solution.strategy),
                action_values=dict(solution.action_values),
                visits=dict(solution.visits),
                prior=prior_log,
            )
        )
        return int(codes[chosen_idx])


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_cfr_search_match(
    deck0: str,
    deck1: str,
    *,
    bridge: ForkBridge,
    seed: int,
    search_players: Sequence[int] = (0,),
    iterations: int = 8,
    depth: int = 1,
    candidate_limit: int | None = None,
    value_spec: str = "",
    search_every: int = 1,
    max_searches: int | None = None,
    max_steps: int = 5000,
    version: str | None = None,
    game_attrs: Mapping[int, int] | None = None,
    base_policies: tuple[Policy | None, Policy | None] | None = None,
    prior_policies: tuple[Any | None, Any | None] | None = None,
    belief_root: bool = True,
    belief_samples: int = 2,
    request_types: tuple[str, ...] | None = None,
    probe_fn: Any = None,
    record_final_state: bool = False,
    built_sink: Any = None,
    full_state_for_agents: bool = False,
) -> CfrMatchResult:
    """Play one game in which ``search_players`` solve each decision with MCCFR.

    Mirrors ``envs.fork_search.run_search_match`` (subprocess-only forking, boundary
    snapshots, positional records); non-search players use an expert policy for
    their deck. Every searched decision logs a ``CFRSearchDecision`` with the root
    average strategy + root value (the S-C teacher targets).

    ``built_sink(player, built, code)`` is called once per decision (after the
    positional record is appended), so a distillation collector can join the
    recorded decision contexts to the logged ``CFRSearchDecision`` targets by
    ``(player, ordinal)``. ``full_state_for_agents`` gives each ``PolicyPlayer`` a
    god-view ``context.full_state`` for **label** encoding only (belief/privileged
    targets); the search root still comes from public-belief determinization and
    only ``player_view`` feeds decision logic, so nothing leaks (charter §5.2).
    """
    from reps.action_hierarchy import reset_default_hierarchical_action_codebook

    if bridge.workers == 0:
        raise ValueError(
            "run_cfr_search_match needs a subprocess fork bridge (workers >= 1): "
            "forking inside a live game callback corrupts the engine's JS runtime."
        )
    reset_default_hierarchical_action_codebook()
    spec0, spec1 = deck_spec(deck0), deck_spec(deck1)
    create_param = _build_create_param(spec0, spec1, seed=seed, version=version)
    game = Game(create_param=create_param)
    attrs = dict(game_attrs or {})
    for attr, value in attrs.items():
        game.set_attr(int(attr), int(value))

    overrides = base_policies or (None, None)
    priors = prior_policies or (None, None)
    search_set = set(int(p) for p in search_players)

    from reps.public_state import PublicStateTracker

    trackers: tuple[Any, Any] = (None, None)
    if belief_root:
        trackers = (PublicStateTracker(player_id=0), PublicStateTracker(player_id=1))

    def _on_decision(player: int, built: Any, code: int) -> None:
        for tracker in trackers:
            if tracker is not None:
                _observe_public_record(tracker, player, built, code)
        if built_sink is not None:
            built_sink(int(player), built, int(code))

    context = SearchContext(
        bridge=bridge,
        game_attrs=attrs,
        rollout_specs=(None, None),
        base_seed=int(seed),
        belief_root=bool(belief_root),
        belief_samples=max(1, int(belief_samples)),
        deck_cards=(
            tuple(int(card) for card in spec0.cards),
            tuple(int(card) for card in spec1.cards),
        ),
        trackers=trackers,
    )

    policies: list[Policy] = []
    for player, deck_slug in ((0, deck0), (1, deck1)):
        base = overrides[player]
        if base is None:
            base = build_policy(
                f"expert:{deck_slug}",
                seed=derive_seed(seed, f"p{player}", "cfr-base"),
                role=f"p{player}",
            )
        if player in search_set:
            policy: Policy = CFRSearchPolicy(
                player=player,
                context=context,
                deck0=deck0,
                deck1=deck1,
                base=base,
                prior=priors[player],
                iterations=iterations,
                depth=depth,
                candidate_limit=candidate_limit,
                value_spec=value_spec,
                search_every=search_every,
                max_searches=max_searches,
                max_steps=max_steps,
                belief_root=bool(belief_root),
                belief_samples=max(1, int(belief_samples)),
                request_types=request_types,
                probe_fn=probe_fn,
            )
        else:
            policy = base
        policies.append(_RecordingPolicy(policy, player, context.records, on_decision=_on_decision))

    players = [PolicyPlayer(p, policies[p]) for p in (0, 1)]
    if full_state_for_agents:
        provider = lambda: game.state().json()  # noqa: E731 - label-only god view
        players = [
            PolicyPlayer(p, policies[p], full_state_provider=provider) for p in (0, 1)
        ]
    game.set_player(0, players[0])
    game.set_player(1, players[1])

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
    return CfrMatchResult(record=record, searches=list(context.searches))
