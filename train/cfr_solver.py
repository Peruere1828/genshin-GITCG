"""External-sampling MCCFR scaffolding for public-belief search (PLAN.md M3-CPU S-B).

This module closes gaps **G2 + G4** of ``train/CONTINUAL_RESOLVING.md``:

* **G2 regret solver** -- the practical resolver (``envs/fork_search``) evaluates
  candidates with Monte-Carlo rollouts, but the GT-CFR main line needs a regret
  solver that returns a *root average strategy* and a *root value*;
* **G4 depth-limited search** -- leaves are scored by a network *frontier value*
  (CVPN value / ``train/resolver.py``) instead of playing every rollout to the
  terminal.

What it provides
----------------
* ``ExternalSamplingMCCFR`` / ``solve`` -- external-sampling MCCFR (Lanctot et
  al., 2009) with regret matching, a running root average strategy and a root
  value. The clean interface ``solve(game, iterations, depth)`` returns a
  ``RootSolution`` that S-C distills back into the network (policy CE + value
  regression + belief scoring).
* ``regret_matching`` -- positive-regret normalisation with a **prior** fallback
  when all regrets are zero. The prior *orders* actions (and seeds the fallback);
  it never deletes a legal action (§5.4 / §11).
* ``belief_root_particles`` / ``public_infoset_key`` -- how the search root and
  the information sets are built from ``reps.belief``'s public-belief
  determinization (S-A): the root is a set of sampled determinizations and the
  infoset key uses only public fields, so **no hidden truth enters the search**
  (charter §5.2 / §7.2).

Searching a public-belief particle tree
---------------------------------------
The solver is engine-free: it only knows the abstract ``ParticleGame`` protocol.
An implementation exposes the root **public-belief particles** (sampled
determinizations, weighted), the local node operations (terminal / acting player
/ legal actions / infoset / transition) and two value oracles -- the terminal
utility and the depth-limited frontier value, both from the (zero-sum) traverser's
perspective. External sampling then samples a root particle per traversal and one
opponent action at opponent nodes, while the traverser enumerates every legal
action and accumulates regret.

The toy Kuhn poker and bandit in ``tests/test_cfr_solver.py`` pin convergence; the
engine-backed adapter (``reps.belief`` roots + fork transitions + CVPN frontier)
implements the same protocol and is the S-C integration point.

Safety invariants (violating them is a rework)
----------------------------------------------
* roots/infosets come from public information only (§5.2);
* ``legal_actions`` is never truncated permanently by a prior (§5.4 / §11);
* depth is bounded: ``depth_remaining == 0`` (or a non-terminal leaf with no legal
  actions) returns the frontier value instead of recursing forever.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

_EPS = 1.0e-8


# --------------------------------------------------------------------------- #
# data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchAction:
    """A legal search action: a stable semantic key plus an optional prior.

    ``key`` is a semantic action key (see AGENTS.md "训练标签/embedding 用语义
    key"), stable across determinizations. ``prior`` is a policy score used only
    to seed regret matching/ordering -- it never removes the action.
    """

    key: str
    prior: float = 0.0


@dataclass(frozen=True)
class RootParticle:
    """One sampled determinization of the belief root (a chance outcome).

    ``state`` is game-defined (e.g. a reconstructed engine state JSON). ``weight``
    is the sampling weight (belief posterior); if every weight is non-positive the
    solver samples uniformly.
    """

    state: Any
    weight: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class RootSolution:
    """The solver's output: root average strategy + root value for ``root_player``."""

    strategy: dict[str, float]
    value: float
    iterations: int
    depth: int
    root_player: int
    action_values: dict[str, float] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    particles: int = 0

    def best_action(self) -> str | None:
        """Highest-probability root action (ties -> lexicographically smallest)."""
        if not self.strategy:
            return None
        return max(self.strategy, key=lambda key: (self.strategy[key], key))

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": {str(k): float(v) for k, v in self.strategy.items()},
            "value": float(self.value),
            "iterations": int(self.iterations),
            "depth": int(self.depth),
            "root_player": int(self.root_player),
            "action_values": {str(k): float(v) for k, v in self.action_values.items()},
            "visits": {str(k): int(v) for k, v in self.visits.items()},
            "particles": int(self.particles),
        }


# --------------------------------------------------------------------------- #
# game protocol
# --------------------------------------------------------------------------- #
class ParticleGame(Protocol):
    """Abstract depth-limited search tree over belief particles.

    Nodes are opaque to the solver. Implementations must guarantee:

    * ``root_particles`` returns the public-belief root particles (sampled
      determinizations) with non-negative weights; roots depend on public
      information only (``reps.belief``), never hidden truth;
    * ``legal_actions`` returns every legal action (prior scores ride along on
      ``SearchAction.prior``) -- a prior ranks, it does not delete;
    * ``terminal_utility`` / ``frontier_value`` are from the traverser's
      perspective (zero-sum: ``f(node, 1) == -f(node, 0)``).
    """

    def root_particles(self) -> Sequence[RootParticle]:
        ...

    def is_terminal(self, node: Any) -> bool:
        ...

    def acting_player(self, node: Any) -> int:
        ...

    def legal_actions(self, node: Any) -> Sequence[SearchAction]:
        ...

    def infoset_key(self, node: Any) -> Hashable:
        ...

    def transition(self, node: Any, action_key: str) -> Any:
        ...

    def terminal_utility(self, node: Any, traverser: int) -> float:
        ...

    def frontier_value(self, node: Any, traverser: int) -> float:
        ...


PriorFn = Callable[[Any, Sequence[SearchAction]], "Mapping[str, float] | None"]


# --------------------------------------------------------------------------- #
# regret matching / distributions
# --------------------------------------------------------------------------- #
def _prior_weights(
    actions: Sequence[SearchAction], prior: Mapping[str, float] | None
) -> list[float]:
    """Per-action prior weights; an explicit ``prior`` overrides ``action.prior``."""
    weights: list[float] = []
    for action in actions:
        if prior is not None and action.key in prior:
            value = float(prior[action.key])
        else:
            value = float(action.prior)
        weights.append(max(0.0, value))
    return weights


def regret_matching(
    regrets: Mapping[str, float],
    actions: Sequence[SearchAction],
    *,
    prior: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Positive-regret matching with a prior/uniform fallback.

    ``regrets`` are the accumulated regrets per semantic action key. Regrets are
    clamped at zero (a negative regret carries no information); if every clamped
    regret is zero the distribution falls back to ``prior`` (normalised) and then
    to uniform. The support is always the full legal action set: a prior only
    orders/seeds -- it never removes an action (§5.4 / §11).
    """
    keys = [action.key for action in actions]
    if not keys:
        return {}
    positives = [max(0.0, float(regrets.get(key, 0.0))) for key in keys]
    total = sum(positives)
    if total > _EPS:
        return {key: value / total for key, value in zip(keys, positives, strict=True)}
    fallback = _prior_weights(actions, prior)
    fallback_total = sum(fallback)
    if fallback_total > _EPS:
        return {
            key: value / fallback_total
            for key, value in zip(keys, fallback, strict=True)
        }
    uniform = 1.0 / float(len(keys))
    return {key: uniform for key in keys}


def normalize_distribution(
    distribution: Mapping[str, float], *, uniform_if_empty: bool = False
) -> dict[str, float]:
    """Normalise non-negative weights to a distribution.

    All-zero (or empty) input returns an empty mapping unless
    ``uniform_if_empty`` is set, in which case a uniform distribution is returned.
    """
    keys = list(distribution.keys())
    if not keys:
        return {}
    values = [max(0.0, float(distribution[key])) for key in keys]
    total = sum(values)
    if total > _EPS:
        return {key: value / total for key, value in zip(keys, values, strict=True)}
    if uniform_if_empty:
        uniform = 1.0 / float(len(keys))
        return {key: uniform for key in keys}
    return {}


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class ExternalSamplingMCCFR:
    """External-sampling MCCFR over a ``ParticleGame``.

    Each iteration traverses once per player (``root_player`` and its opponent),
    sampling a fresh root particle per traversal. At a node owned by the
    traverser every legal action is enumerated (regret update); at an opponent
    node a single action is sampled from the current strategy. The solver keeps a
    running **root** average strategy and a root value; regret tables are kept per
    information-set key.
    """

    def __init__(
        self,
        game: ParticleGame,
        *,
        root_player: int,
        seed: int = 0,
        prior_fn: PriorFn | None = None,
    ) -> None:
        self.game = game
        self.root_player = int(root_player)
        self.seed = int(seed)
        self.prior_fn = prior_fn
        self._rng = random.Random(int(seed))
        self._regrets: dict[Hashable, dict[str, float]] = {}
        self._root_strategy_sum: dict[str, float] = {}
        self._root_action_value_sum: dict[str, float] = {}
        self._root_action_value_count: dict[str, int] = {}
        self._root_value_sum = 0.0
        self._root_traversals = 0
        self.iterations_done = 0

    # -- public API -------------------------------------------------------- #
    def solve(self, iterations: int, depth: int) -> RootSolution:
        """Run ``iterations`` external-sampling sweeps with a depth budget."""
        iterations = max(0, int(iterations))
        depth = max(0, int(depth))
        particles = tuple(self.game.root_particles())
        for _ in range(iterations):
            for traverser in self._traverser_order():
                particle = self._sample_particle(particles)
                if particle is None:
                    continue
                self._cfr(
                    node=particle.state,
                    traverser=traverser,
                    depth_remaining=depth,
                    is_root=True,
                )
            self.iterations_done += 1
        strategy = self._root_average_strategy(particles)
        value = (
            self._root_value_sum / float(self._root_traversals)
            if self._root_traversals
            else 0.0
        )
        action_values = {
            key: self._root_action_value_sum[key] / float(count)
            for key, count in self._root_action_value_count.items()
            if count > 0
        }
        return RootSolution(
            strategy=strategy,
            value=float(value),
            iterations=self.iterations_done,
            depth=depth,
            root_player=self.root_player,
            action_values={key: float(v) for key, v in action_values.items()},
            visits=dict(self._root_action_value_count),
            particles=len(particles),
        )

    # -- traversal --------------------------------------------------------- #
    def _traverser_order(self) -> tuple[int, int]:
        return (self.root_player, 1 - self.root_player)

    def _cfr(
        self, *, node: Any, traverser: int, depth_remaining: int, is_root: bool
    ) -> float:
        game = self.game
        if game.is_terminal(node):
            value = float(game.terminal_utility(node, traverser))
            self._record_root_leaf_value(value, traverser=traverser, is_root=is_root)
            return value
        actions = tuple(game.legal_actions(node))
        if not actions or depth_remaining <= 0:
            # G4: depth budget exhausted -> network frontier value, not a rollout.
            value = float(game.frontier_value(node, traverser))
            self._record_root_leaf_value(value, traverser=traverser, is_root=is_root)
            return value

        acting = int(game.acting_player(node))
        infoset = game.infoset_key(node)
        regrets = self._regrets.get(infoset)
        if regrets is None:
            regrets = {}
            self._regrets[infoset] = regrets
        strategy = regret_matching(regrets, actions, prior=self._prior_for(node, actions))

        if acting == traverser:
            # External sampling: enumerate *every* legal action for the traverser.
            action_values: dict[str, float] = {}
            node_value = 0.0
            for action in actions:
                child = game.transition(node, action.key)
                if child is None:
                    value = float(game.frontier_value(node, traverser))
                else:
                    value = self._cfr(
                        node=child,
                        traverser=traverser,
                        depth_remaining=depth_remaining - 1,
                        is_root=False,
                    )
                action_values[action.key] = float(value)
                node_value += float(strategy[action.key]) * float(value)
            for action in actions:
                regrets[action.key] = float(regrets.get(action.key, 0.0)) + (
                    action_values[action.key] - node_value
                )
            if is_root and acting == self.root_player:
                for action in actions:
                    self._root_strategy_sum[action.key] = (
                        self._root_strategy_sum.get(action.key, 0.0)
                        + float(strategy[action.key])
                    )
            if is_root and traverser == self.root_player:
                for action in actions:
                    self._root_action_value_sum[action.key] = (
                        self._root_action_value_sum.get(action.key, 0.0)
                        + action_values[action.key]
                    )
                    self._root_action_value_count[action.key] = (
                        self._root_action_value_count.get(action.key, 0) + 1
                    )
                self._root_value_sum += float(node_value)
                self._root_traversals += 1
            return float(node_value)

        # Opponent node: sample a single action from the current strategy.
        sampled = self._sample_action(strategy, actions)
        child = game.transition(node, sampled.key)
        if child is None:
            return float(game.frontier_value(node, traverser))
        return self._cfr(
            node=child,
            traverser=traverser,
            depth_remaining=depth_remaining - 1,
            is_root=False,
        )

    # -- helpers ----------------------------------------------------------- #
    def _record_root_leaf_value(
        self, value: float, *, traverser: int, is_root: bool
    ) -> None:
        """Record a leaf value at the root (e.g. the depth-0 frontier value)."""
        if is_root and traverser == self.root_player:
            self._root_value_sum += float(value)
            self._root_traversals += 1

    def _prior_for(
        self, node: Any, actions: Sequence[SearchAction]
    ) -> Mapping[str, float] | None:
        if self.prior_fn is None:
            return None
        return self.prior_fn(node, actions)

    def _sample_particle(self, particles: Sequence[RootParticle]) -> RootParticle | None:
        if not particles:
            return None
        total = sum(max(0.0, float(p.weight)) for p in particles)
        if total <= _EPS:
            return particles[self._rng.randrange(len(particles))]
        threshold = self._rng.random() * total
        running = 0.0
        for particle in particles:
            running += max(0.0, float(particle.weight))
            if threshold <= running:
                return particle
        return particles[-1]

    def _sample_action(
        self, strategy: Mapping[str, float], actions: Sequence[SearchAction]
    ) -> SearchAction:
        threshold = self._rng.random()
        running = 0.0
        last = actions[-1]
        for action in actions:
            running += max(0.0, float(strategy.get(action.key, 0.0)))
            last = action
            if threshold <= running:
                return action
        return last

    def _root_average_strategy(
        self, particles: Sequence[RootParticle]
    ) -> dict[str, float]:
        if self._root_strategy_sum:
            total = sum(max(0.0, v) for v in self._root_strategy_sum.values())
            if total > _EPS:
                return {
                    key: max(0.0, value) / total
                    for key, value in self._root_strategy_sum.items()
                }
        # No traverser expansion at the root (e.g. depth == 0): fall back to the
        # root table's regret matching (prior/uniform) so the strategy stays valid.
        if not particles:
            return {}
        root_node = particles[0].state
        if self.game.is_terminal(root_node):
            return {}
        actions = tuple(self.game.legal_actions(root_node))
        if not actions:
            return {}
        regrets = self._regrets.get(self.game.infoset_key(root_node), {})
        return regret_matching(
            regrets, actions, prior=self._prior_for(root_node, actions)
        )


def solve(
    game: ParticleGame,
    iterations: int,
    depth: int,
    *,
    root_player: int | None = None,
    seed: int = 0,
    prior_fn: PriorFn | None = None,
) -> RootSolution:
    """Convenience wrapper: ``solve(game, iterations, depth) -> RootSolution``.

    ``root_player`` defaults to the acting player of the first root particle.
    A fresh solver is built per call, so identical inputs reproduce the solution.
    """
    if root_player is None:
        particles = tuple(game.root_particles())
        if not particles:
            raise ValueError("ParticleGame.root_particles() returned no particles")
        root_player = int(game.acting_player(particles[0].state))
    solver = ExternalSamplingMCCFR(
        game, root_player=int(root_player), seed=seed, prior_fn=prior_fn
    )
    return solver.solve(iterations, depth)


# --------------------------------------------------------------------------- #
# public-belief rooting (S-A glue) + infoset key
# --------------------------------------------------------------------------- #
def belief_root_particles(
    scaffold_state_json: str,
    *,
    root_player: int,
    opponent_deck_cards: Sequence[int],
    samples: int = 4,
    tracker: Any = None,
    synthetic_hidden_ids: bool = True,
) -> list[RootParticle]:
    """Sample the search root as weighted determinizations of the public belief.

    Thin wrapper over ``reps.belief.sample_belief_roots`` (S-A): the boundary
    snapshot is masked to ``root_player``'s public view, opponent hidden zones are
    overwritten with sampled particles (synthetic hidden ids), and the returned
    roots never depend on hidden truth. Weights come from the public belief's
    opponent-range hypotheses (normalised by the solver at sampling time).

    The actual tree expansion (transition) is game-specific and supplied by the
    S-C engine adapter; this function only fixes the *root* contract.
    """
    from reps.belief import build_public_belief_state, sample_belief_roots
    from reps.public_state import mask_state_for_player
    from reps.snapshot import snapshot_state

    root_player = int(root_player)
    samples = max(1, int(samples))
    roots = sample_belief_roots(
        scaffold_state_json,
        root_player=root_player,
        opponent_deck_cards=opponent_deck_cards,
        samples=samples,
        tracker=tracker,
        synthetic_hidden_ids=synthetic_hidden_ids,
    )
    state = snapshot_state(None, scaffold_state_json)
    public_state = mask_state_for_player(state, perspective_player=root_player)
    belief = build_public_belief_state(
        public_state=public_state,
        root_player=root_player,
        opponent_deck_cards=opponent_deck_cards,
        tracker=tracker,
        belief_samples=samples,
        opponent_dice_count=len(state.players[1 - root_player].dice),
    )
    hypotheses = belief.opponent_range.hypotheses or (None,)
    particles: list[RootParticle] = []
    for index, root in enumerate(roots):
        hypothesis = hypotheses[index % len(hypotheses)]
        weight = float(hypothesis.weight) if hypothesis is not None else 1.0
        particles.append(
            RootParticle(state=root, weight=max(0.0, weight), label=f"particle-{index}")
        )
    return particles


def public_infoset_key(
    state: Any,
    *,
    acting_player: int,
    request_type: Any = None,
    public_history_len: int = 0,
) -> tuple:
    """Hashable information-set key built from **public** fields only.

    ``state`` is re-masked for ``acting_player`` (``reps.public_state``), so the
    opponent's hand/pile definition ids and dice are erased before the key is
    built -- two states that differ only in hidden truth produce the same key
    (charter §5.2). ``public_history_len`` lets a continuous resolver fold the
    public action history into the key.
    """
    from reps.public_state import mask_state_for_player

    masked = mask_state_for_player(state, perspective_player=int(acting_player))
    player_signatures = tuple(
        _public_player_signature(player) for player in masked.players
    )
    return (
        str(masked.phase),
        int(masked.round_number),
        int(masked.current_turn),
        int(masked.iterator_id),
        int(acting_player),
        None if request_type is None else str(request_type),
        int(public_history_len),
        player_signatures,
    )


def _public_player_signature(player: Any) -> tuple:
    """Public (no hidden identity) signature of one player snapshot."""
    return (
        int(player.active_character_id),
        tuple(
            (
                int(character.definition_id),
                int(character.health),
                int(character.max_health),
                int(character.energy),
                int(character.max_energy),
                int(character.defeated),
                bool(character.is_active),
                str(character.aura),
                tuple(sorted(int(entity.definition_id) for entity in character.entities)),
            )
            for character in player.characters
        ),
        # Hidden zones contribute only counts + (for the acting player) definition
        # ids; the mask zeroes the opponent's definition ids, so no hidden identity
        # can survive into the key.
        tuple(sorted(int(card.definition_id) for card in player.hand_cards)),
        len(player.pile_cards),
        len(player.dice),
        tuple(sorted(int(item.definition_id) for item in player.supports)),
        tuple(sorted(int(item.definition_id) for item in player.summons)),
        tuple(sorted(int(item.definition_id) for item in player.combat_statuses)),
    )
