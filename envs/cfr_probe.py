"""Single-step fork probe for the engine MCCFR adapter (PLAN.md M3-CPU S-C).

``envs.cfr_search`` searches a belief-particle tree whose nodes are engine
decisions. The fork bridge's rollout task (``envs.fork_bridge.run_fork_task``)
answers "what happens if I force one option and play out?", but MCCFR needs the
*shape* of the tree: at each node, who acts, what is legal, whether it is
terminal, and a depth-limited frontier value -- without rolling out.

This module adds that primitive as ``ForkTask(kind="probe")``:

* fork the boundary snapshot (an ``is_resumable()`` point) in the worker, mirror
  the game-level attrs, reset the action codebook;
* replay **both** players' recorded per-player prefixes (the same positional
  contract the rollout task uses; the engine re-interleaves them);
* the first time a player is asked *beyond* its prefix, that decision is the
  node: capture its public information (acting player, semantic action keys,
  request type, public infoset key, frontier value) and stop.

Why this is safe / on-contract
------------------------------
* forking happens in a subprocess (``envs.fork_bridge``), never inside a live
  engine callback (D13 discipline C4/C5);
* the fork starts from an ``is_resumable()`` boundary snapshot only;
* infoset keys are built by ``train.cfr_solver.public_infoset_key`` from the
  masked ``notification.state`` public view -- no hidden truth enters the key
  (charter §5.2);
* the frontier value is a *public* estimate: a CVPN value head when a checkpoint
  is supplied, else the documented heuristic
  (:func:`heuristic_frontier_probability`). It is player-0 win probability.

The captured node is described by ``(snapshot, prefix0, prefix1)``: children are
re-probed from the same boundary with one more option appended to the acting
player's prefix, so every node is a fresh, deterministic replay of a boundary
snapshot -- no node ever depends on a non-boundary snapshot.
"""

from __future__ import annotations

from typing import Any

from envs.fork_bridge import ForkTask, _jsonable_key


# --------------------------------------------------------------------------- #
# frontier value (pluggable; heuristic fallback)
# --------------------------------------------------------------------------- #
def heuristic_frontier_probability(state: Any) -> float:
    """Documented heuristic fallback for the depth-limited frontier value.

    Returns an estimate of **player 0's win probability** in ``[0, 1]`` from the
    public board state. The score is a weighted board differential: remaining
    character health (dominant), surviving characters, hand size and dice count --
    all of which the masked public view keeps for both players. It is symmetric
    (swapping players maps ``p -> 1 - p``) and zero-sum once centred, so it needs
    no learned parameters -- the CPU-tier stand-in until a CVPN value head is
    supplied (S-C: ``PriorResolver.value`` / the belief+value heads).

    This is a *rough* public estimate, not a trained value: it exists so the
    depth-limited search is well-defined without a checkpoint, and it is clearly
    labelled as the fallback in the decision log's ``value_spec`` provenance.
    """
    players = state.players

    def score(player: Any) -> float:
        health = float(sum(max(0, int(ch.health)) for ch in player.characters))
        alive = float(sum(1 for ch in player.characters if not ch.defeated))
        hand = float(len(player.hand_cards))
        dice = float(len(player.dice))
        return health + 3.0 * alive + 0.5 * hand + 0.25 * dice

    s0, s1 = score(players[0]), score(players[1])
    total = s0 + s1
    if total <= 0:
        return 0.5
    return float(s0 / total)


_VALUE_CACHE: dict[str, Any] = {}


def _frontier_probability(built: Any, *, value_spec: str) -> float:
    """Player-0 win-probability estimate for a captured decision context."""
    if value_spec:
        try:
            resolver = _VALUE_CACHE.get(value_spec)
            if resolver is None:
                from train.resolver import prior_resolver_from_checkpoint

                resolver = prior_resolver_from_checkpoint(value_spec)
                _VALUE_CACHE[value_spec] = resolver
            probability = float(resolver.value_probability(built))
            # ``PriorResolver`` scores the *acting* player's view; the probe field
            # is player-0 relative. Flipping only when the acting player is 1 fixes
            # the sign at turn hand-offs without changing same-player leaves.
            acting = int(getattr(built.context, "acting_player", 0) or 0)
            return probability if acting == 0 else 1.0 - probability
        except Exception:
            # A missing/incompatible checkpoint must not break the search: fall
            # back to the documented public heuristic.
            pass
    return heuristic_frontier_probability(built.context.player_view)


# --------------------------------------------------------------------------- #
# probe task
# --------------------------------------------------------------------------- #
class _ProbeState:
    """Shared capture slot for both players' probe policies."""

    def __init__(self, *, prefix0: tuple[int, ...], prefix1: tuple[int, ...], value_spec: str) -> None:
        self.totals = {0: len(prefix0), 1: len(prefix1)}
        self.value_spec = value_spec
        self.captured: dict[str, Any] | None = None

    def record(self, who: int, built: Any, codes: list[int]) -> None:
        if self.captured is not None:
            return
        from reps.action_hierarchy import (
            semantic_action_key_for_spec,
            try_low_level_spec_for_code,
            try_semantic_action_key_for_code,
        )
        from train.cfr_solver import public_infoset_key

        context = built.context
        specs = tuple(context.legal_low_level_specs)
        keys: list[Any] = []
        for index, code in enumerate(codes):
            spec = specs[index] if index < len(specs) else try_low_level_spec_for_code(int(code))
            key = (
                semantic_action_key_for_spec(spec)
                if spec is not None
                else try_semantic_action_key_for_code(int(code)) or ("unknown", int(code))
            )
            keys.append(key)
        request_type = context.request_type
        player_view = context.player_view
        infoset = public_infoset_key(
            player_view,
            acting_player=int(who),
            request_type=getattr(request_type, "value", request_type),
            public_history_len=int(self.totals[0] + self.totals[1]),
        )
        self.captured = {
            "acting_player": int(who),
            "legal_options": [int(i) for i in range(len(codes))],
            "legal_keys": [_jsonable_key(key) for key in keys],
            "request_type": str(getattr(request_type, "value", request_type)),
            "infoset_key": _jsonable_key(infoset),
            "frontier_p0": float(_frontier_probability(built, value_spec=self.value_spec)),
        }


class _ProbePolicy:
    """Replay one player's prefix, then capture the next decision."""

    def __init__(self, who: int, prefix: tuple[int, ...], shared: _ProbeState) -> None:
        self.who = int(who)
        self.prefix = tuple(int(i) for i in prefix)
        self.shared = shared

    def choose(self, built: Any) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        if self.shared.captured is not None:
            return int(codes[0]) if codes else -1
        if self._replayed < len(self.prefix):
            index = self.prefix[self._replayed]
            self._replayed += 1
            if not codes:
                return -1
            return int(codes[min(max(index, 0), len(codes) - 1)])
        # Prefix exhausted for this player -> this is the frontier decision.
        self.shared.record(self.who, built, codes)
        return int(codes[0]) if codes else -1

    _replayed: int = 0


def run_probe_task(task: ForkTask) -> dict[str, Any]:
    """Execute one ``kind="probe"`` task in the current process.

    Returns a JSON-serializable dict shaped like :class:`envs.fork_bridge.ForkResult`.
    """
    from gitcg import GameStatus

    from envs.match import PolicyPlayer
    from envs.snapshot import fork_game
    from reps.action_hierarchy import reset_default_hierarchical_action_codebook

    reset_default_hierarchical_action_codebook()
    game = fork_game(task.snapshot, game_attrs=task.game_attrs)
    shared = _ProbeState(
        prefix0=tuple(task.prefix0), prefix1=tuple(task.prefix1), value_spec=str(task.value_spec)
    )
    for player in (0, 1):
        prefix = tuple(task.prefix0) if player == 0 else tuple(task.prefix1)
        game.set_player(player, PolicyPlayer(player, _ProbePolicy(player, prefix, shared)))

    error: str | None = None
    truncated = False
    steps = 0
    try:
        game.start()
        while game.is_running():
            if steps >= task.max_steps:
                truncated = True
                break
            game.step()
            steps += 1
            if shared.captured is not None:
                break
    except Exception as exc:  # engine/IO failures are otherwise swallowed by cffi
        error = f"{type(exc).__name__}: {exc}"

    winner: int | None = None
    rounds = 0
    try:
        rounds = int(game.round_number())
        winner = game.winner()
    except Exception:
        winner = None
    if error is None and game.status() == GameStatus.ABORTED:
        try:
            error = f"aborted: {game.error()}"
        except Exception:
            error = "aborted"

    captured = shared.captured or {}
    node_terminal = shared.captured is None and error is None and not truncated
    payload: dict[str, Any] = {
        "index": task.index,
        "kind": "probe",
        "winner": winner if node_terminal else None,
        "rounds": rounds,
        "error": error,
        "truncated": truncated,
        "tag": task.tag,
        "node_terminal": bool(node_terminal),
        "acting_player": captured.get("acting_player"),
        "legal_options": captured.get("legal_options", []),
        "legal_keys": captured.get("legal_keys", []),
        "request_type": captured.get("request_type"),
        "infoset_key": captured.get("infoset_key"),
        "frontier_p0": captured.get("frontier_p0"),
    }
    return payload
