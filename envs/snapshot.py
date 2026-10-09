"""Mid-game snapshot / clone / branch utilities + the exact fork contract.

This is the WS0 part of the L5.4 "S1 engine bridge" investigation (PLAN.md §4.2,
decisions D12/D13). It documents *precisely* what the ``gitcg`` 0.21.0 pybinding
can and cannot do with a mid-game state.

Verified experimentally on 2026-10-09 (gitcg 0.21.0, engine commit in
``configs/engine.lock``; see ``scripts/probe_engine_snapshot.py`` and
``scripts/probe_boundary_fork.py`` for the runs and ``reports/engine/`` for the
evidence):

* **Round-trip is faithful.** ``State(json=snap).json() == snap`` exactly — the
  serializer keeps random/id iterators, the whole entity graph and the M/N maps.
  (``GameState.data`` is dropped and re-attached from the versioned data table.)
* **A snapshot can be loaded and continued.** ``Game(state=State(json=snap))`` +
  ``start()``/``step()`` runs to ``FINISHED`` without error.
* **Resumed branches are clone-deterministic.** Two resumes from the same snapshot
  with the same policies produce byte-identical terminal state JSON.
* **Boundary snapshots are exact forks.** Snapshots taken at ``canResume:true``
  pause points (``Game.is_resumable()``) resumed with the *same decision sequence*
  reproduce the live terminal byte-for-byte (13/13 boundary snapshots in the
  record-replay probe). The ``canResume`` bit is a reliable replay-safe marker.

Two rules make forking sound — breaking either reproduces the historical
"snapshot cannot fork" conclusion (PLAN.md §0.1 I9, since corrected by I10):

1. **Only snapshot at ``is_resumable()`` points.** ``canResume:false`` pauses
   (``initHands`` after drawing, mid-``skill_executor``, ``gotWinner``) sit
   *inside* phase functions whose suspended work is not serialized; resuming
   re-enters the phase loop from ``state.phase`` and replays that work
   (0/3 non-boundary snapshots reproduced the live terminal).
2. **Mirror game-level attrs on the fork.** Create-param/game attributes such as
   ``ATTR_PLAYER_ALWAYS_OMNI_*`` live outside ``GameState`` and are silently lost
   by ``Game(state=...)``. Losing them changes dice generation and with it every
   downstream request. Pass them via ``fork_game(..., game_attrs=...)``.

Note: early-game phases issue both players' RPCs concurrently (``Promise.all`` in
``initHands``/``initActives``/``rollPhase``), so the RPC interleaving depends on
JS scheduling rather than state. It is deterministic in practice for a fixed
engine build (covered by the D4 engine-lock freeze) but is not guaranteed across
engine versions.

The alternative for bulk offline labeling is replay branching
(``train/replay_branch.py``): replay from the initial state to the decision
point and inject an action. That path is O(depth) per branch instead of O(1),
but crosses process boundaries trivially and needs no fork bookkeeping.
"""

from __future__ import annotations

from typing import Any, Mapping

from gitcg import Game, State

FORK_LIMITATION = (
    "Forking a game snapshot is exact only at canResume:true pause points "
    "(Game.is_resumable()) and only when game-level attrs outside GameState "
    "(e.g. ATTR_PLAYER_ALWAYS_OMNI_*) are mirrored via fork_game(game_attrs=...). "
    "Snapshots at canResume:false pauses (initHands mid-phase, mid-skill, "
    "gotWinner) are NOT fork-safe: resume re-enters the phase loop and replays "
    "phase-internal work. Record-replay probe 2026-10-09: boundary 13/13 exact, "
    "non-boundary 0/3. See envs/snapshot.py and scripts/probe_boundary_fork.py."
)


def capture_snapshot(game: Game) -> str:
    """Return the JSON snapshot of a live game's current state."""
    return game.state().json()


def state_from_snapshot(snapshot: str) -> State:
    """Rebuild a ``gitcg.State`` from a snapshot string."""
    return State(json=snapshot)


def fork_game(
    snapshot: str, *, game_attrs: Mapping[int, int] | None = None
) -> Game:
    """Create a new ``Game`` from a snapshot (no players attached yet).

    ``game_attrs`` must mirror any game-level attributes the source game had set
    (e.g. ``ATTR_PLAYER_ALWAYS_OMNI_*``): they are **not** part of the serialized
    state and silently change dice/RPC behaviour if lost. The returned game can be
    played with ``set_player`` + ``start``/``step`` and reproduces the live game
    exactly when the snapshot is from an ``is_resumable()`` point and the attached
    players make the same decisions (see module docstring).
    """
    game = Game(state=state_from_snapshot(snapshot))
    for attr, value in dict(game_attrs or {}).items():
        game.set_attr(int(attr), int(value))
    return game


def snapshot_roundtrip_is_faithful(snapshot: str) -> bool:
    """True when serializing the rebuilt state reproduces the snapshot exactly."""
    return state_from_snapshot(snapshot).json() == snapshot
