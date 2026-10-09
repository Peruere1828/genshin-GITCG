"""Decisive probe: are ``canResume:true`` boundary snapshots exact forks?

``scripts/probe_engine_snapshot.py`` concluded that resumed games do not
reproduce the live trajectory -- but two confounds made every resume diverge for
reasons unrelated to the snapshot: (a) its resume attached *fresh* expert
policies, whose internal RNG state was already advanced in the live game, and
(b) its fork dropped game-level attrs (``ATTR_PLAYER_ALWAYS_OMNI_*``), so the
resumed game rolled different dice and offered different reroll option sets.

This probe isolates the engine question: resume a snapshot with **record-replay
policies** that answer every request exactly as the live game did, and mirror
the live game's attrs on the fork. If the terminal state then equals the live
terminal byte-for-byte, the snapshot is an exact fork of that point.

Per snapshot we verify, in order of strength:

1. ``terminal_matches``   resumed terminal JSON == live terminal JSON
2. ``requests_match``     every request signature seen on resume equals the
                          recorded suffix (catches divergence even when the
                          game happens to end the same way)

Expected outcome (verified 2026-10-09, gitcg 0.21.0): ``canResume:true``
boundary snapshots reproduce exactly; ``canResume:false`` pauses (initHands
mid-phase, gotWinner) do not -- the module docstring in ``envs/snapshot.py``
carries the full contract.

Usage::

    python -m scripts.probe_boundary_fork --seed 3 --sample 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from typing import Any

from common.lock import runtime_metadata
from common.paths import ensure_dir, reports_dir
from common.seeding import derive_seed
from envs.match import PolicyPlayer
from envs.policy import Policy, expert_policy
from envs.snapshot import capture_snapshot, fork_game
from gitcg import low_level
from scripts.probe_engine_snapshot import make_initial_game

# Game-level attrs that live outside GameState and must be mirrored on a fork.
# They mirror what ``make_initial_game`` sets on the live game.
GAME_ATTRS: dict[int, int] = {
    low_level.ATTR_PLAYER_ALWAYS_OMNI_0: 1,
    low_level.ATTR_PLAYER_ALWAYS_OMNI_1: 1,
}


def _signature(built: Any) -> tuple:
    context = built.context
    specs = tuple(getattr(spec, "label", str(spec)) for spec in context.legal_low_level_specs)
    return (len(specs), specs)


class RecordingPolicy:
    """Wrap a policy and record each choice as a positional option index."""

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self.name = f"rec:{getattr(inner, 'name', '?')}"
        self.records: list[dict[str, Any]] = []

    def choose(self, built: Any) -> int:
        codes = list(built.context.legal_low_level_codes)
        code = int(self.inner.choose(built))
        self.records.append(
            {
                "n": len(codes),
                "idx": codes.index(code) if code in codes else -1,
                "sig": _signature(built),
            }
        )
        return code


class ReplayPolicy:
    """Answer with previously recorded choices; log every observed request."""

    def __init__(self, records: list[dict[str, Any]], start: int) -> None:
        self.records = records
        self.i = start
        self.name = "replay"
        self.observed: list[tuple] = []

    def choose(self, built: Any) -> int:
        codes = list(built.context.legal_low_level_codes)
        self.observed.append(_signature(built))
        if self.i >= len(self.records):
            raise RuntimeError("replay exhausted: resumed game asked for more decisions")
        rec = self.records[self.i]
        self.i += 1
        if rec["n"] != len(codes):
            raise RuntimeError(
                f"replay mismatch at record {self.i - 1}: expected {rec['n']} options, got {len(codes)}"
            )
        return int(codes[rec["idx"]])


def _attach(game, deck: str, opponent: str, policies: tuple[Any, Any]) -> None:
    game.set_player(0, PolicyPlayer(0, policies[0], metadata={"deck": deck}))
    game.set_player(1, PolicyPlayer(1, policies[1], metadata={"deck": opponent}))


def _play_to_terminal(game, *, max_steps: int = 6000) -> str | None:
    err = None
    try:
        game.start()
        steps = 0
        while game.is_running():
            game.step()
            steps += 1
            if steps > max_steps:
                break
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    return err


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponent", default="natlan_battleship")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--sample", type=int, default=8, help="boundary snapshots to test")
    parser.add_argument("--contrast", type=int, default=2, help="non-boundary snapshots to test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deck, opponent, seed = args.deck, args.opponent, args.seed

    def fresh_policies():
        return (
            expert_policy(deck, seed=derive_seed(seed, "p0", deck)),
            expert_policy(opponent, seed=derive_seed(seed, "p1", opponent)),
        )

    recorders = (RecordingPolicy(fresh_policies()[0]), RecordingPolicy(fresh_policies()[1]))
    game = make_initial_game(deck, opponent, seed)
    _attach(game, deck, opponent, recorders)
    game.start()
    snapshots: list[dict[str, Any]] = []
    step = 0
    while game.is_running():
        game.step()
        step += 1
        snapshots.append(
            {
                "step": step,
                "resumable": bool(game.is_resumable()),
                "snap": capture_snapshot(game),
                "n0": len(recorders[0].records),
                "n1": len(recorders[1].records),
            }
        )
    live_terminal = capture_snapshot(game)
    records = (recorders[0].records, recorders[1].records)
    print(f"[fork-probe] {deck} vs {opponent} seed={seed}: {step} steps, "
          f"{sum(1 for s in snapshots if s['resumable'])} boundary snapshots")

    boundary = [s for s in snapshots if s["resumable"]]
    contrast = [s for s in snapshots if not s["resumable"]]
    picks: list[tuple[str, dict[str, Any]]] = []
    if boundary:
        stride = max(1, len(boundary) // max(1, args.sample))
        picks += [("boundary", s) for s in boundary[::stride][: args.sample]]
        picks.append(("boundary", boundary[-1]))
    picks += [("contrast", s) for s in contrast[: args.contrast]]

    details: list[dict[str, Any]] = []
    for kind, snap_info in picks:
        g = fork_game(snap_info["snap"], game_attrs=GAME_ATTRS)
        replay = (
            ReplayPolicy(records[0], snap_info["n0"]),
            ReplayPolicy(records[1], snap_info["n1"]),
        )
        _attach(g, deck, opponent, replay)
        err = _play_to_terminal(g)
        terminal = capture_snapshot(g)
        expected_suffix = (
            [r["sig"] for r in records[0][snap_info["n0"]:]],
            [r["sig"] for r in records[1][snap_info["n1"]:]],
        )
        requests_match = (
            err is None
            and replay[0].observed == expected_suffix[0]
            and replay[1].observed == expected_suffix[1]
        )
        terminal_matches = err is None and terminal == live_terminal
        details.append(
            {
                "kind": kind,
                "step": snap_info["step"],
                "terminal_matches": terminal_matches,
                "requests_match": requests_match,
                "error": err,
            }
        )
        print(f"[fork-probe] {kind:<8} step={snap_info['step']:<4} "
              f"terminal_matches={terminal_matches} requests_match={requests_match}"
              + (f" error={err}" if err else ""))

    boundary_rows = [d for d in details if d["kind"] == "boundary"]
    contrast_rows = [d for d in details if d["kind"] == "contrast"]
    report = {
        "deck": deck,
        "opponent": opponent,
        "seed": seed,
        "steps": step,
        "boundary_snapshots": len(boundary),
        "non_boundary_snapshots": len(contrast),
        "boundary_terminal_match": f"{sum(d['terminal_matches'] for d in boundary_rows)}/{len(boundary_rows)}",
        "boundary_request_match": f"{sum(d['requests_match'] for d in boundary_rows)}/{len(boundary_rows)}",
        "contrast_terminal_match": f"{sum(d['terminal_matches'] for d in contrast_rows)}/{len(contrast_rows)}",
        "details": details,
        "runtime": runtime_metadata(),
    }
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ensure_dir(reports_dir("engine")) / f"probe_boundary_fork_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fork-probe] boundary terminal match: {report['boundary_terminal_match']}, "
          f"request match: {report['boundary_request_match']}, "
          f"contrast terminal match: {report['contrast_terminal_match']}")
    print(f"[fork-probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
