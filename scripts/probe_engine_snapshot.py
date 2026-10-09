"""Diagnostic: characterize the pybinding's mid-game snapshot/fork behaviour (L5.4).

Plays one deterministic game, snapshotting after every step, then for a sample of
snapshots checks:

* round-trip fidelity  ``State(json=snap).json() == snap``
* clone determinism    two resumes from the same snapshot agree exactly
* trajectory reproduction  does a resume reproduce the uninterrupted terminal?

Writes a small JSON report to ``reports/engine/`` and prints a summary. This is
the evidence behind ``envs/snapshot.py``'s ``FORK_LIMITATION``.

Usage::

    python -m scripts.probe_engine_snapshot --seed 3
    python -m scripts.probe_engine_snapshot --seed 3 --all --deck superconduct_aggro \\
        --opponent natlan_battleship
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json

from common.lock import runtime_metadata
from common.paths import ensure_dir, reports_dir
from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import PolicyPlayer, _build_create_param
from envs.policy import Policy, expert_policy
from envs.snapshot import (
    FORK_LIMITATION,
    capture_snapshot,
    fork_game,
    snapshot_roundtrip_is_faithful,
)
from gitcg import low_level


def make_initial_game(deck: str, opponent: str, seed: int):
    from gitcg import Game

    create_param = _build_create_param(deck_spec(deck), deck_spec(opponent), seed=seed)
    game = Game(create_param=create_param)
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_0, 1)
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_1, 1)
    return game


def _attach(game, deck: str, opponent: str, seed: int, policies: tuple[Policy, Policy]):
    game.set_player(
        0, PolicyPlayer(0, policies[0], metadata={"deck": deck})
    )
    game.set_player(
        1, PolicyPlayer(1, policies[1], metadata={"deck": opponent})
    )


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
    parser.add_argument("--sample", type=int, default=12, help="snapshots to test")
    parser.add_argument("--all", action="store_true", help="test every snapshot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deck, opponent, seed = args.deck, args.opponent, args.seed

    def policies(seed_value: int):
        return (
            expert_policy(deck, seed=derive_seed(seed_value, "p0", deck)),
            expert_policy(opponent, seed=derive_seed(seed_value, "p1", opponent)),
        )

    game = make_initial_game(deck, opponent, seed)
    _attach(game, deck, opponent, seed, policies(seed))
    game.start()
    snapshots: list[tuple[int, bool, str]] = []
    step = 0
    while game.is_running():
        game.step()
        step += 1
        snapshots.append((step, bool(game.is_resumable()), capture_snapshot(game)))
    live_terminal = capture_snapshot(game)
    live_winner = game.winner()
    print(f"[probe] {deck} vs {opponent} seed={seed}: {step} steps, winner={live_winner}")

    # choose sample indices
    if args.all or args.sample >= len(snapshots):
        sample_idx = list(range(len(snapshots)))
    else:
        count = max(1, args.sample)
        stride = max(1, len(snapshots) // count)
        sample_idx = sorted(set(list(range(0, len(snapshots), stride))[:count] + [len(snapshots) - 1]))

    roundtrip_ok = 0
    clone_ok = 0
    reproduce_ok = 0
    details: list[dict] = []
    for idx in sample_idx:
        step_index, resumable, snap = snapshots[idx]
        faithful = snapshot_roundtrip_is_faithful(snap)
        roundtrip_ok += int(faithful)

        resume_terminals: list[str] = []
        for _ in range(2):
            g = fork_game(snap)
            _attach(g, deck, opponent, seed, policies(seed))
            _play_to_terminal(g)
            resume_terminals.append(capture_snapshot(g))
        clone = resume_terminals[0] == resume_terminals[1]
        clone_ok += int(clone)
        reproduce = resume_terminals[0] == live_terminal
        reproduce_ok += int(reproduce)
        details.append(
            {
                "step": step_index,
                "can_resume_flag": resumable,
                "roundtrip_faithful": faithful,
                "clone_deterministic": clone,
                "reproduces_live_terminal": reproduce,
            }
        )

    report = {
        "deck": deck,
        "opponent": opponent,
        "seed": seed,
        "steps": step,
        "live_winner": live_winner,
        "snapshots_tested": len(sample_idx),
        "roundtrip_faithful": f"{roundtrip_ok}/{len(sample_idx)}",
        "clone_deterministic": f"{clone_ok}/{len(sample_idx)}",
        "reproduces_live_terminal": f"{reproduce_ok}/{len(sample_idx)}",
        "limitation": FORK_LIMITATION,
        "runtime": runtime_metadata(),
        "details": details,
    }
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ensure_dir(reports_dir("engine"))
    out = out_dir / f"probe_snapshot_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[probe] roundtrip {report['roundtrip_faithful']}, "
        f"clone {report['clone_deterministic']}, "
        f"reproduces-live {report['reproduces_live_terminal']}"
    )
    print(f"[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
