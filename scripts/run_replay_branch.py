"""Run the replay-branch MC teacher on one decision and report action values (L5.4/D12).

Plays a base game, picks a decision of ``--player`` (or the ``--ordinal`` one),
then Monte-Carlo evaluates its candidate actions via replay branching. This is the
offline search teacher used for M3 distillation until a true live-fork engine
bridge exists (see ``envs/snapshot.py``).

Usage::

    python -m scripts.run_replay_branch --seed 3 --rollouts 5
    python -m scripts.run_replay_branch --player 1 --ordinal 10 --top-k 4 \\
        --rollout-spec legal_random --rollouts 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json

from common.lock import runtime_metadata
from common.paths import ensure_dir, reports_dir
from train.replay_branch import capture_trajectory, evaluate_decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponent", default="natlan_battleship")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--player", type=int, default=0, choices=(0, 1))
    parser.add_argument("--ordinal", type=int, default=None, help="decision ordinal (default: smallest option set)")
    parser.add_argument("--rollouts", type=int, default=4, help="rollout games per candidate")
    parser.add_argument("--top-k", type=int, default=None, help="evaluate base choice + (k-1) others")
    parser.add_argument(
        "--rollout-spec",
        default="legal_random",
        help="rollout policy spec (legal_random for MC; empty string = base expert)",
    )
    return parser


def _pick_ordinal(deck: str, opponent: str, seed: int, player: int) -> int:
    _record, points = capture_trajectory(deck, opponent, seed, player=player)
    candidates = [p for p in points if p.option_count > 1]
    if not candidates:
        raise SystemExit("no decision with more than one option")
    return min(candidates, key=lambda p: (p.option_count, p.ordinal)).ordinal


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ordinal = args.ordinal
    if ordinal is None:
        ordinal = _pick_ordinal(args.deck, args.opponent, args.seed, args.player)
    rollout_spec = args.rollout_spec or None

    evaluation = evaluate_decision(
        args.deck,
        args.opponent,
        args.seed,
        player=args.player,
        ordinal=ordinal,
        rollout_spec=rollout_spec,
        rollouts=args.rollouts,
        top_k=args.top_k,
    )
    payload = evaluation.as_dict()
    payload["runtime"] = runtime_metadata()

    print(
        f"[branch] {args.deck} vs {args.opponent} seed={args.seed} player={args.player} "
        f"ordinal={ordinal} ({evaluation.request_type}) base_choice={evaluation.base_choice} "
        f"best={evaluation.best_option}"
    )
    for option_index in sorted(evaluation.options):
        value = evaluation.options[option_index]
        tag = " (base)" if value.is_base_choice else ""
        print(f"  option {option_index:>4}{tag:<7} mean={value.mean:.3f} n={value.n} scores={list(value.scores)}")

    base_value = evaluation.options[evaluation.base_choice]
    best_value = evaluation.options[evaluation.best_option]
    if evaluation.best_option != evaluation.base_choice and best_value.mean > base_value.mean:
        print(
            f"[branch] base choice {evaluation.base_choice} looks suboptimal: "
            f"{base_value.mean:.3f} -> {best_value.mean:.3f}"
        )

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ensure_dir(reports_dir("train")) / f"replay_branch_{args.deck}_vs_{args.opponent}_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[branch] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
