"""Compare pure-policy vs LLM-assisted mode on a fixed pairing (PLAN.md WS4/M5).

Strength acceptance (G1/M4) is pure-policy only; this script quantifies whether the
LLM-assisted mode helps, and logs every intervention. Small-sample by design.

Usage::

    python -m scripts.run_llm_agent --seeds 3 --budget 6
    python -m scripts.run_llm_agent --seeds 10 --budget 6 --model deepseek-chat
    python -m scripts.run_llm_agent --seeds 3 --no-llm      # pure-policy only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from common.llm import build_llm_client, load_llm_config
from common.lock import runtime_metadata
from common.paths import ensure_dir, reports_dir
from common.seeding import derive_seed
from agents.llm_assist import AssistConfig, build_llm_assist_policy
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import expert_policy
from eval.stats import estimate_rate, outcome_score


def _play(deck: str, opponent: str, seed: int, policy0):
    return run_match(
        deck_spec(deck),
        deck_spec(opponent),
        policy0,
        expert_policy(opponent, seed=derive_seed(seed, "p1", opponent)),
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure vs LLM-assisted strength probe.")
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponent", default="natlan_battleship")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=6, help="LLM calls per game")
    parser.add_argument("--model", default="deepseek-chat", help="in-game model")
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(range(args.seeds))
    deck, opponent = args.deck, args.opponent

    pure_scores: list[float] = []
    for seed in seeds:
        policy = expert_policy(deck, seed=derive_seed(seed, "p0", deck))
        record = _play(deck, opponent, seed, policy)
        pure_scores.append(outcome_score(record.winner, 0))
    pure = estimate_rate(sum(pure_scores), len(pure_scores))

    report = {
        "deck": deck,
        "opponent": opponent,
        "seeds": seeds,
        "pure_policy": pure.as_dict(),
        "runtime": runtime_metadata(),
    }

    if not args.no_llm:
        client = build_llm_client()
        if client is None:
            print("[llm] no LLM config; recording pure-policy only", file=sys.stderr)
        else:
            config = AssistConfig(budget_per_game=args.budget, model=args.model)
            llm_scores: list[float] = []
            interventions: list[dict] = []
            for seed in seeds:
                policy = build_llm_assist_policy(
                    deck,
                    client=client,
                    seed=derive_seed(seed, "p0", deck),
                    config=config,
                )
                record = _play(deck, opponent, seed, policy)
                llm_scores.append(outcome_score(record.winner, 0))
                interventions.extend(iv.as_dict() for iv in policy.interventions)
            llm = estimate_rate(sum(llm_scores), len(llm_scores))
            report["llm_assisted"] = llm.as_dict()
            report["llm_usage"] = client.usage.as_dict()
            report["interventions"] = interventions
            report["interventions_changed"] = sum(1 for iv in interventions if iv["changed"])
            report["interventions_failed"] = sum(
                1 for iv in interventions if iv.get("error")
            )

    print(json.dumps({k: v for k, v in report.items() if k != "interventions"}, ensure_ascii=False, indent=2))
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ensure_dir(reports_dir("llm")) / f"llm_probe_{deck}_vs_{opponent}_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[llm] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
