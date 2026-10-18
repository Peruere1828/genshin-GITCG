"""Can a per-matchup *router* over existing checkpoints beat every single policy?

The evaluations show large, matchup-specific differences between arms (e.g. on
``superconduct_aggro`` the mixed model beats its teacher on mualani_double_pyro
0.80 vs 0.40 but loses on superconduct_mika_attack 0.30 vs 0.65). Opponent decks are
public in this game (the opponent's lineup is visible), so choosing a policy per
opponent is a legal pure-policy construction — but picking the arm on the *same*
matches we report would be selection bias.

This script therefore does an honest split: it selects the best arm per opponent on
even seeds and scores the routing on odd seeds (each matchup has 10 seeds x 2 sides).
It also compares against the best single arm and the scripted expert, so we can see
whether the routing gain survives a held-out half.

Usage::

    python -m scripts.matchup_router --deck superconduct_aggro \
        --arm orl=data/eval/orl_bc_e15.jsonl --arm kg50=data/eval/kg_bc_e50.jsonl ...
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error") not in (None, "None"):
            continue
        rows.append(row)
    return rows


def _rate(rows: list[dict]) -> tuple[float, int]:
    if not rows:
        return float("nan"), 0
    return sum(float(r["score"]) for r in rows) / len(rows), len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--arm", action="append", required=True, help="name=path (repeatable)")
    parser.add_argument("--select-mod", type=int, default=2, help="seeds %% mod == 0 go to selection")
    parser.add_argument("--select-rem", type=int, default=0)
    args = parser.parse_args(argv)

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = load_rows(path)
    if not arms:
        raise SystemExit("no arms")

    opponents = sorted({r["opponent"] for rows in arms.values() for r in rows})
    split = lambda rows, s: [r for r in rows if int(r["seed"]) % args.select_mod == s]

    print(f"deck={args.deck} opponents={len(opponents)} arms={list(arms)}")
    print(f"{'arm':14s} {'mean rate(all)':>14s} {'select half':>12s} {'holdout half':>13s}")
    for name, rows in arms.items():
        a, _ = _rate(rows)
        s, _ = _rate(split(rows, args.select_rem))
        h, _ = _rate(split(rows, (args.select_rem + 1) % args.select_mod))
        print(f"{name:14s} {a:14.4f} {s:12.4f} {h:13.4f}")

    # route per opponent by the selection-half rate, score on the holdout half
    routed_holdout: list[float] = []
    teacher_holdout: list[float] = []
    best_single = max(
        ((name, _rate(split(rows, (args.select_rem + 1) % args.select_mod))[0]) for name, rows in arms.items()),
        key=lambda kv: kv[1],
    )
    picks: dict[str, str] = {}
    for opp in opponents:
        scored = []
        for name, rows in arms.items():
            sel = [r for r in split(rows, args.select_rem) if r["opponent"] == opp]
            rate, n = _rate(sel)
            if n:
                scored.append((rate, name))
        if not scored:
            continue
        rate, name = max(scored)
        picks[opp] = name
        holdout = [r for r in split(arms[name], (args.select_rem + 1) % args.select_mod) if r["opponent"] == opp]
        routed_holdout.extend(float(r["score"]) for r in holdout)
        if "teacher" in arms:
            teacher_holdout.extend(float(r["score"]) for r in split(arms["teacher"], (args.select_rem + 1) % args.select_mod) if r["opponent"] == opp)

    routed_rate, routed_n = _rate([{"score": s} for s in routed_holdout])
    print()
    print(f"routed (select on seeds%{args.select_mod}=={args.select_rem}, scored on the rest): "
          f"{routed_rate:.4f} over n={routed_n}")
    print(f"best single arm on the same holdout: {best_single[0]} -> {best_single[1]:.4f}")
    if teacher_holdout:
        tr, tn = _rate([{"score": s} for s in teacher_holdout])
        print(f"teacher on the same holdout: {tr:.4f} over n={tn}")
    print("\npicks per opponent:")
    for opp in opponents:
        print(f"  {opp:28s} -> {picks.get(opp, '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
