"""Arena: run scripted/agent matchups at fixed paired seeds and summarize.

Produces a win-rate matrix (with Wilson CIs) plus an Elo ladder. Raw per-match
records go to ``data/arena/`` (gitignored); the small summary goes to
``reports/arena/`` (committed).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from envs.rollout import MatchTask, make_task_grid, run_tasks
from eval.ladder import compute_elo
from eval.opponents import Opponent, baseline_opponents, scripted_opponents
from eval.stats import estimate_rate


@dataclass
class ArenaSpec:
    contestants: tuple[Opponent, ...]
    pool: tuple[Opponent, ...]
    seeds: tuple[int, ...]
    workers: int = 1
    swap: bool = False
    record_decisions: bool = False
    policy0: str | None = None
    policy1: str | None = None
    tag: str = "arena"
    base_seed: int = 0

    def deck_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for contestant in self.contestants:
            for opponent in self.pool:
                pairs.append((contestant.deck, opponent.deck))
                if self.swap:
                    pairs.append((opponent.deck, contestant.deck))
        return pairs

    def tasks(self) -> list[MatchTask]:
        pairs = self.deck_pairs()
        tasks = make_task_grid(
            pairs,
            self.seeds,
            policy0=self.policy0 or "expert",
            policy1=self.policy1 or "expert",
            tag=self.tag,
            base_seed=self.base_seed,
        )
        if self.record_decisions:
            tasks = [
                replace(task, record_decisions=True, record_views=True) for task in tasks
            ]
        return tasks


@dataclass
class ArenaResult:
    spec: ArenaSpec
    matches: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def pairing_summary(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for match in self.matches:
            key = (match["deck0"], match["deck1"])
            grouped.setdefault(key, []).append(match)
        rows: list[dict[str, Any]] = []
        for (deck0, deck1), group in grouped.items():
            wins0 = sum(1 for m in group if m["winner"] == 0)
            wins1 = sum(1 for m in group if m["winner"] == 1)
            draws = sum(1 for m in group if m["winner"] is None)
            errors = sum(1 for m in group if m["error"])
            rate = estimate_rate(wins0 + 0.5 * draws, len(group))
            rows.append(
                {
                    "deck0": deck0,
                    "deck1": deck1,
                    "games": len(group),
                    "wins0": wins0,
                    "wins1": wins1,
                    "draws": draws,
                    "errors": errors,
                    **{f"rate0_{k}": v for k, v in rate.as_dict().items()},
                }
            )
        rows.sort(key=lambda row: (row["deck0"], row["deck1"]))
        return rows

    def contestant_summary(self) -> list[dict[str, Any]]:
        """Aggregate each contestant over the whole pool (as player 0)."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        contestant_decks = {c.deck for c in self.spec.contestants}
        for match in self.matches:
            if match["deck0"] in contestant_decks:
                grouped.setdefault(match["deck0"], []).append(match)
        rows: list[dict[str, Any]] = []
        for deck, group in grouped.items():
            wins = sum(1 for m in group if m["winner"] == 0)
            draws = sum(1 for m in group if m["winner"] is None)
            losses = sum(1 for m in group if m["winner"] == 1)
            errors = sum(1 for m in group if m["error"])
            rate = estimate_rate(wins + 0.5 * draws, len(group))
            rows.append(
                {
                    "deck": deck,
                    "games": len(group),
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "errors": errors,
                    **rate.as_dict(),
                }
            )
        rows.sort(key=lambda row: (-row["rate"], row["deck"]))
        return rows

    def elo(self) -> dict[str, float]:
        return compute_elo(self.matches)

    def summary(self) -> dict[str, Any]:
        return {
            "tag": self.spec.tag,
            "seeds": list(self.spec.seeds),
            "workers": self.spec.workers,
            "swap": self.spec.swap,
            "n_matches": len(self.matches),
            "metadata": self.metadata,
            "contestants": self.contestant_summary(),
            "pairings": self.pairing_summary(),
            "elo": self.elo(),
        }


def run_arena(spec: ArenaSpec, *, progress: bool = False) -> ArenaResult:
    tasks = spec.tasks()
    matches = run_tasks(tasks, workers=spec.workers, progress=progress)
    metadata = {
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
        "policies": {
            "contestant": spec.policy0 or "expert:<deck>",
            "pool": spec.policy1 or "expert:<deck>",
        },
    }
    return ArenaResult(spec=spec, matches=matches, metadata=metadata)


def _run_id(spec: ArenaSpec) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{spec.tag}"


def write_artifacts(result: ArenaResult, *, run_id: str | None = None) -> dict[str, Path]:
    run_id = run_id or _run_id(result.spec)
    raw_path = ensure_dir(data_dir("arena")) / f"{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for match in result.matches:
            handle.write(json.dumps(match, ensure_ascii=False, sort_keys=True) + "\n")

    out_dir = ensure_dir(reports_dir("arena"))
    summary_path = out_dir / f"{run_id}.json"
    summary_path.write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = out_dir / f"{run_id}.pairings.csv"
    rows = result.pairing_summary()
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {"raw": raw_path, "summary": summary_path, "csv": csv_path}


def _print_report(result: ArenaResult, *, top: int = 30) -> None:
    print(f"[arena] tag={result.spec.tag} matches={len(result.matches)} "
          f"seeds={len(result.spec.seeds)} workers={result.spec.workers}")
    errors = [m for m in result.matches if m["error"]]
    truncated = [m for m in result.matches if m.get("truncated")]
    print(f"[arena] errors={len(errors)} truncated={len(truncated)}")
    print("[arena] contestant win rates over pool (player 0):")
    for row in result.contestant_summary()[:top]:
        print(
            f"  {row['deck']:<34} {row['rate']:.3f} "
            f"[{row['ci_low']:.3f},{row['ci_high']:.3f}] "
            f"({row['wins']}-{row['losses']}-{row['draws']})"
        )
    elo = result.elo()
    if elo:
        print("[arena] Elo:")
        for name, rating in sorted(elo.items(), key=lambda kv: -kv[1])[:top]:
            print(f"  {name:<34} {rating:7.1f}")


def _seeds(count: int, start: int = 0) -> tuple[int, ...]:
    return tuple(range(start, start + count))


def _select_pool(names: Iterable[str] | None) -> tuple[Opponent, ...]:
    pool = scripted_opponents()
    if not names:
        return pool
    wanted = set(names)
    selected = tuple(opp for opp in pool if opp.name in wanted)
    missing = wanted - {opp.name for opp in selected}
    if missing:
        raise SystemExit(f"unknown opponent(s): {sorted(missing)}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the scripted-expert arena.")
    parser.add_argument("--tag", default="arena")
    parser.add_argument("--seeds", type=int, default=50, help="number of paired seeds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="0 = all cores")
    parser.add_argument("--opponents", nargs="*", default=None, help="subset of opponent slugs")
    parser.add_argument("--swap", action="store_true", help="also play each pairing swapped")
    parser.add_argument("--smoke", action="store_true", help="tiny fast subset")
    parser.add_argument("--no-write", action="store_true", help="do not write artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        pool = _select_pool(
            args.opponents
            or ["superconduct_aggro", "natlan_battleship", "dual_mualani_stacks"]
        )
        seeds = _seeds(min(args.seeds, 3), args.seed_start)
        tag = f"{args.tag}_smoke"
    else:
        pool = _select_pool(args.opponents)
        seeds = _seeds(args.seeds, args.seed_start)
        tag = args.tag
    contestants = scripted_opponents()
    if args.smoke:
        contestants = tuple(c for c in contestants if c.deck in {o.deck for o in pool})
    spec = ArenaSpec(
        contestants=contestants,
        pool=pool,
        seeds=seeds,
        workers=args.workers,
        swap=args.swap,
        tag=tag,
    )
    result = run_arena(spec)
    _print_report(result)
    if not args.no_write:
        paths = write_artifacts(result)
        print(f"[arena] wrote {paths['summary']}")
        print(f"[arena] raw     {paths['raw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
