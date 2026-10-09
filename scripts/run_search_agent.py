"""Evaluate the fork-search resolver against the scripted opponent pool.

The L5.4 "search resolver" deliverable (PLAN.md D13/I10, WS3): play the practical
fork-search agent (``envs.fork_search``) against scripted opponents and compare it,
on **paired seeds**, against the *same base expert with search disabled*. Because
both modes share the base policy and seed, the delta isolates the effect of fork
search rather than deck/seed luck.

Design constraints (PLAN.md D7 / §0.1 I7):

- results stream to ``data/search/<tag>.jsonl`` per match (gitignored) so an
  interrupted run resumes instead of restarting;
- a ``.meta.json`` sidecar pins a spec fingerprint + engine lock and refuses to
  resume against a drifting configuration;
- only an **aggregate** report lands in ``reports/search/`` (committed).

Fork search is CPU-heavy (``candidates x rollouts`` full games per searched
decision). Keep ``--max-searches`` / ``--top-k`` small for a smoke.

Usage::

    python -m scripts.run_search_agent --smoke
    python -m scripts.run_search_agent --opponents natlan_battleship --seeds 3 \\
        --rollouts 2 --max-searches 5 --workers 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from common.jsonl import append_jsonl, read_jsonl
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from envs.fork_bridge import ForkBridge
from envs.fork_search import run_search_match
from eval.opponents import scripted_opponents
from eval.stats import estimate_rate, outcome_score

EXPERT = "expert"
SEARCH = "search"


@dataclass(frozen=True)
class SearchSpec:
    deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    modes: tuple[str, ...] = (EXPERT, SEARCH)
    rollouts: int = 1
    top_k: int = 2
    search_every: int = 1
    max_searches: int = 5
    tag: str = "search_agent"

    def tasks(self) -> list["SearchTask"]:
        tasks: list[SearchTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                for mode in self.modes:
                    tasks.append(
                        SearchTask(index=index, opponent=opponent, seed=int(seed), mode=mode)
                    )
                    index += 1
        return tasks

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "deck": self.deck,
                "opponents": list(self.opponents),
                "seeds": list(self.seeds),
                "modes": list(self.modes),
                "rollouts": self.rollouts,
                "top_k": self.top_k,
                "search_every": self.search_every,
                "max_searches": self.max_searches,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SearchTask:
    index: int
    opponent: str
    seed: int
    mode: str

    @property
    def key(self) -> str:
        return f"{self.opponent}|{self.seed}|{self.mode}"


def run_task(spec: SearchSpec, task: SearchTask, *, bridge: ForkBridge) -> dict[str, Any]:
    rollouts = spec.rollouts if task.mode == SEARCH else 0
    result = run_search_match(
        spec.deck,
        task.opponent,
        bridge=bridge,
        seed=task.seed,
        rollouts=rollouts,
        top_k=spec.top_k,
        search_every=spec.search_every,
        max_searches=spec.max_searches if task.mode == SEARCH else 0,
    )
    record = result.record
    return {
        "task_index": task.index,
        "opponent": task.opponent,
        "seed": task.seed,
        "mode": task.mode,
        "winner": record.winner,
        "outcome": outcome_score(record.winner, 0),
        "rounds": record.rounds,
        "decisions0": record.decisions0,
        "fallbacks0": record.fallbacks0,
        "searched_decisions": len(result.searches),
        "wall_seconds": round(record.wall_seconds, 3),
        "error": record.error,
    }


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate(spec: SearchSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def rate(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in rows_subset if r.get("error") is None]
        total = sum(float(r["outcome"]) for r in scored)
        return {**estimate_rate(total, len(scored)).as_dict(), "errors": len(rows_subset) - len(scored)}

    by_mode: dict[str, Any] = {}
    per_opponent: dict[str, Any] = {}
    for mode in spec.modes:
        by_mode[mode] = rate([r for r in rows if r["mode"] == mode])
        per_opponent[mode] = {
            opp: rate([r for r in rows if r["mode"] == mode and r["opponent"] == opp])
            for opp in spec.opponents
        }
    deltas: dict[str, Any] = {}
    if EXPERT in spec.modes and SEARCH in spec.modes:
        for opp in spec.opponents:
            expert = per_opponent[EXPERT][opp]["rate"]
            search = per_opponent[SEARCH][opp]["rate"]
            deltas[opp] = round(search - expert, 4)
        overall = round(by_mode[SEARCH]["rate"] - by_mode[EXPERT]["rate"], 4)
    else:
        overall = None
    return {
        "modes": by_mode,
        "per_opponent": per_opponent,
        "search_minus_expert_by_opponent": deltas,
        "search_minus_expert_overall": overall,
        "matches": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--search-every", type=int, default=1)
    parser.add_argument("--max-searches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="3 opponents x 2 seeds")
    parser.add_argument("--tag", default="search_agent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pool = [opp.name for opp in scripted_opponents()]
    if args.smoke:
        opponents = pool[:3]
        seeds = tuple(range(2))
    else:
        opponents = list(args.opponents) if args.opponents else pool[:5]
        seeds = tuple(range(args.seeds))
    spec = SearchSpec(
        deck=args.deck,
        opponents=tuple(opponents),
        seeds=seeds,
        rollouts=args.rollouts,
        top_k=args.top_k,
        search_every=args.search_every,
        max_searches=args.max_searches,
        tag=args.tag,
    )

    stream_path = data_dir("search", f"{spec.tag}.jsonl")
    meta_path = data_dir("search", f"{spec.tag}.meta.json")
    fingerprint = spec.fingerprint()
    meta = _load_meta(meta_path)
    if meta is not None and meta.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"refusing to resume {stream_path}: spec fingerprint changed "
            f"(delete {stream_path} + {meta_path} to start a new run)"
        )
    if meta is None:
        ensure_dir(stream_path.parent)
        _write_meta(
            meta_path,
            {
                "fingerprint": fingerprint,
                "spec": {
                    "deck": spec.deck,
                    "opponents": list(spec.opponents),
                    "seeds": list(spec.seeds),
                    "modes": list(spec.modes),
                    "rollouts": spec.rollouts,
                    "top_k": spec.top_k,
                    "search_every": spec.search_every,
                    "max_searches": spec.max_searches,
                },
                "engine_lock": load_engine_lock(),
                "runtime": runtime_metadata(),
            },
        )

    done = {f"{r['opponent']}|{r['seed']}|{r['mode']}" for r in read_jsonl(stream_path)}
    tasks = [t for t in spec.tasks() if t.key not in done]
    print(f"[search] {len(done)} done, {len(tasks)} to run "
          f"({len(spec.opponents)} opponents x {len(spec.seeds)} seeds x {len(spec.modes)} modes)")

    bridge = ForkBridge(workers=args.workers)
    try:
        for task in tasks:
            row = run_task(spec, task, bridge=bridge)
            append_jsonl(stream_path, row)
            print(f"[search] {task.opponent:<24} seed={task.seed:<3} {task.mode:<6} "
                  f"winner={row['winner']} outcome={row['outcome']} "
                  f"searched={row['searched_decisions']} {row['wall_seconds']}s"
                  + (f" error={row['error']}" if row["error"] else ""))
    finally:
        bridge.close()

    rows = read_jsonl(stream_path)
    report = {
        "tag": spec.tag,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "spec": {
            "deck": spec.deck,
            "opponents": list(spec.opponents),
            "seeds": list(spec.seeds),
            "modes": list(spec.modes),
            "rollouts": spec.rollouts,
            "top_k": spec.top_k,
            "search_every": spec.search_every,
            "max_searches": spec.max_searches,
        },
        "aggregate": _aggregate(spec, rows),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    out_dir = ensure_dir(reports_dir("search"))
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = out_dir / f"{spec.tag}_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    agg = report["aggregate"]["modes"]
    for mode, stats in agg.items():
        print(f"[search] {mode}: {stats['wins']}/{stats['games']} = {stats['rate']:.3f} "
              f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] errors={stats['errors']}")
    if report["aggregate"]["search_minus_expert_overall"] is not None:
        print(f"[search] search - expert = {report['aggregate']['search_minus_expert_overall']:+.3f}")
    print(f"[search] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
