"""Evaluate the engine MCCFR search policy against scripted opponents (PLAN.md S-C).

The S-C first step (``envs.cfr_search``) plugs the external-sampling MCCFR solver
(``train/cfr_solver.py``) into the real engine. This script plays it against the
scripted opponent pool and compares it, on **paired seeds**, with the *same base
expert with search disabled* -- so the delta isolates the search.

Results stream to ``data/search/<tag>.jsonl`` (gitignored) with a spec-fingerprint
sidecar for resumability; only an aggregate report lands in ``reports/search/``.
The per-decision root solutions (average strategy + value, the S-C teacher
targets) are the standing export point for distillation.

Usage::

    python -m scripts.run_cfr_search_agent --smoke
    python -m scripts.run_cfr_search_agent --opponents natlan_battleship --seeds 3 \\
        --iterations 4 --depth 1 --candidate-limit 3 --workers 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from common.jsonl import append_jsonl, read_jsonl
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from envs.cfr_search import run_cfr_search_match
from envs.fork_bridge import ForkBridge
from eval.opponents import scripted_opponents
from eval.stats import estimate_rate, outcome_score

EXPERT = "expert"
SEARCH = "cfr"


@dataclass(frozen=True)
class CfrSearchSpec:
    deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    modes: tuple[str, ...] = (EXPERT, SEARCH)
    iterations: int = 4
    depth: int = 1
    candidate_limit: int = 3
    belief_root: bool = True
    belief_samples: int = 1
    max_searches: int = 1
    tag: str = "cfr_search_agent"

    def tasks(self) -> list["CfrTask"]:
        tasks: list[CfrTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                for mode in self.modes:
                    tasks.append(CfrTask(index=index, opponent=opponent, seed=int(seed), mode=mode))
                    index += 1
        return tasks

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "deck": self.deck,
                "opponents": list(self.opponents),
                "seeds": list(self.seeds),
                "modes": list(self.modes),
                "iterations": self.iterations,
                "depth": self.depth,
                "candidate_limit": self.candidate_limit,
                "belief_root": self.belief_root,
                "belief_samples": self.belief_samples,
                "max_searches": self.max_searches,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CfrTask:
    index: int
    opponent: str
    seed: int
    mode: str

    @property
    def key(self) -> str:
        return f"{self.opponent}|{self.seed}|{self.mode}"


def run_task(spec: CfrSearchSpec, task: CfrTask, *, bridge: ForkBridge) -> dict[str, Any]:
    search = task.mode == SEARCH
    result = run_cfr_search_match(
        spec.deck,
        task.opponent,
        bridge=bridge,
        seed=task.seed,
        search_players=(0,) if search else (),
        iterations=spec.iterations,
        depth=spec.depth,
        candidate_limit=spec.candidate_limit,
        belief_root=spec.belief_root,
        belief_samples=spec.belief_samples,
        max_searches=spec.max_searches if search else 0,
        request_types=("action",),
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


def _aggregate(spec: CfrSearchSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def rate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in subset if r.get("error") is None]
        total = sum(float(r["outcome"]) for r in scored)
        return {**estimate_rate(total, len(scored)).as_dict(), "errors": len(subset) - len(scored)}

    per_mode = {mode: rate([r for r in rows if r["mode"] == mode]) for mode in spec.modes}
    per_opponent = {
        mode: {opp: rate([r for r in rows if r["mode"] == mode and r["opponent"] == opp]) for opp in spec.opponents}
        for mode in spec.modes
    }
    overall = None
    if EXPERT in spec.modes and SEARCH in spec.modes:
        overall = round(per_mode[SEARCH]["rate"] - per_mode[EXPERT]["rate"], 4)
    return {
        "modes": per_mode,
        "per_opponent": per_opponent,
        "cfr_minus_expert_overall": overall,
        "matches": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--belief-samples", type=int, default=1)
    parser.add_argument("--max-searches", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 opponent x 2 seeds")
    parser.add_argument("--tag", default="cfr_search_agent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pool = [opp.name for opp in scripted_opponents()]
    if args.smoke:
        opponents = tuple(pool[:1])
        seeds = tuple(range(2))
    else:
        opponents = tuple(args.opponents) if args.opponents else tuple(pool[:3])
        seeds = tuple(range(args.seeds))
    spec = CfrSearchSpec(
        deck=args.deck,
        opponents=opponents,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        candidate_limit=args.candidate_limit,
        belief_samples=args.belief_samples,
        max_searches=args.max_searches,
        tag=args.tag,
    )

    stream_path = data_dir("search", f"{spec.tag}.jsonl")
    meta_path = data_dir("search", f"{spec.tag}.meta.json")
    fingerprint = spec.fingerprint()
    meta = _load_meta(meta_path)
    if meta is not None:
        if meta.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"refusing to resume {stream_path}: spec fingerprint changed "
                f"(delete {stream_path} + {meta_path} to start a new run)"
            )
    else:
        # Mirrors run_cfr_ladder._open_stream / eval.arena._stream_state: a raw
        # stream with no sidecar is refused — its fingerprint cannot be verified,
        # so rows from a different budget/config could be silently mixed in.
        if stream_path.exists():
            raise SystemExit(
                f"refusing to resume {stream_path}: {meta_path.name} sidecar is missing, "
                f"so the spec fingerprint cannot be verified (delete the raw file or "
                f"use another --tag)"
            )
        ensure_dir(stream_path.parent)
        meta_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "engine_lock": load_engine_lock(), "runtime": runtime_metadata()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    done = {f"{r['opponent']}|{r['seed']}|{r['mode']}" for r in read_jsonl(stream_path, tolerate_torn=True)}
    tasks = [t for t in spec.tasks() if t.key not in done]
    print(f"[cfr] {len(done)} done, {len(tasks)} to run")

    bridge = ForkBridge(workers=args.workers)
    try:
        for task in tasks:
            row = run_task(spec, task, bridge=bridge)
            append_jsonl(stream_path, row)
            print(
                f"[cfr] {task.opponent:<24} seed={task.seed:<3} {task.mode:<6} "
                f"winner={row['winner']} outcome={row['outcome']} searched={row['searched_decisions']} "
                f"{row['wall_seconds']}s" + (f" error={row['error']}" if row["error"] else "")
            )
    finally:
        bridge.close()

    rows = read_jsonl(stream_path, tolerate_torn=True)
    report = {
        "tag": spec.tag,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "spec": {
            "deck": spec.deck,
            "opponents": list(spec.opponents),
            "seeds": list(spec.seeds),
            "modes": list(spec.modes),
            "iterations": spec.iterations,
            "depth": spec.depth,
            "candidate_limit": spec.candidate_limit,
            "belief_root": spec.belief_root,
            "belief_samples": spec.belief_samples,
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
    for mode, stats in report["aggregate"]["modes"].items():
        print(f"[cfr] {mode}: {stats['wins']}/{stats['games']} = {stats['rate']:.3f} errors={stats['errors']}")
    if report["aggregate"]["cfr_minus_expert_overall"] is not None:
        print(f"[cfr] cfr - expert = {report['aggregate']['cfr_minus_expert_overall']:+.3f}")
    print(f"[cfr] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
