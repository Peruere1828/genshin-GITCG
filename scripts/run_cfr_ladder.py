"""Budget-ladder evaluation of the engine MCCFR search policy vs scripted experts.

PLAN.md D22 follow-up (M3-CPU): the A3 screening found "search distillation ≈ cold
BC", blamed on an under-powered teacher budget. This script decides whether
**MCCFR search itself** can beat the expert control -- the go/no-go for "keep
distilling search" vs "replan to BC/self-play data first".

It reuses ``scripts.run_cfr_search_agent``'s spec/task grid (same pairing and
fingerprint idea) but adds what a budget sweep needs:

* **cross-game parallelism** -- a search is serial *within* a game (the CFR tree
  is built one fork at a time), so a single game is CPU-bound but parallel across
  games; ``--workers`` runs that many games concurrently, each with its own fork
  bridge (``--bridge-workers``);
* **paired-seed delta + CI** (``eval.scoreboard.paired_difference``) on the *same*
  seed for ``cfr`` vs the identical base expert with search disabled;
* **per-search cost** measured against the paired expert game
  (``(wall_cfr - wall_expert) / searched_decisions``);
* **action-change diagnostics** -- how often the search overrides the expert's
  action (``base_option != chosen_option``) and the mean root value.

Rows stream to ``data/search/<tag>.jsonl`` (gitignored) with a fingerprint
sidecar so an interrupted run resumes; only an aggregate lands in
``reports/search/``. Pure-policy only (no LLM).

Usage::

    python -m scripts.run_cfr_ladder --opponents a b c --seeds 6 \\
        --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 \\
        --workers 6 --tag ladder_i8_d1_c4
"""

from __future__ import annotations

import argparse
import concurrent.futures as _futures
import datetime as _dt
import hashlib
import json
import multiprocessing as _mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from common.jsonl import append_jsonl, read_jsonl
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from eval.opponents import scripted_opponents
from eval.scoreboard import MatchOutcome, paired_difference, summarize
from eval.stats import outcome_score

EXPERT = "expert"
CFR = "cfr"


# --------------------------------------------------------------------------- #
# spec / tasks (mirrors scripts.run_cfr_search_agent)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LadderSpec:
    deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    modes: tuple[str, ...] = (EXPERT, CFR)
    iterations: int = 4
    depth: int = 1
    candidate_limit: int = 2
    belief_root: bool = True
    belief_samples: int = 1
    max_searches: int = 8
    search_every: int = 1
    request_types: tuple[str, ...] = ("action",)
    value_spec: str = ""
    # A CVPN checkpoint used as the **base policy and prior** for player 0 (S-C
    # intended design): then the control arm is the pure network and the cfr arm
    # is network + search, so the paired delta isolates the search *on top of a
    # learned prior* instead of the arbitrary no-prior candidate ordering.
    checkpoint: str | None = None
    bridge_workers: int = 1
    workers: int = 6
    tag: str = "cfr_ladder"

    def tasks(self) -> list["LadderTask"]:
        out: list[LadderTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                for mode in self.modes:
                    out.append(LadderTask(index=index, opponent=opponent, seed=int(seed), mode=mode))
                    index += 1
        return out

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
                "search_every": self.search_every,
                "request_types": list(self.request_types),
                "value_spec": self.value_spec,
                "checkpoint": self.checkpoint,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def budget_label(self) -> str:
        cand = "all" if self.candidate_limit is None else str(self.candidate_limit)
        label = (
            f"i{self.iterations}_d{self.depth}_c{cand}_ms{self.max_searches}"
            f"_bs{self.belief_samples}"
        )
        if self.checkpoint:
            label += "_netprior"
        if self.value_spec or self.checkpoint:
            label += "_netvalue"
        return label


@dataclass(frozen=True)
class LadderTask:
    index: int
    opponent: str
    seed: int
    mode: str

    @property
    def key(self) -> str:
        return f"{self.opponent}|{self.seed}|{self.mode}"


# --------------------------------------------------------------------------- #
# one game (same call shape as scripts.run_cfr_search_agent.run_task, + diagnostics)
# --------------------------------------------------------------------------- #
def run_one(spec: LadderSpec, task: LadderTask, *, bridge: Any, prior: Any = None) -> dict[str, Any]:
    from envs.cfr_search import run_cfr_search_match

    search = task.mode == CFR
    kwargs: dict[str, Any] = {}
    value_spec = spec.value_spec or (spec.checkpoint or "")
    if spec.checkpoint and prior is not None:
        # Both arms use the network as the base policy: the control is the pure
        # network, so ``cfr - expert`` isolates the search, not network-vs-expert.
        kwargs["base_policies"] = (prior, None)
        if search:
            kwargs["prior_policies"] = (prior, None)
    result = run_cfr_search_match(
        spec.deck,
        task.opponent,
        bridge=bridge,
        seed=task.seed,
        search_players=(0,) if search else (),
        iterations=spec.iterations,
        depth=spec.depth,
        candidate_limit=spec.candidate_limit,
        value_spec=value_spec,
        belief_root=spec.belief_root,
        belief_samples=spec.belief_samples,
        search_every=spec.search_every,
        max_searches=spec.max_searches if search else 0,
        request_types=spec.request_types,
        **kwargs,
    )
    record = result.record
    searches = list(result.searches)
    changed = sum(1 for s in searches if int(s.base_option) != int(s.chosen_option))
    return {
        "task_index": task.index,
        "opponent": task.opponent,
        "seed": task.seed,
        "mode": task.mode,
        "winner": record.winner,
        "outcome": outcome_score(record.winner, 0),
        "rounds": record.rounds,
        "decisions0": record.decisions0,
        "decisions1": record.decisions1,
        "fallbacks0": record.fallbacks0,
        "searched_decisions": len(searches),
        "changed_decisions": changed,
        "root_value_sum": round(sum(float(s.value) for s in searches), 6),
        "wall_seconds": round(record.wall_seconds, 3),
        "truncated": bool(record.truncated),
        "error": record.error,
    }


# --------------------------------------------------------------------------- #
# worker processes
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _init_worker(spec_dict: dict[str, Any], bridge_workers: int) -> None:
    from envs.fork_bridge import ForkBridge

    spec = LadderSpec(**spec_dict)
    prior = None
    if spec.checkpoint:
        from train.resolver import prior_resolver_from_checkpoint

        prior = prior_resolver_from_checkpoint(spec.checkpoint)
    _WORKER["spec"] = spec
    _WORKER["prior"] = prior
    _WORKER["bridge"] = ForkBridge(workers=int(bridge_workers))


def _worker_run(task_dict: dict[str, Any]) -> dict[str, Any]:
    task = LadderTask(**task_dict)
    return run_one(_WORKER["spec"], task, bridge=_WORKER["bridge"], prior=_WORKER.get("prior"))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _outcomes(rows: Sequence[dict[str, Any]], mode: str) -> list[MatchOutcome]:
    return [
        MatchOutcome(
            opponent=str(r["opponent"]),
            seed=int(r["seed"]),
            side=0,
            score=float(r["outcome"]),
            rounds=int(r["rounds"]),
            decisions=int(r["decisions0"]),
            fallbacks=int(r["fallbacks0"]),
            error=r.get("error"),
            truncated=bool(r.get("truncated")),
        )
        for r in rows
        if r["mode"] == mode
    ]


def _cost(spec: LadderSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Wall-clock cost: totals + per-search estimate vs the paired expert game."""
    cfr = [r for r in rows if r["mode"] == CFR and r.get("error") is None]
    expert = {
        (r["opponent"], int(r["seed"])): r
        for r in rows
        if r["mode"] == EXPERT and r.get("error") is None
    }
    total_searches = sum(int(r["searched_decisions"]) for r in cfr)
    cfr_wall = sum(float(r["wall_seconds"]) for r in cfr)
    expert_wall = sum(float(r["wall_seconds"]) for r in expert.values())
    # Paired per-search estimate: (wall_cfr - wall_expert) / searches.
    per_search: list[float] = []
    for r in cfr:
        base = expert.get((r["opponent"], int(r["seed"])))
        n = int(r["searched_decisions"])
        if base is None or n <= 0:
            continue
        per_search.append((float(r["wall_seconds"]) - float(base["wall_seconds"])) / n)
    per_search.sort()
    return {
        "workers": spec.workers,
        "bridge_workers": spec.bridge_workers,
        "cfr_games": len(cfr),
        "expert_games": len(expert),
        "cfr_wall_total_s": round(cfr_wall, 1),
        "expert_wall_total_s": round(expert_wall, 1),
        "cfr_wall_per_game_s": round(cfr_wall / len(cfr), 3) if cfr else None,
        "expert_wall_per_game_s": round(expert_wall / len(expert), 3) if expert else None,
        "total_searches": total_searches,
        "searches_per_game": round(total_searches / len(cfr), 2) if cfr else None,
        "seconds_per_search_mean": round(sum(per_search) / len(per_search), 3) if per_search else None,
        "seconds_per_search_median": (
            round(per_search[len(per_search) // 2], 3) if per_search else None
        ),
        "cfr_serial_core_hours": round(cfr_wall / 3600.0, 3),
    }


def _change_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cfr = [r for r in rows if r["mode"] == CFR]
    searched = sum(int(r["searched_decisions"]) for r in cfr)
    changed = sum(int(r["changed_decisions"]) for r in cfr)
    games_changed = sum(1 for r in cfr if int(r["changed_decisions"]) > 0)
    return {
        "cfr_games": len(cfr),
        "searched_decisions": searched,
        "changed_decisions": changed,
        "change_rate": round(changed / searched, 4) if searched else None,
        "games_with_any_change": games_changed,
        "mean_changed_per_game": round(changed / len(cfr), 2) if cfr else None,
    }


def _summarize_mode(rows: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    outcomes = _outcomes(rows, mode)
    return summarize(outcomes)


def aggregate(spec: LadderSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    modes = {mode: _summarize_mode(rows, mode) for mode in spec.modes}
    paired = None
    if EXPERT in spec.modes and CFR in spec.modes:
        paired = paired_difference(_outcomes(rows, CFR), _outcomes(rows, EXPERT))
    return {
        "tag": spec.tag,
        "budget": spec.budget_label(),
        "spec": {
            "deck": spec.deck,
            "opponents": list(spec.opponents),
            "seeds": list(spec.seeds),
            "iterations": spec.iterations,
            "depth": spec.depth,
            "candidate_limit": spec.candidate_limit,
            "belief_root": spec.belief_root,
            "belief_samples": spec.belief_samples,
            "max_searches": spec.max_searches,
            "search_every": spec.search_every,
            "request_types": list(spec.request_types),
            "value_spec": spec.value_spec,
            "checkpoint": spec.checkpoint,
        },
        "matches": len(rows),
        "modes": modes,
        "pooled_expert": modes.get(EXPERT, {}).get("pooled_rate"),
        "pooled_cfr": modes.get(CFR, {}).get("pooled_rate"),
        "paired_cfr_minus_expert": paired,
        "cost": _cost(spec, rows),
        "change": _change_stats(rows),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _load_meta(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=2)
    parser.add_argument("--belief-samples", type=int, default=1)
    parser.add_argument("--max-searches", type=int, default=8)
    parser.add_argument("--search-every", type=int, default=1)
    parser.add_argument("--value-spec", default="", help="CVPN checkpoint as the depth-limited frontier value")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CVPN checkpoint as base+prior (control arm = pure network; isolates search on top of it)",
    )
    parser.add_argument("--workers", type=int, default=6, help="concurrent games (each with its own fork bridge)")
    parser.add_argument("--bridge-workers", type=int, default=1)
    parser.add_argument("--tag", default="cfr_ladder")
    args = parser.parse_args(argv)

    pool = [opp.name for opp in scripted_opponents()]
    opponents = tuple(args.opponents) if args.opponents else tuple(pool[:3])
    missing = set(opponents) - set(pool)
    if missing:
        raise SystemExit(f"unknown opponent(s): {sorted(missing)}")
    spec = LadderSpec(
        deck=args.deck,
        opponents=opponents,
        seeds=tuple(range(args.seed_start, args.seed_start + args.seeds)),
        iterations=args.iterations,
        depth=args.depth,
        candidate_limit=args.candidate_limit,
        belief_samples=args.belief_samples,
        max_searches=args.max_searches,
        search_every=args.search_every,
        value_spec=args.value_spec,
        checkpoint=args.checkpoint,
        bridge_workers=args.bridge_workers,
        workers=args.workers,
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
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "spec": asdict(spec),
                    "engine_lock": load_engine_lock(),
                    "runtime": runtime_metadata(),
                    "pid": os.getpid(),
                    "started": _dt.datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    done = {f"{r['opponent']}|{r['seed']}|{r['mode']}" for r in read_jsonl(stream_path, tolerate_torn=True)}
    pending = [t for t in spec.tasks() if t.key not in done]
    print(
        f"[ladder] {spec.budget_label()} {spec.tag}: {len(done)} done, {len(pending)} to run "
        f"({len(spec.opponents)} opp x {len(spec.seeds)} seeds x {len(spec.modes)} modes), "
        f"workers={args.workers} bridge={args.bridge_workers}",
        flush=True,
    )

    start = time.perf_counter()
    if pending:
        ctx = _mp.get_context("spawn")
        with _futures.ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(asdict(spec), int(args.bridge_workers)),
        ) as pool_exec:
            futures = {pool_exec.submit(_worker_run, asdict(t)): t for t in pending}
            for future in _futures.as_completed(futures):
                task = futures[future]
                row = future.result()
                append_jsonl(stream_path, row)
                print(
                    f"[ladder] {task.opponent:<24} seed={task.seed:<3} {task.mode:<6} "
                    f"winner={row['winner']} outcome={row['outcome']} "
                    f"searched={row['searched_decisions']} changed={row['changed_decisions']} "
                    f"{row['wall_seconds']}s" + (f" error={row['error']}" if row["error"] else ""),
                    flush=True,
                )
    wall = time.perf_counter() - start

    rows = read_jsonl(stream_path, tolerate_torn=True)
    report = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
        **aggregate(spec, rows),
    }
    out_dir = ensure_dir(reports_dir("search"))
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = out_dir / f"{spec.tag}_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    paired = report["paired_cfr_minus_expert"] or {}
    print(
        f"[ladder] {spec.budget_label()}: cfr={report['pooled_cfr']} expert={report['pooled_expert']} "
        f"paired_delta={paired.get('delta')} ci=[{paired.get('ci_low')},{paired.get('ci_high')}] "
        f"n={paired.get('n_pairs')} sig+={paired.get('significant_positive')}",
        flush=True,
    )
    print(f"[ladder] cost={report['cost']}", flush=True)
    print(f"[ladder] change={report['change']}", flush=True)
    print(f"[ladder] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
