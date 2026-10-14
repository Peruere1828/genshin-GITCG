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
``reports/search/``. Pure-policy only (no LLM). A single task's exception is
caught per-future, recorded in the report's ``failures`` block, and does not
abort the run (a broken worker pool cancels the queue instead of waiting for
every remaining game); the streamed rows stay resumable either way. A stream
whose ``.meta.json`` sidecar is missing is refused, like ``eval.arena``.

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
from concurrent.futures.process import BrokenProcessPool
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
    import atexit

    from envs.fork_bridge import ForkBridge

    spec = LadderSpec(**spec_dict)
    prior = None
    if spec.checkpoint:
        from train.resolver import prior_resolver_from_checkpoint

        prior = prior_resolver_from_checkpoint(spec.checkpoint)
    bridge = ForkBridge(workers=int(bridge_workers))
    _WORKER["spec"] = spec
    _WORKER["prior"] = prior
    _WORKER["bridge"] = bridge
    # The pool worker owns its fork subprocesses; tear them down on interpreter
    # exit so a finished/idle worker does not leak orphan ``envs.fork_worker``
    # interpreters (mirrors the ``finally: bridge.close()`` in
    # ``scripts.run_search_agent``/``run_cfr_search_agent``).
    atexit.register(bridge.close)


def _worker_run(task_dict: dict[str, Any]) -> dict[str, Any]:
    task = LadderTask(**task_dict)
    return run_one(_WORKER["spec"], task, bridge=_WORKER["bridge"], prior=_WORKER.get("prior"))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _dedup_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse duplicate ``(opponent, seed, mode)`` rows, keeping the last one.

    Two runs sharing a ``--tag`` (e.g. accidentally launched in parallel) can
    append the same task twice. ``summarize`` groups every row, so a duplicate
    double-counts a game, while ``paired_difference`` is keyed by
    ``(opponent, seed, side)`` and collapses it -- the two aggregates then
    disagree. Deduping by task key makes every aggregate see the same games.
    """
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = (str(row["opponent"]), int(row["seed"]), str(row["mode"]))
        if key in by_key:
            duplicates += 1
        by_key[key] = row
    return list(by_key.values()), duplicates


def _median_upper(sorted_values: Sequence[float]) -> float:
    """Median taking the *upper* of the two middle values for an even count.

    Documented convention: with an even number of samples there is no single
    middle element, so we report ``sorted_values[len // 2]`` (the larger middle)
    rather than averaging -- the number stays an actually observed per-search
    cost instead of an interpolation.
    """
    return sorted_values[len(sorted_values) // 2]


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
        # upper middle on an even count (see _median_upper)
        "seconds_per_search_median": round(_median_upper(per_search), 3) if per_search else None,
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
    rows, duplicates = _dedup_rows(rows)
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
        "duplicate_rows": duplicates,
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


def _open_stream(
    stream_path: Path, meta_path: Path, spec: LadderSpec
) -> tuple[set[str], set[str]]:
    """Validate/seed the fingerprint sidecar and return the completed task keys.

    Mirrors ``eval.arena._stream_state``: a raw stream with no sidecar is
    **refused** (the fingerprint cannot be verified, so rows from a different
    budget/config could be silently mixed into the resume), and a fingerprint
    mismatch is refused. Returns ``(done_keys, error_keys)`` — the
    ``opponent|seed|mode`` keys already present (so the caller can skip them) and
    the subset whose streamed row carries an ``error`` (so a caller can retry
    them; see ``--retry-errors``).
    """
    fingerprint = spec.fingerprint()
    meta = _load_meta(meta_path)
    if meta is not None:
        if meta.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"refusing to resume {stream_path}: spec fingerprint changed "
                f"(delete {stream_path} + {meta_path} to start a new run)"
            )
    else:
        if stream_path.exists():
            raise SystemExit(
                f"refusing to resume {stream_path}: {meta_path.name} sidecar is missing, "
                f"so the spec fingerprint cannot be verified (delete the raw file or "
                f"use another --tag)"
            )
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
    done: set[str] = set()
    error_keys: set[str] = set()
    for r in read_jsonl(stream_path, tolerate_torn=True):
        key = f"{r['opponent']}|{r['seed']}|{r['mode']}"
        done.add(key)
        if r.get("error"):
            error_keys.add(key)
    return done, error_keys


def _failure_row(task: LadderTask, exc: BaseException) -> dict[str, Any]:
    """A minimal record for a task whose future raised before producing a row."""
    return {
        "task_index": task.index,
        "opponent": task.opponent,
        "seed": task.seed,
        "mode": task.mode,
        "error": f"{type(exc).__name__}: {exc}",
    }


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
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-queue streamed games whose row carries an error (they are otherwise "
        "treated as done forever, and paired_difference silently drops their pairs)",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        raise SystemExit(f"--workers must be >= 1 (got {args.workers})")
    if args.bridge_workers < 1:
        raise SystemExit(
            f"--bridge-workers must be >= 1 (got {args.bridge_workers}); the search "
            "forks only in subprocesses (workers == 0 would corrupt the JS runtime)"
        )
    if args.seeds < 1:
        raise SystemExit(f"--seeds must be >= 1 (got {args.seeds})")

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
    done, error_done = _open_stream(stream_path, meta_path, spec)
    if args.retry_errors:
        retried = done & error_done
        done -= error_done
        if retried:
            print(
                f"[ladder] --retry-errors: re-queuing {len(retried)} streamed game(s) that errored",
                flush=True,
            )
    elif error_done:
        print(
            f"[ladder] WARNING: {len(error_done)} streamed game(s) errored and stay 'done'; "
            f"paired_difference drops their pairs (survivorship bias). "
            f"Re-run with --retry-errors to retry them.",
            flush=True,
        )
    pending = [t for t in spec.tasks() if t.key not in done]
    print(
        f"[ladder] {spec.budget_label()} {spec.tag}: {len(done)} done, {len(pending)} to run "
        f"({len(spec.opponents)} opp x {len(spec.seeds)} seeds x {len(spec.modes)} modes), "
        f"workers={args.workers} bridge={args.bridge_workers}",
        flush=True,
    )

    start = time.perf_counter()
    failures: list[dict[str, Any]] = []
    if pending:
        ctx = _mp.get_context("spawn")
        executor = _futures.ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(asdict(spec), int(args.bridge_workers)),
        )
        aborted = False
        try:
            futures: dict[Any, LadderTask] = {}
            for t in pending:
                try:
                    futures[executor.submit(_worker_run, asdict(t))] = t
                except BrokenProcessPool as exc:
                    # The pool can die *during* submission too (the initializer
                    # raised on a bad --checkpoint); stop submitting and keep the
                    # rows already streamed.
                    failures.append(_failure_row(t, exc))
                    aborted = True
                    print(
                        f"[ladder] worker pool broke while submitting "
                        f"({type(exc).__name__}: {exc}); streamed rows kept",
                        flush=True,
                    )
                    break
            if not aborted:
                for future in _futures.as_completed(futures):
                    task = futures[future]
                    try:
                        row = future.result()
                    except BrokenProcessPool as exc:
                        # The worker pool itself died: no queued task can run, so
                        # stop and keep whatever already streamed.
                        # ``shutdown(cancel_futures=True)`` below avoids waiting
                        # for the queue only to discard it.
                        failures.append(_failure_row(task, exc))
                        aborted = True
                        print(
                            f"[ladder] worker pool broke ({type(exc).__name__}: {exc}); "
                            f"stopping after {len(failures)} failure(s); streamed rows kept",
                            flush=True,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
                        # A single task's exception (engine build, capture_snapshot,
                        # build_policy, worker init, ...) is recorded and skipped;
                        # the remaining queued games still run and the report is
                        # still written. ``Exception`` (not ``BaseException``) so a
                        # main-thread Ctrl-C/SystemExit still propagates.
                        failures.append(_failure_row(task, exc))
                        print(
                            f"[ladder] {task.opponent:<24} seed={task.seed:<3} {task.mode:<6} "
                            f"FAILED: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    append_jsonl(stream_path, row)
                    print(
                        f"[ladder] {task.opponent:<24} seed={task.seed:<3} {task.mode:<6} "
                        f"winner={row['winner']} outcome={row['outcome']} "
                        f"searched={row['searched_decisions']} changed={row['changed_decisions']} "
                        f"{row['wall_seconds']}s" + (f" error={row['error']}" if row["error"] else ""),
                        flush=True,
                    )
        finally:
            # Normal path: wait for the pool. Aborted path: cancel the queue
            # instead of shutdown(wait=True), which would run every remaining
            # game and then throw the results away.
            executor.shutdown(wait=not aborted, cancel_futures=aborted)
    wall = time.perf_counter() - start

    rows = read_jsonl(stream_path, tolerate_torn=True)
    report = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
        "failures": {
            "count": len(failures),
            "items": failures[:50],
            "items_truncated": len(failures) > 50,
        },
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
    if failures:
        print(
            f"[ladder] {len(failures)} task(s) failed before streaming (see report['failures']); "
            f"re-run the same command to retry them (streamed rows stay resumable)",
            flush=True,
        )
    print(f"[ladder] wrote {report_path}", flush=True)
    # A partial report is still written; a nonzero code flags the failures so a
    # driving job/shell notices, while the streamed rows stay resumable.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
