"""Strength-evaluation scoreboard runner (PLAN.md WS2 / M2 / M3-CPU / M4).

Runs a **contestant** policy spec (``expert:<slug>`` | ``neural:<ckpt>`` |
``heuristic`` | ``legal_random``) against the fixed scripted opponent pool on
**paired seeds** and reports the protocol scoreboard defined in
``reports/explore/eval_protocol.md``:

- pooled win rate with Wilson 95% CI *and* an honest opponent-cluster CI;
- per-opponent win rate + Wilson CI + floor violations (M4: none < 45%);
- a fixed-anchor Bradley-Terry/Elo rating (vs the L5.1 20x20 anchors) and the
  online ``eval.ladder`` Elo;
- decision-quality proxies (fallback/illegal rate must be 0, round/decision
  efficiency, draws, errors);
- a best-response probe (worst-matchup exploitability lower bound);
- optional paired A/B vs ``--baseline`` (non-inferiority training gate).

Both sides of each matchup are played (contestant as player 0 and player 1) on
the same seed, so first-player advantage cancels.

Design constraints (PLAN.md D7 / §0.1 C8):

- raw per-match rows stream to ``data/eval/<tag>.jsonl`` (gitignored) and the run
  **resumes by task_index**; a ``.meta.json`` sidecar pins a spec fingerprint and
  refuses to resume against config drift;
- only the aggregate lands in ``reports/explore/`` (committed).

Usage::

    python -m scripts.eval_protocol --contestant expert:superconduct_aggro \\
        --opponents superconduct_aggro natlan_battleship --seeds 2 --workers 4
    python -m scripts.eval_protocol --contestant neural:<ckpt>#t=0 \\
        --contestant-deck superconduct_aggro --seeds 50 --workers 12
    python -m scripts.eval_protocol --contestant expert:superconduct_aggro \\
        --baseline neural:<old_ckpt> --seeds 20          # paired non-inferiority
    python -m scripts.eval_protocol --sample-sizes       # print planning tables
    python -m scripts.eval_protocol --mode llm --seeds 10 # LLM-assisted (separate report)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from common.jsonl import ensure_trailing_newline, read_jsonl
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from envs.rollout import MatchTask, normalize_workers, run_tasks
from eval.opponents import scripted_opponents
from eval.scoreboard import (
    MatchOutcome,
    anchored_bt_rating,
    best_response_probe,
    elo_from_records,
    evaluate_gates,
    merge_gate,
    noninferiority_gate,
    paired_difference,
    sample_size_table,
    summarize,
)

PURE = "pure"
LLM = "llm"


@dataclass(frozen=True)
class ProtocolTask:
    index: int
    opponent: str
    seed: int
    side: int  # which player the contestant controls

    @property
    def key(self) -> str:
        return f"{self.opponent}|{self.seed}|{self.side}"

    def to_match(self, spec: "ProtocolSpec") -> MatchTask:
        opp_spec = f"expert:{self.opponent}"
        if self.side == 0:
            deck0, deck1 = spec.contestant_deck, self.opponent
            policy0, policy1 = spec.contestant, opp_spec
        else:
            deck0, deck1 = self.opponent, spec.contestant_deck
            policy0, policy1 = opp_spec, spec.contestant
        return MatchTask(
            index=self.index,
            deck0=deck0,
            deck1=deck1,
            seed=spec.base_seed + int(self.seed),
            policy0=policy0,
            policy1=policy1,
            tag=spec.tag,
        )


@dataclass(frozen=True)
class ProtocolSpec:
    contestant: str
    contestant_deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    sides: tuple[int, ...] = (0, 1)
    mode: str = PURE
    workers: int | None = None
    base_seed: int = 0
    tag: str = "eval_protocol"
    llm_budget: int = 6
    llm_model: str = "deepseek-chat"

    def tasks(self) -> list[ProtocolTask]:
        tasks: list[ProtocolTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                for side in self.sides:
                    tasks.append(
                        ProtocolTask(index=index, opponent=opponent, seed=int(seed), side=int(side))
                    )
                    index += 1
        return tasks

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "contestant": self.contestant,
                "contestant_deck": self.contestant_deck,
                "opponents": list(self.opponents),
                "seeds": list(self.seeds),
                "sides": list(self.sides),
                "mode": self.mode,
                "base_seed": self.base_seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# streaming / resume
# --------------------------------------------------------------------------- #
class _Stream:
    """Append-only per-match stream with a fingerprint sidecar (resume-safe)."""

    def __init__(self, spec: ProtocolSpec, raw_path: Path):
        self.raw_path = raw_path
        self.meta_path = raw_path.with_name(raw_path.name.removesuffix(".jsonl") + ".meta.json")
        self.rows: list[dict[str, Any]] = []
        if raw_path.exists():
            if not self.meta_path.exists():
                raise RuntimeError(
                    f"{raw_path} exists but {self.meta_path} is missing; cannot resume "
                    "safely (delete the raw file or use another --tag)"
                )
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != spec.fingerprint():
                raise RuntimeError(
                    f"{raw_path} was produced by a different spec (fingerprint "
                    "mismatch); use another --tag or delete the old raw/meta files"
                )
            self.rows = read_jsonl(raw_path, tolerate_torn=True)
        else:
            ensure_dir(raw_path.parent)
            self.meta_path.write_text(
                json.dumps(
                    {
                        "fingerprint": spec.fingerprint(),
                        "tag": spec.tag,
                        "contestant": spec.contestant,
                        "contestant_deck": spec.contestant_deck,
                        "opponents": list(spec.opponents),
                        "seeds": list(spec.seeds),
                        "match_seeds": [spec.base_seed + int(s) for s in spec.seeds],
                        "sides": list(spec.sides),
                        "mode": spec.mode,
                        "base_seed": spec.base_seed,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        ensure_trailing_newline(raw_path)
        self._handle = raw_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def done_indices(self) -> set[int]:
        return {int(row["task_index"]) for row in self.rows if "task_index" in row}


def _enrich(row: dict[str, Any], task: ProtocolTask, spec: ProtocolSpec) -> dict[str, Any]:
    side = task.side
    winner = row.get("winner")
    score = 0.5 if winner is None else (1.0 if int(winner) == side else 0.0)
    row.update(
        {
            "opponent": task.opponent,
            "side": side,
            "protocol_seed": task.seed,
            "match_seed": spec.base_seed + int(task.seed),
            "score": score,
            "decisions": int(row.get(f"decisions{side}", 0)),
            "fallbacks": int(row.get(f"fallbacks{side}", 0)),
            "contestant": spec.contestant,
            "mode": spec.mode,
        }
    )
    return row


def _run_llm_task(spec: ProtocolSpec, task: ProtocolTask, *, client: Any) -> dict[str, Any]:
    """LLM-assisted mode (in-process; the client is not serializable).

    The LLM re-ranks a base proposer (``spec.contestant``) at a few decisions with
    a hard budget and forced fallback (``agents.llm_assist``). Reported *separately*
    from pure mode (PLAN.md §6.3).
    """
    from agents.llm_assist import AssistConfig, build_llm_assist_policy
    from common.seeding import derive_seed
    from envs.decks import deck_spec
    from envs.match import run_match
    from envs.policy import build_policy, expert_policy

    opp = task.opponent
    base_seed = derive_seed(spec.base_seed + task.seed, f"side{task.side}", "contestant")
    base = build_policy(spec.contestant, seed=base_seed, role="contestant")
    policy = build_llm_assist_policy(
        spec.contestant_deck,
        base=base,
        client=client,
        config=AssistConfig(budget_per_game=spec.llm_budget, model=spec.llm_model),
        seed=base_seed,
        opponent_name=opp,
    )
    opp_policy = expert_policy(opp, seed=derive_seed(spec.base_seed + task.seed, "opp", opp))
    if task.side == 0:
        deck0, deck1 = deck_spec(spec.contestant_deck), deck_spec(opp)
        pol0, pol1 = policy, opp_policy
    else:
        deck0, deck1 = deck_spec(opp), deck_spec(spec.contestant_deck)
        pol0, pol1 = opp_policy, policy
    record = run_match(deck0, deck1, pol0, pol1, seed=spec.base_seed + task.seed)
    interventions = [iv.as_dict() for iv in getattr(policy, "interventions", [])]
    row: dict[str, Any] = {
        "task_index": task.index,
        "deck0": record.deck0,
        "deck1": record.deck1,
        "seed": record.seed,
        "winner": record.winner,
        "rounds": record.rounds,
        "wall_seconds": round(record.wall_seconds, 3),
        "error": record.error,
        "truncated": record.truncated,
        "decisions0": record.decisions0,
        "decisions1": record.decisions1,
        "fallbacks0": record.fallbacks0,
        "fallbacks1": record.fallbacks1,
        "llm_calls": len(interventions),
        "llm_changed": sum(1 for iv in interventions if iv.get("changed")),
        "llm_failed": sum(1 for iv in interventions if iv.get("error")),
    }
    return _enrich(row, task, spec)


def run_protocol(
    spec: ProtocolSpec,
    *,
    raw_path: Path | None = None,
    progress: bool = True,
    client: Any = None,
) -> list[dict[str, Any]]:
    """Run every pending task, streaming enriched rows; resume by task_index."""
    tasks = {t.index: t for t in spec.tasks()}
    stream = _Stream(spec, raw_path) if raw_path is not None else None
    done = stream.done_indices() if stream else set()
    pending = [t for t in tasks.values() if t.index not in done]

    try:
        if spec.mode == LLM:
            assert stream is not None, "LLM mode requires a raw_path to stream to"
            if client is None:
                raise RuntimeError("LLM mode requested but no client is available")
            for i, task in enumerate(pending, 1):
                row = _run_llm_task(spec, task, client=client)
                stream.append(row)
                if progress and (i % 10 == 0 or i == len(pending)):
                    print(f"[eval] {i}/{len(pending)} {task.opponent} seed={task.seed} "
                          f"side={task.side} winner={row['winner']}")
        else:
            match_tasks = [task.to_match(spec) for task in pending]

            def _on_result(row: dict[str, Any]) -> None:
                task = tasks[int(row["task_index"])]
                enriched = _enrich(row, task, spec)
                if stream is not None:
                    stream.append(enriched)

            fresh = run_tasks(match_tasks, workers=spec.workers, on_result=_on_result)
            if stream is None:
                return [_enrich(row, tasks[int(row["task_index"])], spec) for row in fresh]
    finally:
        if stream is not None:
            stream.close()

    if stream is None:
        return []
    # Re-read the whole stream and de-duplicate by task_index (the in-memory
    # snapshot predates this run; the file holds the merged, resumed result).
    by_index: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(stream.raw_path, tolerate_torn=True):
        by_index[int(row["task_index"])] = row
    return [by_index[i] for i in sorted(by_index)]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def outcomes_from_rows(rows: Sequence[dict[str, Any]]) -> list[MatchOutcome]:
    return [
        MatchOutcome(
            opponent=str(row["opponent"]),
            seed=int(row["protocol_seed"]),
            side=int(row["side"]),
            score=float(row["score"]),
            rounds=int(row.get("rounds", 0)),
            decisions=int(row.get("decisions", 0)),
            fallbacks=int(row.get("fallbacks", 0)),
            error=row.get("error"),
            truncated=bool(row.get("truncated")),
        )
        for row in rows
    ]


def load_anchors(path: str | Path | None = None) -> tuple[dict[str, float], str | None]:
    """Fixed reference ratings for the ladder (L5.1 20x20 Elo by default)."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(reports_dir("arena", "20261008T163940_full_matrix_20x20.json"))
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            anchors = payload.get("elo", {})
            if anchors:
                return {str(k): float(v) for k, v in anchors.items()}, str(candidate)
    return {}, None


def aggregate(
    spec: ProtocolSpec,
    rows: Sequence[dict[str, Any]],
    *,
    anchors: dict[str, float],
    anchors_source: str | None,
    pooled_threshold: float,
    floor: float,
    strict_ci: bool,
) -> dict[str, Any]:
    outcomes = outcomes_from_rows(rows)
    summary = summarize(outcomes, floor=floor)
    gates = evaluate_gates(
        summary, pooled_threshold=pooled_threshold, floor=floor, strict_ci=strict_ci
    )
    # Online Elo must key by *player identity*, not deck slug: the contestant's
    # deck can share a slug with a pool opponent (the default invocation does),
    # which would collapse both sides into one rating entry and make the
    # head-to-head games a no-op. Opponents get a ``pool:`` prefix so they never
    # collide with the contestant's policy spec.
    records = [
        {
            "player0": spec.contestant if int(row["side"]) == 0 else f"pool:{row['opponent']}",
            "player1": spec.contestant if int(row["side"]) == 1 else f"pool:{row['opponent']}",
            "deck0": row["deck0"],
            "deck1": row["deck1"],
            "winner": row["winner"],
            "task_index": row["task_index"],
        }
        for row in rows
    ]
    return {
        "summary": summary,
        "gates": gates,
        "rating_anchored": anchored_bt_rating(outcomes, anchors) if anchors else None,
        "anchors_source": anchors_source,
        "elo_online": elo_from_records(records),
        "best_response": best_response_probe(summary["per_opponent"], floor=floor),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _select_opponents(names: Sequence[str] | None) -> tuple[str, ...]:
    pool = [opp.name for opp in scripted_opponents()]
    if not names:
        return tuple(pool)
    known = set(pool)
    missing = [n for n in names if n not in known]
    if missing:
        raise SystemExit(f"unknown opponent(s): {missing}")
    return tuple(names)


def _default_deck(contestant: str) -> str:
    if contestant.startswith("expert:"):
        return contestant.split(":", 1)[1]
    return "superconduct_aggro"


def _protocol_seeds(n: int) -> tuple[int, ...]:
    """Relative task seeds; ``ProtocolSpec.base_seed`` is the engine seed start.

    Keeping this in one place prevents the old double-offset bug where ``main``
    passed ``range(seed_start, seed_start + n)`` *and* ``base_seed=seed_start``.
    """
    return tuple(range(max(0, int(n))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contestant",
        default="expert:superconduct_aggro",
        help="policy spec: expert:<slug> | neural:<ckpt>[#t=<temp>] | heuristic | legal_random",
    )
    parser.add_argument("--contestant-deck", default=None, help="deck slug the contestant pilots")
    parser.add_argument("--opponents", nargs="*", default=None, help="pool subset (slugs)")
    parser.add_argument("--seeds", type=int, default=3, help="number of paired seeds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--sides",
        nargs="*",
        type=int,
        default=[0, 1],
        choices=[0, 1],
        help="which player(s) the contestant occupies (both cancels first-player bias)",
    )
    parser.add_argument("--mode", choices=[PURE, LLM], default=PURE)
    parser.add_argument("--llm-budget", type=int, default=6)
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--baseline", default=None, help="optional A/B baseline policy spec")
    parser.add_argument("--baseline-deck", default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="worker subprocesses (unset = all cores; 1 = single process)")
    parser.add_argument("--floor", type=float, default=0.45, help="per-opponent floor (M4)")
    parser.add_argument("--pooled-threshold", type=float, default=0.50, help="M2/M4 pooled bar")
    parser.add_argument("--strict-ci", action="store_true", help="require cluster CI to clear the bar")
    parser.add_argument("--anchors", default=None, help="path to an arena summary with an 'elo' block")
    parser.add_argument("--tag", default="eval_protocol")
    parser.add_argument("--smoke", action="store_true", help="2 opponents x 2 seeds x 2 sides")
    parser.add_argument("--no-write", action="store_true", help="do not write raw/report artifacts")
    parser.add_argument("--sample-sizes", action="store_true", help="print planning tables and exit")
    return parser


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    quality = summary["quality"]
    print(f"[eval] contestant={report['protocol']['contestant']} "
          f"mode={report['protocol']['mode']} games={summary['games']}")
    print(f"[eval] pooled={summary['pooled_rate']:.3f} "
          f"wilson95=[{summary['pooled_ci_low']:.3f},{summary['pooled_ci_high']:.3f}] "
          f"cluster95=[{summary['cluster_ci_low']:.3f},{summary['cluster_ci_high']:.3f}] "
          f"(bootstrap [{summary['cluster_boot_ci_low']:.3f},{summary['cluster_boot_ci_high']:.3f}])")
    print("[eval] per-opponent (contestant rate):")
    for opponent, row in sorted(summary["per_opponent"].items(), key=lambda kv: kv[1]["rate"]):
        flag = "  <-- floor" if opponent in summary["floor_violations"] else ""
        print(f"  {opponent:<32} {row['rate']:.3f} "
              f"[{row['ci_low']:.3f},{row['ci_high']:.3f}] n={row['games']}{flag}")
    print(f"[eval] quality: fallback_rate={quality['fallback_rate']:.4f} "
          f"error_rate={quality['error_rate']:.4f} draw_rate={quality['draw_rate']:.3f} "
          f"mean_rounds={quality['mean_rounds']:.1f}")
    rating = report.get("rating_anchored")
    if rating:
        print(f"[eval] anchored rating {rating['rating']:.1f} "
              f"[{rating['ci_low']:.1f},{rating['ci_high']:.1f}] (vs {report['anchors_source']})")
    probe = report.get("best_response")
    if probe:
        print(f"[eval] best-response probe: worst={probe['worst_opponent']} "
              f"rate={probe['worst_rate']:.3f} exploit_lb={probe['exploitability_lower_bound']:.3f}")
    gates = report["gates"]
    print(f"[eval] gates: pooled={gates['pooled_pass']} floor={gates['floor_pass']} "
          f"clean={gates['clean_pass']} -> {'PASS' if gates['pass'] else 'FAIL'}")
    paired = report.get("paired_vs_baseline")
    if paired:
        merge = report.get("merge_gate") or {}
        print(f"[eval] paired vs baseline: delta={paired['delta']:+.3f} "
              f"95%=[{paired['ci_low']:+.3f},{paired['ci_high']:+.3f}] "
              f"n={paired['n_pairs']} noninferior={merge.get('noninferior')} "
              f"clean={merge.get('clean')} merge_accept={merge.get('accept')}")


def _print_sample_sizes() -> None:
    print("Pooled Wilson 95% CI half-width -> sample size (p=0.5, K=20 opponents):")
    print(f"  {'halfwidth':>10} {'total':>8} {'per-opp':>8} {'hours@4900gph':>14}")
    for row in sample_size_table():
        print(f"  {row['halfwidth']:>10.3f} {row['total_games']:>8} "
              f"{row['games_per_opponent']:>8} {row['hours_at_4900gph']:>14.2f}")
    print("\nDetect an absolute rate change (two-sided, 80% power, p=0.5) per arm:")
    from eval.scoreboard import detection_sample_size

    for delta in (0.10, 0.075, 0.05, 0.03, 0.02, 0.01):
        print(f"  delta {delta:>5.3f} -> {detection_sample_size(delta):>7} games/arm")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_sizes:
        _print_sample_sizes()
        return 0

    # A neural checkpoint does not carry the deck it was trained for; defaulting
    # to superconduct_aggro silently evaluates the wrong deck (A2 post-mortem).
    if args.contestant.startswith("neural:") and not args.contestant_deck:
        raise SystemExit("--contestant-deck is required when --contestant is neural:<ckpt>")
    if args.baseline and args.baseline.startswith("neural:") and not args.baseline_deck:
        raise SystemExit("--baseline-deck is required when --baseline is neural:<ckpt>")

    if args.smoke:
        opponents = ("superconduct_aggro", "natlan_battleship")
        seeds = tuple(range(min(args.seeds, 2)))
    else:
        opponents = _select_opponents(args.opponents)
        # ``base_seed`` is the first *engine* seed; ProtocolTask.seed is the
        # relative offset (0..n-1) and to_match() adds it. Passing absolute seeds
        # here as well double-offset every run after --seed-start 0 (e.g.
        # --seed-start 900 --seeds 10 actually played 1800..1809).
        seeds = _protocol_seeds(args.seeds)

    contestant_deck = args.contestant_deck or _default_deck(args.contestant)
    try:
        workers = normalize_workers(args.workers)
    except ValueError as exc:
        raise SystemExit(str(exc))
    spec = ProtocolSpec(
        contestant=args.contestant,
        contestant_deck=contestant_deck,
        opponents=opponents,
        seeds=seeds,
        sides=tuple(args.sides),
        mode=args.mode,
        workers=workers,
        base_seed=args.seed_start,
        tag=args.tag,
        llm_budget=args.llm_budget,
        llm_model=args.llm_model,
    )

    raw_path = None if args.no_write else data_dir("eval", f"{spec.tag}.jsonl")
    client = None
    if spec.mode == LLM:
        from common.llm import build_llm_client

        client = build_llm_client()
        if client is None:
            raise SystemExit("LLM mode requested but no client/key is available")

    print(f"[eval] {len(spec.opponents)} opponents x {len(spec.seeds)} seeds x "
          f"{len(spec.sides)} sides = {len(spec.tasks())} matches "
          f"(mode={spec.mode}, contestant={spec.contestant})")
    rows = run_protocol(spec, raw_path=raw_path, client=client)

    anchors, anchors_source = load_anchors(args.anchors)
    report = {
        "tag": spec.tag,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "contestant": spec.contestant,
            "contestant_deck": spec.contestant_deck,
            "opponents": list(spec.opponents),
            "seeds": list(spec.seeds),
            "match_seeds": [spec.base_seed + int(s) for s in spec.seeds],
            "sides": list(spec.sides),
            "mode": spec.mode,
            "workers": spec.workers,
            "base_seed": spec.base_seed,
            "pooled_threshold": args.pooled_threshold,
            "floor": args.floor,
            "strict_ci": args.strict_ci,
        },
        **aggregate(
            spec,
            rows,
            anchors=anchors,
            anchors_source=anchors_source,
            pooled_threshold=args.pooled_threshold,
            floor=args.floor,
            strict_ci=args.strict_ci,
        ),
        "n_matches": len(rows),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }

    if args.baseline:
        baseline_spec = ProtocolSpec(
            contestant=args.baseline,
            contestant_deck=args.baseline_deck or _default_deck(args.baseline),
            opponents=opponents,
            seeds=seeds,
            sides=tuple(args.sides),
            mode=spec.mode,
            workers=workers,
            base_seed=args.seed_start,
            tag=f"{spec.tag}__baseline",
            llm_budget=args.llm_budget,
            llm_model=args.llm_model,
        )
        baseline_raw = None if args.no_write else data_dir("eval", f"{baseline_spec.tag}.jsonl")
        print(f"[eval] baseline: {baseline_spec.contestant} ({len(baseline_spec.tasks())} matches)")
        baseline_rows = run_protocol(baseline_spec, raw_path=baseline_raw, client=client)
        paired = paired_difference(
            outcomes_from_rows(rows), outcomes_from_rows(baseline_rows)
        )
        report["paired_vs_baseline"] = paired
        report["noninferiority"] = (
            noninferiority_gate(paired, tolerance=0.02) if paired else None
        )
        report["merge_gate"] = merge_gate(paired, report["gates"], tolerance=0.02)

    _print_report(report)

    if not args.no_write:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = ensure_dir(reports_dir("explore"))
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        report_path = out_dir / f"{spec.tag}_{stamp}.json"
        report_path.write_text(payload, encoding="utf-8")
        latest_path = out_dir / f"{spec.tag}.json"
        latest_path.write_text(payload, encoding="utf-8")
        print(f"[eval] wrote {report_path}")
        print(f"[eval] latest {latest_path}")
        if raw_path is not None:
            print(f"[eval] raw {raw_path} (streamed; re-run the same command to resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
