"""LLM-assisted vs pure-policy win-rate matrix over the full opponent pool.

This is the L5.2 deliverable (PLAN.md §4.2 / §10): scale the in-game LLM probe
from a single pairing to the whole scripted pool, reporting **pure-policy** and
**LLM-assisted** modes *separately* (PLAN.md §6.3 -- strength acceptance G1/M4 is
pure-policy only), and aggregate the interventions for the coach/BC layers.

Design constraints (PLAN.md D7 / §0.1 I7):

- results stream to ``data/llm/<tag>.jsonl`` per match (gitignored) so an
  interrupted run resumes by ``task_index`` instead of restarting;
- a ``.meta.json`` sidecar pins a spec fingerprint and refuses to resume against
  a drifting configuration;
- only an **aggregate** report lands in ``reports/llm/`` (committed); the
  per-intervention detail stays in the raw stream.

Usage::

    python -m scripts.run_llm_matrix --smoke                 # 3 opponents x 3 seeds
    python -m scripts.run_llm_matrix --seeds 10 --budget 6   # full pool
    python -m scripts.run_llm_matrix --no-llm --seeds 10     # pure-policy only
    python -m scripts.run_llm_matrix --opponents natlan_battleship double_geo_navia
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agents.llm_assist import AssistConfig, build_llm_assist_policy
from common.llm import build_llm_client
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import expert_policy
from eval.opponents import scripted_opponents
from eval.stats import estimate_rate, outcome_score

PURE = "pure"
LLM = "llm"


@dataclass(frozen=True)
class MatrixSpec:
    deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    modes: tuple[str, ...] = (PURE, LLM)
    budget: int = 6
    model: str = "deepseek-chat"
    tag: str = "llm_matrix"

    def tasks(self) -> list["MatrixTask"]:
        """Deterministic task list; the index is stable across resume runs."""
        tasks: list[MatrixTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                for mode in self.modes:
                    tasks.append(
                        MatrixTask(index=index, opponent=opponent, seed=int(seed), mode=mode)
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
                "budget": self.budget,
                "model": self.model,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatrixTask:
    index: int
    opponent: str
    seed: int
    mode: str


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in ("calls", "prompt_tokens", "completion_tokens", "errors")
    }


def run_task(spec: MatrixSpec, task: MatrixTask, *, client: Any) -> dict[str, Any]:
    """Play one match (pure base policy, optionally LLM re-ranked) and record it."""
    p0_seed = derive_seed(task.seed, "p0", spec.deck)
    p1_seed = derive_seed(task.seed, "p1", task.opponent)
    opponent_policy = expert_policy(task.opponent, seed=p1_seed)

    usage_before: dict[str, Any] = {}
    policy = expert_policy(spec.deck, seed=p0_seed)
    interventions: list[dict[str, Any]] = []
    if task.mode == LLM:
        if client is None:
            raise RuntimeError("LLM mode requested but no client is available")
        usage_before = dict(client.usage.as_dict())
        config = AssistConfig(budget_per_game=spec.budget, model=spec.model)
        policy = build_llm_assist_policy(
            spec.deck, client=client, seed=p0_seed, config=config
        )

    record = run_match(
        deck_spec(spec.deck),
        deck_spec(task.opponent),
        policy,
        opponent_policy,
        seed=task.seed,
    )

    row: dict[str, Any] = {
        "task_index": task.index,
        "opponent": task.opponent,
        "seed": task.seed,
        "mode": task.mode,
        "winner": record.winner,
        "score0": outcome_score(record.winner, 0),
        "rounds": record.rounds,
        "wall_seconds": round(record.wall_seconds, 3),
        "error": record.error,
        "truncated": record.truncated,
        "io_errors0": list(record.io_errors0),
        "fallbacks0": record.fallbacks0,
    }
    if task.mode == LLM:
        usage_after = dict(client.usage.as_dict())
        interventions = [iv.as_dict() for iv in getattr(policy, "interventions", [])]
        row["llm_usage"] = _usage_delta(usage_before, usage_after)
        row["interventions"] = interventions
        row["interventions_changed"] = sum(1 for iv in interventions if iv["changed"])
        row["interventions_failed"] = sum(
            1 for iv in interventions if iv.get("error")
        )
    return row


def _open_stream(
    spec: MatrixSpec, raw_path: Path | None
) -> tuple[dict[int, dict[str, Any]], Any]:
    """Load completed rows and open the append-only stream (resume guard)."""
    if raw_path is None:
        return {}, None
    meta_path = raw_path.with_name(raw_path.name.removesuffix(".jsonl") + ".meta.json")
    done: dict[int, dict[str, Any]] = {}
    if raw_path.exists():
        if not meta_path.exists():
            raise RuntimeError(
                f"{raw_path} exists but {meta_path} is missing; cannot resume "
                "safely (delete the raw file or use another --tag)"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != spec.fingerprint():
            raise RuntimeError(
                f"{raw_path} was produced by a different spec (fingerprint "
                "mismatch); use another --tag or delete the old raw/meta files"
            )
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn trailing line after a hard kill
                done[int(row["task_index"])] = row
    else:
        meta_path.write_text(
            json.dumps(
                {"fingerprint": spec.fingerprint(), "tag": spec.tag,
                 "deck": spec.deck, "modes": list(spec.modes)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    handle = raw_path.open("a", encoding="utf-8")
    return done, handle


def run_matrix(
    spec: MatrixSpec,
    *,
    client: Any = None,
    raw_path: Path | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run every pending (opponent, seed, mode) task, streaming results to disk."""
    done, handle = _open_stream(spec, raw_path)
    rows: list[dict[str, Any]] = list(done.values())
    if done and progress:
        print(f"[llm-matrix] resuming: {len(done)} matches already done")
    total = len(spec.tasks())
    if client is None and LLM in spec.modes:
        client = build_llm_client()
        if client is None:
            raise SystemExit(
                "[llm-matrix] no LLM config; cannot run 'llm' mode "
                "(use --no-llm for pure-policy only)"
            )
    try:
        for task in spec.tasks():
            if task.index in done:
                continue
            row = run_task(spec, task, client=client)
            rows.append(row)
            if handle is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if progress:
                extra = ""
                if task.mode == LLM:
                    extra = (
                        f" changed={row.get('interventions_changed')}/"
                        f"{row.get('llm_usage', {}).get('calls')}calls"
                    )
                print(
                    f"[llm-matrix] {len(rows)}/{total} "
                    f"{task.opponent} seed={task.seed} {task.mode} "
                    f"winner={row['winner']}{extra}"
                )
    finally:
        if handle is not None:
            handle.close()
    rows.sort(key=lambda item: item["task_index"])
    return rows


def _rate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score0"]) for row in rows]
    est = estimate_rate(sum(scores), len(scores))
    payload = est.as_dict()
    payload["errors"] = sum(1 for row in rows if row["error"])
    payload["truncated"] = sum(1 for row in rows if row["truncated"])
    return payload


def aggregate(rows: Sequence[dict[str, Any]], spec: MatrixSpec) -> dict[str, Any]:
    """Build the committed (aggregate-only) report from the raw match rows."""
    per_opponent: list[dict[str, Any]] = []
    for opponent in spec.opponents:
        entry: dict[str, Any] = {"opponent": opponent}
        pure = [r for r in rows if r["opponent"] == opponent and r["mode"] == PURE]
        llm = [r for r in rows if r["opponent"] == opponent and r["mode"] == LLM]
        if pure:
            entry[PURE] = _rate(pure)
        if llm:
            entry[LLM] = _rate(llm)
        if pure and llm:
            entry["delta"] = round(entry[LLM]["rate"] - entry[PURE]["rate"], 4)
            changed = sum(int(r.get("interventions_changed", 0)) for r in llm)
            calls = sum(int(r.get("llm_usage", {}).get("calls", 0)) for r in llm)
            entry["llm_interventions_changed"] = changed
            entry["llm_calls"] = calls
        per_opponent.append(entry)

    overall: dict[str, Any] = {}
    for mode in spec.modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if mode_rows:
            overall[mode] = _rate(mode_rows)

    llm_rows = [r for r in rows if r["mode"] == LLM]
    llm_usage = {
        key: sum(int(r.get("llm_usage", {}).get(key, 0)) for r in llm_rows)
        for key in ("calls", "prompt_tokens", "completion_tokens", "errors")
    }
    interventions = {
        "matches_using_llm": len(llm_rows),
        "total": sum(len(r.get("interventions", ())) for r in llm_rows),
        "changed": sum(int(r.get("interventions_changed", 0)) for r in llm_rows),
        "failed": sum(int(r.get("interventions_failed", 0)) for r in llm_rows),
    }
    if interventions["total"]:
        interventions["changed_rate"] = round(
            interventions["changed"] / interventions["total"], 4
        )

    return {
        "deck": spec.deck,
        "modes": list(spec.modes),
        "opponents": list(spec.opponents),
        "seeds": list(spec.seeds),
        "budget_per_game": spec.budget,
        "model": spec.model,
        "n_matches": len(rows),
        "overall": overall,
        "per_opponent": per_opponent,
        "overall_delta": (
            round(overall[LLM]["rate"] - overall[PURE]["rate"], 4)
            if PURE in overall and LLM in overall
            else None
        ),
        "llm_usage": llm_usage,
        "interventions": interventions,
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }


def _print_report(rep: dict[str, Any]) -> None:
    print(f"[llm-matrix] deck={rep['deck']} matches={rep['n_matches']} "
          f"modes={rep['modes']}")
    header = f"  {'opponent':<34} {'pure':>8} {'llm':>8} {'delta':>8}"
    print(header)
    for row in rep["per_opponent"]:
        pure = row.get(PURE, {}).get("rate")
        llm = row.get(LLM, {}).get("rate")
        pure_s = f"{pure:.3f}" if pure is not None else "-"
        llm_s = f"{llm:.3f}" if llm is not None else "-"
        delta_s = f"{row['delta']:+.3f}" if "delta" in row else "-"
        print(f"  {row['opponent']:<34} {pure_s:>8} {llm_s:>8} {delta_s:>8}")
    for mode, stats in rep["overall"].items():
        print(
            f"[llm-matrix] overall {mode:<4} {stats['rate']:.3f} "
            f"[{stats['ci_low']:.3f},{stats['ci_high']:.3f}] "
            f"({stats['wins']:.1f}/{stats['games']})"
        )
    if rep["llm_usage"]["calls"]:
        iv = rep["interventions"]
        print(
            f"[llm-matrix] llm calls={rep['llm_usage']['calls']} "
            f"errors={rep['llm_usage']['errors']} "
            f"interventions changed={iv['changed']}/{iv['total']} "
            f"({iv.get('changed_rate', 0):.0%})"
        )


def _select_opponents(names: Sequence[str] | None) -> tuple[str, ...]:
    pool = tuple(opp.name for opp in scripted_opponents())
    if not names:
        return pool
    wanted = set(names)
    missing = wanted - set(pool)
    if missing:
        raise SystemExit(f"unknown opponent(s): {sorted(missing)}")
    return tuple(name for name in pool if name in wanted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM vs pure-policy matrix.")
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=6, help="LLM calls per game")
    parser.add_argument("--model", default="deepseek-chat", help="in-game model")
    parser.add_argument("--no-llm", action="store_true", help="pure-policy only")
    parser.add_argument("--tag", default="llm_matrix")
    parser.add_argument("--smoke", action="store_true", help="3 opponents x 3 seeds")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        opponents = _select_opponents(
            args.opponents
            or ["natlan_battleship", "double_geo_navia", "skirk_chasca_freeze"]
        )
        seeds = tuple(range(min(args.seeds, 3)))
        tag = f"{args.tag}_smoke"
    else:
        opponents = _select_opponents(args.opponents)
        seeds = tuple(range(args.seeds))
        tag = args.tag
    modes = (PURE,) if args.no_llm else (PURE, LLM)
    spec = MatrixSpec(
        deck=args.deck,
        opponents=opponents,
        seeds=seeds,
        modes=modes,
        budget=args.budget,
        model=args.model,
        tag=tag,
    )

    raw_path = None
    if not args.no_write:
        raw_path = ensure_dir(data_dir("llm")) / f"{tag}.jsonl"
    rows = run_matrix(spec, raw_path=raw_path)
    report = aggregate(rows, spec)
    _print_report(report)

    if not args.no_write:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = ensure_dir(reports_dir("llm")) / f"matrix_{stamp}_{tag}.json"
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[llm-matrix] wrote {out}")
        print(f"[llm-matrix] raw      {raw_path} (streamed; re-run to resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
