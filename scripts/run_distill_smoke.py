"""M3-CPU S-C smoke: collect → distill → paired eval → gate → persist.

Runs one small, CPU-sized search-distillation round end to end and records the
curve + the training merge gate, so the S-C pipeline (search-as-teacher → belief
head → distillation) is exercised with zero manual steps:

1. **collect** -- MCCFR search-teacher data via the resumable collector
   (``train.collect`` with a ``SearchSpec``); every searched decision carries the
   solver's root average strategy (semantic keys) + root value + belief labels.
2. **distill** -- a ``CVPN`` with the belief head is trained by
   ``train.distill.train_distill`` (policy soft-CE + value regression + belief
   scoring); a **cold-start behaviour-cloning** model on the same data is trained
   as the baseline arm.
3. **paired eval** -- both checkpoints play the scripted pool on paired seeds both
   ways (``scripts.eval_protocol.run_protocol``) and the paired A/B
   non-inferiority merge gate (``CI_low(Δ) ≥ -tolerance``) decides whether the
   distilled arm is accepted.
4. **persist** -- checkpoints to ``data/checkpoints`` (gitignored); the aggregate
   report + curve to ``reports/train/distill_smoke_<ts>.json`` (committed).

Usage::

    python -m scripts.run_distill_smoke --deck superconduct_aggro \\
        --opponents natlan_battleship --seeds 2 --iterations 2 --depth 1 \\
        --d-model 64 --epochs 12 --eval-seeds 2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from train.collect import CollectSpec, SearchSpec, collect
from train.data import load_samples, sample_stats
from train.distill import DistillConfig, train_distill
from train.model import CVPN, config_for_encoder
from envs.observation import default_encoder


def _train_arm(samples, encoder, *, d_model: int, epochs: int, seed: int, cold_start: bool):
    model = CVPN(config_for_encoder(encoder, belief=True, d_model=d_model))
    config = DistillConfig(
        epochs=epochs,
        seed=seed,
        use_policy_target=not cold_start,
        use_search_value=not cold_start,
    )
    metrics = train_distill(samples, model, config)
    return model, metrics


def _paired_gate(cand_ckpt: Path, base_ckpt: Path, *, deck: str, opponents, seeds) -> dict:
    from scripts.eval_protocol import ProtocolSpec, outcomes_from_rows, run_protocol
    from eval.scoreboard import (
        evaluate_gates,
        merge_gate,
        noninferiority_gate,
        paired_difference,
        summarize,
    )

    def _run(contestant: str, tag: str):
        spec = ProtocolSpec(
            contestant=contestant,
            contestant_deck=deck,
            opponents=tuple(opponents),
            seeds=tuple(seeds),
            sides=(0, 1),
            workers=1,
            base_seed=7000,
            tag=tag,
        )
        return run_protocol(spec, raw_path=None)

    cand_rows = _run(f"neural:{cand_ckpt}", "distill_smoke_cand")
    base_rows = _run(f"neural:{base_ckpt}", "distill_smoke_base")
    cand_outcomes = outcomes_from_rows(cand_rows)
    base_outcomes = outcomes_from_rows(base_rows)
    paired = paired_difference(cand_outcomes, base_outcomes)
    summary = summarize(cand_outcomes)
    gates = evaluate_gates(summary)
    return {
        "paired": paired,
        "noninferiority": noninferiority_gate(paired, tolerance=0.02) if paired else None,
        "merge_gate": merge_gate(paired, gates, tolerance=0.02),
        "candidate_summary": {
            "pooled_rate": summary["pooled_rate"],
            "pooled_ci_low": summary["pooled_ci_low"],
            "pooled_ci_high": summary["pooled_ci_high"],
            "games": summary["games"],
            "quality": summary["quality"],
        },
        "candidate_gates": gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="+", default=["natlan_battleship"])
    parser.add_argument("--seeds", type=int, default=2, help="collection seeds per opponent")
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=2)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--belief-samples", type=int, default=1)
    parser.add_argument("--bridge-workers", type=int, default=2)
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-opponents", nargs="+", default=["natlan_battleship"])
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--tag", default="distill_smoke")
    parser.add_argument("--replay-tag", default="distill_smoke")
    parser.add_argument("--reuse-replays", action="store_true", help="skip collection, reuse --replay-tag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = default_encoder()
    t0 = time.perf_counter()

    # 1. collect --------------------------------------------------------------
    replay_tag = args.replay_tag
    if args.reuse_replays:
        replay_dir = data_dir("replays", replay_tag)
        collect_summary = {"reused": True, "replay_dir": str(replay_dir)}
    else:
        search = SearchSpec(
            iterations=args.iterations,
            depth=args.depth,
            candidate_limit=args.candidate_limit,
            max_searches=args.max_searches,
            belief_samples=args.belief_samples,
            bridge_workers=args.bridge_workers,
        )
        spec = CollectSpec(
            deck=args.deck,
            opponents=tuple(args.opponents),
            seeds=tuple(range(args.seed_start, args.seed_start + args.seeds)),
            search=search,
        )
        collect_summary = collect(
            spec, tag=replay_tag, workers=args.collect_workers, progress=True
        )
        replay_dir = Path(collect_summary["replay_dir"])
    collect_seconds = time.perf_counter() - t0

    # 2. distill + cold-start BC baseline ------------------------------------
    samples = load_samples(replay_dir)
    if not samples:
        raise SystemExit(f"no samples collected in {replay_dir}")
    stats = sample_stats(samples)
    distill_model, distill_metrics = _train_arm(
        samples, encoder, d_model=args.d_model, epochs=args.epochs, seed=0, cold_start=False
    )
    bc_model, bc_metrics = _train_arm(
        samples, encoder, d_model=args.d_model, epochs=args.epochs, seed=0, cold_start=True
    )

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    ckpt_dir = ensure_dir(data_dir("checkpoints"))
    distill_ckpt = ckpt_dir / f"{stamp}_{args.tag}_distill.pt"
    bc_ckpt = ckpt_dir / f"{stamp}_{args.tag}_bc.pt"
    distill_model.save(str(distill_ckpt), encoder_config=encoder.to_dict())
    bc_model.save(str(bc_ckpt), encoder_config=encoder.to_dict())

    # 3. paired A/B non-inferiority gate -------------------------------------
    gate = _paired_gate(
        distill_ckpt,
        bc_ckpt,
        deck=args.deck,
        opponents=args.eval_opponents,
        seeds=tuple(args.seed_start + 1000 + i for i in range(args.eval_seeds)),
    )

    report = {
        "kind": "distill_smoke",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "deck": args.deck,
        "opponents": list(args.opponents),
        "search": {
            "iterations": args.iterations,
            "depth": args.depth,
            "candidate_limit": args.candidate_limit,
            "max_searches": args.max_searches,
            "belief_samples": args.belief_samples,
        },
        "collect": collect_summary,
        "sample_stats": stats,
        "distill_metrics": distill_metrics.as_dict(),
        "distill_history": distill_metrics.history,
        "baseline_metrics": bc_metrics.as_dict(),
        "baseline_history": bc_metrics.history,
        "checkpoints": {"distill": str(distill_ckpt), "bc": str(bc_ckpt)},
        "gate": gate,
        "timing": {"collect_seconds": collect_seconds},
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    out = ensure_dir(reports_dir("train")) / f"distill_smoke_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. print a compact curve + gate decision -------------------------------
    print("[distill-smoke] sample stats:", json.dumps(stats, ensure_ascii=False))
    print("[distill-smoke] distill epochs (train_loss/acc/value/belief):")
    for row in distill_metrics.history:
        print(
            f"  e{row['epoch']:>2} loss={row.get('train_loss')} "
            f"acc={row.get('train_policy_acc')} value={row.get('train_value_mse')} "
            f"belief={row.get('train_belief_loss')}"
        )
    paired = gate.get("paired") or {}
    ni = gate.get("noninferiority") or {}
    merge = gate.get("merge_gate") or {}
    print(
        f"[distill-smoke] paired distill-vs-BC delta={paired.get('delta')} "
        f"ci=[{paired.get('ci_low')},{paired.get('ci_high')}] n={paired.get('n_pairs')}"
    )
    print(
        f"[distill-smoke] non-inferior(tol=0.02)={ni.get('accept')} "
        f"merge_accept={merge.get('accept')} -> "
        f"{'PASS' if merge.get('accept') else 'FAIL'}"
    )
    print(f"[distill-smoke] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
