"""One toy SoG round: collect -> train -> evaluate -> gate -> persist.

This is the L4 acceptance artifact (PLAN.md §4.1): a full round must complete with
zero errors and the network must demonstrably learn (overfit a small dataset).
Checkpoints go to ``data/checkpoints/`` (gitignored); the small report goes to
``reports/train/`` (committed).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from common.seeding import derive_seed
from agents.neural import NeuralPolicy
from envs.decks import deck_spec
from envs.match import run_match
from envs.observation import default_encoder
from envs.policy import expert_policy
from eval.stats import estimate_rate, outcome_score
from train.data import collect_samples, load_samples, sample_stats
from train.model import CVPN, config_for_encoder
from train.train_loop import TrainConfig, train_bc


def evaluate_neural(
    model: CVPN,
    encoder,
    *,
    deck: str,
    opponents: Sequence[str],
    seeds: Sequence[int],
) -> dict:
    scores: list[float] = []
    errors = 0
    for opponent in opponents:
        for seed in seeds:
            policy = NeuralPolicy(
                model=model,
                encoder=encoder,
                name="neural",
                seed=derive_seed(seed, "p0", deck),
            )
            record = run_match(
                deck_spec(deck),
                deck_spec(opponent),
                policy,
                expert_policy(opponent, seed=derive_seed(seed, "p1", opponent)),
                seed=int(seed),
            )
            errors += 1 if record.error else 0
            scores.append(outcome_score(record.winner, 0))
    est = estimate_rate(sum(scores), len(scores))
    return {**est.as_dict(), "errors": errors}


def evaluate_baseline(
    *, deck: str, opponents: Sequence[str], seeds: Sequence[int]
) -> dict:
    scores: list[float] = []
    for opponent in opponents:
        for seed in seeds:
            record = run_match(
                deck_spec(deck),
                deck_spec(opponent),
                expert_policy(deck, seed=derive_seed(seed, "p0", deck)),
                expert_policy(opponent, seed=derive_seed(seed, "p1", opponent)),
                seed=int(seed),
            )
            scores.append(outcome_score(record.winner, 0))
    return estimate_rate(sum(scores), len(scores)).as_dict()


def run_round(
    *,
    deck: str = "superconduct_aggro",
    train_opponents: Sequence[str] = ("natlan_battleship", "dual_mualani_stacks"),
    eval_opponents: Sequence[str] = ("natlan_battleship", "superconduct_mika_attack"),
    train_seeds: Sequence[int] = tuple(range(6)),
    eval_seeds: Sequence[int] = (100, 101, 102),
    train_config: TrainConfig | None = None,
    tolerance: float = 0.05,
    tag: str = "toy",
    replays_dirs: Sequence[str] | None = None,
    samples: Sequence | None = None,
) -> dict:
    encoder = default_encoder()
    replays_dirs = list(replays_dirs) if replays_dirs else []
    if samples is not None:
        data_source = "samples"
    elif replays_dirs:
        # Cluster path: CPU nodes collected replay dirs; train from them.
        collected = [sample for path in replays_dirs for sample in load_samples(path)]
        samples = collected
        data_source = "replays"
    else:
        collected = []
        for opponent in train_opponents:
            collected.extend(
                collect_samples(deck, opponent, train_seeds, encoder=encoder)
            )
        samples = collected
        data_source = "collected"
    train_samples = list(samples)
    stats = sample_stats(train_samples)

    model = CVPN(config_for_encoder(encoder))
    config = train_config or TrainConfig()
    metrics = train_bc(train_samples, model, config)

    candidate = evaluate_neural(
        model, encoder, deck=deck, opponents=eval_opponents, seeds=eval_seeds
    )
    baseline = evaluate_baseline(deck=deck, opponents=eval_opponents, seeds=eval_seeds)
    gate_passed = candidate["rate"] >= baseline["rate"] - tolerance

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}_{tag}"
    ckpt_dir = ensure_dir(data_dir("checkpoints"))
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    model.save(str(ckpt_path), encoder_config=encoder.to_dict())
    (ckpt_dir / "latest.json").write_text(
        json.dumps({"checkpoint": str(ckpt_path), "deck": deck, "run_id": run_id}),
        encoding="utf-8",
    )

    report = {
        "run_id": run_id,
        "deck": deck,
        "train_opponents": list(train_opponents),
        "eval_opponents": list(eval_opponents),
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "replays_dirs": list(replays_dirs),
        "data_source": data_source,
        "sample_stats": stats,
        "train_config": asdict(config),
        "train_metrics": metrics.as_dict(),
        "candidate": candidate,
        "baseline": baseline,
        "gate": {"passed": gate_passed, "tolerance": tolerance},
        "checkpoint": str(ckpt_path),
        "encoder_config": encoder.to_dict(),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    out = ensure_dir(reports_dir("train")) / f"round_{run_id}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_distill_round(
    *,
    deck: str = "superconduct_aggro",
    train_opponents: Sequence[str] = ("natlan_battleship", "dual_mualani_stacks"),
    eval_opponents: Sequence[str] = ("natlan_battleship", "superconduct_mika_attack"),
    train_seeds: Sequence[int] = tuple(range(6)),
    eval_seeds: Sequence[int] = (100, 101, 102),
    search: Any = None,
    bridge: Any = None,
    d_model: int = 64,
    distill_config: Any = None,
    tolerance: float = 0.02,
    tag: str = "distill",
    replays_dirs: Sequence[str] | None = None,
    samples: Sequence | None = None,
) -> dict:
    """One search-distillation round: collect → distill (policy+value+belief) → eval → persist.

    Data comes from ``samples`` (in-memory, engine-free), ``replays_dirs`` (a
    search-teacher replay from ``train.collect --search``) or an inline MCCFR
    collection (needs ``search`` + ``bridge``; a bridge is created if absent).
    The model is a :class:`~train.model.CVPN` with the S-C belief head, trained by
    :func:`train.distill.train_distill`. The gate mirrors the toy round (candidate
    not worse than the scripted-expert baseline by ``tolerance``); the stricter
    paired non-inferiority merge gate lives in ``scripts/run_distill_smoke.py`` /
    ``scripts.eval_protocol``.
    """
    from train.distill import DistillConfig, train_distill

    encoder = default_encoder()
    replays_dirs = list(replays_dirs) if replays_dirs else []
    owned_bridge = None
    if samples is not None:
        collected = list(samples)
        data_source = "samples"
    elif replays_dirs:
        collected = [sample for path in replays_dirs for sample in load_samples(path)]
        data_source = "replays"
    else:
        from envs.fork_bridge import ForkBridge
        from train.data import collect_search_samples

        if search is None:
            raise SystemExit("run_distill_round needs `search` when collecting inline")
        owned_bridge = bridge is None
        if bridge is None:
            bridge = ForkBridge(workers=max(1, int(getattr(search, "bridge_workers", 2))))
        try:
            collected = []
            for opponent in train_opponents:
                collected.extend(
                    collect_search_samples(
                        deck,
                        opponent,
                        train_seeds,
                        encoder=encoder,
                        bridge=bridge,
                        iterations=getattr(search, "iterations", 8),
                        depth=getattr(search, "depth", 1),
                        candidate_limit=getattr(search, "candidate_limit", None),
                        value_spec=getattr(search, "value_spec", ""),
                        search_every=getattr(search, "search_every", 1),
                        max_searches=getattr(search, "max_searches", None),
                        belief_root=getattr(search, "belief_root", True),
                        belief_samples=getattr(search, "belief_samples", 2),
                        request_types=getattr(search, "request_types", None),
                        capture_targets=getattr(search, "capture_targets", True),
                    )
                )
        finally:
            if owned_bridge:
                bridge.close()
        data_source = "search"

    train_samples = list(collected)
    stats = sample_stats(train_samples)
    model = CVPN(config_for_encoder(encoder, belief=True, d_model=d_model))
    config = distill_config or DistillConfig()
    metrics = train_distill(train_samples, model, config)

    candidate = evaluate_neural(
        model, encoder, deck=deck, opponents=eval_opponents, seeds=eval_seeds
    )
    baseline = evaluate_baseline(deck=deck, opponents=eval_opponents, seeds=eval_seeds)
    gate_passed = candidate["rate"] >= baseline["rate"] - tolerance

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}_{tag}"
    ckpt_dir = ensure_dir(data_dir("checkpoints"))
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    model.save(str(ckpt_path), encoder_config=encoder.to_dict())
    (ckpt_dir / "latest_distill.json").write_text(
        json.dumps({"checkpoint": str(ckpt_path), "deck": deck, "run_id": run_id}),
        encoding="utf-8",
    )

    report = {
        "run_id": run_id,
        "kind": "distill",
        "deck": deck,
        "train_opponents": list(train_opponents),
        "eval_opponents": list(eval_opponents),
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "replays_dirs": list(replays_dirs),
        "data_source": data_source,
        "d_model": int(d_model),
        "belief_head": {
            "group_dim": int(encoder.belief_group_dim),
            "card_dim": int(encoder.belief_histogram_dim),
        },
        "sample_stats": stats,
        "distill_config": asdict(config),
        "train_metrics": metrics.as_dict(),
        "history": metrics.history,
        "candidate": candidate,
        "baseline": baseline,
        "gate": {"passed": gate_passed, "tolerance": tolerance},
        "checkpoint": str(ckpt_path),
        "encoder_config": encoder.to_dict(),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    out = ensure_dir(reports_dir("train")) / f"distill_{run_id}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy SoG round: collect/train/eval/gate.")
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tag", default="toy")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for training (cpu/cuda); D14 WSL MX550 experiments",
    )
    parser.add_argument("--overfit", action="store_true", help="tiny overfit sanity mode")
    parser.add_argument(
        "--precollate",
        action="store_true",
        help="fast path: pad/collate the pool once, slice on device (M3 ops)",
    )
    parser.add_argument(
        "--replays",
        nargs="+",
        default=None,
        help="replay dir(s) from train.collect (train from disk instead of collecting)",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="M3-CPU S-C: search-distillation round (policy CE + value + belief head)",
    )
    parser.add_argument("--d-model", type=int, default=64, help="distill network width")
    parser.add_argument("--iterations", type=int, default=8, help="MCCFR iterations per search")
    parser.add_argument("--depth", type=int, default=1, help="MCCFR depth budget")
    parser.add_argument("--candidate-limit", type=int, default=None, help="root candidate cap")
    parser.add_argument("--max-searches", type=int, default=None, help="max searched decisions/game")
    parser.add_argument("--belief-samples", type=int, default=2, help="belief-root determinizations")
    parser.add_argument("--bridge-workers", type=int, default=2, help="fork-bridge subprocesses")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.overfit:
        report = run_round(
            train_seeds=(0,),
            eval_seeds=(100,),
            train_config=TrainConfig(epochs=60, batch_size=16, lr=3e-3, val_fraction=0.0, device=args.device,
                                     precollate=args.precollate),
            tag="overfit",
        )
    elif args.distill:
        from train.collect import SearchSpec
        from train.distill import DistillConfig

        report = run_distill_round(
            deck=args.deck,
            train_seeds=tuple(range(1000, 1000 + args.train_seeds)),
            eval_seeds=tuple(range(2000, 2000 + args.eval_seeds)),
            search=SearchSpec(
                iterations=args.iterations,
                depth=args.depth,
                candidate_limit=args.candidate_limit,
                max_searches=args.max_searches,
                belief_samples=args.belief_samples,
                bridge_workers=args.bridge_workers,
            ),
            d_model=args.d_model,
            distill_config=DistillConfig(epochs=args.epochs, device=args.device),
            tag=args.tag or "distill",
            replays_dirs=args.replays,
        )
    else:
        report = run_round(
            train_seeds=tuple(range(args.train_seeds)),
            eval_seeds=tuple(range(100, 100 + args.eval_seeds)),
            train_config=TrainConfig(epochs=args.epochs, device=args.device, precollate=args.precollate),
            tag=args.tag,
            replays_dirs=args.replays,
        )
    print(json.dumps({k: v for k, v in report.items() if k != "encoder_config"}, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
