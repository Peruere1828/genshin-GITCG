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
from typing import Sequence

from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir, reports_dir
from common.seeding import derive_seed
from agents.neural import NeuralPolicy
from envs.decks import deck_spec
from envs.match import run_match
from envs.observation import default_encoder
from envs.policy import expert_policy
from eval.stats import estimate_rate, outcome_score
from train.data import collect_samples, sample_stats
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
) -> dict:
    encoder = default_encoder()
    train_samples = []
    for opponent in train_opponents:
        train_samples.extend(
            collect_samples(deck, opponent, train_seeds, encoder=encoder)
        )
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
    model.save(str(ckpt_path))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy SoG round: collect/train/eval/gate.")
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tag", default="toy")
    parser.add_argument("--overfit", action="store_true", help="tiny overfit sanity mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.overfit:
        report = run_round(
            train_seeds=(0,),
            eval_seeds=(100,),
            train_config=TrainConfig(epochs=60, batch_size=16, lr=3e-3, val_fraction=0.0),
            tag="overfit",
        )
    else:
        report = run_round(
            train_seeds=tuple(range(args.train_seeds)),
            eval_seeds=tuple(range(100, 100 + args.eval_seeds)),
            train_config=TrainConfig(epochs=args.epochs),
            tag=args.tag,
        )
    print(json.dumps({k: v for k, v in report.items() if k != "encoder_config"}, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
