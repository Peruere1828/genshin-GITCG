"""CVPN learning-curve sweep (PLAN.md §4.2 L5.3).

Scales the L4 toy validation to thousands of samples and answers two questions:

1. does the CVPN keep improving with more data (train/val loss + accuracy curve)?
2. how do model size (``d_model``) and data size interact?

It collects one sample pool with the scripted teachers, then trains a *fresh*
network per (d_model, training-size) point on a deterministic nested subset, and
writes an aggregate curve to ``reports/train/`` (committed). The validation split
is **reserved once** from the pool before subsetting, so every grid point is scored
on the same fixed holdout (otherwise cross-point deltas are confounded by the
holdout changing size/content). Checkpoints are not persisted (the curve is the
artifact).

Usage::

    python -m train.learning_curve --games 12 --sizes 128 256 512 1024
    python -m train.learning_curve --games 40 --d-models 64 128 256
    python -m train.learning_curve --smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir, reports_dir
from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.observation import default_encoder
from envs.policy import expert_policy
from eval.stats import estimate_rate, outcome_score
from train.data import Sample, collect_samples, sample_stats
from train.model import CVPN, config_for_encoder
from train.train_loop import TrainConfig, evaluate_loss_accuracy, train_bc


@dataclass
class CurvePoint:
    d_model: int
    n_samples: int
    n_train: int
    n_val: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_loss": round(self.train_loss, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "val_loss": round(self.val_loss, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "history": self.history,
        }


def collect_pool(
    deck: str,
    opponents: Sequence[str],
    seeds: Sequence[int],
    *,
    encoder,
) -> list[Sample]:
    """Collect a labelled sample pool across several teacher opponents."""
    pool: list[Sample] = []
    for opponent in opponents:
        pool.extend(collect_samples(deck, opponent, seeds, encoder=encoder))
    return pool


def curve_sizes(pool_size: int, requested: Sequence[int]) -> list[int]:
    """Filter requested sizes to ``1 <= size <= pool_size`` and de-duplicate."""
    sizes = sorted({int(s) for s in requested if 0 < int(s) <= pool_size})
    return sizes


def _subset(pool: Sequence[Sample], size: int, *, seed: int) -> list[Sample]:
    data = list(pool)
    if size >= len(data):
        return data
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:size]


def _fixed_split(
    pool: Sequence[Sample], *, val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Reserve one holdout from ``pool`` before any subsetting.

    Returns ``(train_pool, val_pool)``. The same ``val_pool`` scores every grid
    point, so points are directly comparable; subsets are drawn only from
    ``train_pool`` and are therefore never evaluated on their own training data.
    """
    data = list(pool)
    rng = random.Random(seed)
    rng.shuffle(data)
    if val_fraction <= 0 or len(data) <= 4:
        return data, []
    n_val = max(1, int(len(data) * val_fraction))
    return data[n_val:], data[:n_val]


def run_learning_curve(
    *,
    deck: str = "superconduct_aggro",
    train_opponents: Sequence[str] = ("natlan_battleship", "dual_mualani_stacks"),
    eval_opponents: Sequence[str] = (),
    train_seeds: Sequence[int] = tuple(range(12)),
    eval_seeds: Sequence[int] = (),
    sizes: Sequence[int] = (128, 256, 512, 1024),
    d_models: Sequence[int] = (128,),
    train_config: TrainConfig | None = None,
    encoder=None,
    verbose: bool = False,
) -> dict:
    encoder = encoder or default_encoder()
    base_config = train_config or TrainConfig(epochs=12, batch_size=32, val_fraction=0.2)

    pool = collect_pool(deck, train_opponents, train_seeds, encoder=encoder)
    stats = sample_stats(pool)
    train_pool, val_pool = _fixed_split(
        pool, val_fraction=base_config.val_fraction, seed=base_config.seed
    )
    sizes = curve_sizes(len(train_pool), sizes)
    if not sizes:
        raise SystemExit(
            f"sample pool too small ({len(pool)} samples, "
            f"{len(train_pool)} trainable); increase --games/--train-opponents"
        )
    # Fixed holdout: train on everything in the subset, score on val_pool below.
    train_config = replace(base_config, val_fraction=0.0)

    points: list[CurvePoint] = []
    for d_model in d_models:
        for size in sizes:
            samples = _subset(train_pool, size, seed=base_config.seed)
            model = CVPN(config_for_encoder(encoder, d_model=int(d_model)))
            metrics = train_bc(samples, model, train_config)
            val_loss, val_acc = (
                evaluate_loss_accuracy(model, val_pool, base_config.value_coef)
                if val_pool
                else (0.0, 0.0)
            )
            point = CurvePoint(
                d_model=int(d_model),
                n_samples=len(samples),
                n_train=len(samples),
                n_val=len(val_pool),
                train_loss=metrics.train_loss,
                train_accuracy=metrics.train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_acc,
                history=metrics.history,
            )
            points.append(point)
            if verbose:
                print(
                    f"[curve] d_model={d_model} n={len(samples):>5} "
                    f"train_acc={point.train_accuracy:.3f} "
                    f"val_acc={point.val_accuracy:.3f} "
                    f"val_loss={point.val_loss:.4f}"
                )

    report: dict = {
        "deck": deck,
        "train_opponents": list(train_opponents),
        "train_seeds": list(train_seeds),
        "pool_stats": stats,
        "n_train_pool": len(train_pool),
        "n_val_holdout": len(val_pool),
        "train_config": asdict(base_config),
        "sizes": sizes,
        "d_models": [int(d) for d in d_models],
        "points": [p.as_dict() for p in points],
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }

    if eval_opponents and eval_seeds:
        report["eval"] = _evaluate_best(
            train_pool, encoder, train_config, sizes, d_models,
            deck, eval_opponents, eval_seeds,
        )
    return report


def _evaluate_best(
    train_pool, encoder, config, sizes, d_models, deck, eval_opponents, eval_seeds
) -> dict:
    """Train one model at the largest (d_model, size) and report pool win rate.

    Optional: only runs when ``--eval-seeds``/``--eval-opponents`` are given.
    """
    d_model = max(int(d) for d in d_models)
    size = max(sizes)
    model = CVPN(config_for_encoder(encoder, d_model=d_model))
    train_bc(_subset(train_pool, size, seed=config.seed), model, config)
    from agents.neural import NeuralPolicy

    scores: list[float] = []
    errors = 0
    for opponent in eval_opponents:
        for seed in eval_seeds:
            policy = NeuralPolicy(
                model=model, encoder=encoder, name="neural",
                seed=derive_seed(seed, "p0", deck),
            )
            record = run_match(
                deck_spec(deck), deck_spec(opponent), policy,
                expert_policy(opponent, seed=derive_seed(seed, "p1", opponent)),
                seed=int(seed),
            )
            errors += 1 if record.error else 0
            scores.append(outcome_score(record.winner, 0))
    est = estimate_rate(sum(scores), len(scores))
    return {"d_model": d_model, "n_samples": size, **est.as_dict(), "errors": errors}


def write_artifacts(report: dict, *, stamp: str | None = None) -> dict:
    stamp = stamp or _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ensure_dir(reports_dir("train"))
    json_path = out_dir / f"learning_curve_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / f"learning_curve_{stamp}.csv"
    fields = [
        "d_model", "n_samples", "n_train", "n_val",
        "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in report["points"]:
            writer.writerow({k: point[k] for k in fields})
    return {"json": json_path, "csv": csv_path}


def _print_report(report: dict) -> None:
    print(f"[curve] deck={report['deck']} pool={report['pool_stats'].get('n')} samples "
          f"sizes={report['sizes']} d_models={report['d_models']}")
    print(f"  {'d_model':>7} {'n':>6} {'train_acc':>10} {'val_acc':>8} {'val_loss':>9}")
    for point in report["points"]:
        print(
            f"  {point['d_model']:>7} {point['n_samples']:>6} "
            f"{point['train_accuracy']:>10.3f} {point['val_accuracy']:>8.3f} "
            f"{point['val_loss']:>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVPN learning-curve sweep.")
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--train-opponents", nargs="*",
                        default=["natlan_battleship", "dual_mualani_stacks"])
    parser.add_argument("--games", type=int, default=12,
                        help="teacher games per opponent (sample pool source)")
    parser.add_argument("--sizes", type=int, nargs="*",
                        default=[128, 256, 512, 1024])
    parser.add_argument("--d-models", type=int, nargs="*", default=[128])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-opponents", nargs="*", default=None)
    parser.add_argument("--eval-seeds", type=int, default=0)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny fast sweep")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_opponents and args.eval_seeds <= 0:
        raise SystemExit("--eval-opponents requires --eval-seeds > 0")
    if args.smoke:
        report = run_learning_curve(
            deck=args.deck,
            train_opponents=["natlan_battleship"],
            train_seeds=(0,),
            sizes=[16, 32, 48],
            d_models=[32],
            train_config=TrainConfig(epochs=3, batch_size=16, val_fraction=0.25),
            verbose=True,
        )
    else:
        report = run_learning_curve(
            deck=args.deck,
            train_opponents=args.train_opponents,
            train_seeds=tuple(range(args.games)),
            sizes=args.sizes,
            d_models=args.d_models,
            train_config=TrainConfig(epochs=args.epochs),
            eval_opponents=args.eval_opponents or (),
            eval_seeds=tuple(range(args.eval_seeds)),
            verbose=True,
        )
    _print_report(report)
    if not args.no_write:
        paths = write_artifacts(report)
        print(f"[curve] wrote {paths['json']}")
        print(f"[curve] wrote {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
