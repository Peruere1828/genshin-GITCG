"""Small-scale model comparison: mean-pool CVPN vs token-transformer CVPN.

Exploration harness for "design a larger GITCG model and validate it locally"
(PLAN.md WS3, M3). Trains both architectures on the *same* collected pool and
fixed holdout (same seed), so val loss/accuracy are directly comparable, and
reports wall time, parameter count and (on CUDA) peak memory.

Usage::

    python -m scripts.bench_model --games 8 --d-model 128 --epochs 30 --device cuda
    python -m scripts.bench_model --replays data/replays/coldstart --device cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

from common.paths import ensure_dir
from envs.observation import default_encoder
from train.data import collect_samples, load_samples, sample_stats
from train.model import (
    CVPN,
    TransformerCVPN,
    config_for_encoder,
    transformer_config_for_encoder,
)
from train.train_loop import TrainConfig, evaluate_loss_accuracy, split_samples, train_bc


def _run_one(model, pool, config, holdout, device: str) -> dict:
    import torch

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    metrics = train_bc(pool, model, config)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    val_loss, val_acc = evaluate_loss_accuracy(model, holdout, config.value_coef)
    return {
        "wall_seconds": round(wall, 3),
        "train_loss": round(metrics.train_loss, 4),
        "train_accuracy": round(metrics.train_accuracy, 4),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "params": sum(int(p.numel()) for p in model.parameters()),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if device.startswith("cuda")
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CVPN vs transformer CVPN comparison.")
    p.add_argument("--deck", default="superconduct_aggro")
    p.add_argument("--opponents", nargs="*", default=["natlan_battleship", "dual_mualani_stacks"])
    p.add_argument("--games", type=int, default=8, help="teacher games per opponent")
    p.add_argument("--replays", nargs="*", default=None, help="replay dir(s) instead of collecting")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precollate", action="store_true", default=True)
    p.add_argument("--out", default="reports/explore")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    encoder = default_encoder()
    if args.replays:
        pool = [s for path in args.replays for s in load_samples(path)]
        data_source = "replays"
    else:
        pool = []
        for opponent in args.opponents:
            pool.extend(collect_samples(args.deck, opponent, tuple(range(args.games)), encoder=encoder))
        data_source = "collected"
    if not pool:
        raise SystemExit("empty sample pool")
    stats = sample_stats(pool)
    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, val_fraction=0.2,
        seed=0, device=args.device, precollate=args.precollate,
    )
    train_pool, holdout = split_samples(pool, val_fraction=config.val_fraction, seed=config.seed)

    report: dict = {
        "deck": args.deck,
        "opponents": list(args.opponents),
        "data_source": data_source,
        "pool_stats": stats,
        "d_model": args.d_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "precollate": args.precollate,
        "n_train": len(train_pool),
        "n_val": len(holdout),
        "results": [],
    }

    torch.manual_seed(0)
    toy = CVPN(config_for_encoder(encoder, d_model=args.d_model))
    toy_rec = _run_one(toy, train_pool, config, holdout, args.device)
    toy_rec["arch"] = "cvpn_meanpool"
    report["results"].append(toy_rec)
    print(f"[model] cvpn_meanpool      val_loss={toy_rec['val_loss']:.4f} "
          f"val_acc={toy_rec['val_accuracy']:.3f} {toy_rec['wall_seconds']:.2f}s "
          f"params={toy_rec['params']}")

    torch.manual_seed(0)
    tr = TransformerCVPN(
        transformer_config_for_encoder(
            encoder, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
        )
    )
    tr_rec = _run_one(tr, train_pool, config, holdout, args.device)
    tr_rec["arch"] = "cvpn_transformer"
    tr_rec["n_layers"] = args.n_layers
    tr_rec["n_heads"] = args.n_heads
    report["results"].append(tr_rec)
    print(f"[model] cvpn_transformer   val_loss={tr_rec['val_loss']:.4f} "
          f"val_acc={tr_rec['val_accuracy']:.3f} {tr_rec['wall_seconds']:.2f}s "
          f"params={tr_rec['params']} peak_mem={tr_rec['peak_mem_mb']}MB")

    if not args.no_write:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = ensure_dir(Path(args.out))
        path = out / f"model_compare_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[model] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
