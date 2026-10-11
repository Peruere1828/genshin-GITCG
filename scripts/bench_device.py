"""Device benchmark for CVPN training (PLAN.md D14: WSL MX550 small-net experiment).

Trains the *same* toy CVPN on the same fixed sample pool on each requested torch
device (``cpu`` and, if present, ``cuda``) and reports wall-clock time plus the
train/val metrics, so the "small network experiment decides the training node"
question (PLAN.md §5.2) is answerable with data. The fixed pool and identical
``TrainConfig`` make the devices directly comparable.

Usage::

    python -m scripts.bench_device --games 6 --d-models 64 128 --epochs 12
    python -m scripts.bench_device --replays data/replays/<tag> --devices cpu cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir, reports_dir
from envs.observation import default_encoder
from train.data import collect_samples, load_samples, sample_stats
from train.model import CVPN, config_for_encoder
from train.train_loop import TrainConfig, train_bc


def _available_devices(requested: list[str]) -> list[str]:
    import torch

    out: list[str] = []
    for dev in requested:
        if dev == "cuda" and not torch.cuda.is_available():
            print("[device] cuda requested but unavailable; skipping")
            continue
        out.append(dev)
    return out


def run_bench(
    *,
    deck: str = "superconduct_aggro",
    opponents: tuple[str, ...] = ("natlan_battleship",),
    games: int = 6,
    d_models: tuple[int, ...] = (64, 128),
    epochs: int = 12,
    batch_size: int = 32,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    replays_dirs: tuple[str, ...] = (),
) -> dict:
    import torch

    encoder = default_encoder()
    if replays_dirs:
        pool = [s for path in replays_dirs for s in load_samples(path)]
        data_source = "replays"
    else:
        pool = []
        for opponent in opponents:
            pool.extend(
                collect_samples(deck, opponent, tuple(range(games)), encoder=encoder)
            )
        data_source = "collected"
    stats = sample_stats(pool)
    if not pool:
        raise SystemExit("empty sample pool")

    devices = _available_devices(list(devices))
    results: list[dict] = []
    for d_model in d_models:
        for dev in devices:
            model = CVPN(config_for_encoder(encoder, d_model=int(d_model)))
            config = TrainConfig(
                epochs=epochs, batch_size=batch_size, val_fraction=0.2,
                seed=0, device=dev,
            )
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            metrics = train_bc(pool, model, config)
            if dev == "cuda":
                torch.cuda.synchronize()
            wall = time.time() - t0
            rec = {
                "device": dev, "d_model": int(d_model), "epochs": epochs,
                "batch_size": batch_size, "n_samples": len(pool),
                "wall_seconds": round(wall, 3),
                **{k: v for k, v in metrics.as_dict().items() if k != "epochs"},
            }
            if dev == "cuda":
                rec["cuda_device_name"] = torch.cuda.get_device_name(0)
            results.append(rec)
            print(
                f"[device] {dev:5s} d_model={d_model:4d}: {wall:7.2f}s "
                f"val_loss={metrics.val_loss:.4f} val_acc={metrics.val_accuracy:.3f}"
            )

    # speedup vs cpu within each d_model
    by_key: dict[tuple[int, str], dict] = {(r["d_model"], r["device"]): r for r in results}
    for d_model in d_models:
        cpu = by_key.get((int(d_model), "cpu"))
        cuda = by_key.get((int(d_model), "cuda"))
        if cpu and cuda and cuda["wall_seconds"] > 0:
            cuda["speedup_vs_cpu"] = round(cpu["wall_seconds"] / cuda["wall_seconds"], 3)

    report = {
        "deck": deck,
        "opponents": list(opponents),
        "data_source": data_source,
        "pool_stats": stats,
        "devices": devices,
        "d_models": [int(d) for d in d_models],
        "epochs": epochs,
        "results": results,
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ensure_dir(reports_dir("train")) / f"device_bench_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[device] wrote {out}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CVPN device benchmark (cpu vs cuda).")
    p.add_argument("--deck", default="superconduct_aggro")
    p.add_argument("--opponents", nargs="*", default=["natlan_battleship"])
    p.add_argument("--games", type=int, default=6, help="teacher games per opponent")
    p.add_argument("--d-models", type=int, nargs="*", default=[64, 128])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    p.add_argument("--replays", nargs="*", default=None,
                   help="replay dir(s) from train.collect (skip online collection)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_bench(
        deck=args.deck,
        opponents=tuple(args.opponents),
        games=args.games,
        d_models=tuple(args.d_models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=tuple(args.devices),
        replays_dirs=tuple(args.replays or ()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
