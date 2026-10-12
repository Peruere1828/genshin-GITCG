"""Data-size x model-size learning curve over a compact BC tensor store.

Reuses :mod:`scripts.train_bc_scale` (streaming compact store + variable-length
batch collation) so it can sweep subset sizes far beyond what the in-memory
``train.learning_curve`` path can hold.  A single fixed validation store scores
every grid point (like ``train.learning_curve``), and each point's checkpoint is
saved so its *strength* (pool win rate) can be measured separately with
``scripts.eval_protocol``.

Nested subsets are drawn from one seeded permutation of the train store, so the
``size`` axis is genuinely nested (a bigger point is a superset of the smaller).

Usage::

    python -m scripts.bc_learning_curve --store-cache data/store/sa500.pt \\
        --sizes 2000 8000 30000 100000 --d-models 128 --epochs 12 \\
        --tag sa500 --report reports/train/bc_curve_sa500.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
from pathlib import Path

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir
from envs.observation import default_encoder
from train.model import CVPN, TransformerCVPN, config_for_encoder, transformer_config_for_encoder
from scripts.train_bc_scale import Config, Store, evaluate, load_or_build_store, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="*", default=None)
    p.add_argument("--store-cache", default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--max-val-samples", type=int, default=8000)
    p.add_argument("--max-val", type=int, default=8000, help="cap the val store used for scoring")
    p.add_argument("--sizes", type=int, nargs="+", default=[2000, 8000, 30000, 100000])
    p.add_argument("--d-models", type=int, nargs="+", default=[128])
    p.add_argument("--arch", choices=["cvpn", "transformer"], default="cvpn")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--belief", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt-dir", default="data/checkpoints")
    p.add_argument("--tag", default="curve")
    p.add_argument("--report", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.store_cache and not args.replays:
        raise SystemExit("provide --store-cache or --replays")
    encoder = default_encoder()
    class _A:  # shim so load_or_build_store can be reused
        pass

    a = _A()
    for name in ("replays", "store_cache", "seed_start", "val_every",
                 "max_train_samples", "max_val_samples", "device"):
        setattr(a, name, getattr(args, name, None))
    a.max_train_samples = None
    train_store, val_store, meta, cached = load_or_build_store(a)
    if args.max_val and len(val_store) > args.max_val:
        val_store = val_store.subset(list(range(args.max_val)))
        print(f"[curve] capped val to {len(val_store)}", flush=True)

    sizes = sorted({int(s) for s in args.sizes if 0 < int(s) <= len(train_store)})
    n = len(train_store)
    perm = list(range(n))
    random.Random(args.seed).shuffle(perm)

    points = []
    ckpt_dir = ensure_dir(Path(args.ckpt_dir))
    for d_model in args.d_models:
        for size in sizes:
            index = perm[:size]
            sub = train_store.subset(index)
            if args.arch == "transformer":
                cfg = transformer_config_for_encoder(
                    encoder, belief=args.belief, d_model=d_model,
                    n_layers=args.n_layers, n_heads=args.n_heads,
                )
                model = TransformerCVPN(cfg)
            else:
                cfg = config_for_encoder(encoder, belief=args.belief, d_model=d_model)
                model = CVPN(cfg)
            has_belief = bool(getattr(model, "has_belief", False))
            config = Config(
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                seed=args.seed, device=args.device,
            )
            history, secs = train(sub, val_store, model, config, has_belief)
            val = evaluate(model, val_store, config, args.device, has_belief)
            ckpt = ckpt_dir / f"curve_{args.tag}_d{d_model}_n{size}.pt"
            model.save(str(ckpt), encoder_config=encoder.to_dict())
            point = {
                "d_model": d_model, "arch": args.arch, "n_train": len(sub),
                "n_val": len(val_store), "train_seconds": secs,
                "val_ce": round(val["ce"], 5), "val_acc": round(val["acc"], 4),
                "val_value": round(val["value"], 5), "val_belief": round(val["belief"], 5),
                "final_train_ce": history[-1]["ce"] if history else None,
                "final_train_acc": history[-1]["acc"] if history else None,
                "checkpoint": str(ckpt),
            }
            points.append(point)
            print(f"[curve] {json.dumps(point)}", flush=True)

    report = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag, "arch": args.arch, "belief_head": bool(args.belief),
        "epochs": args.epochs, "store": meta, "sizes": sizes,
        "d_models": [int(d) for d in args.d_models], "points": points,
        "engine_lock": load_engine_lock(), "runtime": runtime_metadata(),
    }
    if args.report:
        rp = Path(args.report)
        ensure_dir(rp.parent)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[curve] wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
