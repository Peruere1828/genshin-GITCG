"""Data-first BC trainer over a prebuilt compact tensor store (PLAN D23).

A thin wrapper around the compact ``Store`` / ``train`` / ``evaluate`` primitives
from ``scripts.train_bc_scale`` (imported read-only) that adds the knobs the
data-first scale runs actually need:

* **bounded, reproducible sub-sampling** — ``--train-limit`` draws a seeded random
  subset so a size axis is comparable and a 3e5-sample SA store can train in
  minutes rather than hours;
* **bounded validation** — the stock loop re-scores the *full* holdout every epoch
  (~3.7e4 samples on the reused SA store), which dominates wall time; here the
  holdout is itself capped (``--val-limit``) so the curve/accuracy is still
  measured but training is not val-bound.

Policy targets (H2 of the 2-day strength plan): ``--loss {ce,awr,q}`` — plain BC,
advantage-weighted BC, or DouZero-style action value (``--awr-beta`` / ``--q-target``).

Checkpoints are byte-compatible with the ``neural:<ckpt>`` policy spec (saved via
``CVPN.save`` with the encoder config) so they plug straight into
``scripts.eval_protocol``.

Usage::

    python -m scripts.datafirst_bc_train --store-cache data/store/sa_all.pt \\
        --d-model 256 --epochs 15 --train-limit 80000 --val-limit 8000 \\
        --out data/checkpoints/dfbc_sa_d256.pt \\
        --report reports/train/dfbc_sa_d256.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir
from envs.observation import default_encoder
from train.model import CVPN, TransformerCVPN, config_for_encoder, transformer_config_for_encoder
from scripts.train_bc_scale import Config, train


def _subsample(store, limit: int | None, seed: int):
    if limit is None or limit >= len(store):
        return store, len(store)
    perm = list(range(len(store)))
    random.Random(seed).shuffle(perm)
    return store.subset(perm[:limit]), limit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-cache", required=True)
    p.add_argument("--arch", choices=["cvpn", "transformer"], default="cvpn")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--belief", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=8000)
    p.add_argument("--loss", choices=["ce", "awr", "q"], default="ce",
                   help="policy target: ce = BC (default), awr = advantage-weighted CE, q = DouZero-style action value")
    p.add_argument("--awr-beta", type=float, default=1.0, help="advantage temperature for --loss awr")
    p.add_argument("--awr-max-weight", type=float, default=10.0, help="clip for the AWR weight")
    p.add_argument("--q-target", choices=["episode", "discount"], default="episode",
                   help="MC return for --loss q: raw episode outcome, or discounted by remaining steps")
    p.add_argument("--q-gamma", type=float, default=0.97, help="discount per remaining step for --q-target discount")
    p.add_argument("--value-loss", choices=["mse", "brier", "bce"], default="mse",
                   help="episode-outcome loss for the value head (H3 calibration)")
    p.add_argument("--value-activation", choices=["identity", "sigmoid"], default="identity",
                   help="value head activation stored in the checkpoint (sigmoid = probability head)")
    p.add_argument("--value-smooth", type=float, default=0.0, help="label smoothing for the value target")
    p.add_argument("--matchup-weight", choices=["none", "balance", "deficit"], default="none",
                   help="H5 floor attack: reweight samples by (deck, opponent) group "
                        "(balance = inverse frequency, deficit = boost matchups below --matchup-floor)")
    p.add_argument("--matchup-weight-strength", type=float, default=1.0)
    p.add_argument("--matchup-floor", type=float, default=0.45)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    encoder = default_encoder()
    t0 = time.perf_counter()
    payload = torch.load(args.store_cache, map_location="cpu", weights_only=False)
    train_full, val_full, meta = payload["train"], payload["val"], payload["meta"]
    train_store, n_train = _subsample(train_full, args.train_limit, args.seed)
    val_store, n_val = _subsample(val_full, args.val_limit, args.seed + 1)
    print(
        f"[load] cache {args.store_cache} in {time.perf_counter() - t0:.0f}s; "
        f"train={n_train} (of {len(train_full)}) val={n_val} (of {len(val_full)})",
        flush=True,
    )
    if args.arch == "transformer":
        cfg = transformer_config_for_encoder(
            encoder, belief=args.belief, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
            value_activation=args.value_activation,
        )
        model = TransformerCVPN(cfg)
    else:
        cfg = config_for_encoder(encoder, belief=args.belief, d_model=args.d_model,
                                 value_activation=args.value_activation)
        model = CVPN(cfg)
    has_belief = bool(getattr(model, "has_belief", False))
    config = Config(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, device=args.device, weight_decay=args.weight_decay,
        loss=args.loss, awr_beta=args.awr_beta, awr_max_weight=args.awr_max_weight,
        q_target=args.q_target, q_gamma=args.q_gamma,
        value_loss=args.value_loss, value_smooth=args.value_smooth,
        matchup_weight=args.matchup_weight,
        matchup_weight_strength=args.matchup_weight_strength,
        matchup_floor=args.matchup_floor,
    )
    history, train_seconds = train(train_store, val_store, model, config, has_belief)
    out = Path(args.out)
    ensure_dir(out.parent)
    model.save(str(out), encoder_config=encoder.to_dict())
    report = {
        "kind": "datafirst_bc",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "store_cache": args.store_cache,
        "arch": args.arch,
        "d_model": args.d_model,
        "belief_head": has_belief,
        "config": asdict(config),
        "store_meta": meta,
        "n_train_full": len(train_full),
        "n_val_full": len(val_full),
        "n_train": n_train,
        "n_val": n_val,
        "train_limit": args.train_limit,
        "train_seconds": train_seconds,
        "history": history,
        "checkpoint": str(out),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    if args.report:
        rp = Path(args.report)
        ensure_dir(rp.parent)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ckpt={out} train_seconds={train_seconds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
