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
from reps.observation_encoder import TokenObservationEncoder, definition_feature_indices
from train.model import (
    CVPN,
    TransformerCVPN,
    config_for_encoder,
    load_state_dict_compatible,
    transformer_config_for_encoder,
    zero_init_identity_branch,
)
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
    p.add_argument("--id-buckets", type=int, default=0,
                   help="A2: learned card/entity identity embeddings (0 = historical "
                        "architecture, which is what every pre-A2 checkpoint expects)")
    p.add_argument("--id-emb-dim", type=int, default=32)
    p.add_argument("--id-zero-init", action="store_true",
                   help="A2 fine-tune: start the identity branch as a no-op so a pre-A2 "
                        "--init-from checkpoint begins bit-identical to its old outputs")
    p.add_argument("--belief", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=8000)
    p.add_argument("--loss", choices=["ce", "awr", "q", "pg"], default="ce",
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
    p.add_argument("--pg-entropy", type=float, default=0.01, help="entropy bonus for --loss pg")
    p.add_argument("--grad-clip", type=float, default=0.0, help="gradient norm clip (PG rounds use 1.0)")
    p.add_argument("--no-pg-advantage-norm", action="store_true",
                   help="disable per-batch advantage normalisation for --loss pg")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--save-best", action="store_true",
                   help="save the best-validation checkpoint in addition to the last epoch")
    p.add_argument("--select-metric", choices=["val_ce", "val_acc"], default="val_ce",
                   help="validation metric for --save-best (lower val_ce / higher val_acc)")
    p.add_argument("--best-out", default=None,
                   help="best checkpoint path (default: <out>.best.pt)")
    p.add_argument("--init-from", default=None,
                   help="initialize the weights from a checkpoint (continued training / PG rounds); "
                        "the architecture must match (same --arch / --d-model / head config)")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    # P0: seed *before* constructing the model. Historically the model was built
    # here and train() seeded torch only afterwards, so two runs with the same
    # --seed started from different random weights (A2/A0 comparisons were
    # contaminated by init variance).
    torch.manual_seed(int(args.seed))
    t0 = time.perf_counter()
    payload = torch.load(args.store_cache, map_location="cpu", weights_only=False)
    train_full, val_full, meta = payload["train"], payload["val"], payload["meta"]

    # P0 parity: prefer the encoder pinned in the store meta, and refuse an
    # identity-embedding run over a corpus whose definition ids were lost to
    # float16 (the A2 failure mode).
    encoder = default_encoder()
    if meta.get("encoder_config"):
        encoder = TokenObservationEncoder.from_dict(meta["encoder_config"])
    expected_indices = [int(v) for v in definition_feature_indices()]
    stored_indices = meta.get("definition_indices")
    if stored_indices is not None and [int(v) for v in stored_indices] != expected_indices:
        raise SystemExit(
            f"store definition columns {stored_indices} do not match the current encoder "
            f"{expected_indices}; rebuild the store or pin the old encoder layout"
        )
    if len(train_full) and train_full.tokens[0].shape[-1] != encoder.token_dim:
        raise SystemExit(
            f"store token_dim={train_full.tokens[0].shape[-1]} != encoder.token_dim="
            f"{encoder.token_dim}; wrong encoder/config or store"
        )
    if len(train_full) and train_full.options[0].shape[-1] != encoder.option_dim:
        raise SystemExit(
            f"store option_dim={train_full.options[0].shape[-1]} != encoder.option_dim="
            f"{encoder.option_dim}; wrong encoder/config or store"
        )
    n_events = int(meta.get("events", len(train_full) + len(val_full)))
    n_definition_ids = int(meta.get("definition_ids_present", 0))
    if args.id_buckets > 1 and n_definition_ids < n_events:
        raise SystemExit(
            f"--id-buckets={args.id_buckets} needs lossless definition ids, but only "
            f"{n_definition_ids}/{n_events} samples carry them (legacy fp16 corpus). "
            "Re-collect with --format compact (current collector writes integer ids) "
            "or use --compact-dtype float32; see reports/explore/literature_review.md §1."
        )
    if n_events and n_definition_ids < n_events:
        print(
            f"[warn] {n_events - n_definition_ids}/{n_events} samples lack exact definition "
            "ids (legacy fp16 corpus); identity features are unreliable for this run",
            flush=True,
        )
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
            id_buckets=args.id_buckets, id_emb_dim=args.id_emb_dim,
        )
        model = TransformerCVPN(cfg)
    else:
        cfg = config_for_encoder(encoder, belief=args.belief, d_model=args.d_model,
                                 value_activation=args.value_activation,
                                 id_buckets=args.id_buckets, id_emb_dim=args.id_emb_dim)
        model = CVPN(cfg)
    has_belief = bool(getattr(model, "has_belief", False))
    if args.id_zero_init and zero_init_identity_branch(model):
        print("[init] identity branch zero-initialised (no-op at step 0)", flush=True)
    if args.init_from:
        from train.model import load_cvpn_model

        source = load_cvpn_model(args.init_from)
        try:
            missing_ids = load_state_dict_compatible(model, source.state_dict())
        except RuntimeError as exc:
            raise SystemExit(
                f"--init-from architecture mismatch ({args.init_from}): {exc}"
            ) from exc
        note = f" ({len(missing_ids)} new identity tensors)" if missing_ids else ""
        if missing_ids and not args.id_zero_init:
            # Missing id tensors are randomly initialised by load_state_dict;
            # make the new branch a no-op instead of silently perturbing a
            # pre-A2 trunk (A2 post-mortem, P0).
            zero_init_identity_branch(model)
            note += " zero-initialised (no-op)"
        print(f"[init] weights <- {args.init_from}{note}", flush=True)
    out = Path(args.out)
    ensure_dir(out.parent)
    best_out = Path(args.best_out) if args.best_out else out.with_name(
        f"{out.stem}.best{out.suffix or '.pt'}"
    )
    config = Config(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, device=args.device, weight_decay=args.weight_decay,
        loss=args.loss, awr_beta=args.awr_beta, awr_max_weight=args.awr_max_weight,
        q_target=args.q_target, q_gamma=args.q_gamma,
        value_loss=args.value_loss, value_smooth=args.value_smooth,
        matchup_weight=args.matchup_weight,
        matchup_weight_strength=args.matchup_weight_strength,
        matchup_floor=args.matchup_floor,
        pg_entropy=args.pg_entropy,
        grad_clip=args.grad_clip,
        pg_advantage_norm=not args.no_pg_advantage_norm,
        save_best=args.save_best,
        select_metric=args.select_metric,
        best_path=str(best_out) if args.save_best else None,
    )
    history, train_seconds = train(
        train_store, val_store, model, config, has_belief,
        encoder_config=encoder.to_dict(),
    )
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
        "best_checkpoint": str(best_out) if args.save_best else None,
        "best_epoch": config.best_epoch,
        "best_metric": config.best_metric,
        "n_definition_ids": n_definition_ids,
        "n_events": n_events,
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
