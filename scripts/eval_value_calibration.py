"""Is the CVPN value head usable as a search frontier? (H3, D23 precondition)

D23 says search-as-teacher may only be reopened once there is a *calibrated public
belief value head*. This measures the value head we already have, on the frozen val
split of a compact store, with proper scoring rules:

* **Brier** = mean((p - s)^2) with ``s`` the episode score (0 loss / 0.5 draw / 1 win);
* **ECE** = expected calibration error over 15 equal-count-ish bins;
* **AUC** = how well the value *ranks* wins above losses (0.5 = no discrimination —
  the failure D23 observed as "根价值区分度 ≈ 0");
* **bias** = mean(p) - mean(s), plus a reliability table (bin -> mean p vs mean s).

A value head that cannot rank wins above losses makes any search that leans on it
(prior pruning, frontier evaluation) worthless, regardless of budget.

Usage::

    python -m scripts.eval_value_calibration --store data/store/sa_all.pt --val-limit 8000 \
        --ckpt bc_e15=data/checkpoints/orl_bc_e15.pt --ckpt kg_bc_e50=data/checkpoints/kaggle/sa_bc_d256_e50.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.train_bc_scale import _collate
from train.model import load_cvpn_model


def _val_subset(store_path: str, limit: int, seed: int):
    payload = torch.load(store_path, map_location="cpu", weights_only=False)
    val = payload["val"]
    if limit and limit < len(val):
        perm = list(range(len(val)))
        random.Random(seed).shuffle(perm)
        val = val.subset(perm[:limit])
    return val


@torch.no_grad()
def _predictions(model, store, batch: int = 256) -> tuple[list[float], list[float]]:
    model.eval()
    preds: list[float] = []
    targets: list[float] = []
    for start in range(0, len(store), batch):
        index = list(range(start, min(start + batch, len(store))))
        tokens, token_mask, options, option_mask, _teacher, values = _collate(store, index, "cpu")
        _, value = model(tokens, token_mask, options, option_mask)
        preds.extend(float(x) for x in value.tolist())
        targets.extend(float(x) for x in values.tolist())
    return preds, targets


def _auc(preds: list[float], targets: list[float]) -> float:
    """Rank wins above losses (ties count a half); returns 0.5 when undecided."""
    wins = [p for p, s in zip(preds, targets) if s > 0.5]
    losses = [p for p, s in zip(preds, targets) if s < 0.5]
    if not wins or not losses:
        return float("nan")
    pairs = 0.0
    for w in wins:
        for l in losses:
            pairs += 1.0 if w > l else 0.5 if w == l else 0.0
    return pairs / (len(wins) * len(losses))


def _reliability(preds: list[float], targets: list[float], bins: int = 15) -> list[dict]:
    order = sorted(zip(preds, targets))
    out: list[dict] = []
    size = max(1, len(order) // bins)
    for i in range(0, len(order), size):
        chunk = order[i : i + size]
        if not chunk:
            continue
        p = sum(x for x, _ in chunk) / len(chunk)
        s = sum(y for _, y in chunk) / len(chunk)
        out.append({"n": len(chunk), "mean_p": round(p, 4), "mean_score": round(s, 4),
                    "gap": round(p - s, 4)})
    return out


def calibrate(preds: list[float], targets: list[float], bins: int = 15) -> dict:
    n = len(preds)
    brier = sum((p - s) ** 2 for p, s in zip(preds, targets)) / n
    rel = _reliability(preds, targets, bins)
    ece = sum(r["n"] / n * abs(r["gap"]) for r in rel)
    base = sum(targets) / n
    brier_base = sum((base - s) ** 2 for s in targets) / n
    return {
        "n": n,
        "mean_pred": round(sum(preds) / n, 4),
        "mean_score": round(base, 4),
        "bias": round(sum(preds) / n - base, 4),
        "brier": round(brier, 4),
        "brier_baseline": round(brier_base, 4),
        "brier_skill": round(1 - brier / brier_base, 4) if brier_base else None,
        "ece": round(ece, 4),
        "auc": round(_auc(preds, targets), 4),
        "reliability": rel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/store/sa_all.pt")
    parser.add_argument("--val-limit", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", action="append", required=True, help="name=path (repeatable)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    store = _val_subset(args.store, args.val_limit, args.seed)
    scores = [float(v) for v in store.value]
    print(f"[calib] n={len(store)} mean_score={sum(scores) / len(scores):.4f} "
          f"draw_frac={sum(1 for s in scores if s == 0.5) / len(scores):.4f}", flush=True)
    report: dict = {"store": args.store, "n_val": len(store), "arms": {}}
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        model = load_cvpn_model(path)
        preds, targets = _predictions(model, store)
        row = calibrate(preds, targets)
        report["arms"][name] = row
        print(f"[calib] {name:14s} pred={row['mean_pred']:.3f} score={row['mean_score']:.3f} "
              f"bias={row['bias']:+.3f} brier={row['brier']:.4f} (skill {row['brier_skill']}) "
              f"ece={row['ece']:.4f} auc={row['auc']:.3f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[calib] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
