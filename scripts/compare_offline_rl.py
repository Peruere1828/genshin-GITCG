"""Compare the offline-RL grid arms on the same val subset (H2 diagnostics).

For each checkpoint it reports, on the *same* val subset the trainer used:

* agreement with the teacher overall, split by episode outcome (won / lost) — the
  point of AWR/Q is to keep the winning-play mass and drop the losing-play mass,
  so the gap between the two columns is the mechanism's fingerprint;
* options per decision where the arm disagrees with the teacher (raw policy change);
* pairwise action agreement between arms (how different the policies actually are);
* greedy value (Q) ranking sanity: mean Q of the chosen action.

Usage::

    python -m scripts.compare_offline_rl --store data/store/sa_all.pt \
        --val-limit 4000 --ckpt bc_e15=... --ckpt awr_b1_e15=... --ckpt q_disc_e15=...
"""

from __future__ import annotations

import argparse
import itertools
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
def _argmax(model, store, indices, device="cpu"):
    model.to(device)
    model.eval()
    picks: list[int] = []
    values: list[float] = []
    for start in range(0, len(indices), 128):
        batch = indices[start : start + 128]
        tokens, token_mask, options, option_mask, _, _ = _collate(store, batch, device)
        logits, value = model(tokens, token_mask, options, option_mask)
        picks.extend(int(x) for x in logits.argmax(dim=-1).tolist())
        values.extend(float(x) for x in value.tolist())
    return picks, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/store/sa_all.pt")
    parser.add_argument("--val-limit", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", action="append", required=True,
                        help="name=path (repeatable)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    store = _val_subset(args.store, args.val_limit, args.seed)
    indices = list(range(len(store)))
    teacher = [int(t) for t in store.teacher]
    won = [i for i in indices if store.value[i] == 1.0]
    lost = [i for i in indices if store.value[i] == 0.0]
    has_values = bool(won and lost)

    report: dict = {"store": args.store, "n_val": len(store), "n_won": len(won), "n_lost": len(lost),
                    "arms": {}}
    picks_by_arm: dict[str, list[int]] = {}
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        model = load_cvpn_model(path)
        picks, values = _argmax(model, store, indices)
        picks_by_arm[name] = picks
        agree = [p == teacher[i] for i, p in enumerate(picks)]
        row = {
            "checkpoint": path,
            "teacher_agreement": round(sum(agree) / len(agree), 4),
            "mean_value": round(sum(values) / len(values), 4),
        }
        if has_values:
            row["agreement_won"] = round(sum(agree[i] for i in won) / len(won), 4)
            row["agreement_lost"] = round(sum(agree[i] for i in lost) / len(lost), 4)
            row["win_minus_lost"] = round(row["agreement_won"] - row["agreement_lost"], 4)
        report["arms"][name] = row
        print(f"[cmp] {name}: {json.dumps(row)}", flush=True)

    report["pairwise_action_agreement"] = {}
    for a, b in itertools.combinations(picks_by_arm, 2):
        same = sum(1 for x, y in zip(picks_by_arm[a], picks_by_arm[b]) if x == y) / len(store)
        report["pairwise_action_agreement"][f"{a}~{b}"] = round(same, 4)
        print(f"[cmp] {a}~{b}: action agreement {same:.4f}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cmp] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
