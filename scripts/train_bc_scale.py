"""Memory-efficient large-scale behaviour-cloning from ``train.collect`` replay dirs.

The stock ``train.learning_curve`` / ``train.pipeline`` paths call
``train.data.load_samples`` which materialises every sample as an
``EncodedObservation`` (Python tuples of floats).  That is fine for the ~1e3-1e4
sample M3-CPU tier but blows past RAM at the 1e5-1e6 sample scale the "data-first"
reschedule targets.

This script streams the collector's per-game JSONL records, converts each
decision **straight to compact per-sample tensors** (tokens ``(T,65)``, options
``(K,68)``, belief labels), drops the Python objects, and keeps only the tensors
in RAM.  Batches are padded to the *batch* max at train time (originals stay
variable length, so no global ``K=256`` padding blow-up).  Memory is therefore
~16 KB/sample instead of ~200 KB/sample, which is what makes a few 1e5-sample run
fit on a 30 GB box.

Selection is deterministic and order-independent: games are split train/val by a
rule on ``(seed - seed_start)`` so the result does not depend on the
(completion-order) file layout.  Samples are then sorted by ``(opponent, seed,
step_index)`` before shuffling, so a rerun reproduces the same batches.

Pure-policy BC: hard ``teacher_index`` CE + episode-outcome value regression +
optional belief head, i.e. the cold-start arm of ``train.distill``.

Usage::

    python -m scripts.train_bc_scale --replays data/replays/datafirst_bc_sa \\
        --seed-start 1000000 --val-every 10 --max-train-samples 500000 \\
        --d-model 256 --epochs 30 --out data/checkpoints/big_d256.pt \\
        --report reports/train/big_d256_train.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir
from envs.observation import default_encoder
from train.model import CVPN, TransformerCVPN, config_for_encoder, transformer_config_for_encoder


@dataclass
class Store:
    """Compact per-sample tensors (variable length, no padding)."""

    tokens: list[torch.Tensor]      # each (T_i, 65) float32
    options: list[torch.Tensor]     # each (K_i, 68) float32
    teacher: list[int]
    value: list[float]
    # belief labels (god's-eye targets); empty list when capture_targets=False
    b_hand: list[torch.Tensor]      # (G,) float32
    b_deck: list[torch.Tensor]      # (G,)
    b_hist: list[torch.Tensor]      # (V,)
    b_hsize: list[float]
    b_burst: list[float]
    b_mask: list[bool]
    keys: list[tuple[str, int, int]]  # (opponent, seed, step_index) for sorting

    def __len__(self) -> int:
        return len(self.tokens)

    def subset(self, indices: Sequence[int]) -> "Store":
        take = lambda xs: [xs[i] for i in indices]
        return Store(
            tokens=take(self.tokens),
            options=take(self.options),
            teacher=take(self.teacher),
            value=take(self.value),
            b_hand=take(self.b_hand),
            b_deck=take(self.b_deck),
            b_hist=take(self.b_hist),
            b_hsize=take(self.b_hsize),
            b_burst=take(self.b_burst),
            b_mask=take(self.b_mask),
            keys=take(self.keys),
        )


def _is_val(seed: int, seed_start: int, val_every: int) -> bool:
    if val_every <= 0:
        return False
    return ((int(seed) - int(seed_start)) % int(val_every)) == 0


def load_store(
    replay_dirs: Sequence[str],
    *,
    seed_start: int,
    val_every: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    device: str = "cpu",
    progress_every: int = 500,
) -> tuple[Store, Store, dict[str, Any]]:
    """Stream replay dirs into compact (train, val) stores."""
    train = Store([], [], [], [], [], [], [], [], [], [], [])
    val = Store([], [], [], [], [], [], [], [], [], [], [])
    n_games = n_skipped = n_dup = 0
    seen: set[tuple[str, int]] = set()
    t0 = time.perf_counter()
    for replay_dir in replay_dirs:
        samples_path = Path(replay_dir)
        if samples_path.is_dir():
            samples_path = samples_path / "samples.jsonl"
        if not samples_path.exists():
            raise SystemExit(f"no samples.jsonl under {replay_dir}")
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn trailing line
                if record.get("error"):
                    n_skipped += 1
                    continue
                game_key = (str(record["opponent"]), int(record["seed"]))
                if game_key in seen:
                    n_dup += 1
                    continue
                seen.add(game_key)
                is_val = _is_val(int(record["seed"]), seed_start, val_every)
                dest = val if is_val else train
                cap = max_val_samples if is_val else max_train_samples
                if cap is not None and len(dest) >= cap:
                    continue
                opponent = str(record["opponent"])
                seed = int(record["seed"])
                for payload in record.get("samples", ()):
                    if cap is not None and len(dest) >= cap:
                        break
                    obs = payload["observation"]
                    dest.tokens.append(
                        torch.tensor(obs["token_features"], dtype=torch.float32, device=device)
                    )
                    dest.options.append(
                        torch.tensor(obs["option_features"], dtype=torch.float32, device=device)
                    )
                    dest.teacher.append(int(payload["teacher_index"]))
                    dest.value.append(float(payload["value_target"]))
                    bt = obs.get("belief_target") or {}
                    dest.b_hand.append(
                        torch.tensor(bt.get("group_hand_presence", ()), dtype=torch.float32, device=device)
                    )
                    dest.b_deck.append(
                        torch.tensor(bt.get("group_remaining_deck_presence", ()), dtype=torch.float32, device=device)
                    )
                    dest.b_hist.append(
                        torch.tensor(bt.get("hand_histogram", ()), dtype=torch.float32, device=device)
                    )
                    dest.b_hsize.append(float(bt.get("hand_size", 0.0)))
                    dest.b_burst.append(float(bt.get("burst_ready_count", 0.0)))
                    sig = (
                        float(bt.get("hand_size", 0.0)) > 0.0
                        or any(float(v) > 0.0 for v in bt.get("group_hand_presence", ()))
                        or any(float(v) > 0.0 for v in bt.get("group_remaining_deck_presence", ()))
                        or any(float(v) > 0.0 for v in bt.get("hand_histogram", ()))
                    )
                    dest.b_mask.append(bool(sig))
                    dest.keys.append((opponent, seed, int(payload.get("step_index", 0))))
                n_games += 1
                if n_games % progress_every == 0:
                    print(
                        f"[store] games={n_games} train={len(train)} val={len(val)} "
                        f"({time.perf_counter() - t0:.0f}s)",
                        flush=True,
                    )
    # deterministic ordering independent of completion order
    for store in (train, val):
        order = sorted(range(len(store)), key=lambda i: store.keys[i])
        _reorder(store, order)
    meta = {
        "replay_dirs": list(replay_dirs),
        "games": n_games,
        "skipped_games": n_skipped,
        "duplicate_games": n_dup,
        "train_samples": len(train),
        "val_samples": len(val),
        "load_seconds": round(time.perf_counter() - t0, 1),
        "seed_start": seed_start,
        "val_every": val_every,
    }
    return train, val, meta


def _reorder(store: Store, order: Sequence[int]) -> None:
    for field in (
        "tokens", "options", "teacher", "value",
        "b_hand", "b_deck", "b_hist", "b_hsize", "b_burst", "b_mask", "keys",
    ):
        seq = getattr(store, field)
        setattr(store, field, [seq[i] for i in order])


def _collate(store: Store, index: Sequence[int], device: str):
    """Pad a batch to its own max and stack (vectorised via ``pad_sequence``)."""
    from torch.nn.utils.rnn import pad_sequence

    toks = [store.tokens[i] for i in index]
    opts = [store.options[i] for i in index]
    b = len(index)
    t_lens = torch.tensor([t.shape[0] for t in toks])
    k_lens = torch.tensor([o.shape[0] for o in opts])
    tokens = pad_sequence(toks, batch_first=True).to(device)
    options = pad_sequence(opts, batch_first=True).to(device)
    token_mask = (
        torch.arange(tokens.shape[1]).unsqueeze(0) < t_lens.unsqueeze(1)
    ).to(device)
    option_mask = (
        torch.arange(options.shape[1]).unsqueeze(0) < k_lens.unsqueeze(1)
    ).to(device)
    teacher = torch.tensor([store.teacher[i] for i in index], dtype=torch.long, device=device)
    values = torch.tensor([store.value[i] for i in index], dtype=torch.float32, device=device)
    return tokens, token_mask, options, option_mask, teacher, values


def _belief_batch(store: Store, index: Sequence[int], device: str):
    if not store.b_hand[index[0]].numel() and not store.b_hist[index[0]].numel():
        return None
    return {
        "b_hand": torch.stack([store.b_hand[i] for i in index]).to(device),
        "b_deck": torch.stack([store.b_deck[i] for i in index]).to(device),
        "b_hist": torch.stack([store.b_hist[i] for i in index]).to(device),
        "b_hsize": torch.tensor([store.b_hsize[i] for i in index], dtype=torch.float32, device=device),
        "b_burst": torch.tensor([store.b_burst[i] for i in index], dtype=torch.float32, device=device),
        "b_mask": torch.tensor([store.b_mask[i] for i in index], dtype=torch.bool, device=device),
    }


def _belief_loss(belief, bt) -> torch.Tensor:
    if belief is None or bt is None:
        return torch.zeros((), device=bt["b_hsize"].device if bt else "cpu")
    from train.belief_head import BeliefTargetBatch, belief_loss

    targets = BeliefTargetBatch(
        group_hand_presence=bt["b_hand"],
        group_deck_presence=bt["b_deck"],
        hand_histogram=bt["b_hist"],
        hand_size=bt["b_hsize"],
        burst_ready=bt["b_burst"],
        mask=bt["b_mask"],
    )
    return belief_loss(belief, targets)


def run_batch(model, store, index, device, value_coef, belief_coef, has_belief):
    tokens, token_mask, options, option_mask, teacher, values = _collate(store, index, device)
    if has_belief:
        logits, value, belief = model.forward_all(tokens, token_mask, options, option_mask)
    else:
        logits, value = model(tokens, token_mask, options, option_mask)
        belief = None
    ce = F.cross_entropy(logits, teacher)
    mse = F.mse_loss(value, values)
    bl = _belief_loss(belief, _belief_batch(store, index, device)) if has_belief else torch.zeros((), device=device)
    loss = ce + value_coef * mse + belief_coef * bl
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == teacher).float().mean()
    return loss, {"ce": float(ce.item()), "value": float(mse.item()), "belief": float(bl.item()), "acc": float(acc.item())}


@torch.no_grad()
def evaluate(model, store, config, device, has_belief):
    model.eval()
    if len(store) == 0:
        return {"ce": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0, "n": 0}
    n = len(store)
    agg = {"ce": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0}
    for start in range(0, n, config.batch_size):
        index = list(range(start, min(start + config.batch_size, n)))
        _, info = run_batch(model, store, index, device, config.value_coef, config.belief_coef, has_belief)
        for k in agg:
            agg[k] += info[k] * len(index)
    for k in agg:
        agg[k] /= n
    agg["n"] = n
    model.train()
    return agg


@dataclass
class Config:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    value_coef: float = 0.5
    belief_coef: float = 0.5
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1


def train(train_store: Store, val_store: Store, model, config: Config, has_belief: bool):
    torch.manual_seed(config.seed)
    model.to(config.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n = len(train_store)
    order0 = list(range(n))
    import random as _random

    rng = _random.Random(config.seed)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        order = list(order0)
        rng.shuffle(order)
        agg = {"loss": 0.0, "ce": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0}
        nb = 0
        for start in range(0, n, config.batch_size):
            index = order[start : start + config.batch_size]
            if not index:
                continue
            opt.zero_grad(set_to_none=True)
            loss, info = run_batch(model, train_store, index, config.device, config.value_coef, config.belief_coef, has_belief)
            loss.backward()
            opt.step()
            agg["loss"] += float(loss.item())
            for k in ("ce", "value", "belief", "acc"):
                agg[k] += info[k]
            nb += 1
        for k in agg:
            agg[k] /= max(1, nb)
        row = {"epoch": epoch, **{k: round(v, 5) for k, v in agg.items()}}
        if val_store is not None and len(val_store):
            v = evaluate(model, val_store, config, config.device, has_belief)
            row["val_ce"] = round(v["ce"], 5)
            row["val_acc"] = round(v["acc"], 4)
            row["val_value"] = round(v["value"], 5)
            row["val_belief"] = round(v["belief"], 5)
        history.append(row)
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(f"[train] {json.dumps(row)}", flush=True)
    return history, round(time.perf_counter() - t0, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="*", default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--val-every", type=int, default=10, help="every Nth game seed -> val (0=no val)")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None, help="cap the val store used for scoring")
    p.add_argument("--train-limit", type=int, default=None, help="cap train store after load (nested subsets)")
    p.add_argument("--arch", choices=["cvpn", "transformer"], default="cvpn")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--belief", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="checkpoint .pt path")
    p.add_argument("--report", default=None, help="aggregate report .json path")
    p.add_argument(
        "--store-cache",
        default=None,
        help="cache the compact tensor store as a .pt (reused across runs; skips JSON re-parse)",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="build/save the store cache and exit (no training)",
    )
    return p


def load_or_build_store(args):
    if args.store_cache:
        import os

        if os.path.exists(args.store_cache):
            print(f"[store] loading cache {args.store_cache}", flush=True)
            payload = torch.load(args.store_cache, map_location="cpu", weights_only=False)
            return payload["train"], payload["val"], payload["meta"], True
    if not args.replays:
        raise SystemExit("provide --replays or an existing --store-cache")
    train_store, val_store, meta = load_store(
        args.replays,
        seed_start=args.seed_start,
        val_every=args.val_every,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        device=args.device,
    )
    if args.store_cache:
        ensure_dir(Path(args.store_cache).parent)
        torch.save({"train": train_store, "val": val_store, "meta": meta}, args.store_cache)
        print(f"[store] wrote cache {args.store_cache}", flush=True)
    return train_store, val_store, meta, False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = default_encoder()
    train_store, val_store, meta, _cached = load_or_build_store(args)
    if args.build_only:
        print(f"[store] built train={len(train_store)} val={len(val_store)} (build-only)", flush=True)
        return 0
    if args.train_limit is not None and args.train_limit < len(train_store):
        train_store = train_store.subset(list(range(args.train_limit)))
    if args.max_val is not None and len(val_store) > args.max_val:
        val_store = val_store.subset(list(range(args.max_val)))
    print(f"[store] train={len(train_store)} val={len(val_store)} meta={meta}", flush=True)
    if args.arch == "transformer":
        cfg = transformer_config_for_encoder(
            encoder, belief=args.belief, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        model = TransformerCVPN(cfg)
    else:
        cfg = config_for_encoder(encoder, belief=args.belief, d_model=args.d_model)
        model = CVPN(cfg)
    has_belief = bool(getattr(model, "has_belief", False))
    config = Config(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, device=args.device, weight_decay=args.weight_decay,
    )
    history, train_seconds = train(train_store, val_store, model, config, has_belief)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.out:
        out = Path(args.out)
        ensure_dir(out.parent)
    else:
        out = Path("data/checkpoints") / f"{stamp}_bc_scale.pt"
        ensure_dir(out.parent)
    model.save(str(out), encoder_config=encoder.to_dict())
    report = {
        "kind": "bc_scale",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "arch": args.arch,
        "d_model": args.d_model,
        "belief_head": has_belief,
        "config": asdict(config),
        "store": meta,
        "n_train": len(train_store),
        "n_val": len(val_store),
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
    print(f"[done] ckpt={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
