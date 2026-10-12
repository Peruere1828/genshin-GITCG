"""Compact-capable streaming BC store builder (data-first scale, PLAN D23/A6).

A5's ``scripts.train_bc_scale`` streams the collector's per-game JSONL records but
assumes the *legacy* dense-list payload (it reads
``obs["token_features"]`` as a list of lists). The A6 compact format stores those
same dense arrays as base64 fp16 buffers, so a data-first corpus written with
``train.collect --format compact`` cannot go through that loader.

This module builds the **same** compact per-sample ``Store`` (imported read-only
from ``scripts.train_bc_scale``) while decoding *both* payload formats directly as
numpy buffers (``train.data.is_dense_payload`` / ``_unpack_dense``), so the result
is byte-for-byte compatible with A5's trainer / learning-curve paths:

    python -m scripts.datafirst_bc_store --replays data/replays/dfbc_a_c80 \\
        --store-cache data/store/dfbc_a.pt --val-every 10
    python -m scripts.train_bc_scale --store-cache data/store/dfbc_a.pt \\
        --d-model 256 --epochs 25 --out data/checkpoints/a_d256.pt

Selection is deterministic and order-independent: games are de-duplicated by
``(opponent, seed)`` (so overlapping replay dirs — e.g. the reused SA probes — do
not double-count), split train/val by a rule on ``(seed - seed_start)`` and sorted
by ``(opponent, seed, step_index)``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir
from train.data import _unpack_dense, is_dense_payload
from scripts.train_bc_scale import Store


def _tensor(value, dtype=torch.float32) -> torch.Tensor:
    """Decode one dense field (compact dict or legacy list) to a tensor."""
    if is_dense_payload(value):
        arr = _unpack_dense(value)
        if arr.size == 0:
            return torch.empty((0,), dtype=dtype)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return torch.empty((0,), dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _is_val(seed: int, seed_start: int, val_every: int) -> bool:
    if val_every <= 0:
        return False
    return ((int(seed) - int(seed_start)) % int(val_every)) == 0


def _reorder(store: Store, order: Sequence[int]) -> None:
    for field in (
        "tokens", "options", "teacher", "value",
        "b_hand", "b_deck", "b_hist", "b_hsize", "b_burst", "b_mask", "keys",
    ):
        seq = getattr(store, field)
        setattr(store, field, [seq[i] for i in order])


def load_store_any(
    replay_dirs: Sequence[str],
    *,
    seed_start: int,
    val_every: int,
    device: str = "cpu",
    progress_every: int = 500,
) -> tuple[Store, Store, dict]:
    """Stream replay dirs (jsonl and/or compact) into compact train/val stores."""
    empty = lambda: Store([], [], [], [], [], [], [], [], [], [], [])
    train, val = empty(), empty()
    n_games = n_skipped = n_dup = n_events = 0
    seen: set[tuple[str, int]] = set()
    t0 = time.perf_counter()
    for replay_dir in replay_dirs:
        path = Path(replay_dir)
        if path.is_dir():
            path = path / "samples.jsonl"
        if not path.exists():
            raise SystemExit(f"no samples.jsonl under {replay_dir}")
        with path.open("r", encoding="utf-8") as handle:
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
                key = (str(record["opponent"]), int(record["seed"]))
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                is_val = _is_val(int(record["seed"]), seed_start, val_every)
                dest = val if is_val else train
                opponent, seed = key
                for payload in record.get("samples", ()):
                    obs = payload["observation"]
                    dest.tokens.append(_tensor(obs["token_features"]).to(device))
                    dest.options.append(_tensor(obs["option_features"]).to(device))
                    dest.teacher.append(int(payload["teacher_index"]))
                    dest.value.append(float(payload["value_target"]))
                    bt = obs.get("belief_target") or {}
                    hand = _tensor(bt.get("group_hand_presence", ()))
                    deck = _tensor(bt.get("group_remaining_deck_presence", ()))
                    hist = _tensor(bt.get("hand_histogram", ()))
                    hsize = float(bt.get("hand_size", 0.0))
                    burst = float(bt.get("burst_ready_count", 0.0))
                    dest.b_hand.append(hand.to(device))
                    dest.b_deck.append(deck.to(device))
                    dest.b_hist.append(hist.to(device))
                    dest.b_hsize.append(hsize)
                    dest.b_burst.append(burst)
                    # Label is "present" when any god's-eye marginal is non-zero.
                    sig = (
                        hsize > 0.0
                        or bool((hand > 0.0).any())
                        or bool((deck > 0.0).any())
                        or bool((hist > 0.0).any())
                    )
                    dest.b_mask.append(bool(sig))
                    dest.keys.append((opponent, seed, int(payload.get("step_index", 0))))
                    n_events += 1
                n_games += 1
                if progress_every and n_games % progress_every == 0:
                    print(
                        f"[store] games={n_games} train={len(train)} val={len(val)} "
                        f"({time.perf_counter() - t0:.0f}s)",
                        flush=True,
                    )
    for store in (train, val):
        order = sorted(range(len(store)), key=lambda i: store.keys[i])
        _reorder(store, order)
    meta = {
        "replay_dirs": list(replay_dirs),
        "games": n_games,
        "skipped_games": n_skipped,
        "duplicate_games": n_dup,
        "events": n_events,
        "train_samples": len(train),
        "val_samples": len(val),
        "load_seconds": round(time.perf_counter() - t0, 1),
        "seed_start": int(seed_start),
        "val_every": int(val_every),
    }
    return train, val, meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--store-cache", required=True)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--val-every", type=int, default=10, help="every Nth seed -> val (0=no val)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_store, val_store, meta = load_store_any(
        args.replays,
        seed_start=args.seed_start,
        val_every=args.val_every,
        device=args.device,
    )
    cache = Path(args.store_cache)
    ensure_dir(cache.parent)
    torch.save({"train": train_store, "val": val_store, "meta": meta}, cache)
    print(
        f"[store] train={len(train_store)} val={len(val_store)} games={meta['games']} "
        f"dup={meta['duplicate_games']} -> {cache} ({meta['load_seconds']}s)",
        flush=True,
    )
    if args.report:
        rp = Path(args.report)
        ensure_dir(rp.parent)
        rp.write_text(
            json.dumps(
                {
                    "kind": "datafirst_bc_store",
                    "generated": _dt.datetime.now().isoformat(timespec="seconds"),
                    "store_cache": str(cache),
                    **meta,
                    "engine_lock": load_engine_lock(),
                    "runtime": runtime_metadata(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
