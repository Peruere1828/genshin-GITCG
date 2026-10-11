"""MX550 / generic-device operator + CVPN throughput benchmark (PLAN.md §5.2).

Exploration harness for "optimize matmul etc. to speed up training/inference".
Measures, on the current torch device:

1. raw matmul throughput across dtype (fp32/fp16/bf16) and size;
2. memory bandwidth (device copy);
3. CVPN training throughput (samples/s) for a *baseline* per-batch pipeline
   (collate + host->device every step) vs a *precollated* pipeline (pad once,
   slice on device) and, optionally, ``torch.compile``;
4. single-sample and batched inference latency (kernel-launch bound), with and
   without ``torch.compile``.

Findings on the WSL MX550 (cc 7.5, no tensor cores): fp16/bf16 matmul is
*slower* than fp32, so AMP is a loss here; keep fp32. The tiny CVPN is
launch/transfer bound, not FLOP bound.

Usage::

    python -m scripts.bench_ops --device cuda --out reports/explore
    python -m scripts.bench_ops --device cpu --no-compile
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from common.paths import ensure_dir
from envs.observation import default_encoder
from train.model import CVPN, collate, config_for_encoder


@dataclass
class Obs:
    token_features: list
    option_features: list


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(fn, *, reps: int = 10, warmup: int = 3, device: str = "cpu") -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / reps


def bench_matmul(device: str, sizes=(256, 512, 1024, 2048)) -> list[dict]:
    torch.set_grad_enabled(False)
    rows: list[dict] = []
    for n in sizes:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        dt = _timed(lambda: a @ b, reps=20, device=device)
        rows.append({"n": n, "dtype": "fp32", "ms": dt * 1e3, "tflops": 2 * n**3 / dt / 1e12})
        for dtype in (torch.float16, torch.bfloat16):
            ah, bh = a.to(dtype), b.to(dtype)
            try:
                dt = _timed(lambda: ah @ bh, reps=20, device=device)
                rows.append(
                    {"n": n, "dtype": str(dtype).split(".")[-1], "ms": dt * 1e3,
                     "tflops": 2 * n**3 / dt / 1e12}
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"n": n, "dtype": str(dtype), "error": str(exc)})
    return rows


def bench_bandwidth(device: str) -> float:
    torch.set_grad_enabled(False)
    x = torch.empty(64 * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    dt = _timed(lambda: y.copy_(x), reps=30, device=device)
    return 2 * x.numel() * 4 / dt / 1e9


def _make_pool(encoder, n: int, *, tokens: int, options: int, seed: int = 0) -> list[Obs]:
    g = torch.Generator().manual_seed(seed)
    pool: list[Obs] = []
    for _ in range(n):
        t = 1 + int(torch.randint(1, tokens + 1, (1,), generator=g).item())
        k = 1 + int(torch.randint(1, options + 1, (1,), generator=g).item())
        pool.append(
            Obs(
                token_features=torch.randn(t, encoder.token_dim, generator=g).tolist(),
                option_features=torch.randn(k, encoder.option_dim, generator=g).tolist(),
            )
        )
    return pool


def _precollate(pool, device: str):
    tokens, token_mask, options, option_mask = collate([o for o in pool])
    return (
        tokens.to(device, non_blocking=True),
        token_mask.to(device, non_blocking=True),
        options.to(device, non_blocking=True),
        option_mask.to(device, non_blocking=True),
    )


def bench_training(device: str, encoder, *, pool_size: int, batch_size: int, compile_model: bool):
    from train.train_loop import TrainConfig

    torch.set_grad_enabled(True)
    torch.manual_seed(0)
    config = TrainConfig(val_fraction=0.0)
    model = CVPN(config_for_encoder(encoder, d_model=128)).to(device)
    if compile_model:
        model = torch.compile(model)

    # baseline: per-batch collate + host->device
    pool = _make_pool(encoder, pool_size, tokens=48, options=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    def baseline_epoch():
        correct = 0
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size]
            tokens, token_mask, options, option_mask = collate(
                [o for o in batch]
            )
            step_model_step(
                model, optimizer, config, device,
                tokens.to(device), token_mask.to(device), options.to(device), option_mask.to(device),
            )
            correct += len(batch)
        return correct

    epoch_dt_baseline = _timed(baseline_epoch, reps=3, warmup=1, device=device)
    baseline_sps = len(pool) / epoch_dt_baseline

    # precollated: pad once, slice on device
    tensors = _precollate(pool, device)
    batch_t = tuple(t.view(-1, batch_size, *t.shape[1:]) for t in tensors)

    def pre_epoch():
        for i in range(tensors[0].size(0) // batch_size):
            step_model_step(model, optimizer, config, device,
                            batch_t[0][i], batch_t[1][i], batch_t[2][i], batch_t[3][i])

    epoch_dt_pre = _timed(pre_epoch, reps=3, warmup=1, device=device)
    pre_sps = (tensors[0].size(0) // batch_size) * batch_size / epoch_dt_pre
    return {
        "pool_size": pool_size,
        "batch_size": batch_size,
        "compile": compile_model,
        "baseline_samples_per_s": baseline_sps,
        "baseline_epoch_s": epoch_dt_baseline,
        "precollated_samples_per_s": pre_sps,
        "precollated_epoch_s": epoch_dt_pre,
        "speedup": pre_sps / max(1e-9, baseline_sps),
    }


def step_model_step(model, optimizer, config, device, tokens, token_mask, options, option_mask):
    import torch.nn.functional as F

    logits, value = model(tokens, token_mask, options, option_mask)
    teacher = torch.zeros(tokens.size(0), dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, teacher) + config.value_coef * F.mse_loss(
        value, torch.zeros_like(value)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def bench_inference(device: str, encoder, *, compile_model: bool, batch_sizes=(1, 8, 64)):
    torch.manual_seed(0)
    model = CVPN(config_for_encoder(encoder, d_model=128)).to(device).eval()
    if compile_model:
        model = torch.compile(model)
    rows = []
    with torch.no_grad():
        for b in batch_sizes:
            tokens = torch.randn(b, 48, encoder.token_dim, device=device)
            token_mask = torch.ones(b, 48, dtype=torch.bool, device=device)
            options = torch.randn(b, 24, encoder.option_dim, device=device)
            option_mask = torch.ones(b, 24, dtype=torch.bool, device=device)
            dt = _timed(lambda: model(tokens, token_mask, options, option_mask), reps=50, warmup=10, device=device)
            rows.append({"batch": b, "ms": dt * 1e3, "ms_per_sample": dt * 1e3 / b,
                         "samples_per_s": b / dt})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MX550/torch operator + CVPN benchmark.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="reports/explore")
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    encoder = default_encoder()
    do_compile = not args.no_compile and args.device.startswith("cuda")
    report: dict = {
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "matmul": bench_matmul(args.device),
        "copy_bandwidth_gbs": bench_bandwidth(args.device),
        "training": [],
        "inference": [],
    }
    for batch in (32, 128, 512):
        report["training"].append(
            bench_training(args.device, encoder, pool_size=args.pool_size,
                            batch_size=batch, compile_model=do_compile)
        )
    report["inference"] = {
        "compiled": bench_inference(args.device, encoder, compile_model=do_compile),
        "eager": bench_inference(args.device, encoder, compile_model=False),
    }
    if not args.no_write:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = ensure_dir(Path(args.out))
        path = out / f"mx550_ops_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench_ops] wrote {path}")
    print(json.dumps({k: report[k] for k in ("device_name", "copy_bandwidth_gbs")}, indent=2))
    for row in report["matmul"]:
        if "tflops" in row:
            print(f"  matmul {row['dtype']:>4} {row['n']:>4}: {row['tflops']:5.2f} TFLOPS")
    for row in report["training"]:
        print(f"  train batch={row['batch_size']:>3} compile={row['compile']}: "
              f"base={row['baseline_samples_per_s']:7.0f} pre={row['precollated_samples_per_s']:7.0f} "
              f"s/s  x{row['speedup']:.2f}")
    for tag, rows in report["inference"].items():
        for row in rows:
            print(f"  infer[{tag}] B={row['batch']:>3}: {row['ms']:6.2f} ms "
                  f"({row['ms_per_sample']:.3f} ms/sample, {row['samples_per_s']:.0f} samples/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
