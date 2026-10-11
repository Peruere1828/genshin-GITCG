"""MX550 fused-kernel benchmark: CVPN eager vs custom CUDA kernels vs torch.compile.

The toy CVPN forward is launch/memory bound at the batch sizes the search and
batched inference actually use (B=1..64). ``train/cuda/cvpn_fused.cu`` fuses the
elementwise/reduction parts (LayerNorm+SiLU, masked pooling, scorer tail, value
MLP); GEMMs stay on cuBLAS. This script measures the end-to-end forward and each
kernel against its torch equivalent.

Usage::

    python -m scripts.bench_kernels --device cuda --out reports/explore
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import torch

from common.paths import ensure_dir
from envs.observation import default_encoder
from train.cuda_kernels import cvpn_kernels
from train.model import CVPN, config_for_encoder


def _event_ms(fn, *, reps: int = 500, warm: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _batch(encoder, b: int, device: str, t: int = 42, k: int = 26):
    return (
        torch.randn(b, t, encoder.token_dim, device=device),
        torch.ones(b, t, dtype=torch.bool, device=device),
        torch.randn(b, k, encoder.option_dim, device=device),
        torch.ones(b, k, dtype=torch.bool, device=device),
    )


def bench_forward(encoder, device: str, batches=(1, 8, 64)) -> list[dict]:
    torch.manual_seed(0)
    eager = CVPN(config_for_encoder(encoder, d_model=128)).to(device).eval()
    fast = CVPN(config_for_encoder(encoder, d_model=128, use_cuda_kernels=True)).to(device).eval()
    fast.load_state_dict(eager.state_dict())
    graph = CVPN(
        config_for_encoder(encoder, d_model=128, use_cuda_kernels=True, use_cuda_graph=True)
    ).to(device).eval()
    graph.load_state_dict(eager.state_dict())
    compiled_base = CVPN(config_for_encoder(encoder, d_model=128)).to(device).eval()
    compiled_base.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_base)

    rows = []
    for b in batches:
        inputs = _batch(encoder, b, device)
        with torch.no_grad():
            lo, vo = eager(*inputs)
            lf, vf = fast(*inputs)
            lg, vg = graph(*inputs)
        max_diff = max(
            (lo - lf).abs().max().item(),
            (vo - vf).abs().max().item(),
            (lo - lg).abs().max().item(),
            (vo - vg).abs().max().item(),
        )
        e_ms = _event_ms(lambda: eager(*inputs))
        k_ms = _event_ms(lambda: fast(*inputs))
        g_ms = _event_ms(lambda: graph(*inputs))
        c_ms = _event_ms(lambda: compiled(*inputs))
        rows.append(
            {
                "batch": b,
                "eager_ms": round(e_ms, 4),
                "kernels_ms": round(k_ms, 4),
                "graph_ms": round(g_ms, 4),
                "compile_ms": round(c_ms, 4),
                "kernels_speedup": round(e_ms / k_ms, 3),
                "graph_speedup": round(e_ms / g_ms, 3),
                "compile_speedup": round(e_ms / c_ms, 3),
                "max_abs_diff": max_diff,
            }
        )
    return rows


def bench_ops(encoder, device: str) -> list[dict]:
    import torch.nn.functional as F

    k = cvpn_kernels()
    rows: list[dict] = []

    def add(name, shape, kernel_fn, torch_fn):
        rows.append(
            {
                "op": name,
                "shape": shape,
                "kernel_us": round(_event_ms(kernel_fn, reps=300) * 1e3, 2),
                "torch_us": round(_event_ms(torch_fn, reps=300) * 1e3, 2),
            }
        )

    for rows_n in (64, 2688):
        x = torch.randn(rows_n, 128, device=device)
        w = torch.randn(128, device=device)
        b = torch.randn(128, device=device)
        add("layernorm_silu", f"{rows_n}x128",
            lambda: k.layernorm_silu(x, w, b, 1e-5),
            lambda: F.silu(F.layer_norm(x, (128,), w, b, 1e-5)))
        xt = torch.randn(rows_n, 65, device=device)
        wt = torch.randn(128, 65, device=device)
        bt = torch.randn(128, device=device)
        lw = torch.randn(128, device=device)
        lb = torch.randn(128, device=device)
        add("token_proj_ln_silu", f"{rows_n}x65",
            lambda: k.token_proj_ln_silu(xt, wt, bt, lw, lb, 1e-5),
            lambda: F.silu(F.layer_norm(xt @ wt.T + bt, (128,), lw, lb, 1e-5)))
        xo = torch.randn(rows_n, 68, device=device)
        wo = torch.randn(128, 68, device=device)
        bo = torch.randn(128, device=device)
        add("linear_silu", f"{rows_n}x68",
            lambda: k.linear_silu(xo, wo, bo),
            lambda: F.silu(xo @ wo.T + bo))
    for batch in (1, 64):
        p = torch.randn(batch, 42, 128, device=device)
        m = torch.ones(batch, 42, dtype=torch.bool, device=device)
        mm = m.unsqueeze(-1).to(p.dtype)
        add("masked_mean_pool", f"{batch}x42x128",
            lambda: k.masked_mean_pool(p, m),
            lambda: (p * mm).sum(1) / mm.sum(1).clamp(min=1))
        hid = torch.randn(batch, 26, 256, device=device)
        w2 = torch.randn(256, device=device)
        om = torch.ones(batch, 26, dtype=torch.bool, device=device)
        add("scorer_tail", f"{batch}x26x256",
            lambda: k.scorer_tail(hid, w2, 0.1, om),
            lambda: (F.silu(hid) @ w2 + 0.1).masked_fill(~om, float("-inf")))
        pooled = torch.randn(batch, 128, device=device)
        w1 = torch.randn(128, 128, device=device)
        b1 = torch.randn(128, device=device)
        w2v = torch.randn(128, device=device)
        add("value_head", f"{batch}x128",
            lambda: k.value_head(pooled, w1, b1, w2v, 0.1),
            lambda: F.silu(pooled @ w1.T + b1) @ w2v + 0.1)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MX550 fused-kernel benchmark.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="reports/explore")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    if cvpn_kernels() is None:
        raise SystemExit("cvpn_fused extension unavailable (need CUDA + nvcc)")

    encoder = default_encoder()
    report = {
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch": torch.cuda.get_device_capability(0),
        "forward": bench_forward(encoder, args.device),
        "ops": bench_ops(encoder, args.device),
    }
    for row in report["forward"]:
        print(f"forward B={row['batch']:>3}: eager {row['eager_ms']:.3f} ms | "
              f"kernels {row['kernels_ms']:.3f} ms (x{row['kernels_speedup']}) | "
              f"kernels+graph {row['graph_ms']:.3f} ms (x{row['graph_speedup']}) | "
              f"compile {row['compile_ms']:.3f} ms (x{row['compile_speedup']})")
    for row in report["ops"]:
        print(f"  {row['op']:>16} {row['shape']:>12}: kernel {row['kernel_us']:8.2f} us | "
              f"torch {row['torch_us']:8.2f} us")
    if not args.no_write:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = ensure_dir(Path(args.out))
        path = out / f"mx550_kernels_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench_kernels] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
