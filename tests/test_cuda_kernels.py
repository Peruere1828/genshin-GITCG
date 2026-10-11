"""MX550 fused CUDA kernels (train/cuda/cvpn_fused.cu): build + numerical parity.

Slow-gated: needs CUDA + nvcc to build the extension. Skips cleanly otherwise.
"""

from __future__ import annotations

import pytest
import torch

from envs.observation import default_encoder
from train.cuda_kernels import cvpn_kernels
from train.model import CVPN, config_for_encoder, enable_cuda_kernels

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    module = cvpn_kernels()
    if module is None:
        pytest.skip("cvpn_fused extension could not be built")
    return module


def test_kernels_match_torch(kernels):
    import torch.nn.functional as F

    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(37, 128, device=dev)
    w = torch.randn(128, device=dev)
    b = torch.randn(128, device=dev)
    ref = F.silu(F.layer_norm(x, (128,), w, b, 1e-5))
    assert torch.allclose(kernels.layernorm_silu(x, w, b, 1e-5), ref, atol=1e-5)

    p = torch.randn(5, 9, 16, device=dev)
    m = torch.ones(5, 9, dtype=torch.bool, device=dev)
    m[0, 5:] = False
    mm = m.unsqueeze(-1).to(p.dtype)
    ref = (p * mm).sum(1) / mm.sum(1).clamp(min=1)
    assert torch.allclose(kernels.masked_mean_pool(p, m), ref, atol=1e-5)

    hid = torch.randn(4, 7, 32, device=dev)
    w2 = torch.randn(32, device=dev)
    om = torch.ones(4, 7, dtype=torch.bool, device=dev)
    om[1, 3] = False
    ref = (F.silu(hid) @ w2 + 0.3).masked_fill(~om, float("-inf"))
    out = kernels.scorer_tail(hid, w2, 0.3, om)
    assert torch.allclose(out[om], ref[om], atol=1e-5)
    assert torch.isinf(out[~om]).all()

    pooled = torch.randn(6, 16, device=dev)
    w1 = torch.randn(24, 16, device=dev)
    b1 = torch.randn(24, device=dev)
    w2v = torch.randn(24, device=dev)
    ref = F.silu(pooled @ w1.T + b1) @ w2v + 0.1
    assert torch.allclose(kernels.value_head(pooled, w1, b1, w2v, 0.1), ref, atol=1e-5)


def test_cvpn_kernel_fast_path_matches_eager(kernels):
    encoder = default_encoder()
    dev = "cuda"
    torch.manual_seed(0)
    eager = CVPN(config_for_encoder(encoder, d_model=64)).to(dev).eval()
    fast = CVPN(config_for_encoder(encoder, d_model=64, use_cuda_kernels=True)).to(dev).eval()
    fast.load_state_dict(eager.state_dict())
    assert enable_cuda_kernels(fast)
    b, t, k = 8, 20, 12
    tokens = torch.randn(b, t, encoder.token_dim, device=dev)
    token_mask = torch.ones(b, t, dtype=torch.bool, device=dev)
    options = torch.randn(b, k, encoder.option_dim, device=dev)
    option_mask = torch.ones(b, k, dtype=torch.bool, device=dev)
    option_mask[0, k - 1] = False
    with torch.no_grad():
        lo, vo = eager(tokens, token_mask, options, option_mask)
        lf, vf = fast(tokens, token_mask, options, option_mask)
    assert torch.allclose(lo[option_mask], lf[option_mask], atol=1e-4)
    assert torch.allclose(vo, vf, atol=1e-4)
    assert torch.isinf(lf[~option_mask]).all()
