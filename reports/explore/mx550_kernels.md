# MX550 融合 CUDA kernel（M3 特化，sm_75）

> 2026-10-11，WSL MX550（cc 7.5 / torch 2.6.0+cu124 / nvcc 12.3）。源码 `train/cuda/cvpn_fused.cu`，
> 加载器 `train/cuda_kernels.py`，基准 `scripts/bench_kernels.py`，测试 `tests/test_cuda_kernels.py`。
> 原始数据 `reports/explore/mx550_kernels_20261011T165044.json`。

## 0. 结论速览

- CVPN 前向在搜索/批推理实际用的 batch（B=1..64）是**启动/访存受限**（profiler：**每次前向 ~22 次 kernel 启动**，transformer ~72 次），不是算力受限。
- 自写 **6 个融合 kernel**：LayerNorm+SiLU、掩码池化、scorer 末层、value MLP、token 投影(+LN+SiLU)、option 投影(+SiLU)。
  **GEMM 的取舍**：token/option 投影的融合 GEMM **只在行数少时**胜 cuBLAS，故按行数门控——小 batch 用融合 GEMM，大 batch 用 cuBLAS + 融合 epilogue。
- **在线搜索的 B=1..8：前向 x1.8–2.2**（且优于 `torch.compile`）；**逐算子最多快 ~9×**。
- B=64 由 cuBLAS GEMM 主导，前向 ≈持平（0.98×）；`torch.compile` 在大 batch 略优。
- 数值与 eager 一致（max abs diff ≤ 2e-6，fp32）。

## 1. 实现的 kernel（`train/cuda/cvpn_fused.cu`）

| kernel | 融合内容 | 使用条件 |
|--------|----------|----------|
| `layernorm_silu` | LayerNorm + SiLU（token_proj epilogue） | 大行数（cuBLAS GEMM 后） |
| `masked_mean_pool` | token 掩码池化 | 总是 |
| `scorer_tail` | option_scorer 的 SiLU + 末层线性 + 掩码(-inf) | 总是 |
| `value_head` | value MLP（Linear+SiLU+Linear） | 总是 |
| `token_proj_ln_silu` | token 投影 GEMM + LayerNorm + SiLU（W 入共享、16 行/块） | 行数 ≤ 128 |
| `linear_silu` | option 投影 GEMM + SiLU（W 入共享、16 行/块） | 行数 ≤ 256 |

启动数 ~22 → ~7；其余 GEMM（scorer 首层、cat）保持 cuBLAS + torch。

## 2. 前向端到端（同权重，fp32，CUDA events，500 rep）

| batch | eager | kernels | compile | kernels 加速 | compile 加速 |
|------:|------:|--------:|--------:|-------------:|-------------:|
| 1 | 0.794 ms | **0.357 ms** | 0.610 ms | **2.23×** | 1.30× |
| 8 | 0.710 ms | **0.404 ms** | 0.663 ms | **1.76×** | 1.07× |
| 64 | 0.670 ms | 0.682 ms | 0.579 ms | 0.98× | 1.16× |

## 3. 逐算子 vs torch 等价实现（µs）

| 算子 | 形状 | kernel | torch | 备注 |
|------|------|-------:|------:|------|
| layernorm_silu | 64×128 | 12.7 | 30.6 | |
| layernorm_silu | 2688×128 | 55.0 | 78.1 | |
| token_proj_ln_silu | 64×65 | 45.0 | 170.4 | 小行数胜 |
| token_proj_ln_silu | 2688×65 | 568.3 | 149.5 | **大行数败给 cuBLAS** |
| linear_silu | 64×68 | 32.3 | 46.7 | 小行数胜 |
| linear_silu | 2688×68 | 407.5 | 140.7 | **大行数败给 cuBLAS** |
| masked_mean_pool | 1×42×128 | 16.1 | 80.1 | |
| masked_mean_pool | 64×42×128 | 18.6 | 80.4 | |
| scorer_tail | 1×26×256 | 18.4 | 106.2 | |
| scorer_tail | 64×26×256 | 46.2 | 93.4 | |
| value_head | 1×128 | 13.0 | 107.3 | |
| value_head | 64×128 | 78.7 | 79.1 | |

**关键发现**：朴素的「每输出点乘」GEMM 在行数大时远慢于 cuBLAS（L2 带宽受限）；融合投影只在**小行数**有价值。故 `CVPN` 用行数门控（`_FUSED_PROJ_TOKEN_ROWS=128` / `_FUSED_PROJ_OPTION_ROWS=256`）。

## 4. 接入方式

- 配置：`CVPNConfig(use_cuda_kernels=True)`；或 `train.model.enable_cuda_kernels(model)`。
- 仅对 **eager CVPN**、**eval 模式**、**dropout=0**、**fp32 CUDA** 生效；否则自动回退 eager（扩展构建失败也回退）。
- 批推理：`python -m train.batched_inference --serve --checkpoint <ckpt> --device cuda --cuda-kernels`
  （`BatchedInferenceService(..., cuda_kernels=True)`）。
- 编译：首次调用 `torch.utils.cpp_extension.load`（`TORCH_CUDA_ARCH_LIST=7.5`，torch 缓存）；`GITCG_CUDA_ARCH`/`GITCG_CUDA_BUILD_DIR` 可覆盖。

## 5. 复现

```bash
python -m scripts.bench_kernels --device cuda --out reports/explore
python -m pytest tests/test_cuda_kernels.py -q      # 慢测（需 CUDA+nvcc）
```

## 6. 取舍与下一步

- 收益集中在**小 batch**（在线搜索叶子打分、单局推理）；大 batch 由 cuBLAS GEMM 主导，融合收益有限。
- 未做：真正的分块 SGEMM（小 K/小 N 下仍难胜 cuBLAS，收益低）、把 `cat`/`expand` 融掉、attention 专用 kernel（transformer 每次 72 次启动，是更大的下一步目标）、CUDA Graph 捕获。
- 2 GB 显存下只做 fp32（fp16 在本卡更慢，见 `reports/explore/mx550_ops.md`）。
