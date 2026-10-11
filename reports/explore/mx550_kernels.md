# MX550 融合 CUDA kernel + CUDA Graph（M3 特化，sm_75）

> 2026-10-11，WSL MX550（cc 7.5 / torch 2.6.0+cu124 / nvcc 12.3）。源码 `train/cuda/cvpn_fused.cu`、
> `train/cuda_graph.py`，加载器 `train/cuda_kernels.py`，基准 `scripts/bench_kernels.py`，测试 `tests/test_cuda_kernels.py`。
> 原始数据 `reports/explore/mx550_kernels_20261011T170026.json`。

## 0. 结论速览

- CVPN 前向在搜索/批推理实际用的 batch（B=1..64）是**启动/访存受限**（profiler：每次前向 ~22 次启动，transformer ~72 次），不是算力受限。
- 两步特化：
  1. **6 个融合 kernel**（elementwise/归约 + 小行数投影 GEMM），启动数 ~22 → ~7；
  2. **CUDA Graph** 把整个前向捕获成一次重放，消掉剩余启动/Python 开销。
- **结果（fp32，CUDA events）**：B=1 **4.39×**、B=8 **4.90×**、B=64 **1.77×**（vs eager），全面优于 `torch.compile`。
- 数值与 eager 一致（max abs diff ≤ 2e-6；CUDA Graph 逐字节一致）。
- 关键正确性修复：自定义 kernel 原来在**默认流**上启动，CUDA Graph 捕获会错；改为在 **PyTorch 当前流**（`at::cuda::getCurrentCUDAStream()`）启动后，图重放逐字节一致，且普通路径也更快。

## 1. 实现的 kernel（`train/cuda/cvpn_fused.cu`）

| kernel | 融合内容 | 使用条件 |
|--------|----------|----------|
| `layernorm_silu` | LayerNorm + SiLU（token_proj epilogue） | 大行数（cuBLAS GEMM 后） |
| `masked_mean_pool` | token 掩码池化 | 总是 |
| `scorer_tail` | option_scorer 的 SiLU + 末层线性 + 掩码(-inf) | 总是 |
| `value_head` | value MLP（Linear+SiLU+Linear） | 总是 |
| `token_proj_ln_silu` | token 投影 GEMM + LayerNorm + SiLU（W 入共享、16 行/块） | 行数 ≤ 128 |
| `linear_silu` | option 投影 GEMM + SiLU（W 入共享、16 行/块） | 行数 ≤ 256 |

所有 kernel 在 **当前 CUDA 流**上启动（图捕获/多流正确）；其余 GEMM（scorer 首层、cat）保持 cuBLAS + torch。

## 2. 前向端到端（同权重，fp32，CUDA events，500 rep）

| batch | eager | kernels | kernels+graph | compile | graph 加速 |
|------:|------:|--------:|--------------:|--------:|-----------:|
| 1 | 0.866 ms | 0.234 ms | **0.198 ms** | 0.539 ms | **4.39×** |
| 8 | 0.861 ms | 0.315 ms | **0.176 ms** | 0.668 ms | **4.90×** |
| 64 | 0.877 ms | 0.478 ms | 0.497 ms | 0.746 ms | 1.77× |

（B=1..8 是搜索叶子打分/单局推理的主战场，收益最大。）

## 3. 逐算子 vs torch 等价实现（µs）

| 算子 | 形状 | kernel | torch | 备注 |
|------|------|-------:|------:|------|
| layernorm_silu | 64×128 | 13.9 | 28.4 | |
| layernorm_silu | 2688×128 | 54.5 | 81.3 | |
| token_proj_ln_silu | 64×65 | 46.0 | 68.4 | 小行数胜 |
| token_proj_ln_silu | 2688×65 | 566.2 | 148.8 | **大行数败给 cuBLAS** |
| linear_silu | 64×68 | 33.4 | 46.3 | 小行数胜 |
| linear_silu | 2688×68 | 412.6 | 159.9 | **大行数败给 cuBLAS** |
| masked_mean_pool | 1×42×128 | 18.5 | 82.3 | |
| scorer_tail | 1×26×256 | 20.1 | 107.9 | |
| value_head | 1×128 | 12.9 | 98.5 | |

**关键发现**：朴素「每输出点乘」GEMM 在大行数时远慢于 cuBLAS（L2 带宽受限），故投影融合按行数门控（`_FUSED_PROJ_TOKEN_ROWS=128` / `_FUSED_PROJ_OPTION_ROWS=256`）。

## 4. CUDA Graph（`train/cuda_graph.py`）

- 按**输入形状**惰性捕获，静态缓冲每调用 `copy_` 后 `replay`；输出 clone 返回（调用方可保留）。
- 仅推理（eval、无梯度、静态形状）；`use_cuda_graph=True` 时先走融合 kernel 路径再捕获。
- 与 kernel 叠加：B=8 从 eager 0.861ms → **0.176ms（4.9×）**。
- 注意：捕获时**不能有 CPU-GPU 同步**——把 scorer/value 的标量 bias 缓存在 host（`_cuda_scalars`）避免 `float(tensor)` 同步。

## 5. 接入方式

- `CVPNConfig(use_cuda_kernels=True, use_cuda_graph=True)`；或 `train.model.enable_cuda_kernels(model)` / `enable_cuda_graph(model)`。
- 仅对 **eager CVPN**、**eval 模式**、**dropout=0**、**fp32 CUDA** 生效；否则自动回退 eager。
- 批推理：`python -m train.batched_inference --serve --checkpoint <ckpt> --device cuda --cuda-kernels`。
- 编译：首次调用 `torch.utils.cpp_extension.load`（`TORCH_CUDA_ARCH_LIST=7.5`，torch 缓存）。

## 6. 复现

```bash
python -m scripts.bench_kernels --device cuda --out reports/explore
python -m pytest tests/test_cuda_kernels.py -q      # 慢测（需 CUDA+nvcc）
```

## 7. 取舍与下一步

- 收益集中在**小 batch**；大 batch 由 cuBLAS GEMM 主导。
- CUDA Graph 有静态形状/显存代价（按形状缓存，超上限清空）；对频繁变化的形状收益下降。
- 未做：真正的分块 SGEMM、把 `cat`/`expand` 融掉、attention 专用 kernel（transformer 每次 72 次启动，是更大的目标）。
- 2 GB 显存下只做 fp32（fp16 在本卡更慢，见 `reports/explore/mx550_ops.md`）。
