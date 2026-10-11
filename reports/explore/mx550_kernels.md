# MX550 融合 CUDA kernel（M3 特化，sm_75）

> 2026-10-11，WSL MX550（cc 7.5 / torch 2.6.0+cu124 / nvcc 12.3）。源码 `train/cuda/cvpn_fused.cu`，
> 加载器 `train/cuda_kernels.py`，基准 `scripts/bench_kernels.py`，测试 `tests/test_cuda_kernels.py`。
> 原始数据 `reports/explore/mx550_kernels_20261011T163849.json`。

## 0. 结论速览

- CVPN 前向在搜索/批推理实际用的 batch（B=1..64）是**启动/访存受限**，不是算力受限（profiler：**每次前向 ~22 次 kernel 启动**，transformer ~72 次）。
- 自写 4 个融合 kernel 把 elementwise/归约部分从 ~14 次启动压到 4 次；**GEMM 仍交给 cuBLAS**（这些尺寸 cuBLAS 更快）。
- **在线搜索的 B=1..8 区间：前向 x1.4–1.8**，且**逐算子最多快 18×**（scorer_tail B=1：12 µs vs torch 221 µs）。
- B=64 时前向被 cuBLAS GEMM 主导，融合收益被摊薄（≈持平）；`torch.compile` 同区间无明显优势。
- 数值与 eager 一致（max abs diff ≤ 2e-6，fp32）。

## 1. 实现的 kernel（`train/cuda/cvpn_fused.cu`）

| kernel | 融合内容 | 启动数 |
|--------|----------|-------|
| `layernorm_silu` | token_proj 的 LayerNorm + SiLU（块内均值/方差归约） | ~5 → 1 |
| `masked_mean_pool` | token 掩码池化（乘+求和+归一） | ~3 → 1 |
| `scorer_tail` | option_scorer 的 SiLU + 末层线性 + 掩码（-inf） | ~3 → 1 |
| `value_head` | value MLP（Linear+SiLU+Linear，块内归约） | ~3 → 1 |

启动数 ~22 → ~9；GEMM（token_proj / option_proj / scorer 首层 / cat）保持 cuBLAS + torch。

## 2. 前向端到端（同权重，fp32，CUDA events，500 rep）

| batch | eager | kernels | compile | kernels 加速 | compile 加速 |
|------:|------:|--------:|--------:|-------------:|-------------:|
| 1 | 0.748 ms | **0.535 ms** | 0.549 ms | **1.40×** | 1.36× |
| 8 | 0.744 ms | **0.413 ms** | 0.613 ms | **1.80×** | 1.21× |
| 64 | 0.681 ms | 0.696 ms | 0.783 ms | 0.98× | 0.87× |

## 3. 逐算子 vs torch 等价实现（µs）

| 算子 | 形状 | kernel | torch | 加速 |
|------|------|-------:|------:|-----:|
| layernorm_silu | 64×128 | 12.6 | 35.3 | 2.8× |
| layernorm_silu | 2688×128 | 54.0 | 107.3 | 2.0× |
| masked_mean_pool | 1×42×128 | 12.8 | 75.1 | 5.9× |
| masked_mean_pool | 64×42×128 | 18.1 | 81.1 | 4.5× |
| scorer_tail | 1×26×256 | 12.0 | 221.4 | **18.4×** |
| scorer_tail | 64×26×256 | 45.7 | 93.3 | 2.0× |
| value_head | 1×128 | 13.6 | 101.7 | 7.5× |
| value_head | 64×128 | 77.4 | 138.8 | 1.8× |

（torch 等价实现多次小 kernel + 中间张量；融合后省掉中间物化与启动。）

## 4. 接入方式

- 配置：`CVPNConfig(use_cuda_kernels=True)`；或对已加载模型 `train.model.enable_cuda_kernels(model)`。
- 仅对 **eager CVPN**、**eval 模式**、**dropout=0**、**fp32 CUDA** 生效；否则自动回退 eager（扩展构建失败也回退）。
- 批推理服务：`python -m train.batched_inference --serve --checkpoint <ckpt> --device cuda --cuda-kernels`
  （`BatchedInferenceService(..., cuda_kernels=True)`）。
- 编译：首次调用用 `torch.utils.cpp_extension.load` 编译（`TORCH_CUDA_ARCH_LIST=7.5`，缓存于 torch 扩展目录）；
  可用 `GITCG_CUDA_ARCH`/`GITCG_CUDA_BUILD_DIR` 覆盖。

## 5. 复现

```bash
python -m scripts.bench_kernels --device cuda --out reports/explore
python -m pytest tests/test_cuda_kernels.py -q      # 慢测（需 CUDA+nvcc）
```

## 6. 取舍与下一步

- 收益集中在**小 batch**（在线搜索叶子打分、单局推理）；大 batch 由 cuBLAS GEMM 主导，融合收益有限。
- 未做：自定义 GEMM（这些尺寸难胜 cuBLAS）、把 `cat`/`expand` 也融掉、attention 专用 kernel（transformer 每次 72 次启动，是下一步更大目标）、CUDA Graph 捕获。
- 2 GB 显存限制下，kernel 只做 fp32（fp16 在本卡更慢，见 `reports/explore/mx550_ops.md`）。
