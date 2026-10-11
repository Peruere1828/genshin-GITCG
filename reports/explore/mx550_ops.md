# MX550 算子与训练/推理加速探索（M3 侧线）

> 2026-10-11，WSL MX550（cc 7.5 / 2.15 GB / 驱动 560.94 / torch 2.6.0+cu124，接电满电 P0）。
> 复现：`python -m scripts.bench_ops --device cuda --out reports/explore`
> 原始数据：`reports/explore/mx550_ops_20261011T131705.json`。

## 0. 结论速览

| 问题 | 结论 |
|------|------|
| fp16/bf16/AMP 能加速吗？ | **不能**。TU117 无 tensor core，fp16 matmul 反而慢 6–8×，bf16 与 fp32 相近。**这条 GPU 上用 fp32**。 |
| matmul 甜点 | 2048³ fp32 ≈ **2.5 TFLOPS**；≥4096³ 掉到 0.76（2 GB 显存吃紧）。单算子规模别超 ~2048。 |
| 训练瓶颈 | 不在 FLOPs，而在**每 batch 的 collate + 主机→设备拷贝**。 |
| 训练加速 | **precollate（整池一次 pad、设备上切片）**：真实数据 ~**5.4×**，`bench_device` 小池 ~2×。已落地为 `TrainConfig(precollate=True)` 快路径。 |
| 推理 | 批量推理 B=64 ≈ **147k samples/s**；`torch.compile` 把单样本延迟 0.96→**0.51 ms**（约 2×）。 |
| 带宽 | H2D/D2D copy ≈ **94 GB/s**（与 `LOCAL_BASELINE` 101 一致）。 |

## 1. 原始算子（`scripts/bench_ops.py`）

matmul TFLOPS（N×N×N）：

| N | fp32 | fp16 | bf16 |
|--:|-----:|-----:|-----:|
| 256 | 0.43 | 0.27 | 0.66 |
| 512 | **1.65** | 0.26 | 1.06 |
| 1024 | 1.19 | 0.30 | 1.67 |
| 2048 | **2.51** | 0.39 | 1.53 |

- **fp16 全面劣于 fp32**（cuBLAS fp16 无 TC 路径慢），**bf16 ≈ fp32 且不稳定**；`torch.autocast` 在本机是负优化，训练/推理一律 fp32。
- 小 matmul（<512）受 kernel 启动/占用限制，TFLOPS 低；把 batch 拼大比调 dtype 更有效。
- 显存 2 GB：4096³ 的中间量开始抖动，cvpn 放大时用梯度累积而非更大矩阵。

## 2. 训练加速：precollate 快路径（已落地）

`train/train_loop.py` 新增 `TrainConfig(precollate=True)`：把整池 `collate` 一次并搬到设备，之后每步只做设备上索引切片（同一 shuffle 顺序，指标与默认路径一致；见 `tests/test_train.py::test_precollate_*`）。

| 场景 | 默认 | precollate | 加速 |
|------|-----:|-----------:|-----:|
| 真实池 615 样本，d_model 128，10 epoch（脚本实测） | 3.20 s | 0.59 s | **5.4×** |
| `bench_device --games 8 --devices cuda` d_model 64 | 3.62 s | 2.24 s | 1.6× |
| 同上 d_model 128 | 1.94 s | 0.94 s | **2.1×** |

- 加速来源：默认路径每步 `collate`（Python 构造 + pad）再 `.to(device)` 拷贝，kernel/拷贝启动开销占主导；precollate 把这些移出训练循环。
- 代价：整池按**全局** max token/option 数 pad（真实池 max_options 可达 192），显存约 88 MB 量级（4099 样本），2 GB 够用；池极大时可分批 precollate。
- 开关：`train.pipeline --precollate`、`train.learning_curve --precollate`、`scripts.bench_device --precollate`。默认关，行为不变。

## 3. 推理：批量 + compile

CVPN 前向延迟（d_model 128）：

| 批大小 | eager | compiled |
|-------:|------:|---------:|
| 1 | 0.96 ms | **0.51 ms** |
| 8 | 0.56 ms | 0.46 ms |
| 64 | 0.58 ms | 0.44 ms（≈147k samples/s） |

- 单样本延迟是 kernel 启动受限，`torch.compile` 约减半——对搜索叶子打分（逐步前向）有意义。
- 批大小 ≥8 后 kernel 已饱和，再大只提吞吐不提单样本延迟。
- 注意：`torch.compile` 首次编译有 warmup（数十秒），短作业/多进程场景要权衡；建议在**长时批推理服务**（`train/batched_inference.py`）里用，而不是每个对局子进程各编译一遍。

## 4. 建议

1. GPU 训练/推理一律 **fp32**；不要开 AMP（本机负优化）。
2. 训练用 **`--precollate`**（本机 MX550 与 CPU 都可）；池 > ~5 万样本时改为分块 precollate 或只 precollate 热门 request 类型。
3. 批推理服务用 **batch ≥ 8 + `torch.compile`**；单机在线搜索的逐步前向可用 compile 但注意 warmup。
4. 2 GB 显存是硬约束：网络放大（d_model↑/transformer）优先用**梯度累积**与更小矩阵，而不是堆大 batch。

## 5. 未做 / 下一步

- 未写自定义 CUDA kernel（`sm_75`）：当前瓶颈是启动/拷贝而非算力，收益有限；若日后 batched inference 成为热点可评估一个融合打分 kernel。
- 未系统调 `torch.compile(mode="reduce-overhead")`/CUDA Graph：对固定形状的批推理可能有额外收益，列入机会窗口实验。
