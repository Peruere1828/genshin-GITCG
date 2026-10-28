# wsl-mx550 — WSL（4C/8G，MX550 2G）

## 事实

| 项 | 值 |
|---|---|
| 路径 | `~/projects/genshin-GITCG`（以实际 WSL 用户为准） |
| 硬件 | 4 核 / 8 GiB；MX550 2 GB（Turing cc 7.5，无 Tensor Core） |
| Python | conda env `gitcg`；torch CUDA 版（以 `bench_device` 为准） |
| 角色 | 并行采集/评测节点；小网络训练试验；**不是提强度训练节点** |

## 吞吐/限制

- mean-pool d128 ~8.6k 样本/s；2L transformer ~3.5k 样本/s（precollated）。
- 2 GB 显存：d_model ≤128、2 层、fp32；不要开 AMP（fp16/bf16 在 cc7.5 更慢）。
- 8 GB 内存：限制 precollate 池大小；4 worker 采集/评测。

## 已知坑

- `torch.compile` 对 cc7.5 收益未测，不要默认开。
- 大 store 加载会吃满 8G；用 `--train-limit/--val-limit` 或分片。
- 与 WSL 网络/文件系统 IO 有关的长任务先用小样本测吞吐。

## 冒烟

```bash
export PY=~/miniconda3/envs/gitcg/bin/python
"$PY" -m scripts.bench_device --devices cpu cuda --d-models 64 128 \
  --epochs 12 --batch-size 32 --precollate --games 6
```
