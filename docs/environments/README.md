# environments — 机器矩阵与通用契约

> 通用文档禁止写死 `/home/...`、`/root/...`、具体核数/显存。机器事实只放在这里。
> 所有命令用 `$REPO` / `$PY` / `$DATA`，各机文档给具体值。

## 机器矩阵

| 机器 | 硬件 | 定位 | Python | 入口 |
|---|---|---|---|---|
| 本机 `local-12c` | 6C/12T，~30–32 GiB，无 GPU | 采集/评测/CPU 训练主力 | conda `gitcg` (3.12) | [local-12c.md](./local-12c.md) |
| WSL `wsl-mx550` | 4C/8G，MX550 2G | 并行采集/评测 + 小网络试验 | conda `gitcg` | [wsl-mx550.md](./wsl-mx550.md) |
| A800 `a800` | A800 80G；宿主 128 核，cgroup ~10 核；~37G 存储 | **只做 GPU 训练/推理** | `/root/blockdata/conda/gitcg/bin/python` | [a800.md](./a800.md) |
| 南大集群 `nju-cluster` | LSF；CentOS7/glibc 2.17；CPU/GPU 队列 | 暂缓；离线依赖 | `fa_env` (3.10) | [nju-cluster.md](./nju-cluster.md) |
| Kaggle `kaggle-t4` | T4×2 16G；4 vCPU/29G；12h | **只训练** | kernel 内 | [kaggle-t4.md](./kaggle-t4.md) |

## 通用变量

| 变量 | 用途 | 备注 |
|---|---|---|
| `$REPO` | 仓库根 | 不写死路径 |
| `$PY` | 该机 conda env 的 python | 本机/A800 `gitcg`；NJU `fa_env` |
| `$DATA` | 数据根 | `GITCG_DATA_DIR` 可覆盖 |
| `GITCG_REPORTS_DIR` | 报告根 | Kaggle/只读输入用 |
| `GITCG_ASSETS_CACHE` | 资产缓存 | 离线必须设 |
| `GITCG_ASSETS_OFFLINE=1` | 禁止联网 | CI/NJU/A800 建议 |
| `NO_SHUFFLE=1` | 禁引擎洗牌 | `envs/match.py` 内部设置 |
| `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` | 防评测 worker 线程超订 | 多 worker 评测必须 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU | Kaggle 双卡分卡 |
| `PYTHONPATH=$REPO` + `cwd=$REPO` | 子进程协议 | `envs/rollout_worker` / `envs/fork_worker` |

## 共同纪律

- 采集/评测是 CPU 密集；A800/Kaggle 不做采集/评测（D21）。
- 长任务挂 tmux/setsid，结果流式落盘 + 按 `task_index` 断点续跑。
- 数据/checkpoint 不入库；跨机传输用 `rsync`/`scp`，checkpoint 回本地认证。
- 任何机器跑完先做冒烟：`python -m pytest -q -m "not slow"`。
