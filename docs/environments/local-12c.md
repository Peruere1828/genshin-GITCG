# local-12c — 本机（6C/12T，无 GPU）

## 事实

| 项 | 值 |
|---|---|
| 路径 | `/home/ubuntuaaa/projects/genshin-GITCG` |
| 硬件 | 6 物理核 / 12 线程；~30–32 GiB（以 `free -g` 为准）；无 GPU |
| Python | conda env `gitcg`，`/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python`（3.12） |
| 引擎 | `gitcg 0.21.0`；torch CPU（版本以 `pip show torch` 为准） |
| 角色 | 采集、评测、CPU 档训练主力 |
| 数据 | `data/replays/`、`data/store/`、`data/checkpoints/`（gitignore） |

## 吞吐（实测，供排产）

- 单核 ~819 局/h；12 worker 稳态 ~4,900 局/h（L5.1，理想/干净窗口）。
- 8 worker 采集实测 3,299–3,700 局/h；11 worker 4,020 局/h（受竞争）。
- 评测 ~1,500 局/h；CPU 训练 d256 ~150–300 样本/s。
- compact 语料 ~0.65 MB/局（fp16 + exact id sidecar）。

## 并发

- 默认 8–12 worker；不要同时跑“多个 10–12 worker 采集 + 训练 + 评测”（会 swap/OOM）。
- 评测 worker 必须设 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
- 长作业用 `tmux`/`setsid`；结果流式落盘。

## 已知坑

- cffi 回调会吞 `KeyboardInterrupt`；终端人机用 `HumanQuit`。
- 每局开始必须 `reset_default_hierarchical_action_codebook()`。
- `envs/match.py` 在 Python 侧预洗牌并设 `NO_SHUFFLE=1`；不要手改。
- `train.collect` 父进程内存已由 `stream_task_shard` 修复；旧命令不要再用 `--max-tasks`。
- 旧 compact 语料没有 exact definition id；不要用于 id 特征训练。

## 冒烟

```bash
export PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python
export REPO=/home/ubuntuaaa/projects/genshin-GITCG
cd "$REPO"
"$PY" -m pytest -q -m "not slow"
"$PY" -m envs.benchmark --games 12 --workers 1
"$PY" -m scripts.eval_protocol --smoke --seeds 2 --workers 4
```
