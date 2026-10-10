# NJU 超算（LSF）作业模板与说明（PLAN.md §5 / §10）

> **状态（2026-10-10，D14）：暂缓。** 南大集群 glibc/环境适配成本高，当前默认算力为本机 + WSL（PLAN §5）。
> 本目录保留为将来机会算力的适配起点（C13）；启用前置：glibc ≥ 2.34 环境（`module load glibc/2.36-gcc12.1.0`）
> + `vendor` 分支离线依赖（gitcg wheel + 资产缓存）。

工程约束（D7/D14）：**无常驻服务**、**一切长任务可断点续跑**、热数据在作业节点本地盘、归档走大容量存储。

## 环境事实（2026-10-09 实测）

- 调度：**LSF**（`bsub`）；提交方式 `bsub < xxx.lsf`，脚本第一行起 `#BSUB ...`。
- 登录/计算节点均为 **CentOS 7（glibc 2.17）**；而 `gitcg` 0.21.0 的 `libgitcg.so` 是
  **manylinux_2_34（需 glibc ≥ 2.34）** → 必须 `module load glibc/2.36-gcc12.1.0`。
- Python：集群**无 conda 频道访问**（`conda create` 拉不到包），复用已装 torch 的已有环境
  （默认 `$HOME/.conda/envs/fa_env`，Python 3.10 + torch 2.6.0）。代码已做 3.10 兼容
  （`tomllib` → `tomli` 回退）。
- 集群**不通外网**：代码走 njugit（master 镜像，同步 origin）；`gitcg` 预编译 wheel 与专家**资产缓存**走 njugit 的 `vendor` 分支。
- 包镜像：`https://mirror.nju.edu.cn/pypi/web/simple`（内网可达，用于 cffi/protobuf/tomli 等）。

## 一次性准备（登录节点）

```bash
cd ~
git clone              git@git.nju.edu.cn:Fmyh1828/genshin-GITCG.git genshin-GITCG        # master 镜像（同步 origin）
git clone -b vendor    git@git.nju.edu.cn:Fmyh1828/genshin-GITCG.git genshin-GITCG-vendor # 离线依赖（wheel + 资产缓存）
cd genshin-GITCG
GITCG_VENDOR_DIR=$HOME/genshin-GITCG-vendor bash scripts/cluster/bootstrap.sh
```

`bootstrap.sh` 幂等：装依赖到仓库内 `.pydeps/`（不改动共享 env）、解开 gitcg wheel、
装资产缓存、跑冒烟。运行任何东西前 `source scripts/cluster/env.sh`。

## 队列对应（PLAN.md §5；公共共享，不被抢占）

| 用途 | 队列 | 规格 |
|------|------|------|
| CPU（小作业 ≤24 核） | `cpu1` | e5v3ib/6140ib/7702ib/6330ib 低优先级 |
| CPU（自博弈/arena/采集） | `7702ib` | 128c/节点（公共共享） |
| CPU 备选 | `6140ib`(72c)、`6330ib`(56c) | 公共共享 |
| GPU（训练/推理） | `945090ib`(5090 32G)、`734090ib`(4090 24G) | 公共共享 |
| GPU 备选 | `83a100ib`(A100 40G)、`62v100ib`(V100 32G) | 公共共享 |

GPU 作业用 `#BSUB -gpu "num=1"`；CPU 核按 GPU 比例自动分配。

## 存储

- 家目录 `/fsb/home/...`（Tier1）；高速全闪缓存 `/bbfs/fsb/home/...`（同一文件，读写自动回写，热数据优先）。
- 临时/中间 checkpoint：`/bbfs/scratch/<user>`（30 天未访问删除，适合 checkpoint）。
- 节点本地 `/tmp`、`/var/tmp`、部分节点 `/ssd`、`/tmp/ssd`、`/dev/shm`。

## 作业模板

| 模板 | 队列 | 用途 |
|------|------|------|
| `lsf/benchmark.lsf` | cpu1 | 单核吞吐基线（M0 换算） |
| `lsf/arena.lsf` | 7702ib | 脚本对手 arena / checkpoint 评测（可 `--policy0 neural:<ckpt>`） |
| `lsf/collect.lsf` | 7702ib | 可续跑 replay 采集（单节点多 worker） |
| `lsf/collect_array.lsf` | 7702ib | 海量数组作业采集（每元素独立分片目录） |
| `lsf/train.lsf` | cpu1 | 玩具/小规模 SoG 轮次 |
| `lsf/train_gpu.lsf` | 945090ib | 从 replay 目录训练 CVPN（`--replays`） |

```bash
# 冒烟
bsub < scripts/cluster/lsf/benchmark.lsf
# CPU 采集（可续跑）
GITCG_SEED_COUNT=500 GITCG_TAG=coldstart bsub < scripts/cluster/lsf/collect.lsf
# 海量数组作业
GITCG_WINDOW=50 GITCG_TAG=coldstart bsub < scripts/cluster/lsf/collect_array.lsf
# GPU 训练（消费分片）
GITCG_REPLAYS="data/replays/coldstart" bsub < scripts/cluster/lsf/train_gpu.lsf
# checkpoint 评测
GITCG_CONTESTANTS="superconduct_aggro" GITCG_POLICY0="neural:data/checkpoints/x.pt" \
    bsub < scripts/cluster/lsf/arena.lsf
```

## 关键做法

- **数据平面（CPU 采集 → 存储 → GPU 训练）**：`python -m train.collect` 写**可续跑** replay 目录，
  `train.pipeline/learning_curve --replays` 读盘训练（§5.1，CPU/GPU 解耦）。
- **海量作业**：`collect_array.lsf` 每数组元素一个种子窗口、各自分片目录，天然断点续跑。
- **资产离线**：作业里 `GITCG_ASSETS_OFFLINE=1`，缓存由 `bootstrap.sh` 从 `vendor` 分支装入。
- **确定性**：`envs/match.py` 同种子可复现（预洗牌 + codebook reset），跨节点一致。
- **产物归属**：checkpoint/数据只追加；小报告回 `reports/` 用 git 同步，大 JSONL 留簇上（不入库）。
