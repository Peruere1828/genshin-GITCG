# nju-cluster — 南大集群（暂缓）

## 事实

| 项 | 值 |
|---|---|
| 调度 | LSF（`bsub < xxx.lsf`），模板 `scripts/cluster/lsf/` |
| OS/glibc | CentOS 7（glibc 2.17）；`gitcg` manylinux_2_34 需 `module load glibc/2.36-gcc12.1.0` |
| Python | 无 conda 频道；复用 `$HOME/.conda/envs/fa_env`（3.10 + torch 2.6.0）；`tomli` 回退 |
| 网络 | 集群不通外网；代码走 njugit master 镜像；wheel/资产走 `vendor` 分支 |
| 存储 | `/fsb/home`（Tier1）、`/bbfs/fsb/home`（全闪）、`/bbfs/scratch`（30 天） |
| 队列 | `cpu1` ≤24 核；`7702ib` 128c；`6140ib` 72c；`6330ib` 56c；GPU `945090ib` 5090 32G / `734090ib` 4090 24G |

## 状态

**暂缓（D14）**。启用前置：glibc/资产离线适配完成；`scripts/cluster/` 保留为适配起点。
不要把本机 `conda run -n gitcg` 照搬。

## 冒烟

```bash
export GITCG_VENDOR_DIR=$HOME/genshin-GITCG-vendor
bash scripts/cluster/bootstrap.sh
bsub < scripts/cluster/lsf/benchmark.lsf
```

## 坑

- 公共共享队列；按队列规格申请，GPU 作业 `#BSUB -gpu "num=1"`。
- 数据盘/scratch 有清理周期；长任务把 checkpoint 及时搬回 home。
- 集群代码只镜像 origin master；不要指望实时同步分支。
