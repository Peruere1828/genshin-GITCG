# kaggle/ — training-arm kernel（Kaggle GPU 只当「epoch 预算」）

> 机器特化/坑见 [`docs/environments/kaggle-t4.md`](../docs/environments/kaggle-t4.md)；
> 当前排产见 [`docs/STATUS.md`](../docs/STATUS.md)。

定位（见 `reports/explore/plan_strength_2day_kaggle.md`）：**Kaggle 只做训练**。
对局步进/采集/`eval_protocol` 认证全在 V8/cffi（CPU），留在本机 / WSL；本目录只有一个
script kernel，把 compact store 训练跑完、把 checkpoint + 报告回传。

## 组成

| 文件 | 作用 |
|---|---|
| `train_arm.py` | kernel 入口：解包代码 → 解包 gitcg wheel → 挂 store → 按 run 列表逐个子进程训练（每卡一个并发、`CUDA_VISIBLE_DEVICES` 固定）→ 写 `checkpoints/ logs/ reports/ summary.json` |
| `kernel-metadata.json` | script kernel、私有、`machine_shape: NvidiaTeslaT4`（= GPU T4 ×2）、挂 `genshin-gitcg-code` + `genshin-gitcg-corpus` |
| `mode.txt` | 由 `scripts/kaggle_push.sh` 写进**暂存目录** `build/kaggle/kernel/`（不打进仓库）：`auto`（挂到 corpus 就跑真网格）/ `probe` / `smoke` / `runs` / `h4` / `cap` / `wave1` / `wave2` / `groupb` / `groupc` 等；**完整白名单以 `scripts/kaggle_push.sh` 为准** |
| `grids/` | 可复现的实验编排（本地/Kaggle 两侧）：`run_targeted_collect.sh`（定向补采）、`run_offline_rl_grid.sh`（BC/AWR/Q A/B）、`run_value_calib_grid.sh`（价值目标 A/B）、`run_h3_search_ladder.sh`（搜索门禁）、`certify_*.sh`（认证） |

幂等：已有 `<name>.pt` 的 run 直接跳过、checkpoint 先写 `.part.pt` 再改名 → 12 h 会话被砍后
重推一版 kernel 即可续跑；`--max-seconds`（默认 10.5 h）到点不再启新 run。

## 数据面

* 代码不从 GitHub 拉（私有仓 + 无网）：`scripts/kaggle_push.sh` 打 `build/kaggle/bundle-code/gitcg-code.tar.gz`
  （含 `PLAN.md`，`common.paths.repo_root()` 靠它定位），并 `datasets version` 刷新。
* 语料 `build/kaggle/bundle-corpus/*.pt`（硬链接到 `data/store/*.pt`）→ `datasets create/version`，挂到
  `/kaggle/input/genshin-gitcg-corpus`。
* **上传必须走本地代理**：Kaggle 的 blob 端点是 `googleapis.com`，本机直连超时（实测挂在 0 B/s）。
  `scripts/kaggle_push.sh` / `kaggle_pull.sh` 默认 `HTTPS_PROXY=http://127.0.0.1:7890`（`KAGGLE_PROXY` 覆盖），
  实测上传 ~5–30 MB/s。

## 用法

```bash
scripts/kaggle_push.sh                     # 默认：刷新 code 数据集 + 推 kernel（T4×2，11h 超时）
KAGGLE_MODE=smoke scripts/kaggle_push.sh   # 冒烟（合成 store + 1 个真 store 探针）
KAGGLE_MODE=runs  REFRESH_CODE=0 scripts/kaggle_push.sh
scripts/kaggle_pull.sh status|logs|out|collect
```

认证仍走本机：`python -m scripts.eval_protocol --contestant neural:<ckpt> --contestant-deck <deck> --seeds 20 --strict-ci --baseline expert:<deck>`。
