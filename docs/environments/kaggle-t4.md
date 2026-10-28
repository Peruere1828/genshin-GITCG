# kaggle-t4 — Kaggle 训练节点

## 事实

| 项 | 值 |
|---|---|
| 硬件 | T4×2 16 GB；4 vCPU / 29 GB；`/kaggle/working` 20 GB |
| 时限 | 单会话 12h；GPU 30h/周 |
| 角色 | **只训练**；无引擎（V8/cffi 在 4 vCPU 太弱）；采集/评测留本机 |
| 数据 | `/kaggle/input/<slug>` 只读；数据集 200 GB/个 |
| 入口 | `scripts/kaggle_push.sh` / `scripts/kaggle_pull.sh`；`kaggle/train_arm.py` |

## 坑

- P100 已停用且 sm_60 不兼容；用 T4。
- 上传走代理（`googleapis.com` 直连超时）；先确认数据集版本 ready。
- `mode.txt` 白名单曾把 `cap` 静默回落 `runs`；改 mode 后核对 kernel 日志。
- 单 run 时限 4h/11h；按 epoch 估算，先跑 1 epoch 冒烟。
- `--precollate` 必做；CPU collate 是瓶颈（334k 样本 ~36s/epoch）。

## 冒烟

```bash
KAGGLE_MODE=smoke bash scripts/kaggle_push.sh
kaggle datasets status <slug>
```

## 数据/认证

训练产物回传本机，不在 Kaggle 做最终认证；评测用本机 `eval_protocol`。
