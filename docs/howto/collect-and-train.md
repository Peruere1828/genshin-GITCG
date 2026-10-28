# howto — 采集与训练

> 详细实现见 `train/README.md`、`train/CONTINUAL_RESOLVING.md`、`docs/PLAYBOOK.md`。

## 采集（CPU 节点）

```bash
python -m train.collect --deck <deck> --all-opponents \
  --seed-start 0 --seed-count 20 --workers 8 \
  --tag <tag> --format compact --no-privileged-state
```

- 一行一局，`data/replays/<tag>/samples.jsonl`；manifest 指纹护栏、按 `task_index` 断点续跑。
- 新 compact 语料带 `token_definition_ids` / `option_definition_ids`（exact id）；旧 fp16 语料不能用于 id 特征训练。
- 采集种子避开 val 残差（`val_every=10` 时取模 10 余 1–9）。

## 建 store

```bash
python -m scripts.datafirst_bc_store --replays data/replays/<tag> \
  --store-cache data/store/<tag>.pt --val-every 10
```

- `meta` 记录 `encoder_config`、`definition_indices`、`definition_ids_present`。
- id 训练要求 `definition_ids_present == events`；否则 `datafirst_bc_train --id-buckets` 直接拒绝。

## 训练

```bash
python -m scripts.datafirst_bc_train --store-cache data/store/<tag>.pt \
  --arch cvpn --d-model 256 --epochs 15 --batch-size 256 \
  --save-best --select-metric val_ce \
  --out data/checkpoints/<tag>.pt --report reports/train/<tag>.json
```

- `--save-best` 保存 best（默认 `val_ce`）与 last；A2 的 best 是 ep70，不是 final。
- `--loss ce|pg`；`awr/q` 已有负结果，见 `docs/failures/002`。
- CPU 档 d256 ~150–300 样本/s；A800 用 `--device cuda --batch-size 512`。

## 评测

见 `docs/EVAL_PROTOCOL.md`；认证用 `scripts/eval_protocol.py`，配对种子、双方、纯策略。
