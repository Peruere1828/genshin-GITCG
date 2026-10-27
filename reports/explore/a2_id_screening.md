# A2 identity embeddings：全量训练与胜率认证（2026-10-27）

> 口径：**纯策略**（PLAN §6.3），seed 900–909（全新），20 套脚本对手双向，400 局/deck/模型。
> 预注册判据：A0i 对 A0 的配对 Δ 的下界 > 0，且没有 deck 显著变差；**val_acc 只作参考**。

## 0. 结论

**A2 在胜率口径下失败，且不是小幅回退。**

| deck | A0（h4m e150） | A0i（CVPN d256 + id1024） | 配对 Δ（A0−A0i） | 95% CI |
|---|---:|---:|---:|---:|
| double_geo_navia | **0.720** | **0.266** | +0.454 | [+0.400, +0.508] |
| unyielding_geo | **0.828** | **0.358** | +0.470 | [+0.418, +0.522] |
| dvalin_bonk | **0.760** | **0.678** | +0.083 | [+0.031, +0.134] |

- 三个 deck 全部显著变差；dgeo/ungeo 直接掉 45–47pp。
- A0i 的 quality：`fallback_rate=0`、`error_rate=0`、`illegal_rate=0`——不是非法动作/崩溃，是「合法地下输」。
- A0i 的 store val_acc **0.866**（best 0.884 @ ep70），A0 0.838；**val_acc 涨了，胜率崩了**。这次 A2 的冒烟信号（4k/2ep 0.400→0.535）是早训/过拟合信号，不是强度信号。
- 去掉 id 分支的同一 trunk（`A0i_noid`）dgeo 0.741 / ungeo 0.670 / dvalin 0.303：**dgeo 略高于 A0，ungeo 低 16pp，dvalin 直接崩**。id 分支与 trunk 都是强 deck 依赖，**没有稳定超过 A0 的组合**；A0i（id 开）整体不可用。

## 1. 全量训练（A0i）

| 项 | 值 |
|---|---|
| 数据 | `mixed_four_decks.pt` 542,096 train / 28,838 val |
| 模型 | CVPN d256，`id_buckets=1024`，`id_emb_dim=32`，0.35M 参数 |
| 预算 | 150 epoch，batch 512，lr 1e-3，seed 0 |
| 耗时 | 4,529.6 s（75.5 min，含一次 eval 并发干扰） |
| val | ep0 0.7109 / ep70 **0.8839**（best） / ep149 **0.8658**；h4m 0.8377 |
| val_ce | ep70 0.3747 / ep149 0.5316；h4m 0.5898 |

- ep70 后 val_ce 回升、val_acc 回落，**最终 checkpoint 已过拟合**。
- 当前 `datafirst_bc_train` 只保存最后一个 epoch；best checkpoint 没有落盘。下一轮应加 `--save-best` 或把预算压到 70–100 ep。

## 2. 认证（seed 900–909）

原始文件：`data/eval/a0*_*_s900.jsonl`；报告：`reports/explore/a0*_s900*.json`。
命令：`scripts/certify_a2.sh`（A0 与 A0i 都跑 `--baseline expert:<deck> --strict-ci`）。

- A0 的 seed-900 数字比旧的 seeds 0–9 略高（dgeo 0.720 vs 0.659；ungeo 0.828 vs 0.823；dvalin 0.760 vs 0.758），属于种子波动，不影响配对结论。
- A0i 的 dvalin 只掉 8pp；dgeo/ungeo 崩 45pp，说明不是所有 deck 同等受损，**deck 依赖性很强**。

## 3. 诊断：id 分支 vs trunk

为区分「eval 时 id 分支坏了」与「trunk 在引擎上过拟合」，已生成去掉 6 个 id 参数的
`data/checkpoints/a2/A0i_noid.pt`（`reports/explore/a0inoid_double_geo_navia_s900.json`）：

| 模型 | store val_acc (dgeo) | 引擎 dgeo | 引擎 ungeo | 引擎 dvalin |
|---|---:|---:|---:|---:|
| A0（h4m） | 0.838 | 0.720 | 0.828 | 0.760 |
| A0i（id 开） | **0.866** | **0.266** | **0.358** | **0.678** |
| A0i_noid（同一 trunk，id 关） | 0.397 | **0.741** | 0.670 | **0.303** |

结论：**trunk 与 id 分支是强 deck 依赖的，二者都没有稳定超过 A0。**
- dgeo：id 分支在引擎上把 trunk 从 0.741 拉到 0.266，是纯伤害；trunk 本身略高于 A0。
- ungeo：trunk 0.670 好于 id 版 0.358，但仍低于 A0 0.828。
- dvalin：反过来，id 版 0.678 好于 trunk 0.303，但仍低于 A0 0.760。
- id 分支在 store val 上是必需的（0.866 vs 0.397），但在引擎上收益/伤害随 deck 翻转——这是典型的
  **store→engine 特征不一致 / id 分支过拟合 store**，不是简单的列错位。

已核对：store 的 definition 列确实是 token 42 / option 30，与当前
`reps.observation_encoder.definition_feature_indices()` 一致；token_dim 65 / option_dim 68 也一致。
下一步需要把同一 `(deck, opponent, seed, step)` 的 store 样本与引擎观测逐元素对比，重点看
definition 列的值域、option 的 definition 列、以及 T>50 的 token。

## 4. 当前可用/不可用

- **可用**：A0（h4m e150）仍是当前 4-deck 通用 BC 基线；seed-900 dgeo 0.720 / ungeo 0.828 / dvalin 0.760。
- **不可用**：A0i（id 开）在任何 deck 上都不如 A0；A0i_noid 只在 dgeo（0.741 vs 0.720）可用，ungeo/dvalin 都不如 A0。**不要把 A0i/A0i_noid 当通用底座**。
- **未受影响**：per-deck specialist（dgeo 0.880 / ungeo 0.909，20 seeds PASS）与 exploit ladder；本轮没有改 specialist 代码/数据。

## 5. 下一步（按优先级）

1. **已定位**：id 分支与 trunk 都是强 deck 依赖，且 id 分支在 store val 上必需、在引擎上随 deck 翻转（dgeo 害、dvalin 帮）；没有稳定超过 A0 的组合。
2. 对比同一 `(deck, opponent, seed, step)` 的 store 样本与引擎观测：definition 列（token 42 / option 30）的值域、option 的 definition、T>50 的 token；加一个逐元素对比测试，找出 id 分支在引擎上读到的「不同值」。优先看 dgeo（伤害最大）和 dvalin（反向）。
3. 同时加 `--save-best`（或 70–100 ep 预算 + early stop），避免最终 checkpoint 过拟合；用 ep70 的 best 再认证一次。
4. **A3/A4/A5 暂缓**：A2 的 val_acc 假阳性已经说明，先把 train/eval 一致性和胜率口径修好，再谈决策记忆/选项注意力/辅助头。若 id 分支修复后仍无胜率增益，回到 D24（specialist/RL/数据覆盖）。

## 6. 复现

```bash
# A0i 全量
python -m scripts.datafirst_bc_train --store-cache data/store/mixed_four_decks.pt \
  --arch cvpn --d-model 256 --epochs 150 --lr 1e-3 --batch-size 512 --threads 8 \
  --device cuda --seed 0 --loss ce --id-buckets 1024 --id-emb-dim 32 \
  --val-limit 8000 --out data/checkpoints/a2/A0i.pt --report reports/train/a2_a0i.json

# 认证（A0 + A0i，3 deck，seed 900，10 workers）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PY=/root/blockdata/conda/gitcg/bin/python SEEDS=10 SEED_START=900 WORKERS=10 \
CANDIDATES="a0i:data/checkpoints/a2/A0i.pt" bash scripts/certify_a2.sh

# 配对 Δ
python -m scripts.compare_eval_runs a0_double_geo_navia_s900 a0i_double_geo_navia_s900
```
