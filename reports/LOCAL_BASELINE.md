# 本地先行基线（L0/L1，本机 12 核 CPU）

> 2026-10-07，本地 12 核；`gitcg` 0.21.0，引擎 commit 见 `configs/engine.lock`。
> 观测口径：脚本对手互打（纯 CPU，无 GPU / 无 LLM）。

## L0 吞吐基线（验收替代原"1 万局/小时"）

固定对局 `superconduct_aggro` vs `natlan_battleship`（含完整决策构造，走 `reps` 表示层）：

| 指标 | 实测 |
|------|------|
| 单核稳态吞吐 | **~819 局/小时/核** |
| 单核决策速率 | **~23 决策/秒/核** |
| 12 worker 突发（24 局） | ~2984 局/小时（受每 worker 启动/资产加载摊薄，局数越多越贴近单核×核数） |
| 平均每局 | ~4.4 秒/局（含 ~55 决策/方） |

换算（用于 M0/M4 排产）：
- M0 "1 万局/小时" ≈ **13 核**即可达成（远低于单机 128 核假设）。
- 100 万局自博弈 ≈ **~1220 核·时**（单核口径）；超算 128 核单节点 ≈ 10 小时。
- 20 套对手全矩阵、每对局 50 局、双向 = 20×20×50×2 = 40000 局 ≈ 49 核·时 ≈ 12 核 4 小时（隔夜充裕）。
**⚠️ 2026-10-08 实测修正：此线性外推不成立**，见下方 L5.1（并行效率 ~50%，实际 ≈ 96 核·时）。

原始报告：`reports/benchmarks/bench_20261007T135343.json`。

## L1 arena 冒烟

3 套装对、3 配对种子、含自对局共 27 局，12 worker，0 error / 0 truncated（约 36 秒）：

| 卡组 | 对池胜率 | Wilson 95% CI |
|------|----------|----------------|
| superconduct_aggro | 0.778 | [0.453, 0.937] |
| dual_mualani_stacks | 0.722 | [0.402, 0.910] |
| natlan_battleship | 0.222 | [0.063, 0.547] |

样本极小，CI 极宽，**不作强度结论**，仅证明矩阵/CI/Elo/落盘管线通。原始：`reports/arena/20261007T135215_arena_smoke.json`。

## 复现命令

```bash
python -m envs.benchmark --games 12 --workers 1
python -m eval.arena --smoke --seeds 3 --workers 12
python -m pytest -q -m "not slow"   # 快速；去掉 -m 跑全量（含慢测约 3 分钟）
```

## L4 SoG 玩具管线（2026-10-07）

`python -m train.pipeline --overfit`（1 局采集 → CVPN → 60 epoch 训练 → 对脚本对手评测 → 门禁 → 落盘）：
- 采集样本 94（含 action/choose_active/reroll/select/switch 五类请求），零报错完成。
- 训练集一致率 ~0.74（多数类 ~0.3）→ 网络可学；`reports/train/round_*_overfit.json`。
- checkpoint 落 `data/checkpoints/`（不入库）；`NeuralPolicy` 已能在真引擎里对脚本对手完成合法对局。

## L5.1 对手池全矩阵底座（2026-10-08）

20×20×50 局双向 = **40,000 局，0 error / 0 truncated**；12 worker 一次跑完（`resumed_matches=0`，
断点续跑机制未触发即完成）。原始记录 `data/arena/full_matrix_20x20.jsonl`（40,000 行，不入库）。

**吞吐修正（排产以此为准）**：12 worker 并行实测 **~4,900 局/h**（墙钟 ~8 小时）；每局并行口径
~8.8 秒/核，为单核 819 局/h 的 **~50% 并行效率**。上方"49 核·时 ≈ 12 核 4 小时"的线性外推偏乐观
约 2 倍，实际 ≈ 96 核·时；M0"1 万局/h"的核数换算相应修正为 ~25 核（并行口径）。

产物（含 `engine.lock` 元数据）：`reports/arena/20261008T163940_full_matrix_20x20.json` /
`.pairings.csv`（400 对位行 × 100 局/行，Wilson CI）。

强度基线（对池内对手，player 0 视角；完整 20 行见 summary json）：

| 卡组 | 胜率 | Wilson 95% CI | Elo |
|------|------|---------------|-----|
| unyielding_geo | 0.831 | [0.814, 0.847] | 1875 |
| double_geo_navia | 0.723 | [0.702, 0.742] | 1740 |
| dvalin_bonk | 0.733 | [0.713, 0.751] | 1734 |
| … | | | |
| ice_water_battleship | 0.186 | [0.170, 0.204] | 1092 |
| skirk_ayaka_navia | 0.181 | [0.165, 0.199] | 1213 |
| skirk_chasca_freeze | 0.175 | [0.159, 0.193] | 1148 |

对手池强度分层清晰（0.175–0.831），作为 M2/M4 门禁的固定底座合适。

**注意（2026-10-09，I10/D13）**：搜索分叉能力已钉死——`canResume:true` 边界暂停点的快照**即精确分叉**
（record-replay 决策序列复现活体终局 13/13，证据 `reports/engine/probe_boundary_fork_*.json`；早期
"不能分叉"结论系探针混淆，见 PLAN §0.1 I9/I10）。纪律：只在 `is_resumable()` 点取快照 + fork 补
游戏级 attrs（`envs/snapshot.fork_game`）；`canResume:false` phase 内部点不可分叉。批量离线 teacher
用 replay-branch MC（`train/replay_branch.py`）。

## L5.4 引擎桥实测 + replay-branch teacher（2026-10-09）

- 无 `njugit` 依赖，纯本地；证据与命令见 `scripts/probe_engine_snapshot.py`（`--all` 可扫全部快照）、
  `scripts/probe_boundary_fork.py`（record-replay 分叉保真判定）。
- `envs/snapshot.py`：`capture_snapshot` / `fork_game` / `snapshot_roundtrip_is_faithful` + `FORK_LIMITATION`。
- `train/replay_branch.py`：`capture_trajectory` / `replay_branch` / `evaluate_decision` / `aggregate_option_values`。
- 验收：`python -m pytest tests/test_engine_bridge.py -q`（含「注入=base 选择时逐局复现 base」分支确定性）。

## L5.2 LLM 矩阵 runner + 冒烟（2026-10-09）

把「纯策略 vs LLM 辅助」从单局小样本放大为**全对手池矩阵**，两模式分开报分（§6.3），结果逐局流式落盘 +
按 `task_index` 断点续跑 + spec 指纹护栏（D7）；聚合报告落 `reports/llm/matrix_*.json`，逐干预明细只在原始流。

`scripts/run_llm_matrix.py` 冒烟（3 对手 × 3 种子 × 2 模式 = 18 局，budget 6，`deepseek-chat`）：

| 对手 | 纯策略 | LLM 辅助 | Δ |
|------|--------|----------|---|
| natlan_battleship | 1.000 | 1.000 | +0.000 |
| double_geo_navia | 0.167 | 0.333 | +0.167 |
| skirk_chasca_freeze | 1.000 | 1.000 | +0.000 |
| **整体** | **0.722** [0.402,0.910] | **0.778** [0.453,0.937] | **+0.056** |

LLM：54 次调用、0 失败；干预改动作 45/54（83%）。样本极小、CI 宽，**不作强度结论**；与 L5.2 早期 5×10 冒烟
一致（0 API 失败，弱对局提升更明显）。产物：`reports/llm/matrix_<ts>_llm_matrix_smoke.json`。
放量到池内全部对手：`python -m scripts.run_llm_matrix --seeds 10 --budget 6`（隔夜；可断点续跑）。

## L5.3 CVPN 学习曲线（2026-10-09）

`train/learning_curve.py`：一份样本池（2 脚本 teacher × 12 种子 = 1277 样本），对每个
(网络规模 `d_model`, 数据量) 组合训练**全新**网络，记录 train/val 损失与一致率。

| d_model | n=128 | n=256 | n=512 | n=1024 |
|--------:|------:|------:|------:|-------:|
| 64 val_loss | 1.863 | 1.771 | 1.345 | **1.163** |
| 64 val_acc | 0.480 | 0.529 | 0.676 | 0.647 |
| 128 val_loss | 1.818 | 1.694 | 1.163 | **1.146** |
| 128 val_acc | 0.520 | 0.529 | 0.657 | 0.623 |

数据量↑ → val_loss 单调下降、一致率上升（64 维降幅最大：−0.70），**网络可学、数据有效**；
两档网络规模差异在小数据量下不明显，符合「本地仅玩具规模、全量放大待 GPU」的定位（M3）。
产物：`reports/train/learning_curve_<ts>.json/.csv`。复现：
`python -m train.learning_curve --games 12 --sizes 128 256 512 1024 --d-models 64 128`。

