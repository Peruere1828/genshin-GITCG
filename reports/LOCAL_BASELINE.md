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

**注意（2026-10-09 修正）**：搜索（continual resolving）不能靠 pybinding 分叉——快照 JSON 往返无损、
可载入续跑、克隆确定，但**不是活体决策点的忠实分叉**（续跑重放 phase；`canResume` 不可靠）。
证据 `reports/engine/probe_snapshot_*.json`（9/9 往返、9/9 克隆、1/9 复现活体轨迹）。已落 D12 兜底
replay-branch MC teacher（`train/replay_branch.py`）；真活体分叉桥（TS server）仍待做。详见
`train/README.md`、PLAN §0.1 I9。

## L5.4 引擎桥实测 + replay-branch teacher（2026-10-09）

- 无 `njugit` 依赖，纯本地；证据与命令见 `scripts/probe_engine_snapshot.py`（`--all` 可扫全部快照）。
- `envs/snapshot.py`：`capture_snapshot` / `fork_game` / `snapshot_roundtrip_is_faithful` + `FORK_LIMITATION`。
- `train/replay_branch.py`：`capture_trajectory` / `replay_branch` / `evaluate_decision` / `aggregate_option_values`。
- 验收：`python -m pytest tests/test_engine_bridge.py -q`（含「注入=base 选择时逐局复现 base」分支确定性）。
