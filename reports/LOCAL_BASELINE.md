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

换算（用于 M0/M4 排产；并行口径以 L5.1 实测为准）：
- M0 "1 万局/小时" ≈ **13 核**（单核口径）/ **~25 核**（并行口径）。
- 100 万局自博弈 ≈ **~1,220 核·时**（单核口径）/ **~2,400 核·时**（12 核并行口径）。
- 20 套对手全矩阵、每对局 50 局、双向 = 20×20×50×2 = 40000 局 ≈ **49 核·时**（单核口径）/ **~96 核·时**（并行口径，12 核 ~8 小时，隔夜充裕）。

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

**分叉纪律（D13）**：只在 `is_resumable()` 点取快照 + fork 补游戏级 attrs（`envs/snapshot.fork_game`）；
`canResume:true` 边界快照即精确分叉（record-replay 决策序列复现活体终局 13/13，证据
`reports/engine/probe_boundary_fork_*.json`）。批量离线 teacher 用 replay-branch MC（`train/replay_branch.py`）。

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

LLM：54 次调用、0 失败；干预改动作 45/54（83%）。样本极小、CI 宽，**不作强度结论**；与 L5.2 的 5×10 冒烟
一致（0 API 失败，弱对局提升更明显）。产物：`reports/llm/matrix_<ts>_llm_matrix_smoke.json`。
报告区分 `llm_quality`：客户端失败的比赛逐局标 `degraded` 并单列（`clean` 为非降级胜率）；
连续 N 次（默认 5，`--max-consecutive-errors`）客户端失败即中止，已完成的流式行保留，重跑续跑——
避免一次网络中断把 llm 列的后半段悄悄变成纯策略测量。
放量到池内全部对手：`python -m scripts.run_llm_matrix --seeds 10 --budget 6`（隔夜；可断点续跑）。

## L5.3 CVPN 学习曲线（2026-10-09）

`train/learning_curve.py`：采集一份样本池（2 脚本 teacher × 14 种子 = 1465 样本），**先固定一份验证
留出集（20%）**，再对每个 (网络规模 `d_model`, 训练样本量) 组合在**同一留出集**上训练/评测一个全新网络。
固定 holdout 使各数据点直接可比。

| d_model | 128 | 256 | 512 | 1024 |
|--------:|----:|----:|----:|-----:|
| 64 val_loss | 2.107 | 1.845 | 1.343 | **1.106** |
| 64 val_acc | 0.495 | 0.590 | 0.625 | **0.665** |
| 128 val_loss | 2.041 | 1.746 | 1.257 | **1.083** |
| 128 val_acc | 0.468 | 0.584 | 0.621 | **0.662** |

数据量↑ → val_loss 单调下降、val_acc 单调上升（64 维 val_loss 降 ~1.0），**网络可学、数据有效**；
两档网络规模差异在小数据量下不明显，符合「本地玩具~中等规模、全量放大并入 M3 GPU 档」的定位。
产物：`reports/train/learning_curve_<ts>.json/.csv`（含 `n_train_pool`/`n_val_holdout`）。复现：
`python -m train.learning_curve --games 14 --sizes 128 256 512 1024 --d-models 64 128`。

## WSL / MX550 设备与训练对照（D14，2026-10-11）

WSL 侧（4 核 / 8G / MX550 2G）装 **torch 2.6.0+cu124**（PyPI 默认 `torch 2.14.1` 为 CUDA 13 构建，
与 WSL 驱动 560.94 的 CUDA 12.6 不匹配，故用 cu124 档）。MX550（cc 7.5、2.15 GB）接电充满时满速：

| 指标 | 满速实测 |
|------|---------|
| P-state / 显存时钟 | P0 / 7000 MHz（最大 7001） |
| 功率 | ~39 W（墙 40 W；default 40 / max 60） |
| 显存带宽 | copy ~101 GB/s、读 ~109 GB/s |
| matmul 2048³ | 2.64 TFLOPS |
| 纯 FP32 ALU | ~3.6 TFLOPS（满速，SM ≈1.8 GHz） |

**电源保障（正确做法）**：MX550 的可用功率由**整机供电**决定——
- **接电 + 电池充满（不在充电）**：dGPU 满速 ~40 W，这是默认工作状态；
- **电池充电中**：EC 给充电让电，dGPU 降到 ~15 W；高负载 + 屏幕亮 + 供电不足会叠加触发降功率。

故用 WSL 跑 GPU 训练时应**保持接电、电池充满**，并在 Windows 关闭「限充到 80%」类设置。

**CVPN 训练对照**（同池 615 样本、d_model 64/128、15 epoch，`scripts/bench_device.py`）：

| d_model | CPU(4核) | CUDA(MX550) | 加速 |
|--------:|---------:|------------:|-----:|
| 64 | 6.40s | 3.57s | **1.79×** |
| 128 | 7.39s | 3.00s | **2.46×** |

train/val 指标两设备一致（val_loss/val_acc 相同）。产物 `reports/train/device_bench_<ts>.json`。
复现：`python -m scripts.bench_device --games 6 --d-models 64 128`。

**用法**：WSL = 「4 核并行采集/评测 + 满电时 GPU 训练/推理」双用节点；GPU 训练用 `--device cuda`
（`train.pipeline` / `train.learning_curve` / `scripts.bench_device`），device 默认 cpu。压测工具：
WSL `nvcc -O3 -arch=sm_75 -o gpu_stress scripts/gpu_stress.cu`；Windows 原生 `python scripts/gpu_stress_native.py`。

### WSL / MX550 CVPN 学习曲线（L5.3，2026-10-11，满电）

在 WSL 用 `--device cuda` 跑一轮放大学习曲线（`python -m train.learning_curve --games 40 --sizes
256 512 1024 2048 --d-models 64 128 256 --epochs 30 --device cuda`）：教师采集 4099 样本（2 脚本对手 ×
40 种子），固定留出 20%，每点训练一个全新网络（12 点；采集 ~1 核·时，训练全在 MX550 上）：

| d_model | n=256 | n=512 | n=1024 | n=2048 |
|--------:|------:|------:|-------:|-------:|
| 64  val_loss | 1.198 | 1.127 | 1.005 | **0.922** |
| 128 val_loss | 1.177 | 1.131 | 0.991 | **0.893** |
| 256 val_loss | 1.146 | 1.056 | 0.957 | **0.861** |
| 64  val_acc  | 0.602 | 0.600 | 0.709 | **0.702** |
| 128 val_acc  | 0.604 | 0.665 | 0.705 | **0.717** |
| 256 val_acc  | 0.597 | 0.614 | 0.719 | **0.723** |

数据量↑ → val_loss 单调下降、val_acc 单调上升；同数据量下 d_model 越大越好（4099 样本、d_model 256 时
val_loss 0.861 / val_acc 0.723，明显强于本机 1465 样本的 64 维 1.106 / 0.665）。产物
`reports/train/learning_curve_20261011T124632.json/.csv`。复现同上命令；WSL 满电（接电、电池充满）时全在
MX550 上训练，CPU 只做采集。

