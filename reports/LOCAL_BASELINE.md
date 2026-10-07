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
python -m pytest -q -m "not slow"   # 快速；去掉 -m 跑全量（含慢测约 2 分钟）
```
