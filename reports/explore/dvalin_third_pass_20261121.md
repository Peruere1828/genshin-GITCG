# dvalin 第三个 PASS：从"教师天花板"到全门禁通过（2026-11-21）

> 口径：`deck=dvalin_bonk`，纯策略（无 LLM、无测试期搜索），固定 20 套脚本池，双方各半、配对种子。
> 认证：`eval_protocol --strict-ci --baseline expert:dvalin_bonk`，**40 seeds / 1600 配对局**（seeds 10300–10339，全新）。
> 状态汇总见 `docs/STATUS.md §1/§6.8`；方法见 `docs/PLAYBOOK.md §7.7`；负结果见 `docs/failures/008–010`。

## 1. 结果

| 项 | 值 |
|---|---|
| contestant | `neural:data/checkpoints/r7_gb50_lr2e5_e2_s0.pt` |
| pooled | **0.7650**（n=1600，Wilson95 [0.744,0.785]） |
| 同 deck 专家 | 0.7344 |
| 配对 Δ | **+0.0306 [+0.0080,+0.0533]** |
| floor | **PASS**（`floor_mode=wilson_low_d25`，违规 0；豁免 `dvalin_bonk` 0.525/专家 0.500、`unyielding_geo` 0.300/专家 0.350） |
| 质量 | fallback 0 / error 0 / draw 0，mean_rounds 5.6 |
| 筛选复现 | 800 配对局（seeds 10200–10219）：pooled 0.774、Δ +0.045 [+0.014,+0.076] |

`pytest -q -m "not slow"` → exit 0；新增 `tests/test_game_batches.py`（6 例）、`tests/test_floor_exemption.py`（4 例）。

## 2. 走到这一步的四段证据（每段都改变了下一个动作）

| # | 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | 补 history（32 token）与 PTIE 特权 critic | 纯模仿下三臂 val_acc 0.8779/0.8769/0.8770；引擎配对 `A1−A0=+0.0000`、`A2−A1=+0.0000` | 两个通道**零增益**（`failures/008`）；但暴露 mean-pool 稀释（0.871→0.368）→ 零初始化旁路 |
| 2 | 定向补采（expert teacher） | 单 matchup +15pp（dgeo 0.50→0.65），整体 Δ 不可复现（0.75→0.25 跨种子集） | floor 的差距是**局部**的，但 40 局/matchup 无法判定（`failures/009`） |
| 3 | 1-epoch PG + KL 锚 | 锚把 seed 极差 7pp→0.48pp，但 Δ 不动（+0.021@800） | 卡在"≈5 局/批"的 advantage 方差（`failures/010`，已 superseded） |
| 4 | **整局 batch（≥50 局/步）+ 重新标定 lr** | 先量步长响应：1e-5→KL 0.005（不动）、2e-5→0.013（可用）、3e-5→0.173（掉 7pp）、1.5e-4×4ep→崩 | 取 lr 2e-5 + anchor 0.2 + 2 ep → **认证 PASS** |

关键机制：局内回报是常数，5 局一批时 advantage 的方差由"这批赢了哪几局"主导；≥50 局/批把它平均掉，
但每步梯度也强得多，**lr 必须按响应曲线重标**（沿用旧 lr 会把策略推崩）。

## 3. 复现命令

```bash
# 1) 从当前 BC 臂采集 2000 局自博弈（seeds 11000-11099）
python -m train.collect --deck dvalin_bonk --all-opponents \
  --seed-start 11000 --seed-count 100 --workers 12 \
  --teacher-spec neural:data/checkpoints/tg3_bc_e20_s0.pt#t=1.0 \
  --tag r6_dvalin_sp2000 --format compact --no-history --no-privileged-state

python -m scripts.datafirst_bc_store --replays data/replays/r6_dvalin_sp2000 \
  --store-cache data/store/r6_dvalin_sp2000.pt --val-every 10 --seed-start 11000

# 2) 整局 batch + KL 锚
python -m scripts.datafirst_bc_train --store-cache data/store/r6_dvalin_sp2000.pt \
  --init-from data/checkpoints/tg3_bc_e20_s0.pt --loss pg \
  --batch-games 50 --batch-samples-cap 8192 --epochs 2 --lr 2e-5 \
  --pg-anchor-kl 0.2 --pg-entropy 0.01 --grad-clip 1.0 --no-privileged \
  --out data/checkpoints/r7_gb50_lr2e5_e2_s0.pt

# 3) 认证
python -m scripts.eval_protocol --contestant neural:data/checkpoints/r7_gb50_lr2e5_e2_s0.pt \
  --contestant-deck dvalin_bonk --seeds 40 --seed-start 10300 --workers 12 --strict-ci \
  --baseline expert:dvalin_bonk --tag cert_r7_gb50_s10300
```

## 4. 注意事项（不要把这条结果外推过头）

1. **Δ 余量只有 0.023**（CI 下界 +0.008）：任何回归都可能把它推回 CI 含 0，复验必须用全新种子。
2. **pooled 与 floor 的阻塞 matchup 随种子集漂移**（同一臂 0.765–0.774；阻塞 matchup 在 dgeo/klee/ungeo 间轮换），
   所以认证一律 **40 seeds / 1600 配对局**，400–800 局只作筛选。
3. **D25 豁免是文档化规则**，实现在 `eval/scoreboard.evaluate_gates(..., expert_per_opponent=...)`；
   本 deck 的两个豁免 matchup 上专家自己分别是 0.500 / 0.350（Wilson low 0.393 / 0.255）。
4. 该臂是在 **expert 定向补采 + 自博弈**两批数据上训练的；换 deck 需重跑整条链，不能直接搬 checkpoint。
