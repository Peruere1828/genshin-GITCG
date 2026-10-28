# PLAYBOOK — 已认证的有效方法

> 只收“通过完整门禁”的方法。每条必须写：适用条件、证据、命令、停止条件。
> 失败方法见 `docs/failures/`；一次性实验证据见 `reports/`。

## 1. 逐 deck 的 PG 剥削阶梯（当前最强方法）

**适用条件**

- 固定 20 套脚本对手池；目标是**剥削**而不是 Nash。
- 只对 **parity+** 的 deck 开阶梯：底座 BC 对同 deck 专家的配对 Δ 下界 ≥ −0.02，且 `fallback/error = 0`。
- 结构性劣势 deck（Skirk 系、SA）不进阶梯；先换 deck 或补 teacher。

**方法**

1. 底座：该 deck 认证最优的 BC 臂（不是共享 anchor）。
2. 采集：当前策略 `#t=1.0` 自对弈 2,000 局（20 对手 × 100 seeds），每轮换新种子。
3. 更新：`--loss pg`（REINFORCE + value baseline + entropy 0.01 + grad-clip 1.0），1 epoch、lr 3e-5，从当前策略 `--init-from`。
4. 认证：`eval_protocol`，20 对手 × 10 seeds × 双向 = 400 配对局；**每轮换全新种子**；预注册 `CI_low ≥ −0.02` 且点估计 >0。
5. 重复 2–4，典型 3 轮；第 1–2 轮在 parity 附近多为噪声。

**证据**

| deck | 起点 | r1 | r2 | r3 | 最终 |
|---|---|---:|---:|---:|---:|
| `double_geo_navia` | 0.66–0.72 | +0.100 | +0.058 | **+0.188** | **0.880 / Δ +0.168**，800 配对，gate PASS |
| `unyielding_geo` | 0.82–0.83 | −0.040 | +0.043 | **+0.105** | **0.909 / Δ +0.055**，800 配对，gate PASS |
| `dvalin_bonk` | 0.7575 | +0.038 | +0.030 | +0.033 | 平台 +3pp，未过 CI |

来源：`reports/EXPLOIT_LADDER.md`、`reports/SPECIALISTS.md`、`reports/explore/plan_strength_2day_kaggle.md`。

**命令**

```bash
bash kaggle/grids/run_single_deck_rl.sh <deck> <tag> <teacher-ckpt> <collect-seed> 100 <eval-seed>
python -m scripts.specialist_table    # 生成 reports/SPECIALISTS.md
```

**停止条件**

- 连续两轮 Δ 不涨或为负 → 停，不要继续加轮。
- 底座不是 parity+ → 先修底座，不要开 PG。
- 同种子对照不复现 → 视为种子运气，不进 PLAYBOOK。

## 2. 逐 deck 的稠密 BC 底座

**适用条件**

- 每个 deck 独立训练；不要 20-deck 单模型（见 failures/004）。
- 语料是 expert-vs-expert，且每 matchup 有足够局数；episode 预算 ≥15（6 ep 会被欠训练截断）。

**证据**

- 3/7 组首次过 pooled ≥0.50（strict cluster CI）：dvalin 0.701 / ungeo 0.709 / dgeo 0.689。
- BC 从未显著超过同 deck 专家；最好统计打平。
- 数据量 8k→48k（同 epoch）不显著；5→15 ep 有显著增益（+0.076）。
- 来源：`reports/datafirst_bc.md`、`reports/train/bc_scale_datafirst.md`。

**命令**

```bash
python -m train.collect --deck <deck> --all-opponents --seed-start 0 --seed-count 80 \
  --workers 8 --tag dfbc_<deck>_c80 --format compact --no-privileged-state
python -m scripts.datafirst_bc_store --replays data/replays/dfbc_<deck>_c80 \
  --store-cache data/store/<deck>.pt --val-every 10
python -m scripts.datafirst_bc_train --store-cache data/store/<deck>.pt \
  --d-model 256 --epochs 15 --train-limit 50000 --val-limit 4000 \
  --save-best --out data/checkpoints/<deck>_d256.pt
```

**停止条件**

- 配对 Δ 显著低于专家 → 先加 epoch/密度，不要换更大的网络。
- val_acc 涨但引擎不涨 → 回到 parity/特征检查，不要继续调结构。

## 3. 固定评测协议（所有强度声明必须走它）

- `scripts/eval_protocol.py`：固定 20 脚本池、配对种子、双方各半、Wilson + opponent-cluster CI、floor、Elo、exploitability 下界。
- 认证：≥20 seeds（800 局）只判 ≥6pp；非劣/4–5pp 用 1,400–1,600 局；3pp 需 2,000–3,000 局。
- 预注册：假设 / 臂 / 主指标 / 样本量 / 停止条件写在实验前；不在同一批种子上选型又报分。
- **不要用 `val_acc` 做 model selection**；A2 的 val_acc 与引擎胜率可完全反向。

**命令**

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.eval_protocol --contestant neural:<ckpt> --contestant-deck <deck> \
  --seeds 20 --seed-start <fresh> --workers 10 --strict-ci \
  --baseline expert:<deck> --tag <tag>
python -m scripts.compare_eval_runs <tag_a> <tag_b>
```

协议全文：`docs/EVAL_PROTOCOL.md`。

## 4. Parity 门禁（新结构训练前）

- 同一 `(deck, opponent, seed, step)` 对比 replay/store 与 live engine 的 token/option；definition 列按**整数 id** 比。
- 旧 compact fp16 语料会失败（152 卡→28 id）；新 compact 必须带 exact int id sidecar。
- 在 parity PASS 前，冻结新结构训练（A3/A4/A5/id/FiLM/candidate attention）。

**命令**

```bash
python -m scripts.parity_store_live --replay-dir data/replays/<tag> \
  --deck <deck> --opponent <opponent> --seed <seed> --limit 5 --strict
python -m pytest tests/test_feature_parity.py -q
```

证据：`reports/explore/p0_parity_20261027.md`。

## 5. 可续跑采集 / compact 数据平面

- `train.collect`：一行一局、manifest 指纹护栏、按 `task_index` 断点续跑；父进程流式落盘，不再线性吃内存。
- compact：fp16 base64（体积 ~0.54×/0.45×）；**新语料必须带 exact definition id**（`token_definition_ids`/`option_definition_ids`）。
- 采集/训练解耦：本机/WSL 采集 → 存储 → A800 GPU 训练；A800 不做采集/评测（D21）。

**命令**

```bash
python -m train.collect --deck <deck> --all-opponents --seed-start 0 --seed-count 20 \
  --workers 8 --tag <tag> --format compact --no-privileged-state
python -m scripts.datafirst_bc_store --replays data/replays/<tag> --store-cache data/store/<tag>.pt
```

## 6. NPC PvE 调试系统（独立门禁）

- 定位：调试/训练辅助，不进口径验收。
- 门禁 PvE-v1：decklab 组牌 × 16 关 × 多种子，三星 ≥60%；当前 PASS 74.7%，但支援区/多轮挑战与游戏内机制不一致需标注。
- 入口：`npc/README.md`、`configs/npc/README.md`、`docs/howto/npc-pve.md`。

## 7. Web 人机调试接口（独立线）

- 定位：对 AI 的调试接口，不进口径验收。
- 能力：standalone overlay、浏览器↔Python 桥、按 `(session, who)` 隔离、`/notify` seq 护栏、反选、NPC 阵容、续局、逐决策日志。
- 入口：`web/README.md`、`docs/howto/web-debug.md`。

## 8. 不要写进 PLAYBOOK 的东西

- 只有一次 run、没有独立种子复验的 +Δ。
- `val_acc`/训练损失涨、引擎不涨的“改进”。
- AWR/Q/DMC/搜索蒸馏/胜局筛选/matchup 重加权/20-deck 单模型（均已有负结果，见 `docs/failures/`）。
- 未过 parity 的新结构（id、FiLM、candidate attention）。
