# STATUS — 当前状态（2026-11-21）

> 只写“现在有效”的事实。历史与一次性结果见 `reports/`；失败归因见 `docs/failures/`。
> 所有胜率默认是**纯策略**、固定 20 套脚本池、双方各半、配对种子；门禁口径见 `docs/EVAL_PROTOCOL.md`。

## 1. 当前最强的已认证策略

| deck | arm | pooled | 同 deck 专家 | 配对 Δ [95% CI] | 配对数 | floor 违规 | 门禁 |
|---|---|---:|---:|---:|---:|---:|---|
| `double_geo_navia` | `final20_dgeo` | **0.8800** | 0.7125 | **+0.1675** [+0.134,+0.201] | 800 | 0 | **PASS** |
| `unyielding_geo` | `final20_ungeo` | **0.9087** | 0.8538 | **+0.0550** [+0.027,+0.083] | 800 | 0 | **PASS** |
| `dvalin_bonk` | `r7_gb50_lr2e5_e2_s0` | **0.7650** | 0.7344 | **+0.0306** [+0.008,+0.053] | **1600** | 0（D25 豁免 2 个） | **PASS**（2026-11-21） |
| `shenhe_double_swirl` | `rl_shenhe1_...` | 0.7025 | 0.6650 | +0.0375 [−0.014,+0.089] | 400 | 2 | fail |
| `electro_manifestation_nuke` | `rl_electro1_...` | 0.6850 | 0.6550 | +0.0300 [−0.025,+0.085] | 400 | 2 | fail |

- 19 个 deck 有认证臂；**3 个完整门禁 PASS**（dgeo / ungeo / **dvalin**）；6 个达到专家 parity 或更好。
- dvalin 的 PASS 是 2026-11-21 的**第五轮**：game-level batch（≥50 局/步）+ KL 锚 0.2 + lr 2e-5，
  40 seeds / 1600 配对局，pooled 0.765、D25 floor 干净、Δ +0.0306 CI 不含 0。见 §6.8。
- 生成源：`python -m scripts.specialist_table` → `reports/SPECIALISTS.md`；方法见 `docs/PLAYBOOK.md`。
- 历史：`dvalin` 曾长期停在 +3.75pp 平台期（CI 含 0）；2026-11-21 用 game-level batch 突破，见 §6.8。

## 2. 通用 BC 基线（A0）

| deck | A0 pooled | 说明 |
|---|---:|---|
| `double_geo_navia` | 0.720 | `h4m e150` 4-deck 通用 BC；seed-900（旧协议实际 1800–1809），400 配对 |
| `unyielding_geo` | 0.828 | 同上 |
| `dvalin_bonk` | 0.760 | 同上 |

- **A2（identity embeddings）判负无效**：compact fp16 把 definition id 压成不可逆假 id；A0i 的 val_acc 0.866 / 引擎 dgeo 0.266 是数据事故，不是 id 路线结论。见 D25、`reports/explore/p0_parity_20261027.md`。
- A0 仍是当前 4-deck 通用 BC 基线；不要用 A0i/A0i_noid 当通用底座。

## 3. P0 状态（2026-10-27）

| 项 | 状态 | 证据 |
|---|---|---|
| parity harness（store↔engine 逐元素） | ✅ | `envs/parity.py`、`scripts/parity_store_live.py`、`tests/test_feature_parity.py` |
| exact definition id（compact int sidecar） | ✅ | 旧 fp16 2 决策 10 token/15 option id 失配；新 4-deck 85,573/85,573 决策有 exact id |
| 干净 4-deck 语料 / store | ✅ | `data/store/p0_clean_four_decks.pt`（1,600 局 / 74,407 train / 11,166 val） |
| `--save-best`（val_ce）+ last | ✅ | `datafirst_bc_train --save-best`；A2 best ep70 不再丢 |
| init / eval / 温度采样 seed | ✅ | 模型构造前 seed；`#t>0` 用 per-policy `torch.Generator`；`--seed-start` 不再双倍偏移 |
| 价值语义 / dice | ✅ | `PriorResolver.value_probability()`；identity→概率；acting→p0；dice i.i.d. |
| A2 重做（150ep，dgeo/dvalin） | ⏳ 等 A800 | `scripts/p0_a2_rerun.sh` 已封装 |
| seat 覆盖 A/B（p0-only vs 双边） | ⏳ 未跑 | f07 已确认 store `acting_player` 1128/1128 为 p0；需 200 局 A/B |

## 4. 当前排产

| 优先级 | 实验 | 目的 | 预算 | 停止条件 |
|---|---|---|---|---|
| P1 | DAgger 诊断（200 局，0 训练） | 模型状态 vs 专家状态一致率 gap | ~0.5h | gap <3pp → 停 |
| P1 | 可修 matchup 靶向采集 pilot（nahida × 1 target） | floor 修补 | 1,400 采集 + 960 评测，2–3h | target Δ <+10pp → 停 |
| P1 | specialist 扩张（dvalin/shenhe/electro） | 把 PASS 从 2 个扩到更多 deck | 18k 采集，6–10h CPU | 3 轮 0/3 → 停 |
| P1 | portfolio/router（0 新训练） | 现有 arm 的 floor 路由 | 12k 评测 | routed <最佳单 arm +3pp → 不做 |
| P2 | league/PFSP + anchor | specialist 防漂移 | 每轮 2k 局 | 非劣 CI 下界 <−0.02 → 回滚 |
| P2 | value 拆头 + 搜索重开 | 修复 D23 的测量问题后再判 | 1 次训练 + 3.2k 局 | parity/价值门禁不过 → 不重开 |
| P3 | 候选 self-attention probe | 只在 0 成本诊断有信号时做 | 0 新采集 | store 涨引擎不涨 → 停 |

## 5. 评测与统计口径（当前）

- **筛选**：400 配对局（10 seeds）只抓大效应；不能做“非劣”判定。
- **确认/非劣**：1,400–1,600 配对局；`CI_low ≥ −0.02` 需要 ~1,400 局。
- ** superiority 5pp**：20–28 seeds（800–1,120 局）；3pp 需 2,000–3,000 局。
- **floor**：绝对 floor + 相对同 deck 专家的配对 Δ；teacher-limited matchup（expert Wilson low <45%）不要求绝对 45%（D25）。
- **model selection**：`val_ce` early stop + 小样本配对引擎 screen + 冻结认证种子；**不看 val_acc**。

## 6. 阻塞与已知风险

1. **A800 不可随时可用**：A2 150ep 重做、GPU 放大训练依赖窗口；采集/评测不要放窗口（D21）。
2. **本机 CPU 是共享资源**：采集/评测/训练不要同时开满；评测 worker 设 `OMP_NUM_THREADS=1`。
3. **旧 compact 语料不能用于 id 特征训练**：缺 exact id；只能用于非 id 模型或重采。
4. **P0-E seat 覆盖未验证**：side=1 的分布偏移可能仍在；在 A/B 前不要把 side=1 的强度差异归因于模型。
5. **文档迁移已落地主体**：`docs/` 骨架、`AGENTS.md`/`README.md`、`STATUS`/`PLAYBOOK`/`EVAL_PROTOCOL`、
   `failures/`、`environments/`、`decisions/`、`howto/` 已建；旧报告保留在 `reports/`。
   仍待持续清理：模块 README 的历史状态、全部报告的 front-matter、`PLAN §10` 之外的旧状态句。

## 6.5 D30 gap 收口（2026-11-20）

| 项 | 状态 | 证据 |
|---|---|---|
| 公开动作历史通道（32 token，此前恒空） | ✅ 已接通，**纯模仿下零增益** | 采集/推理同源 `MatchHistory`；`docs/failures/008`：配对 Δ `A1−A0 = +0.0000 [−0.030,+0.030]` n=400 |
| 新输入通道的接入方式 | ✅ 结构分支（历史行单独池化 + 零初始化加性投影） | 直接 mean-pool 会把 teacher val_acc 0.871→0.368；分支后逐位回到 0.871 |
| PTIE 特权 critic（`value = value_head(g + priv_proj(x))`） | ✅ 已接通，**策略零影响、价值泛化反而略差** | `A2−A1 = +0.0000 [−0.016,+0.016]`；val_value 0.1244→0.1274 |
| checkpoint 输入格式标记 | ✅ | `encoder_config['include_history']`；旧臂无标记 → 继续无 history 推理，历史数字可比 |
| Kaggle 训练面（`hist1` 9 runs, T4×2, 20.6 min） | ✅ 跑通 | `build/kaggle/out_hist1/summary.json` |
| dvalin 第三个 PASS | ❌ 未达成 | floor 两个结构弱 matchup（0.35@20 局，n=400）；见 `failures/008` 与预注册 §4.1 |

**当前 2 个 PASS 未变**（`double_geo_navia` 0.880、`unyielding_geo` 0.909）。

### 6.6 dvalin 第三次认证（2026-11-20，40 seeds / 1600 配对局）

定向补采（expert teacher，`unyielding_geo` / `shenhe_double_swirl` / `dvalin_bonk` /
`double_geo_navia` 各 200 局）+ BC 20ep，Kaggle `hist4`：

| 项 | 数值 | 判定 |
|---|---:|---|
| pooled（seeds 9900–9939） | **0.7725** | ≥0.75 ✅ |
| floor（D25：专家自己 Wilson low <0.45 的 matchup 豁免） | 无违规 | ✅（`unyielding_geo` 0.35 / 镜像 0.55 均豁免；`double_geo_navia` 0.61 > 专家 0.54） |
| 配对 Δ vs `expert:dvalin_bonk` | **+0.019** [−0.003,+0.041] | ❌ CI_low < 0（差 0.3pp） |

即 **dvalin 已经达到绝对强度 + floor 两关，只差"显著高于专家"这一条**（+0.3pp）。
注意种子集敏感：同一 deck 的 teacher 在 seeds 9600 上是 +0.040 [+0.010,+0.070]。

### 6.7 再走一轮自博弈 PG（2026-11-21，`failures/010`）

在 `tg3_bc_e20_s0` 上用 1,200 局新自博弈数据做 `--loss pg --pg-anchor-kl 0.1`、3 seeds：

| 项 | 结果 |
|---|---|
| 训练稳定性（锚） | val_acc 0.8454 / 0.8502 / 0.8492，极差 **0.48pp** |
| 筛选（seeds 10100–10119, 800 局） | pooled **0.721**，Δ **+0.021 [−0.012,+0.055]** |
| floor（D25） | 违规 `klee_mavuika_mualani`（0.575 vs 专家 0.650） |

结论：**该轮没有越过 parity**（1-epoch、≈5 局/批），下一步换更新方式——见 §6.8（已做到）。

**门禁修复（本轮）**：`eval/scoreboard.evaluate_gates` 此前**没有实现 D25 豁免**，
把"专家自己也过不了"的 matchup 记成违规；现已实现（`floor_mode="wilson_low_d25"`、
`floor_exempt` 字段、`expert_per_opponent` 入参），并加测试 `tests/test_floor_exemption.py`。

### 6.8 第三个 PASS：game-level batch 的 PG（2026-11-21）

按 §6.7 的重开条件换的是**更新方式**，不是种子：

1. `--batch-games 50`：按 `(deck, opponent, seed)` 聚成整局，每步 ≥50 局（旧配方 256 样本 ≈5 局）；
2. 先量出**步长响应曲线**（同一份 2,000 局自博弈数据）：
   lr 1e-5 → KL 0.005（不动）、2e-5 → KL 0.013（稳）、3e-5 → KL 0.173（掉 7pp val_acc）、
   1.5e-4×4ep → val_acc 0.28–0.49（崩）。
3. 取 lr 2e-5 + `--pg-anchor-kl 0.2` + 2 epochs，`--init-from tg3_bc_e20_s0.pt`。

认证（40 seeds / 1600 配对局，seeds 10300–10339，全新）：

```
pooled=0.765  wilson95=[0.744,0.785]
gates: pooled=True floor=True clean=True -> PASS      (floor_mode=wilson_low_d25)
paired vs baseline: delta=+0.031 95%=[+0.008,+0.053] n=1600
floor_exempt: ['dvalin_bonk', 'unyielding_geo']   # 专家自己也只有 0.350 / 0.500
```

即 **dvalin 首次同时满足三项**：pooled ≥0.75、非豁免 matchup Wilson low ≥0.45、配对 Δ CI 不含 0。
PASS 集合 **2 → 3**（dgeo、ungeo、dvalin）。

**仍有效的两个方法**（写入 PLAYBOOK）：
1. 新输入通道一律走**零初始化加性旁路**（§7.5）；
2. 1-epoch PG 一律加 **`--pg-anchor-kl 0.1`**（`KL(π‖π_warmstart)`）：3 seed 的 val_acc 极差从
   7pp 降到 **3pp**，引擎 Δ 从 −0.128…+0.015 收敛到与 teacher 一致的 +0.040。

## 7. 相关入口

- 有效方法：`docs/PLAYBOOK.md`
- 失败/重开：`docs/failures/README.md`
- 机器环境：`docs/environments/README.md`
- 评测协议：`docs/EVAL_PROTOCOL.md`
- 决策索引：`docs/decisions/README.md`
