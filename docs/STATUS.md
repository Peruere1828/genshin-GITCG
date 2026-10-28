# STATUS — 当前状态（2026-10-27）

> 只写“现在有效”的事实。历史与一次性结果见 `reports/`；失败归因见 `docs/failures/`。
> 所有胜率默认是**纯策略**、固定 20 套脚本池、双方各半、配对种子；门禁口径见 `docs/EVAL_PROTOCOL.md`。

## 1. 当前最强的已认证策略

| deck | arm | pooled | 同 deck 专家 | 配对 Δ [95% CI] | 配对数 | floor 违规 | 门禁 |
|---|---|---:|---:|---:|---:|---:|---|
| `double_geo_navia` | `final20_dgeo` | **0.8800** | 0.7125 | **+0.1675** [+0.134,+0.201] | 800 | 0 | **PASS** |
| `unyielding_geo` | `final20_ungeo` | **0.9087** | 0.8538 | **+0.0550** [+0.027,+0.083] | 800 | 0 | **PASS** |
| `dvalin_bonk` | `rl_dvalin1_dvalin_bonk` | 0.7800 | 0.7425 | +0.0375 [−0.009,+0.084] | 400 | 1 | fail（平台期 +3pp） |
| `shenhe_double_swirl` | `rl_shenhe1_...` | 0.7025 | 0.6650 | +0.0375 [−0.014,+0.089] | 400 | 2 | fail |
| `electro_manifestation_nuke` | `rl_electro1_...` | 0.6850 | 0.6550 | +0.0300 [−0.025,+0.085] | 400 | 2 | fail |

- 19 个 deck 有认证臂；**2 个完整门禁 PASS**（dgeo/ungeo）；6 个达到专家 parity 或更好。
- 生成源：`python -m scripts.specialist_table` → `reports/SPECIALISTS.md`；方法见 `docs/PLAYBOOK.md`。
- `dvalin` 的 +3.75pp 在 10 seeds 上 CI 含 0，不能写成“超过专家”；只算 plateau。

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

## 7. 相关入口

- 有效方法：`docs/PLAYBOOK.md`
- 失败/重开：`docs/failures/README.md`
- 机器环境：`docs/environments/README.md`
- 评测协议：`docs/EVAL_PROTOCOL.md`
- 决策索引：`docs/decisions/README.md`
