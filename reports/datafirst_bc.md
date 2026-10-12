# 数据先行重采 + BC 认证（D23；2026-10-12，A7 子代理）

> 口径：**纯策略**（不接 LLM，PLAN §6.3）。机器：本机 **6 物理核 / 12 逻辑、30 GB**，
> 无 GPU（torch CPU）。**与另一会话并行**（其同等管线 `bc_learning_curve` 实测 177% CPU），
> 墙钟被挤占 ~2–3×；峰值并发触发过 **OOM（worker rc=-9）**。本文所有吞吐/时间均为**受竞争下**的实测。
> 原始逐局数据 / checkpoint / store 全在 `data/`（gitignore）；聚合在 `reports/`。
> **并发会话**（A5 续）在同仓跑同一数据先行管线（其报告 `reports/train/bc_scale_datafirst.md`）；
> 两会话的 SA-15ep 结论**互相印证**（见 §4 末），本文仅报告本会话产出的多 deck 语料与逐 deck 认证。

---

## 0. 结论速览（诚实数字）

| deck | BC pooled | expert 同 deck | 配对 Δ [95% CI] | pooled≥0.5 strict-CI | M4 floor | seeds/epochs |
|---|---:|---:|---:|:--:|:--:|--:|
| `dvalin_bonk` (d256) | **0.701** | 0.736 | **−0.034** [−0.067,−0.001] | ✅ | ✗(7/20) | 20 / 12 |
| `unyielding_geo` (d256) | **0.709** | 0.799 | **−0.090** [−0.137,−0.043] | ✅ | ✗(6/20) | 10 / 6 |
| `double_geo_navia` (d256) | **0.689** | 0.681 | **+0.007** [−0.045,+0.060] | ✅ | ✗(8/20) | 10 / 6 |
| `superconduct_aggro` (d256) | 0.405 | 0.509 | −0.104 [−0.161,−0.047] | ✗ | ✗(18/20) | 10 / 6 |
| `superconduct_aggro` (d256, **15 ep**) | **0.480** | 0.509 | −0.029† [−0.081,+0.023] | ✗ | ✗(15/20) | 10 / 15 |
| `superconduct_aggro` (d128) | 0.395 | 0.509 | −0.114 [−0.170,−0.057] | ✗ | ✗(16/20) | 10 / 6 |
| `natlan_battleship` (d256) | 0.140 | 0.225 | −0.085 [−0.127,−0.043] | ✗ | ✗(20/20) | 10 / 6 |

† SA-15ep 本会话 10 seeds（功效不足）；**并发会话 20 seeds 得 −0.061 显著负**（见 §4 末）。

**判定**：
- **BC 从未「显著超过」同 deck 脚本专家**：`double_geo_navia` 统计**打平**（Δ=+0.007，CI 含 0）；
  `superconduct_aggro`(15 ep) 本会话 10-seed Δ=−0.029（CI 含 0），但**并发会话 20-seed 复跑为 −0.061 显著负**（见 §4 末）
  → SA 实际仍落后专家 ~6 pp。其余臂显著低 3.4–11.4 pp。
  → 与 A3/A4「网络尚未追上专家」一致；**数据先行的价值在“把 BC 顶到专家附近”而非“超过专家”**。
- **3/7 臂首次通过 M3-CPU 的 pooled 门禁（≥0.50，strict cluster CI）**：`dvalin/ unyielding/ double_geo`
  均落在 **0.689–0.709** —— 远高于 A3 全池 ~0.50 与 A4 专家 `superconduct_aggro` 0.561。
  **差距来自 deck 选择**：这些 deck 的**专家本身**就强（0.68–0.80），BC 能跟到 −0.3~ −9 pp。
- **epoch 预算是被低估的变量**：`superconduct_aggro` 6 ep→15 ep 从 **0.405→0.480**（配对 gap 从 −0.104*
  缩到 −0.029/20-seed −0.061）。**本报告其余 deck 只跑 6–12 ep，其强度是下界**（尤其 6 ep 的 unyielding/double_geo/natlan）。
- **全部 deck 的 M4 floor（无单套 <45%）仍 FAIL**：长尾（岩/雷/草系）是**结构性**瓶颈，未随数据/规模收敛。
- **SA 的瓶颈是 epoch 而非容量**：d128≈d256（0.395 vs 0.405），但 6→15 ep +0.075 → 需足够 epoch 预算（见 §9）。

---

## 1. 盘点与复用（`data/replays/`，采集前快照）

| tag | deck | 对手×种子 | 局数 | 格式 | 模式 | 体积 | 结论 |
|---|---|---:|---:|---|---|---:|---|
| `datafirst_bc_sa500` | superconduct_aggro | 20×500 | 3004 | jsonl | BC | 4.2 G | **复用**（SA 前 7 对手，各 ~500） |
| `df_*`（14 个） | superconduct_aggro | 1×500 | 346–448/个 | jsonl | BC | ~0.45 G/个 | **复用**（11 个不同对手，各 ~320–450） |
| `a3_corpus_v1` | superconduct_aggro | 20×8 | 160 | jsonl | **search** | 0.21 G | 不用（D23：搜索 teacher 有害） |
| `distill_smoke*`, `probe_bc` | superconduct_aggro | — | 2–20 | jsonl | 混合 | <40 M | 太小，丢弃 |

**复用清单**：SA（`superconduct_aggro`）已由 A5 探针覆盖**全 20 池**（每对手 ≥ ~280 局，多数 ~350–500），
去重后 **8,061 局 / 372,270 决策**（`data/store/sa_all.pt`，6.4 GB）。→ **SA 无需重采**。
**缺口**：除 SA 外的所有 meta deck 无数据 → 本次重采 4 套。

> 注：A5 的 `df_*` 在本会话期间仍被其自身循环**追加**（峰值 14×1 worker 并发）；本盘点为采集前静止快照，
> 去重按 `(opponent, seed)`，重跑不受影响。

---

## 2. 重采（A6 紧凑格式，多套 deck × 全 20 池 × 80 seeds）

命令模板（可续跑，逐局流式落盘）：
```bash
python -m train.collect --deck <deck> --all-opponents \
  --seed-start 0 --seed-count 80 --workers 8 \
  --tag dfbc_<deck>_c80 --format compact --no-privileged-state
```

| tag | deck | 局数 | errors | 体积 | MB/局 | 对手 | seeds/对手 |
|---|---|---:|---:|---:|---:|---:|---:|
| `dfbc_dvalin_bonk_c80` | dvalin_bonk | 1600 | 0 | 894 MB | 0.559 | 20 | 80 |
| `dfbc_natlan_battleship_c80` | natlan_battleship | 1600 | 0 | 1169 MB | 0.731 | 20 | 80 |
| `dfbc_unyielding_geo_c80` | unyielding_geo | 1600 | 0 | 1031 MB | 0.645 | 20 | 80 |
| `dfbc_double_geo_navia_c80` | double_geo_navia | 1600 | 0 | 1079 MB | 0.675 | 20 | 80 |
| **合计** | 4 deck | **6400** | **0** | **4.17 GB** | **0.652** | — | 80 |

- **体积**：compact fp16 + 丢 `privileged_state` → **0.56–0.73 MB/局**（长局 deck 更高），均值 **0.65 MB/局**；
  同数据 jsonl 约 1.2–1.5 MB/局（A6 实测 0.54×/0.45×，此处一致）。**无 `privileged_state`**（无任何训练/评测路径消费）。
- **吞吐（实测）**：
  - **干净窗口 4,248 局/h**（8 worker，150 s 采样）——与 AGENTS 的 12 核 ~4,900 局/h 同量级；
  - 单 deck 完整跑：`dvalin` 1600 局 / 26 min ≈ **3,700 局/h**；`natlan` 1600 / 40 min ≈ **2,400 局/h**；
  - **受竞争平均 ~1,450 局/h**（与 A5 的 14×1 worker 并发时；6400 局墙钟 ~4.4 h）。
  - 结论：**紧凑格式下 8 worker 稳态 ≥ 3,700 局/h**；瓶颈是 6 物理核被跨会话共享，不是格式。
- 复现/断点：同命令重跑自动跳过已完成局（`manifest.json` 指纹护栏）。

---

## 3. CPU 档训练（每 deck 一个 d256；SA 另 d128）

紧凑 store 由本会话新增的 `scripts/datafirst_bc_store.py` 构建（**吃 A6 compact**，A5 的 `train_bc_scale`
流式 loader 只认旧 jsonl 的稠密列表）；训练复用 A5 的 `Store/train`（只读）经
`scripts/datafirst_bc_train.py`（加 `--train-limit/--val-limit` 以在有界内存/时间内跑完）。
**纯策略 BC**：硬 `teacher_index` CE + episode-outcome 值回归，无搜索目标。

| model | 数据（train/val） | epochs | train_seconds | final train acc | **final val acc** |
|---|---:|---:|---:|---:|---:|
| `dfbc_sa_d256` | 40000 / 3000（of 334734） | 6 | 1786 | 0.755 | **0.764** |
| `dfbc_sa_d256_e15` | 50000 / 3000 | 15 | 1160 | 0.812 | **0.808** |
| `dfbc_sa_d128` | 40000 / 3000 | 6 | 1334 | 0.726 | 0.729 |
| `dfbc_dvalin_bonk_d256` | 30000 / 4000 | 12 | 1864 | 0.752 | 0.735 |
| `dfbc_unyielding_geo_d256` | 20000 / 3000 | 6 | 835 | 0.640 | 0.630 |
| `dfbc_double_geo_navia_d256` | 20000 / 2801 | 6 | 1440 | 0.686 | 0.658 |
| `dfbc_natlan_battleship_d256` | 20000 / 3000 | 6 | 1434 | 0.496 | **0.475** |

**训练吞吐**：CPU d256 ≈ **150–300 样本/s**（受竞争）；加载 6.4 GB SA store 单次 ~120–200 s。
**重要口径**：受共享 CPU 限制，本会话各 run 只跑 **6–12 epochs**、每条 ≤40k 样本（A3 用 40 epochs / 6k）。
**SA 的 6-epoch 预算最可疑**（其 val acc 最高 0.764 但胜率最低，见 §9）；已补一个 15-epoch 重训（`dfbc_sa_d256_e15`）验证。

---

## 4. 全池认证（`scripts/eval_protocol`，配对种子，`--strict-ci`）

- **dvalin_bonk**：20 seeds × 20 池 × 2 边 = **800 局/臂**（Wilson 半宽 ≈ ±0.035），配对 vs `expert:dvalin_bonk`。
- 其余 4 臂：**10 seeds = 400 局/臂**（Wilson ≈ ±0.049 ≤ 0.05）。**因跨会话共享 CPU，20 seeds×5 臂不可行**，
  故除 dvalin 外降到 10 seeds（仍满足「Wilson ≤0.05」；`--seeds 20` 一行命令即可复跑加严）。
- 设计：同时跑候选臂与 `expert:<同 deck>` 基线臂，`paired_difference` 按 `(opponent, seed, side)` 配对。
- 质量：**全部臂 fallback=0 / error=0**（硬门禁 PASS）。

| deck | BC pooled | Wilson95 | cluster95 | expert pooled | 配对 Δ | n_pairs | best / worst 对手 |
|---|---:|---|---|---:|---:|---:|---|
| dvalin_bonk | 0.701 | [0.669,0.732] | [0.610,0.792] | 0.736 | −0.034 | 800 | ice_water 0.96 / unyielding 0.24 |
| unyielding_geo | 0.709 | [0.662,0.751] | [0.606,0.812] | 0.799 | −0.090 | 400 | ice_water 0.97 / electrocharged 0.20 |
| double_geo_navia | 0.689 | [0.642,0.732] | [0.568,0.809] | 0.681 | **+0.007** | 400 | ice_water 0.97 / electrocharged 0.05 |
| natlan_battleship | 0.140 | [0.109,0.177] | [0.074,0.206] | 0.225 | −0.085 | 400 | skirk_ayaka 0.45 / aywolfskirk 0.00 |
| superconduct_aggro | 0.405 | [0.358,0.454] | [0.294,0.516] | 0.509 | −0.104 | 400 | skirk_chasca 0.95 / unyielding 0.05 |
| superconduct_aggro (d256, 15 ep) | **0.480** | [0.431,0.529] | [0.367,0.593] | 0.509 | **−0.029** | 400 | skirk_chasca / unyielding |
| superconduct_aggro (d128) | 0.395 | [0.348,0.444] | [0.274,0.516] | 0.509 | −0.114 | 400 | skirk_chasca 0.95 / unyielding 0.05 |

**读法**：
1. **BC 强度基本跟随 deck 的专家强度**：专家 0.225(natlan) → BC 0.140；专家 0.799(unyielding) → BC 0.709。
   deck 难度是 BC 上限的**主导变量**，不是模型规模或数据量。SA 15-ep 臂说明 **epoch 预算同样关键**（0.405→0.480）。
2. **`natlan_battleship` 是一套对池极弱的 deck**（专家仅 0.225，20 个对手中 20 个 <45%）——
   它与 6-ep 的 `superconduct_aggro` 都**不是“数据能救”**的 run；但多 deck 语料正好暴露了这一点。
3. **长尾一致**：所有 deck 的天花板都被 `unyielding_geo / electrocharged_moon / double_geo_navia / nahida_kokomi_raiden`
   等岩/雷/草系压住；最优反应探针 exploitability_lb 0.76–0.95。

> **SA-15ep 的「打平」需谨慎**：本会话 10 seeds（n=400）得 Δ=−0.029（CI 含 0，功效不足）。
> **并发会话**对同类模型（d128@48k, 15 ep）跑到 **20 seeds（n=800）**，得 pooled **0.458 vs expert 0.519、
> Δ=−0.061 [−0.100,−0.022]（显著负）**（`reports/train/bc_scale_datafirst.md`）。
> 即 **SA 的 BC 仍显著低于专家约 6 pp**；本会话的「ns」是 10-seed 功效不足所致。**认证级须 ≥20 seeds。**

---

## 5. 数据量 × 强度曲线（SA，d128，5 seeds/10 seeds）

为把「数据量」轴单独看清（模型容量、epoch 固定 6、d128）：

| n_train | 局数 | pooled | Wilson95 |
|---:|---:|---:|---|
| 10,000 | 200 | 0.405 | [0.339,0.474] |
| 20,000 | 200 | 0.375 | [0.311,0.444] |
| 40,000 | 400 | 0.395 | [0.348,0.444] |

**结论（6-epoch 预算下）：对 `superconduct_aggro`，强度在 10k→40k 数据量上平坦（0.375–0.405，全在噪声内）。**
**但该平坦是「6 epoch 预算」下的现象，不能单独解释为「数据无用」**：把 SA 提到 **15 epoch / 50k 样本**后，
强度跳到 **0.480**（本会话 10-seed 配对 gap −0.029，功效不足；并发会话 20-seed 为 −0.061 显著负）。
→ **SA 的真实瓶颈是「训练预算（epoch×样本）」而非模型容量**（d128≈d256），且当前平坦曲线主要被 epoch 上限截断
（需在 15+ epoch 重扫数据轴才能分离两个变量）。
> 说明：曲线点用 5 seeds（200 局含双边）为**趋势级**，非认证级；认证级需 ≥20 seeds。逐点 checkpoint 在 `data/checkpoints/dfbc_sacurve_*`。

---

## 6. 门禁判定（逐项）

| 门禁项 | 口径 | 结果 |
|---|---|---|
| pooled ≥ 0.50 | M2/M3-CPU（point） | ✅ dvalin/unyielding/double_geo；✗ SA/natlan |
| pooled ≥ 0.50 | 认证级（**strict cluster CI 下界**） | ✅ dvalin[0.610]/unyielding[0.606]/double_geo[0.568]；✗ SA[0.294→e15 0.367]/natlan[0.074] |
| floor 无单套 <45% | M4 | ✗ **全部 5 deck FAIL**（6–20 个对手违规） |
| quality fallback/error=0 | 硬门禁 | ✅ 全 0 |
| BC ≥ 专家（同 deck 配对） | 进度标尺 | ✗ 未「显著超过」；`double_geo` 打平（+0.007 ns）；SA-15ep 20-seed 复跑 −0.061 显著负 |

**结论**：**M3-CPU 仍未整体通过**——pooled 门禁在 3/7 臂上首次达标，但 **M4 floor 全面崩盘**，
且 **BC 未「显著」追平专家**（仅 `double_geo` 打平；SA 认证级 20 seeds 仍 −6 pp）。数据先行把「每 deck 的可达强度」显著抬高了
（0.69–0.71 vs A3 的 ~0.50；SA 15-ep 0.48 vs 6-ep 0.40），但**长尾 matchup 的欠拟合是结构性瓶颈**，
规模放大不会自动消除。

---

## 7. 瓶颈

1. **算力共享（首要）**：本机 6 物理核被**另一个运行相同数据先行管线的会话**（`bc_learning_curve`，177% CPU）
   与本会话的采集/训练/评测三线并占；峰值并发触发 **OOM（worker rc=-9，重链 resume 无数据损失）**。
   净效果：墙钟膨胀 ~2–3×，采集 1450–4250 局/h、训练 150–300 样本/s、评测 ~1500 局/h。
2. **长尾 matchup（结构性）**：岩/雷/草系稳定压制所有 BC 与多数 deck；floor 是真正未过的门禁。
3. **epoch 预算**：受 CPU 限制多数 run 只跑 6–12 epochs（A3 用 40）。SA 的 6→15 ep 已把 pooled 0.405→0.480
   （gap −0.104*→−0.029 ns），**证明 epoch 是独立于数据量的有效杠杆**；其余 deck 需在 A800 上补足 epoch。
4. **无 GPU**：网络停在 d256；更宽网络/自博弈无法在 CPU 上推进。
5. **无搜索增益**（D23 已定）：本会话未再碰搜索 teacher。

---

## 8. 给 A800 窗口的放量参数（D21：窗口只做 GPU 训练/推理）

**数据（本地采集 → 传入）**
- 已就绪：SA **8,061 局**（jsonl，`data/store/sa_all.pt` 6.4 GB）+ 新 **6,400 局**（compact，4.17 GB）。
- 放量建议：**全部 meta deck（20 套）× 全 20 池 × 100–150 seeds**。粗算 20×20×120 ≈ **48,000 局**。
  - compact fp16 + no-priv ≈ **0.65 MB/局** → 48k 局 ≈ **31 GB**（37 GB 传输预算内；jsonl 需 ~58 GB，装不下）。
  - 采集侧本机 8–12 worker，约 **4,000–4,900 局/h** → 48k 局 ≈ 10–12 h（可分片断点续跑，不占窗口 CPU）。
- 每条 run：复制 `configs/engine.lock.example` → `engine.lock` 填 commit/卡池版本，随 run 元数据保存（D4）。

**训练（A800 GPU）**
| 维度 | 本会话 CPU 档 | A800 建议 |
|---|---|---|
| 网络 | `CVPN` d256（n_layers 2） | d_model **256–384**（或 `TransformerCVPN` 4 层 / 8 头） |
| 样本 | ≤40k/run，6–12 ep | **全量**（每条 8–50k 局 → 0.4M–2M 样本），**epochs 100–200** |
| batch / lr | 128 / 1e-3 | **512–1024** / 1e-3 → 3e-4（cosine），暖启 5% |
| 精度 | fp32 CPU | **bf16**（A800 原生），loss scale 动态 |
| 目标 | — | 先「BC 追平同 deck 专家（配对 CI 下界 ≥0）」；再考虑自博弈/搜索重开 |
| 读盘 | JSON 直读 | 用 `--format compact`（解码 ~2× 快）+ 预建 tensor store（`scripts/datafirst_bc_store.py`） |

**放量前的判定闸（不满足则收益有限）**：至少 1 个 deck 的 BC 在 **≥20 seeds + `--strict-ci`** 下配对 Δ 的下界 ≥ −2%，
且 floor 违规数较本报告的 6–20 明显下降。**当前仍未满足**（SA-15ep 20-seed Δ 下界 −0.100；floor 全 FAIL），
但 **6→15 epoch 已把 SA 的 gap 从 −0.104* 拉到 −0.029/20-seed −0.061** → 优先在 A800 上把 **epoch/样本预算**
放大到 100–200 ep × 全量数据，再判 floor 是否随规模松动。

**评分/认证（回流本机，不放窗口）**：`scripts/eval_protocol --contestant neural:<ckpt> --contestant-deck <deck>
--seeds 50 --workers 12 --strict-ci --baseline expert:<deck>`（≥50 seeds → Wilson ±0.03）。

---

## 9. 未决问题

1. **epoch 预算已证伪为 SA 的主因**：6→15 ep 把 SA 从 0.405 抬到 0.480（Δ 从 −0.104* → −0.029 ns）。
   仍需在 **15+ epoch** 下重扫数据轴（10k/20k/50k）以**分离**「数据量」与「epoch」两个变量；
   其余 deck 的 6–12 ep 结果同样是**下界**。
2. **floor 归因**：岩/雷/草系崩盘是**观测表示**、**动作抽象**还是**数据覆盖**？需 replay 复盘（`coach/replay.py`）。
3. **20 seeds 加严**：非 dvalin 的 4 臂需复跑到 20 seeds（响应面 Wilson ±0.035）出认证级结论。
4. **per-deck vs 多 deck 单模型**：本会话用 per-deck 专家；观测含自身角色 token（隐含 deck 身份），
   多 deck 单模型可行性未验证。
5. **多 deck 语料的混合训练价值**：本会话每 deck 独立训练；跨 deck 共享是否提升长尾未知。

---

## 10. 纪律 / 命令 / 产物

**新增文件（本会话，可提交）**
- `scripts/datafirst_bc_store.py`（compact-capable 流式 store 构建）
- `scripts/datafirst_bc_train.py`（有界 train/val 子采样 + 复用 A5 trainer）
- 本报告 `reports/datafirst_bc.md`；聚合 `reports/train/dfbc_summary.json`；各训练/认证报告
  `reports/train/dfbc_*.json`、`reports/explore/dfbc_cert_*.json`、`reports/explore/dfbc_curve_*.json`。

**未改动**：`npc/ envs/ web/ engine-overlay/ PLAN.md AGENTS.md refs/`，以及 A5 的未提交脚本
（`scripts/train_bc_scale.py`、`scripts/bc_learning_curve.py` 仅只读 import/调用）。

**关键命令**
```bash
PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python
# 采集（每 deck）
$PY -m train.collect --deck dvalin_bonk --all-opponents --seed-start 0 --seed-count 80 \
    --workers 8 --tag dfbc_dvalin_bonk_c80 --format compact --no-privileged-state
# store（compact 直读）
$PY -m scripts.datafirst_bc_store --replays data/replays/dfbc_dvalin_bonk_c80 \
    --store-cache data/store/dvalin_bonk.pt --val-every 25 --seed-start 0
# 训练
$PY -m scripts.datafirst_bc_train --store-cache data/store/dvalin_bonk.pt \
    --d-model 256 --epochs 12 --train-limit 30000 --val-limit 4000 \
    --out data/checkpoints/dfbc_dvalin_bonk_d256.pt --report reports/train/dfbc_dvalin_bonk_d256.json
# 认证
$PY -m scripts.eval_protocol --contestant neural:data/checkpoints/dfbc_dvalin_bonk_d256.pt \
    --contestant-deck dvalin_bonk --seeds 20 --workers 12 --strict-ci \
    --baseline expert:dvalin_bonk --tag dfbc_cert_dvalin_bonk
```

**产物索引（gitignore 的 `data/`）**：`data/replays/dfbc_*`（6400 局 compact）、`data/store/{sa_all,dvalin_bonk,
natlan_battleship,unyielding_geo,double_geo_navia}.pt`、`data/checkpoints/dfbc_*.pt`、
`data/eval/dfbc_cert_*.jsonl`（逐局原始）。
