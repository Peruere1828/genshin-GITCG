# 数据先行：规模化 BC 采集 + 认证（D23 改排产）

> 日期：2026-10-12｜作者：**数据先行（D23）子代理**｜口径：**纯策略**（不接 LLM，PLAN §6.3）
> 目标（D23）：用**规模化 BC 数据 + 更大网络**把强度顶过**专家基线**，并给出认证级对比。
> 机器：本机 12 核 CPU、无 GPU；**与 A6/其它子代理共用同一台机器**（多作业并发，见 §6）。
> 原始逐局 JSONL 在 `data/`（gitignore）；聚合报告在 `reports/`；本报告为唯一入库摘要。

---

## 0. 结论速览（诚实数字）

**核心判定（修订后）**：数据先行方向下，本机 CPU 档 BC 网络（d128，48k 样本）**先前的"显著劣于专家"
是 5-epoch 未收敛造成的假象**。把同一配方训到 15 epoch（val acc 0.727→**0.794**）后，全池
pooled 从 0.403 升到 **0.479**，与**同种子专家**（0.509）的配对 Δ = **−0.030**，95% CI
[−0.086, +0.026]（n=400）——**与专家统计无显著差异（打平），但仍未"顶过"专家**。
"顶过专家基线 0.561"在本机算力/当前配方下【未达成】。

**真正的杠杆是训练时长/优化，而不是数据量**：8k→48k（同为 5 epoch）配对 Δ 不显著；
48k 的 5→15 epoch 配对 Δ = **+0.076** [0.023, 0.129]（**显著**）。

| 量 | 值 |
|---|---|
| 采集规模（唯一局） | **8,061 局**（20 池 × 314–500 种子；见 §1） |
| 采集样本 | **372,270** 条决策（46.2/局） |
| 语料落盘 | ≈ 11 GB（jsonl，约 1.4 MB/局） |
| 训练集/验证集（store） | 334,734 / 37,536（按局种子 9:1 划分） |
| d128 @ 48k：val 策略准确率 | 5ep **0.7266** → 15ep **0.7944**（8k/5ep → 0.5916） |
| **d128 @ 48k（15ep）：全池 pooled** | **0.479**（10 seeds，n=400；Wilson95 [0.430,0.528]） |
| **同种子专家基线 pooled** | **0.509**（10 seeds；20 seeds 口径 0.519，A4 全池 0.561/n=400） |
| **配对 Δ（net − expert）** | **−0.030**，95% CI [−0.086, +0.026]，n=400，**与专家无显著差异** |
| 5ep → 15ep 的在线强度变化 | **Δ=+0.076**，95% CI [+0.023, +0.129]（n=400，**显著**） |
| 数据量 8k → 48k（同为 5ep） | **Δ=+0.023**，95% CI [−0.036, +0.081]（n=400，**不显著**） |
| d128 @ 48k（5ep）：全池 pooled | 0.385（20 seeds，n=800）；配对 Δ −0.134 [−0.175,−0.094]（**未收敛，劣于专家**） |
| d256 @ 48k（3ep，容量轴） | pooled **0.225**，配对 Δ **−0.284**（更差；epoch 不匹配，见 §5） |
| M4 floor（strict：per-opp Wilson 下界 ≥0.45） | 5ep **13/20**、15ep **11/20** 违规 → FAIL |
| 质量门禁（fallback/error/draw） | 0/0/0 → PASS |

**一句话**：BC 在 5 epoch 时看着"显著劣于专家"（Δ=−0.134），但那主要是**没训够**——
同一 48k 配方训到 15 epoch（val acc 0.79），在线 pooled 升到 **0.479**，与专家**打平**
（Δ=−0.030，CI 跨 0）。**放大数据（8k→48k）在同 epoch 下没有显著增益，而训练时长 5→15 epoch
有显著增益（+0.076）**。结论：当前 BC 路线**能逼近/打平专家，但未超过**；要"顶过"，靠"更多专家
BC 数据"希望不大（目标上限即专家），需要**自博弈/RL 目标**或**更充分的优化**。

> **修订说明（重要）**：本报告初稿（5-epoch）曾给出"数据放大不提强度、显著劣于专家"的结论；
> 经 15-epoch 长训复核，该结论**主要是训练不足的假象**，已按上表修订。保留 5-epoch 数字以
> 记录这一混淆。

---

## 1. 规模化采集（本机，可续跑/流式落盘）

### 1.1 命令与任务编排

`train.collect --deck superconduct_aggro --all-opponents`，种子 `1000000+`：

- 第一段：`--seed-start 1000000 --seed-count 500 --workers 11 --tag datafirst_bc_sa500`
  （20 池 × 500 种子 = 10,000 局目标）。
- 补采（平衡化）：因 `collect` 的任务序是**对手主序**（先跑完一个对手的全部种子再下一个），
  单进程跑到一半会让"已完成"严重偏向靠前对手。为让"任意时刻都 20 池均衡"，把剩余 14 个对手
  各自拆成一个独立 tag（`df_<opp>`）并行补采，每个 `--seed-count 500`。

BC 采集（teacher = `expert:<deck>`，搜索关闭）由 `train.collect` 驱动，逐局落盘到
`data/replays/<tag>/samples.jsonl`（一行一局），可断点续跑。

### 1.2 吞吐与体积（实测）

| 观测 | 值 |
|---|---|
| 6 worker 探针 | 20 局 / 27 s（含启动）≈ **2,665 局/h** |
| 11 worker（稳态探针，仅本线） | ≈ **4,020 局/h**（60 s 增量实测） |
| 与其他子代理并发时 | **1,700–4,800 局/h** 波动（整机争用，见 §6） |
| 落盘 | 约 **1.4 MB/局**（≈ 28–30 KB/决策） |
| 样本/局 | **46.2**（372,270 / 8,061） |

### 1.3 ⚠️ 关键工程发现：`train.collect` 父进程内存随局数线性增长

`envs/rollout.run_task_shard` 会把**每局全部编码样本**放进返回列表
（`results.extend(...)` / `results.append(...)`），`train.collect.collect()` 虽然用
`on_result` 流式落盘，但**返回列表仍全程驻留父进程**。实测：

- 单 worker 单进程跑到 ~194 局时父进程 RSS ≈ **2 GB**；外推 500 局 ≈ 5 GB/进程。
- 14 个并行 `collect`（每进程 500 局）峰值合计 ≈ **28 GB**，直接把 30 GB 机器打到
  swap 耗尽（`free` 归零、swap 打满）。
- **规避**：把长作业切成 `--max-tasks 50` 的短块，每块一个**新进程**（块结束即退出、内存回收），
  同一 spec 指纹下可续跑。本报告所有采集均用该"块循环"跑法；等价命令：

  ```bash
  for i in $(seq 1 14); do   # 每块 ~50 局、进程退出即回收
    $PY -m train.collect --deck superconduct_aggro --opponents <opp> \
        --seed-start 1000000 --seed-count 500 --workers 1 --max-tasks 50 --tag df_<opp>
  done
  ```

> 这是"数据先行"放量的**真实阻碍**之一（不是撮合、不是样本编码，而是父进程汇聚）。
> 建议上游把 `on_result` 路径改为**不保留返回结果**（或分片落盘后即弃），否则大机器也难放量。

---

## 2. 训练侧：省内存的紧凑 tensor store

`train.data.load_samples` 把每条样本物化成 `EncodedObservation`（Python 浮点元组），
实测 **≈ 150–200 KB/样本**；334k 样本需 ~50 GB，本机内存不可行。

因此新增 `scripts/train_bc_scale.py`（本报告新增脚本，入库）：

- 直接从采集 JSONL **流式**把每条样本转成**紧凑、变长**张量（`tokens (T,65)`、`options (K,68)`、
  belief 标签），丢弃 Python 对象；**≈ 17 KB/样本**（省 ~10×）。
- 批内按 batch 最大长度 `pad_sequence`，避开全局 `K=256` 的 padding 爆炸。
- 落一份 **tensor store 缓存**（`data/store/dfbc_sa.pt`，6.4 GB），后续多配置训练秒级加载。
- 按 `(seed - seed_start) % 10` 划验证；按 `(opponent, seed, step_index)` 排序后再洗牌，保证可复现。
- 学习曲线：`scripts/bc_learning_curve.py`（同一 store 上扫嵌套子集 × 网络宽度）。

训练目标 = 冷启动 BC：硬 `teacher_index` 交叉熵 + episode outcome 值回归 + belief 头
（与 A3 的 BC 臂一致；搜索 teacher 已由 A4 证否，本线不接搜索）。

---

## 3. 数据量 × val 策略准确率（曲线）

| d_model | n_train | epochs | val 策略 acc | val CE | val value MSE | 训练墙钟 |
|---|---|---|---|---|---|---|
| 128 | 8,000 | 5 | **0.5916** | 1.0266 | 0.2094 | 186 s |
| 128 | 48,000 | 5 | **0.7266** | 0.8222 | 0.1732 | 896 s |
| 128 | 48,000 | **15** | **0.7944** | 0.6336 | 0.1640 | 1830 s |
| 256 | 48,000 | 3 | **0.6894** | 0.8794 | 0.1859 | 1293 s |

**val 准确率随数据、也随训练时长上升**：8k→48k（同 5ep）+0.135；48k 上 5ep→15ep **+0.068**
（0.7266→0.7944，且 15ep 仍未见顶）。§5 显示**在线强度跟"训练时长"显著相关，跟"数据量"不显著**。

> **d256 消融的局限（诚实标注）**：d256 点因整机争用只跑了 **3 个 epoch**（d128 点 5 个），
> 二者**不同 epoch**，不是干净的容量对比；d256(3ep) val acc 0.689 < d128(5ep) 0.727 主要反映
> **训练不足**而非容量劣势。这一轴的正经消融应放 A800（§7）。

> 说明：本项目内 128k 点的训练在整机 swap 抖动下被中断（§6）；48k 点为已在同机跑完的最大点。

---

## 4. 认证级评测（全 20 池 × 20 种子 × 双先手 = 800 局/臂）

命令（纯策略）：

```bash
PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/curve_dfbcsa_d128_n48000.pt \
  --baseline expert:superconduct_aggro \
  --seeds 20 --workers 8 --strict-ci --tag dfbcsa_d128_48k
```

（`--baseline` 以**同种子同对手**跑同 deck 脚本专家，给出配对 Δ。）

### 4.1 整体

| 量 | d128 @48k | 同种子专家 | 配对 Δ |
|---|---|---|---|
| pooled | **0.385** | **0.519** | **−0.134** |
| Wilson95 | [0.352, 0.419] | — | [−0.175, −0.094] |
| opponent-cluster 95 | [0.271, 0.499] | — | [−0.190, −0.079] |
| 门禁 | pooled ✗（<0.50）、floor ✗、clean ✓ → **FAIL** | | `significant_positive=false` |

**认证级结论：BC（48k）显著劣于专家（Δ 的 95% CI 整体在 0 以下）。**

### 4.2 逐对手（contestant 胜率 / 与专家配对 Δ，n=40/对手）

| 对手 | rate | 95% | Δ | | 对手 | rate | 95% | Δ |
|---|--:|---|---:|---|---|--:|---|---:|
| unyielding_geo | **0.025** | [0.004,0.129] | −0.125 | | electro_manifestation_nuke | 0.300 | [0.181,0.454] | −0.150 |
| dvalin_bonk | **0.050** | [0.014,0.165] | −0.175 | | mualani_double_pyro | 0.300 | [0.181,0.454] | −0.200 |
| double_geo_navia | 0.175 | [0.087,0.320] | +0.013 | | yelan_clorinde_ororon | 0.300 | [0.181,0.454] | −0.338 |
| electrocharged_moon | 0.175 | [0.087,0.320] | −0.075 | | superconduct_mika_attack | 0.350 | [0.221,0.505] | −0.250 |
| nahida_kokomi_raiden | 0.175 | [0.087,0.320] | −0.350 | | aywolfskirk | 0.425 | [0.285,0.578] | +0.050 |
| dual_mualani_stacks | 0.225 | [0.123,0.375] | −0.087 | | trial_skirk_ayaka_navia | 0.500 | [0.352,0.648] | −0.062 |
| klee_mavuika_mualani | 0.250 | [0.142,0.402] | −0.087 | | superconduct_aggro | 0.550 | [0.398,0.693] | +0.050 |
| shenhe_double_swirl | 0.275 | [0.161,0.428] | −0.150 | | natlan_battleship | 0.575 | [0.422,0.715] | −0.325 |
| | | | | | ice_water_battleship | 0.675 | [0.520,0.799] | −0.212 |
| | | | | | beidou_skirk_ayaka | 0.725 | [0.572,0.839] | −0.062 |
| | | | | | skirk_ayaka_navia | 0.750 | [0.598,0.858] | −0.125 |
| | | | | | skirk_chasca_freeze | 0.900 | [0.769,0.960] | −0.025 |

- **floor 崩盘**：13/20 对手 <0.45（strict 口径 per-opp Wilson 下界 <0.45）；最弱
  `unyielding_geo` 0.025、`dvalin_bonk` 0.050 —— 与 A3（搜索小网）弱项同族（岩/龙/雷长尾）。
- 少数非负项（`double_geo_navia`/`aywolfskirk`/`superconduct_aggro` +0.013~+0.050）远不抵全局 −0.134。
- 最优反应探针：worst=`unyielding_geo`，exploitability_lower_bound=0.975。

### 4.3 收敛模型（48k，15 epoch）——与专家打平

把同一 48k 配方训到 15 epoch（val acc 0.794）后复评（10 seeds × 20 池 × 双先手 = 400 局）：

```bash
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/curve_dfbcsa_e15_d128_n48000.pt \
  --seeds 10 --workers 10 --strict-ci --tag dfbcsa_d128_48k_e15
```

| 量 | 15ep 模型 | 同种子专家（seeds 0–9） | 配对 Δ |
|---|---|---|---|
| pooled | **0.479**（Wilson95 [0.430,0.528]） | **0.509** | **−0.030** [−0.086, +0.026] |
| opponent-cluster 95 | [0.346,0.611] | — | [−0.088, +0.028] |
| 门禁 | pooled ✗（<0.50 point，但 CI 含 0.50）、floor ✗、clean ✓ | | `significant_positive=false` |

**结论**：收敛后，BC 网络与同 deck 脚本专家**无显著差异（打平，Δ 点估计 −0.03）**，
但**仍未超过**专家。5ep→15ep 的配对 Δ = **+0.076** [0.023,0.129]（**显著**，n=400）——
说明**先前"显著劣于专家"主要是欠训练**。**floor 仍崩**（strict：**11/20** 对手 Wilson 下界 <0.45；
最弱 `unyielding_geo` 0.000/20）——长尾 matchup 未随收敛解决。

---

## 5. 数据量 × 强度（在线）曲线

见 §3 的 val 准确率曲线 + 下表的**在线胜率**点（全 20 池，配对同一专家基线行）：

| 配置 | val acc | 在线 pooled | 配对 Δ vs expert | 说明 |
|---|---|---|---|---|
| 8k / 5ep | 0.5916 | **0.425**（10 seeds，n=400） | **−0.084** [−0.145,−0.023] | 欠训练 |
| 48k / 5ep | 0.7266 | 0.4025（10 seeds，n=400） | −0.106 [−0.163,−0.050] | 欠训练 |
| **48k / 15ep** | **0.7944** | **0.479**（10 seeds，n=400） | **−0.030** [−0.086,+0.026] | **与专家打平** |
| 48k / 5ep | 0.7266 | 0.385（20 seeds，n=800） | −0.134 [−0.175,−0.094] | 欠训练（认证 20 seeds 口径） |
| 48k / 3ep，d256 | 0.6894 | 0.225（10 seeds，n=400） | −0.284 [−0.338,−0.229] | 容量轴，epoch 更不匹配 |
| 372,270（已采集） | — | — | — | 本机未训完，留 A800 |

**同种子（seeds 0–9，n=400）关键配对对比**：

- **训练时长是显著杠杆**：`15ep − 5ep` 配对 Δ = **+0.076** [0.023, 0.129]（**显著**）。
- **数据量不是**：`8k − 48k`（同为 5ep）配对 Δ = **+0.0225** [−0.036, +0.081]（**不显著**）。
- **收敛后与专家打平**：`15ep − expert` Δ = **−0.030** [−0.086, +0.026]（不显著，CI 跨 0）。
- 8k（5ep）与 15ep 相比：`15ep − 8k` Δ = +0.054 [−0.005, +0.113]（不显著）。

**读法**：val acc 随数据与 epoch 都上升，但**在线强度只对"训练是否收敛"敏感**：
欠训练（5ep）时"数据更多反而点估计更低"（0.425→0.403，不显著）；一旦训到 15ep，
强度跳到 0.479、追平专家。**在"模仿 expert"目标下，把优化做足即可逼近专家上限，但不会超过**。

（8k/48k-5ep/d256 无 own baseline 臂，配对复用 48k run 的专家基线行
`dfbcsa_d128_48k__baseline.jsonl`；同种子交集 400 对。）

**容量轴（d256）**：d256@48k 只跑了 3 epoch（争用限制），pooled **0.225**，配对 Δ vs expert
= **−0.284**；且 d256 显著劣于 8k 的 d128（Δ=+0.20 [0.145,0.255]）。这**主要是训练步数不足**，
**不能**读作"大网更差"。容量轴的干净消融（同 epoch）留 A800。

---

## 6. 瓶颈分析（本机实测）

1. **上游内存汇聚（§1.3）**：`train.collect` 父进程把全部样本留在 RAM，500 局/进程 ≈ 5 GB；
   多进程并行直接把 30 GB 机器打满 swap。**这是放量的首要工程阻碍。**
2. **整机多作业争用**：本线运行时 A6 在跑"每 deck × 全池 × 80 种子"的 32,000 局多 deck 采集，
   另有并行子代理在训练/评测。实测 **load average 长期 20–67**、`swap` 打满、持续 swap-out
   （vmstat `so≈9 MB/s`）。同一训练点在有/无争用时**每样本每 epoch 差 ~4–7×**
   （d128：0.67 ms → 2.3–5 ms）。128k 点因此被中断。
3. **"离线准确率"与"在线强度"非线性**：同一 48k 数据，5ep（val acc 0.727）在线 0.403，
   15ep（val acc 0.794）在线 0.479——**val acc 只 +0.068，在线强度 +0.076 且显著**；而 8k→48k
   （val acc +0.135）在线**不显著**。说明**收敛程度**（优化步数）比"数据量"更能决定强度。
4. **数据量仍偏小，但不是当前瓶颈**：48k 样本 / 20 池 ≈ 每对手 1.4–2.7k 样本。既然收敛 BC
   只能"打平"专家（目标上限即专家），**要超过专家需换目标（自博弈/RL/搜索），而非堆 BC 数据**。

---

## 7. 给 A800 窗口的放量参数建议（遵守 D21）

| 维度 | 本机 CPU 档（本次） | A800 窗口建议 |
|---|---|---|
| 采集 | 本机块循环（`--max-tasks 50`），8,061 局 | 数据本机采好后传入；**窗口不放采集** |
| 语料 | 372,270 样本 / 11 GB jsonl（或 A6 `--format compact` ≈0.54×） | 20 池 × 500–1000 种子 / **1e6 样本**量级 |
| 网络 | d128 CVPN（48k/5ep）；d256 CVPN（48k/3ep，受争用限制） | **d256–d512**，n_heads 8；bf16；同 epoch 做干净容量消融 |
| epochs/lr | 5ep 未收敛；**15ep 才追平专家**（val acc 0.79） | 训到**收敛**（val acc 平台期），50–200 ep / 1e-3→3e-4 cosine；早停按 val acc |
| batch | 128 | 512–2048（GPU） |
| 训练硬件 | CPU（争用下 0.67–5 ms/样本/epoch） | A800（`--device cuda`）|
| 认证 | 20 种子全池，配对专家（本报告 §4） | 同口径复跑，≥20 种子、`--strict-ci` |
| **前置** | — | 先决定"模仿 vs 超越"：若仍 BC→上限=专家；建议配 **自博弈/DAgger** 或**校准后的搜索** |

**放量前必读**：本报告显示"更多 BC 数据 + 更大网络"**未必**能超过专家基线——因为 BC 的目标
（复现 expert）不包含"超过 expert"。A800 应优先投在**自博弈/RL 目标**与**价值头校准**上，
纯 BC 扩容按"逼近专家、补齐长尾"定位。

---

## 8. 未决问题

1. **d256 只跑了 3 epoch**（整机 swap 争用），与 d128 的 5 epoch 不匹配，**容量轴非干净消融**（结论：大网需 A800 同 epoch 重跑；本机 d256 0.225 主要反映训练不足）。
2. **128k/372k 点的在线强度未测**：数据量能否把 0.385 抬到专家附近，仍需更大机器的短回归。
3. **BC 模仿 expert 的理论上限**：本报告实证"模仿准确率高 ≠ 强度高"；是否 `unyielding_geo`
   等长尾是**观测表示**欠拟合还是**策略分布**问题，未归因。
4. **多 deck 数据**（A6 在做）能否提升单 deck 强度，未验证。
5. **上游内存汇聚修复**后会显著改善放量吞吐（建议优先）。

---

## 9. 产物索引

- 本报告：`reports/train/bc_scale_datafirst.md`
- 聚合 JSON（入库）：`reports/explore/dfbcsa_d128_48k.json`（5ep 认证评测 + 配对 Δ）、
  `reports/explore/dfbcsa_d128_48k_e15.json`（**15ep，打平专家**）、
  `reports/explore/dfbcsa_d128_8k.json`、`reports/explore/dfbcsa_d256_48k.json`、
  `reports/train/bc_curve_dfbcsa.json`（d128 val 曲线）、`reports/train/bc_curve_dfbcsa_e15.json`、
  `reports/train/bc_curve_dfbcsa256.json`（d256）
- 脚本（入库，本报告新增）：`scripts/train_bc_scale.py`、`scripts/bc_learning_curve.py`
- 原始（gitignore）：`data/replays/datafirst_bc_sa500/`、`data/replays/df_*/`（语料）、
  `data/store/dfbc_sa.pt`（tensor store 6.4 GB）、
  `data/checkpoints/curve_dfbcsa_d128_n{8000,48000}.pt`、`curve_dfbcsa_e15_d128_n48000.pt`、
  `curve_dfbcsa256_d256_n48000.pt`、
  `data/eval/dfbcsa_d128_48k*.jsonl`、`data/eval/dfbcsa_d128_8k.jsonl`、
  `data/eval/dfbcsa_d128_48k_e15.jsonl`、`data/eval/dfbcsa_d256_48k.jsonl`
- 关联：`reports/train/a3_m3_cpu_screening.md`、`reports/search/a4_search_budget_ladder.md`、PLAN D22/D23
