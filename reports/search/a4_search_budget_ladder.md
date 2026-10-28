# A4 — MCCFR 搜索智能体直接强度 × 预算阶梯（D22 关键判定实验）

> **status: limited（2026-10-27，D26）**：本文的“根价值区分度≈0/搜索净负”依赖当时
> `envs/cfr_probe.py` 的价值语义；该处已修（identity→概率、acting→p0），因果结论降级为
> “在当前测量/预算下无增益”。排产决策“不放大搜索”仍有效；重开条件见
> `docs/failures/001-search-as-teacher.md`。原文保留。

> 日期：2026-10-12｜作者：A4 子代理｜口径：**纯策略**（不接 LLM，PLAN §6.3）
> 问题（PLAN D22 / §4.3 下一步）：**「MCCFR 搜索本身能否显著强于 expert/BC？」**——决定"继续搜索蒸馏"还是"改排产（先补 BC/自博弈数据）"。
> 判定口径：`run_cfr_search_agent` 式**配对同种子**评测（cfr 组 = 同 deck expert 基座 + 搜索；专家组 = 同一 expert、搜索关闭），看 **配对 Δ 的 95% CI 下界是否 > 0**。
> 机器：本机 12 核 CPU、无 GPU；`--workers 6` 并发 6 局（每局 1 个 fork 子进程，CPU 绑定）。
> 原始逐局 JSONL 在 `data/search/`（gitignore）；聚合 JSON 在 `reports/search/`；本报告为唯一入库摘要。

---

## 0. 结论速览（诚实数字）

**判定：「搜索 teacher 无可测增益」——不仅无正增益，在全池 n=400 上搜索显著 *劣于* 同基座 expert（配对 CI 上界 < 0）。不支持"继续搜索蒸馏"；支持 D22 的改排产分支。**

| 档 | 预算（i/d/c/ms） | pairs | expert | cfr | 配对 Δ | 95% CI | CI 下界>0 | 每搜索 | 每局(cfr) |
|---|---|---:|---:|---:|---:|---|---:|--:|--:|
| A（A3 等价） | 4/1/2/8 | 36 | 0.431 | 0.389 | −0.042 | [−0.218,+0.135] | ✗ | 0.37 s | 10.1 s |
| B | 8/1/4/8 | 36 | 0.431 | 0.389 | −0.042 | [−0.200,+0.116] | ✗ | 0.47 s | 10.2 s |
| C | 16/1/4/8 | 36 | 0.431 | 0.389 | −0.042 | [−0.200,+0.116] | ✗ | 0.48 s | 10.3 s |
| D | 8/2/4/4 | 36 | 0.431 | 0.528 | +0.097 | [−0.048,+0.242] | ✗ | 6.28 s | 31.5 s |
| E | 16/2/4/4 | 36 | 0.431 | 0.361 | −0.069 | [−0.194,+0.055] | ✗ | 6.64 s | 32.9 s |
| **B 全池** | 8/1/4/8 | **400** | **0.561** | **0.495** | **−0.066** | **[−0.125,−0.008]** | ✗（**显著为负**） | 0.44 s | 8.6 s |
| **D 全池** | 8/2/4/4 | **400** | **0.561** | **0.448** | **−0.114** | **[−0.169,−0.058]** | ✗（**显著为负**） | 5.78 s | 28.2 s |

（A–E 为 6 对手×6 种子的敏感子集，n=36，用于扫档；B/D 在**全 20 池 × 20 种子 = 400 配对**复测，Wilson 半宽 ≈0.049，达 ≤0.05 目标。）

**三条硬结论**：

1. **没有任何档位配对 CI 下界 > 0**。子集 n=36 的 5 档全部跨 0；唯一正点估计（D，+0.097）在**同预算仅换 iterations（E）时翻成 −0.069**，即噪声，且全池复测 D 直接落到 −0.114。
2. **全池 n=400 上搜索显著劣于 expert**：B Δ=−0.066 CI[−0.125,−0.008]，D Δ=−0.114 CI[−0.169,−0.058]，两侧 CI 完全在 0 以下。搜索把"该赢的局"改成"该输的局"的次数是反向的 **1.5–2.0 倍**（B：85 个 win→loss vs 58 个 loss→win；D：90 vs 44）。
3. **搜索越"强"越差，且 iterations 不是瓶颈**：d1 档 it8 与 it16 **逐局结果逐位相同**（72/72 行 outcome/searched/changed 全同），成本也相同（107.8 s vs 107.7 s）——在 d1 且单粒子根下，`iterations ≥ 8` 完全不改变所选动作。把预算从 B 提到 D（depth 2，6.3 s/搜索，17×）反而把 Δ 从 −0.066 推到 −0.114。

---

## 1. 实验设计（命令与口径）

评测脚本：`scripts/run_cfr_ladder.py`（复用 `scripts/run_cfr_search_agent` 的 spec/任务网格与指纹护栏；搜索在**局内串行**，故按**跨局并行**加速）。

```bash
PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python
FULL="natlan_battleship klee_mavuika_mualani superconduct_mika_attack dual_mualani_stacks superconduct_aggro mualani_double_pyro yelan_clorinde_ororon electrocharged_moon aywolfskirk dvalin_bonk shenhe_double_swirl electro_manifestation_nuke double_geo_navia unyielding_geo beidou_skirk_ayaka skirk_chasca_freeze skirk_ayaka_navia ice_water_battleship nahida_kokomi_raiden trial_skirk_ayaka_navia"
SUB="unyielding_geo double_geo_navia electrocharged_moon dvalin_bonk nahida_kokomi_raiden natlan_battleship"

# 专家控制组画像（20 池 × 3 种子；max_searches=0 时 cfr 组退化=expert）
$PY -m scripts.run_cfr_ladder --opponents $FULL --seeds 3 --max-searches 0 \
  --workers 10 --bridge-workers 1 --tag profile_expert

# 预算阶梯（6 对手 × 6 种子敏感子集）：
#   A: i4 d1 c2 ms8 | B: i8 d1 c4 ms8 | C: i16 d1 c4 ms8 | D: i8 d2 c4 ms4 | E: i16 d2 c4 ms4
$PY -m scripts.run_cfr_ladder --opponents $SUB --seeds 6 \
  --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 --belief-samples 1 \
  --workers 6 --bridge-workers 1 --tag ladder_B_i8_d1_c4_ms8

# 全池放量（20 池 × 20 种子 = 400 配对/档）
$PY -m scripts.run_cfr_ladder --opponents $FULL --seeds 20 \
  --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 --belief-samples 1 \
  --workers 6 --bridge-workers 1 --tag ladder_fullB_i8_d1_c4_ms8
$PY -m scripts.run_cfr_ladder --opponents $FULL --seeds 20 \
  --iterations 8 --depth 2 --candidate-limit 4 --max-searches 4 --belief-samples 1 \
  --workers 6 --bridge-workers 1 --tag ladder_fullD_i8_d2_c4_ms4

# 网络 prior+value 敏感性（S-C 设计本意：学习到的 prior 排序 + 价值前沿）
$PY -m scripts.run_cfr_ladder --opponents $SUB --seeds 6 --iterations 8 --depth 2 \
  --candidate-limit 4 --max-searches 4 --checkpoint data/checkpoints/20261012T101435_a3_distill_v1_d128.pt \
  --workers 6 --bridge-workers 1 --tag ladder_netH_i8_d2_c4_ms4
$PY -m scripts.run_cfr_ladder --opponents $SUB --seeds 6 --iterations 8 --depth 1 \
  --candidate-limit 4 --max-searches 8 --value-spec data/checkpoints/20261012T101435_a3_distill_v1_d128.pt \
  --workers 6 --bridge-workers 1 --tag ladder_netI_i8_d1_c4_ms8
```

设计要点：

- **配对同种子**：cfr 组与 expert 组共用同一 deck、同一基座 expert、同一 seed；基座 RNG 在两组推进一致，故分叉前轨迹逐位相同，`Δ` 只反映搜索。
- **`candidate_limit` 无 prior 时按选项序**：`EngineParticleGame.legal_actions` 在根按 `(-prior, index)` 取前 C 个，无 prior 时 prior≡0 → 取**引擎选项序的前 C 个**（再并入 expert 当前动作）。即 A–E 档的候选裁剪是**任意序**而非"专家更看好的几个"——这正是网络 prior 敏感性（第 4 节）要检验的点。
- **成本口径**：`s/搜索 = (wall_cfr − wall_expert_同配对) / searched_decisions`（中位数与均值基本一致，见 JSON）；`s/局` 为 cfr 组墙钟均值。
- 计算总量 ≈ 4.5e3 s 墙钟（~75 min，含所有档与验证）。

---

## 2. 全池放量结果（n=400，认证级样本）与门禁

### 2.1 整体

| 档 | cfr pooled (Wilson95) | expert pooled (Wilson95) | 配对 Δ (95% CI) | opponent-cluster 95% | 门禁：Δ CI 下界>0 | M4 floor（单套<0.45） |
|---|---|---|---|---|---|---|
| B `i8/d1/c4/ms8` | 0.495 [0.446,0.544] | 0.561 [0.512,0.609] | **−0.066 [−0.125,−0.008]** | [−0.143,+0.010] | **FAIL（显著为负）** | cfr 8 套 / expert 8 套 |
| D `i8/d2/c4/ms4` | 0.448 [0.400,0.496] | 0.561 [0.512,0.609] | **−0.114 [−0.169,−0.058]** | [−0.191,−0.036] | **FAIL（显著为负）** | cfr 11 套 / expert 8 套 |

（注：M4 floor 对**专家控制组本身也 FAIL**——20 池有高方差长尾，expert 在 `aywolfskirk`/`double_geo_navia`/`dual_mualani_stacks`/`dvalin_bonk`/`electrocharged_moon`/`klee_mavuika_mualani`/`mualani_double_pyro`/`unyielding_geo` 8 套上 <0.45。搜索只是**在同一条长尾上重新分配胜负**：在 expert 弱项（`aywolfskirk`、`klee_mavuika_mualani`）赢回来，在 expert 强项上大量输掉。）

**D22 判定的答案：否。不存在（在 0.37–6.6 s/搜索 区间内）可承受预算使搜索显著强于 expert；相反，全池 n=400 上搜索显著更弱。**

### 2.2 逐对手配对 Δ（全池，cfr − expert；n=20/对手）

| 对手 | B Δ | D Δ | | 对手 | B Δ | D Δ |
|---|--:|--:|---|---|--:|--:|
| aywolfskirk | **+0.200** | 0.000 | | natlan_battleship | −0.050 | −0.100 |
| klee_mavuika_mualani | **+0.350** | +0.100 | | skirk_ayaka_navia | −0.150 | −0.100 |
| electrocharged_moon | +0.150 | +0.100 | | skirk_chasca_freeze | −0.050 | −0.100 |
| double_geo_navia | +0.075 | −0.025 | | superconduct_mika_attack | −0.050 | **−0.350** |
| yelan_clorinde_ororon | +0.050 | **−0.350** | | unyielding_geo | −0.100 | −0.050 |
| dual_mualani_stacks | −0.100 | +0.100 | | dvalin_bonk | **−0.300** | −0.100 |
| beidou_skirk_ayaka | **−0.250** | +0.050 | | shenhe_double_swirl | −0.200 | **−0.300** |
| nahida_kokomi_raiden | **−0.250** | −0.150 | | superconduct_aggro | −0.050 | **−0.500** |
| trial_skirk_ayaka_navia | −0.200 | −0.100 | | electro_manifestation_nuke | −0.150 | −0.250 |
| mualani_double_pyro | −0.100 | 0.000 | | ice_water_battleship | −0.150 | −0.150 |

- **B 档：5 升 / 15 降**；**D 档：4 升 / 2 平 / 14 降**。搜索**在 expert 的强项上破坏最大**（D：`superconduct_aggro` −0.50、`superconduct_mika_attack` −0.35、`yelan_clorinde_ororon` −0.35）——即搜索"自信地"把专家对的招换成错的招。
- 少数正项（B 的 `klee_mavuika_mualani` +0.35、D 的 `dual_mualani_stacks`/`electrocharged_moon` +0.10）不抵全局负项。

---

## 3. teacher 质量侧证：搜索改变了动作，但更常输

| 量 | B（d1） | D（d2） | 说明 |
|---|---:|---:|---|
| 每局搜索决策数 | 8.0 | 4.0 | ms 上限 |
| **动作改变率** | **86.6%** | **88.8%** | cfr 覆盖 expert 动作的比例 |
| 每局改变数 | 6.92 | 3.55 | |
| win→loss 翻转 | 85 | 90 | expert 赢、cfr 输 |
| loss→win 翻转 | 58 | 44 | expert 输、cfr 赢 |
| **净翻转** | **−27/400** | **−46/400** | 与配对 Δ 完全一致 |

**「搜索改变 expert 动作后是否更常赢」的答案：否。** 搜索覆盖了 ~86–89% 的被搜决策，但 **win→loss 是 loss→win 的 1.5–2.0 倍**，净负。搜索的"改动"平均是有害的。

**根价值（root value）校准**（搜索自评的根价值能否区分胜负，从搜索方视角）：

| 档 | 赢局均根价值 | 输局均根价值 | 分离度 |
|---|---:|---:|---:|
| fullB (d1) | +0.047 (n=198) | +0.024 (n=202) | 0.023（微弱） |
| fullD (d2) | +0.043 (n=179) | +0.037 (n=221) | 0.006（几乎为零） |
| netH (网络 prior+价值) | −0.718 | −0.776 | 0.058（且整体偏负、无法对齐胜负） |
| netI (网络价值前沿) | −0.625 | −0.600 | 负向、无分离 |

搜索的根价值**几乎没有区分度**（d1 仅 0.023；d2 仅 0.006），网络价值头也**严重失准**（均值 −0.6~−0.8 却与胜负无关）。**这从价值侧解释了为什么软目标没有信息量：teacher 的 value/strategy 是噪声。**

> 侧记：**d1 下 iterations 8→16 逐位不变**（B vs C：72/72 行 outcome/searched/changed 全同；墙钟 107.8 s vs 107.7 s）。即"预算太弱"的最直接改法（加 iterations）在本实现里**完全无效**——瓶颈不是迭代数，而是**前沿价值 + 候选 prior**。

---

## 4. 网络 prior + 学习价值前沿的敏感性（S-C 设计本意）

无 prior 时候选裁剪是任意选项序（§1）；用 CVPN checkpoint 作 **base+prior+value**（控制组=纯网络）复测：

| 档 | 配置 | pairs | 控制(纯网络) | cfr | 配对 Δ | 95% CI | 改变率 | s/搜索 |
|---|---|---:|---:|---:|---:|---|---:|--:|
| H | 网络 prior+value, d2/c4/ms4 | 36 | 0.403 | 0.361 | −0.042 | [−0.200,+0.116] | 79.9% | 6.66 s |
| I | 专家基座 + 网络 value 前沿, d1/c4/ms8 | 36 | 0.431 | 0.444 | +0.014 | [−0.145,+0.172] | 76.0% | 0.57 s |

- 加网络 prior 只把改变率从 89% 降到 80%，**仍是单调覆盖专家动作**，配对 Δ 依旧 ≤0（H：−0.042，ns）。**先验没有把搜索从"噪声覆盖"救回来。**
- 加网络 value 前沿（I）把 d1 的 Δ 从 −0.042 抬到 +0.014（仍跨 0，n=36，不可信为正）——方向上比启发式前沿略好，但远不足以推出"可承受预算下显著为正"。
- 同一网络作为"控制组"（纯策略）在子集上 pooled 0.403 vs 专家 0.431，**低于专家**，与 A3「distill≈BC≈0.50」一致：当前 CPU 档网络本身就不是靠谱的搜索先验/价值源。

---

## 5. 成本表（供 A800/本机排产）

| 档 | s/搜索（均值） | s/局(cfr) | s/局(expert) | 12 核吞吐（含两组） | 400 配对串行核时 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| expert 基线（无搜索） | — | — | ~5.1 | ~4.3e3 局/h | （800 局 ≈ 0.57 核时） | `profile_expert`，10 worker |
| **B** `i8/d1/c4/ms8` | **0.44** | **8.6** | 5.1 | **~3.1e3 局/h** | 1.53 | 全池实测 921.6 s / 800 局 |
| **D** `i8/d2/c4/ms4` | **5.78** | **28.2** | 5.1 | **~1.3e3 局/h** | 3.70 | 全池实测 2230.7 s / 800 局 |
| H `netprior+value d2/c4/ms4` | 6.66 | 33.0 | 6.3 | ~1.1e3 局/h | — | 6 子集 |
| I `netvalue d1/c4/ms8` | 0.57 | 10.4 | 5.8 | ~2.7e3 局/h | — | 6 子集 |
| A（A3 teacher 同档 i4/d1/c2/ms8） | 0.37 | 10.1 | 7.1 | ~2.5e3 局/h | — | 6 子集 |

- 搜索成本**由 depth 主导**（d2 ≈ 13–17× d1：根候选 C 个 → 每个深一层的子节点对**全部合法动作**开叉，无 `candidate_limit` 保护），iterations 几乎免费（缓存命中），`ms` 线性。
- 与 A3 汇报一致：搜索是 **CPU 密度**任务，**A800 窗口（仅 ~10 CPU 配额）不适合做搜索采集**；若走搜索路线，1000 局/组按 B 档 ≈ 0.7 核时、D 档 ≈ 2.5 核时——但**本实验证明这钱不该花**。

---

## 6. 判定与建议

### 判定：搜索 teacher 无可测增益（且显著有害）

按 D22 的事先判据（配对 CI 下界 > 0）：

- **否**。5 档敏感子集全部跨 0；全池 n=400 上 B/D **显著为负**（CI 上界 < 0）。搜索覆盖 86–89% 的专家动作、win→loss 翻转 1.5–2.0×、根价值几乎无区分度、`iterations` 在 d1 完全不改变动作。

### 建议：支持 D22 改排产分支——**先补 BC/自博弈数据，搜索蒸馏暂缓**

1. **不要把搜索蒸馏放量到 A800**（D21）：当前 teacher 的软目标劣于 expert 硬标签，放量只会放大有害信号（与 A3「distill≈BC、Δ=−0.0175」互证）。
2. **优先补数据面**：BC/自博弈（或 expert 自对弈）扩数据 + 扩卡组覆盖，先把非搜索策略的强度曲线拉起来；这是当前 ~0.50 结构性瓶颈（A3：9–10/20 套 <45%）的**真正瓶颈方向**。
3. 若**仍要**保留搜索 teacher 探针：只推荐**最便宜、伤害最小**的 B 档（`i8/d1/c4/ms8`，0.44 s/搜索、8.6 s/局、~3.1e3 局/h）——但其 Δ 已是 −0.066（显著负），**不构成放量理由**。
4. **不推荐 D 档**（d2）：贵 13–17× 且更差（−0.114）。

### 若将来要重启搜索路线，先满足的前置（否则会重蹈覆辙）

- **校准的公共信念价值头**：当前根价值区分度 ≈0.006–0.023；价值不校准，MCCFR 的 regret 目标就是噪声。
- **非任意候选裁剪**：无 prior 时 `candidate_limit` 取引擎选项序前 C 个（任意）；须由**可用的** prior 排序，或对专家动作做"仅在显著更优时才改"的门控（本实验显示默认"总是覆盖"是有害的）。
- **不止一次可信的预算扫描**：本次 d1 上 iterations 无效、depth 2 更差，说明"加预算"不是解法；需要先解决 representation/value，再谈搜索预算。

---

## 7. 未决问题

1. **value 头质量是上限还是实现缺陷？** 用更强的价值网络（大网/更多数据）能否让搜索转为正增益——本实验只用了弱 d128 网络的 value，未能分辨"搜索路线不成立"与"价值太差"。
2. **depth ≥ 3 / belief_samples > 1 未测**（成本 ~10×以上），无法排除极深搜索翻正。
3. **门控式搜索**（`search 只在根价值优势 > τ 时覆盖专家`）未测；由"搜索覆盖越多越差"（D<B）推测门控可能止血，但未验证。
4. **候选裁剪 vs 价值噪声的归因**：NetH 只降低改变率（89%→80%）未转正，尚不能完全分离两者。
5. **样本量**：子集档 n=36（CI ±0.14），只够"无显著正增益"，不足以下"某档恰好略负"的定论；全池 n=400（CI ±0.049）才给出显著负的认证级证据——后续如做正结论需 ≥50 seeds/对手。

---

## 8. 产物索引

- **本报告**：`reports/search/a4_search_budget_ladder.md`
- 聚合 JSON（入库）：`reports/search/a4_search_budget_ladder.json`（阶梯+全池+网络敏感性汇总）、
  `reports/search/ladder_{A,B,C,D,E}_*.json`、`reports/search/ladder_full{B,D}_*.json`、
  `reports/search/ladder_net{H,I}_*.json`、`reports/search/profile_expert_*.json`
- 脚本（入库）：`scripts/run_cfr_ladder.py`
- 原始逐局（gitignore）：`data/search/ladder_*.jsonl`、`data/search/profile_expert.jsonl`
- 关联：`reports/train/a3_m3_cpu_screening.md`（A3：M3-CPU 未过、distill≈BC）、PLAN.md D22
