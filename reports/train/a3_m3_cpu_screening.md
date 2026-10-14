# A3 — M3-CPU 筛选认证报告（搜索+小网 纯策略 vs 20 池）

> 日期：2026-10-12｜作者：A3 子代理｜口径：**纯策略**（不接 LLM，PLAN §6.3）
> 门禁：**「搜索+小网」纯策略 vs 评测池全部 20 套脚本对手 → 整体胜率 ≥ 50%**（M3-CPU，PLAN §4.3 / M2 口径）
> 机器：本机 12 核 CPU，无 GPU，fp32 CPU 档（d_model 64–128）
> 原始逐局 JSONL 在 `data/eval/`、`data/replays/`（gitignore）；聚合 scoreboard 在 `reports/explore/`；本报告为唯一入库摘要。

---

## 0. 结论速览（诚实数字）

| # | 候选 | 局数 | pooled | Wilson95 | cluster95 | 门禁 pooled | 门禁 overall |
|---|------|------|--------|----------|-----------|------------|--------------|
| 1 | smoke `d64`（20 局/1 对手） | 200 | **0.460** | [0.392,0.529] | [0.321,0.599] | ✗ | **FAIL** |
| 2 | distill `d128`（160 局/20 对手） | 200 | **0.485** | [0.417,0.554] | [0.357,0.613] | ✗ | **FAIL** |
| 2b | 同上，加大样本 | 400 | **0.500** | [0.451,0.549] | [0.371,0.629] | ✓（恰好压线） | **FAIL**（floor） |
| 3 | BC 对照 `d128`（消融组） | 200 | **0.502** | [0.434,0.571] | [0.374,0.631] | ✓ | **FAIL**（floor） |

**判定**：M3-CPU 门禁（整体 ≥50%）**在 point-estimate 口径下恰好压线（400 局 pooled=0.500，pooled_pass=True），但不能通过认证级口径**——Wilson 95% 下界 [0.451] 低于 0.50，`--strict-ci`（cluster CI 下界 ≥0.50 = 0.371）明确 FAIL。M4「无单套 <45%」floor 门禁**明确 FAIL**（20 套里 9–10 套 <45%，最弱 `unyielding_geo` 仅 0.05）。**诚实结论：当前 CPU 档「搜索+小网」在池上约打平（~0.49–0.50），远未达到可信 ≥50%，更不说 0.70（M4）。**

**关键诊断（搜索蒸馏是否有效）**：同等数据（7505 样本）与设置下，**搜索蒸馏组 ≈ 冷启动 BC 组**——配对差 distill−BC = **−0.0175**，95% CI [−0.084,+0.049]，不显著且不为正（`significant_positive=false`）。**在当前 teacher 预算（iterations=4 / depth=1 / candidate_limit=2）与数据量（1277 个搜索目标）下，MCCFR 搜索 teacher 没有给网络带来可测量的增益。** 这是本报告最重要的负面发现。

---

## 1. 命令清单（全部 env 内 python 直调，纯策略）

```bash
PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python

# 步骤 1：对已有 smoke checkpoint 做 200 局 screening
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/20261012T094651_distill_smoke_big_distill.pt \
  --seeds 5 --workers 10 --tag a3_screen_smoke

# 步骤 2a：采集（search teacher，20 池全量，CPU 档）
$PY -m train.collect --deck superconduct_aggro --all-opponents \
  --seed-start 6000 --seed-count 8 \
  --search --iterations 4 --depth 1 --candidate-limit 2 --max-searches 8 \
  --belief-samples 2 --bridge-workers 2 --workers 4 --tag a3_corpus_v1

# 步骤 2b：蒸馏训练（d_model 128，40 epochs；脚本 /tmp/opencode/a3_train.py）
#   同脚本的 cold-start 版训练 BC 消融组（/tmp/opencode/a3_train_bc.py）

# 步骤 3：对新 checkpoint screening（200 局 + 400 局）
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/20261012T101435_a3_distill_v1_d128.pt \
  --seeds 5 --workers 10 --tag a3_screen_v1
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/20261012T101435_a3_distill_v1_d128.pt \
  --seeds 10 --workers 10 --tag a3_screen_v1_s10

# 消融：BC 组 200 局（与 distill 200 局同种子 0–4）
$PY -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/20261012T103011_a3_bc_v1_d128.pt \
  --seeds 5 --workers 10 --tag a3_screen_bc_v1

# 规划口径
$PY -m scripts.eval_protocol --sample-sizes
```

---

## 2. 门禁原始行数核对（C8）

| 原始文件 | 行数 | 期望 | 核对 |
|----------|------|------|------|
| `data/eval/a3_screen_smoke.jsonl` | 200 | 20×5×2 | ✅ |
| `data/eval/a3_screen_v1.jsonl` | 200 | 20×5×2 | ✅ |
| `data/eval/a3_screen_v1_s10.jsonl` | 400 | 20×10×2 | ✅ |
| `data/eval/a3_screen_bc_v1.jsonl` | 200 | 20×5×2 | ✅ |
| `data/replays/a3_corpus_v1/samples.jsonl` | 160 | 20 对手×8 种子 | ✅ errors=0 |

采集汇总（manifest）：**160 局 / 7505 样本 / 0 错误**；搜索目标 **1277 条**（`has_search_target`），其中带软策略分布 `policy_target` 的 **1158 条**；`search_fraction=17.0%`；`win_fraction=0.447`（teacher 侧）。采集墙钟 10:03:29→10:09:43 ≈ **6.2 min**（4 collect worker × 2 bridge worker）。

训练：distill 组 175.6 s，BC 组 153.6 s（6004 训练 / 1501 验证，40 epochs，CPU）。
- distill 终值：train policy acc 0.746 / val acc 0.724，val value MSE 0.094，val belief loss 3.145。
- BC 终值：val acc 0.730，val value MSE 0.116，val belief loss 3.197。
（BC 的 policy acc 略高、value MSE 略差，符合"硬 one-hot CE vs 软搜索目标"的预期。）

评测墙钟：200 局 ≈ 5–8 min，400 局 ≈ 8 min（10 worker）。

---

## 3. 逐对手表（contestant 胜率；200 局列 n=10 = 5 seeds × 2 sides，400 局列 n=20 = 10 seeds × 2 sides）

| 对手 | smoke d64 | distill d128 | distill d128 (400 局) | BC d128 |
|------|:---------:|:------------:|:---------------------:|:-------:|
| unyielding_geo | 0.10 | 0.10 | **0.05** ⚠floor | 0.20 |
| electrocharged_moon | 0.20 | 0.30 | **0.20** ⚠ | 0.30 |
| double_geo_navia | 0.20 | 0.20 | **0.25** ⚠ | 0.15 |
| nahida_kokomi_raiden | 0.30 | 0.40 | **0.25** ⚠ | 0.20 |
| electro_manifestation_nuke | 0.00 | 0.10 | **0.30** ⚠ | 0.10 |
| mualani_double_pyro | 0.50 | 0.30 | **0.30** ⚠ | 0.50 |
| superconduct_mika_attack | 0.10 | 0.10 | **0.30** ⚠ | 0.40 |
| dvalin_bonk | 0.30 | 0.40 | **0.35** ⚠ | 0.20 |
| shenhe_double_swirl | 0.20 | 0.50 | **0.40** ⚠ | 0.50 |
| aywolfskirk | 0.50 | 0.50 | 0.45 | 0.30 |
| superconduct_aggro | 0.20 | 0.50 | 0.45 | 0.60 |
| klee_mavuika_mualani | 0.60 | 0.40 | 0.50 | 0.50 |
| trial_skirk_ayaka_navia | 0.50 | 0.40 | 0.50 | 0.40 |
| dual_mualani_stacks | 0.60 | 0.60 | 0.55 | 0.80 |
| yelan_clorinde_ororon | 0.70 | 0.50 | 0.60 | 0.60 |
| natlan_battleship | 0.60 | 0.90 | 0.85 | 0.80 |
| ice_water_battleship | 0.90 | 0.80 | 0.90 | 1.00 |
| skirk_ayaka_navia | 0.90 | 0.90 | 0.90 | 0.80 |
| beidou_skirk_ayaka | 0.90 | 0.90 | 0.95 | 0.90 |
| skirk_chasca_freeze | 0.90 | 0.90 | 0.95 | 0.80 |

（smoke/distill/BC 三列均为 200 局、同种子 0–4；400 局列为 distill 加大样本。⚠ = <0.45 floor。）

**最弱项（400 局口径）**：`unyielding_geo`（0.05）、`electrocharged_moon`（0.20）、`double_geo_navia`/`nahida_kokomi_raiden`（0.25）。**最强项**：`skirk_chasca_freeze`/`beidou_skirk_ayaka`（0.95）。**最优反应探针**：worst=`unyielding_geo`，exploitability_lower_bound=0.95。

**均值 vs 结构**：整体 ~0.50，但**方差极大**（0.05–0.95）：说明网络已能稳定吃掉一小半对局（冰队/纳塔系），但被另一半（岩/雷/部分草系）稳定压制，且**没有随数据/规模增长而收敛**——polished d128 相对 d64 仅 +0.025（0.460→0.485），落在一个标准差内。

---

## 4. 门禁判定（逐项）

| 门禁项 | 口径 | smoke d64 | distill d128 (400) | 判定 |
|--------|------|:---------:|:------------------:|------|
| pooled ≥ 0.50（point） | M2/M3-CPU | 0.460 ✗ | **0.500 ✓（压线）** | **边界通过** |
| pooled ≥ 0.50（strict-CI，cluster 下界） | 认证级 | 0.321 ✗ | 0.371 ✗ | **FAIL** |
| Wilson 95% 下界 ≥ 0.50 | 认证级 | 0.392 ✗ | 0.451 ✗ | **FAIL** |
| floor：无单套 <45%（M4） | M4 | 9 套违规 ✗ | 9 套违规 ✗ | **FAIL** |
| quality：fallback/error = 0 | 硬门禁 | 0/0 ✓ | 0/0 ✓ | PASS |
| distill ≥ BC（搜索蒸馏有效） | 消融 | — | Δ=−0.0175,CI[−0.084,+0.049] | **无增益** |

**M3-CPU go/no-go 结论**：**不通过认证**。screening-grade 的 pooled 恰好触线（0.500），但 (a) 认证级 CI 下界远低于 0.50，(b) M4 floor 门禁崩盘（9/20 <45%），(c) 搜索蒸馏相对 BC 无可测量增益。**不满足"放大规模"的前置条件**——把当前配方直接丢给 A800 只会放大计算量、不改变 ~0.50 的结构性瓶颈。

---

## 5. 瓶颈分析

1. **搜索 teacher 每局成本（首要瓶颈，CPU 绑定）**
   - 实测：`max_searches=8, iterations=4, depth=1, candidate_limit=2, belief_samples=2` 下 **≈9.4 s/局/worker**（160 局、4 worker、6.2 min 墙钟；单 worker 探针 11.5 s/局）。
   - 折算 **≈1.2 s / 搜索**（含 fork 子进程 + 2 belief 确定化 × 4 iteration × 2 candidate rollout）。搜索目标占样本 17%。
   - 采集 4000 局（本机 4 worker）≈ 2.6 h；这是 CPU 密度任务，**A800 无助于此**（D21：窗口 CPU 配额仅 ~10 核）。
2. **teacher 质量过低**：`depth=1` 近 1-ply + rollout、`candidate_limit=2`、`iterations=4` → 根平均策略噪声大。**这正是"蒸馏 ≈ BC"的最可能原因**：teacher 的软目标没有比 expert 硬标签更有信息量。需先做 teacher 预算消融（iterations/depth/candidate_limit ↑）再放量。
3. **网络容量/数据量**：CPU 档 d_model 64→128、20→160 局、0→1277 搜索目标，pooled 仅 +0.025–0.04。7.5e3 样本对 20 套异构对手过少（每对手仅 8 局训练样本），**长尾 matchup 欠拟合**（岩/雷系崩盘）。
4. **无 GPU**：torch 2.14.1+cpu，训练非瓶颈（d128×40ep≈3 min），但**无法用 GPU 放大网络/批推理**，只能停留在 CPU 档。

---

## 6. 给 A800 窗口的放量参数建议（遵守 D21：窗口只做 GPU 训练/推理）

**职责切分（D21 硬约束）**：采集/评测留在本机 12 核 + WSL（A800 窗口 CPU 仅 ~10 核，不放采集/评测）；**A800 只做 CVPN 放大训练 + 批推理**；数据本机采集后传入，checkpoint 回流本机做认证。

| 维度 | CPU 档（本次，ing） | A800 窗口建议 |
|------|--------------------|----------------|
| 采集地点 | 本机 12 核（4 worker） | **本机/WSL**（不放 A800）；用 12 worker + 每 worker 1 bridge 提吞吐 |
| 采集规模 | 160 局 / 7.5e3 样本 / 1277 搜索目标 | **3000–6000 局 / 1.5e5–4e5 样本**（20 对手 × 150–300 种子）；同 `--all-opponents` |
| teacher 预算 | iter 4 / depth 1 / cand 2 / max_searches 8 | **先做消融**：iter 16–32 / depth 2 / cand 4–8 / max_searches 16–32；确认 soft-target 有增益再全量放量 |
| 网络 | d_model 128，belief head，fp32 | **d_model 256–512**（n_heads 8），fp32 或 bf16；batch 256–1024 |
| epochs/lr | 40 / 1e-3 | **100–300 / 1e-3→3e-4（cosine）**；checkpoint 每 20 ep 落盘，按断点续跑 |
| 训练硬件 | CPU（3 min/run） | A800（`train/*gpu*.lsf` / `train.pipeline --replays --device cuda`）；注意窗口内装 CUDA 版 torch |
| 认证 seeds | 5（200 局）/10（400 局） | **≥50 seeds/对手 → 1000+ 局/组**，Wilson 半宽 ≈0.03；跑 `--strict-ci` 出认证级判定 |
| 消融 | distill vs BC 200 局 | 每轮 search-distill vs BC **配对**（同 seeds），`significant_positive` 才推进 |
| 窗口排产（PLAN §5.1） | — | 训练 > 评测 > 采集 > LLM；采集可离线错峰，不占窗口 |

**放量前的 checkpoint（不满足则 A800 窗口收益有限）**：teacher 消融证明 search-distill 显著优于 BC（配对 CI 下界 >0），且 pooled 随 d_model/数据单调上升。**当前两项均未满足。**

---

## 7. 未决问题

1. **teacher 预算消融未做**：iterations/depth/candidate_limit 的最优点未知；这是判断"搜索蒸馏路线是否成立"的前置实验。
2. **规模-强度曲线仅两点**（d64/20 局 vs d128/160 局），且种子集不同；需要同 seeds 的受控扫描（`train/learning_curve.py` 式）确认是否单调。
3. **长尾 matchup 归因**：岩/雷系（`unyielding_geo` 等）崩盘是观测表示、动作抽象还是数据不足？需 replay 复盘（`coach/replay.py`）。
4. **BC 组是否已足够**：若 teacher 消融后仍 ≈BC，则应重新评估"先补 BC/自博弈数据、后搜索蒸馏"的排产。
5. **评测样本量**：400 局 Wilson 半宽 ±0.049，仍无法在 0.50 附近做认证；认证需 ≥1000 局/组。

---

## 8. 产物索引

- 聚合 scoreboard（入库，`reports/explore/`）：`a3_screen_smoke.json`、`a3_screen_smoke_<ts>.json`、`a3_screen_v1.json`、`a3_screen_v1_s10.json`、`a3_screen_bc_v1.json`。
- 训练聚合（入库，`reports/train/`）：`a3_distill_v1.json`、`a3_bc_v1.json`。
- 原始逐局/语料（gitignore，`data/`）：`data/eval/a3_screen_*.jsonl`、`data/replays/a3_corpus_v1/`。
- checkpoint（gitignore，`data/checkpoints/`）：`20261012T101435_a3_distill_v1_d128.pt`、`20261012T103011_a3_bc_v1_d128.pt`。
