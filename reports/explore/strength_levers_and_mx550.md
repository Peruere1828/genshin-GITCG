# 强度提升杠杆 + MX550 可训性（研究备忘，2026-10-16）

> **status: superseded（2026-10-27）**：文中把 AWR/Q 当“最便宜杠杆”，10-17 后已被证否；
> 见 `docs/failures/002-awr-q-dmc.md`。当前方法见 `docs/PLAYBOOK.md`。

> 口径：**纯策略**（PLAN §6.3）。本文只做两件事：① 依据现有实测（D22–D24 + `reports/train/`、`reports/search/`）
> 排出「提升 AI 博弈水平」的杠杆次序与验收；② 回答「MX550 能否训模型」并给出可复现的 WSL 命令。
> **未在本机复跑 MX550**（本机为 12 逻辑核 / 30 G / `torch 2.14.1+cpu`，无 CUDA）；MX550 结论来自
> `reports/explore/model_design.md`（2026-10-11 WSL 实测）与 `scripts/bench_ops.py` 的实测记录。

---

## 0. 结论速览

1. **当前 0.46–0.71 的强度不是算力问题，是「优化目标 + 长尾 matchup」问题**：数据 8k→48k（同 epoch）无显著增益，
   而 5→15 epoch 显著 +0.076（D24）；容量 d128≈d256。**主杠杆＝训练目标（自博弈/离线 RL）与逐 matchup 覆盖**，
   不是更多专家 BC 数据，也不是更大网络。
2. **BC 的上限就是专家**（同 deck 最好也只统计打平）→ 要「顶过专家」必须引入**非专家来源的信号**：
   离线 RL（对现有语料按回报重加权）、自博弈、或修好的搜索（D23 三个前置）。
3. **验收门禁的真正拦路虎是 M4 floor**（无单套 <45%）：全部 deck FAIL，长尾岩/雷/草系是结构性瓶颈，且与数据量无关。
   floor 是 per-opponent 指标 → **按 matchup 定向补数据 + 对手条件化**是最直接、最便宜的攻坚手段。
4. **MX550 能训小网络（已实测：d128 2 层 transformer，431k 参数，265 MB 峰值显存，30 epoch / 2k 样本 ≈ 14 s），
   但它是 Turing（cc 7.5、无 tensor core、2 GB / 64-bit、~2.7 TFLOPS fp32）**：
   fp16/bf16 比 fp32 更慢（AMP 有害），只能 fp32、小 batch、d≤128~256。**它不是提强度的路径**，
   正确定位＝CI/冒烟节点 + 并行采集/评测节点；训练默认仍在本机 CPU，规模化等 A800 窗口。

---

## 1. 强度诊断（全部为已有实测）

| 观测 | 数字 | 证据 |
|---|---|---|
| 评测池 | 20 套脚本专家（`eval.opponents`） | D11 |
| 池内最强专家（分 deck） | 0.681–0.799（dvalin / unyielding / double_geo）；`superconduct_aggro` 0.509–0.561 | `reports/datafirst_bc.md` §4、D23 |
| BC 在同等 deck 的强度 | 专家 −0.3 ~ −11.4 pp；最好一组（double_geo）打平（Δ=+0.007，CI 含 0） | `reports/datafirst_bc.md` §0 |
| 认证级（20 seeds / n=800） | pooled 0.458 vs 专家 0.519，Δ=−0.061 [−0.100,−0.022] **显著负** | `reports/train/bc_scale_datafirst.md` §0 |
| 数据量轴（8k→48k，同 5 ep） | Δ=+0.023 [−0.036,+0.081] **不显著** | 同上 |
| 训练时长轴（5→15 ep，同 48k） | Δ=+0.076 [+0.023,+0.129] **显著** | 同上 |
| 容量轴 | d128@48k 0.458 vs d256@48k（3 ep，epoch 不匹配）0.225；d128≈d256（6 ep） | `reports/datafirst_bc.md` §3/§5 |
| 搜索 teacher（可行预算） | 全池配对 Δ **显著为负**；搜索覆盖专家动作 ~87–89% 但净负，根价值区分度 ≈0 | D23、`reports/search/a4_search_budget_ladder.md` |
| M4 floor（per-opponent Wilson 下界 ≥0.45） | 5 deck **全部 FAIL**（6–20 个对手违规）；SA 5ep 13/20、15ep 11/20 违规 | `reports/datafirst_bc.md` §6 |
| 最优反应探针（exploitability 下界） | 0.76–0.95（岩/雷/草系压住全部 deck） | `reports/datafirst_bc.md` §4 |
| 模型侧 | 注意力**策略头**略优（val_acc 0.741→0.758），**价值头明显变差**（val_loss 0.856→1.176） | `reports/explore/model_design.md` §2 |

**归因**：① 网络从未追上专家（BC 天花板＝teacher）；② 长尾 matchup 与数据/容量无关，说明是**表示/目标/覆盖**问题；
③ 价值头是全局短板——它同时卡住搜索（D23 前置之一）与任何 RL/自博弈信号。

---

## 2. 提升杠杆（按「千卡·时收益」排序）

| # | 杠杆 | 做什么 | 成本 | 验收 |
|---|---|---|---|---|
| **L1** | **测量先修** | 逐 deck 报分 + ≥20 seeds + `--strict-ci`，把 floor 违规数当一等指标（D24 已定） | 0（已有一半） | 每次改动都出「pooled / Δ vs 同 deck 专家 / floor 违规数」三元组 |
| **L2** | **离线 RL（对现有语料）** | 在已有 372k 决策 / 8k 局语料上做 AWR/CRR 式重加权（权重 = exp(β·(episode 回报 − V(state)))，退化即 BC）或 DMC 式 Q 回归（DouZero 蓝本 `refs/AI`，用 score/margin 做回报）。**不改采集，不改引擎** | 低（CPU 可跑；d128 数百秒/轮） | 同数据同 epoch 的 A/B：AWR vs BC 配对 Δ 及其 CI；**这是最便宜的一次「超过 teacher」尝试** |
| **L3** | **价值头标定** | 把 value 目标从「终局胜率」改成 HL-Gauss/分布化 + 用 S-C 已有 `search_value` 目标；测 ECE 与 |根价值−实际胜率| | 低（复用 `train/distill.py`） | 标定误差显著下降；**这是重开搜索的硬前置**（D23） |
| **L4** | **长尾 matchup 覆盖** | ① 定向采集：`--opponents <最差 3–5 套>` × 更多种子（当前每 deck×池均匀 80 seeds，最差 matchup 只有 ~数十局）；② **对手条件化**：把对手公开的 3 个角色/archetype 编码成 token（对手阵容是公共信息），并试点**多 deck 单模型**（`reports/datafirst_bc.md` §9.4/§9.5 未决问题） | 中（采集 = 引擎 CPU 时间，~4,000 局/h） | floor 违规数从 6–20 降到个位数；最差 matchup 净涨 |
| **L5** | **自博弈闭环** | 以已「打平专家」的 3 套 deck（dvalin / unyielding / double_geo）当种子：`train.collect --teacher-spec neural:<ckpt> --opponent-spec neural:<ckpt2>` + 温度采样探索 + 保留 `expert:<deck>` anchor 防漂移；候选 checkpoint 逐轮过门禁，只追加不覆盖（WS5 护栏） | 中高（数据平面已有，缺循环脚本 + checkpoint 池） | 阶梯：net_t vs net_{t−1} 与 vs 池内锚点；**只有 L2/L3 有信号后才值得投入** |
| **L6** | **表示/动作** | 分层动作头（high-level 分类 + 目标/骰子参数头）；把 belief 从「辅助头」升级为**策略输入特征**（隐藏信息下的反制更准）；token 分段 + 公开事件历史 | 中 | 动作头 CE 与 floor 变化；信息泄漏测试仍须 PASS |
| **L7** | **LLM 辅助档（分开报分）** | 候选重排/校验/兜底（≤10 次/局），实测收益已可量化（`scripts/run_llm_matrix.py`） | 低（API） | 单列「LLM 辅助」口径，**不进 G1 验收**（§6.3） |
| **L8** | **修好的搜索（重开）** | 仅在 L3 达标 + 非任意候选裁剪 + 门控式覆盖**三者齐备**后再跑预算阶梯 | 高 | 复制 A4 阶梯协议，配对 Δ 需 ≥0 且 CI 下界 >−0.02 |

### 建议的推进顺序（每步都是可独立验收的 A/B）

1. **L2 + L3 并行**（都不需要新算力/新数据，1–2 天量级）：先证明「离线 RL 能超过 BC」且「价值头能标定」。
2. **L4**：靶向 worst-matchup 补采 + 对手条件化 —— 这是唯一直接打 floor 的手段。
3. **L5**：三个打平 deck 上起自博弈闭环，跑阶梯。**L8 不先开**。

---

## 3. 明确不做（有负面证据或违反 charter）

- 同 epoch 下继续堆专家 BC 数据（D24：不显著）；
- 「先上 A800 放大网络/bf16」（D22：容量不是瓶颈，放大只放大结构问题）；
- 在价值头未标定前把搜索蒸馏放量（D23：软目标劣于硬标签）；
- 用 PPO/GAE 作主训练信号（charter §8/§13.1，`train/CONTINUAL_RESOLVING.md` §4）。

---

## 4. 算力现实（决定杠杆可行性）

| 环节 | 所在硬件 | 实测 |
|---|---|---|
| 对局步进/采集/评测/MCCFR 搜索 | **纯 CPU（V8/cffi）**，GPU 无用 | 12 核采集 4,000–4,900 局/h；评测 ~1,500 局/h |
| CVPN 训练（d256，本机 CPU） | CPU | 150–300 样本/s（含读盘、受并发竞争） |
| CVPN 训练（A800） | GPU | batch 512 → 276k 样本/s（≈ 10 核数据平面的 100×） |
| 结论 | — | **窗口瓶颈是数据不是算力**；工程保持「本地采集 → 传入 GPU 训练 → checkpoint 回流本地认证」 |

---

## 5. MX550 能否训练模型

### 5.1 硬件事实

| 项 | 值 |
|---|---|
| 架构 | Turing **TU117-670-A1**（笔记本入门 dGPU，2022-03） |
| 计算能力 | **cc 7.5**（与 T4/RTX 20 系同代 CUDA 特性） |
| 核心 | 1024 CUDA / 32 TMU / 16 ROP（**无 Tensor Core**） |
| 显存 | **2 GB GDDR6，64-bit**，~96 GB/s |
| 算力 | fp32 **~2.70 TFLOPS**（fp16 同速，无 TC 加速）；TDP 15–30 W |

### 5.2 已实测（WSL MX550，fp32，2 GB）

| 模型 | 样本 | 设置 | 墙钟 | 峰值显存 | 备注 |
|---|---|---|---|---|---|
| `CVPN` d128（mean-pool，100k 参数） | 2,067（30 ep） | precolate，batch 64 | **5.7 s** | **185 MB** | val_acc 0.724 |
| `TransformerCVPN` d128 2L（431k 参数） | 2,067（30 ep） | precolate，batch 64 | **14.2 s** | **265 MB** | val_acc 0.729；策略头略优、价值头变差 |
| `TransformerCVPN` 3L | 2,067 | 同上 | 更慢 | 更高 | 更差（过拟合/欠优化） |

推算网络前反向吞吐：mean-pool ≈ **8.6k 样本/s**、2L transformer ≈ **3.5k 样本/s**（precollated，纯计算）。

**dtype 结论**：`scripts/bench_ops.py` 记录了 MX550（cc 7.5，无 tensor core）上 **fp16/bf16 matmul 比 fp32 更慢**
→ **不要开 AMP**，统一 fp32。小网络是 **kernel launch / H2D 传输 bound**，不是 FLOP bound（precollate 收益最大）。

### 5.3 边界与结论

- ✅ **能训**：现有 `CVPN` / 2 层 `TransformerCVPN` 系列（≤ ~0.5M 参数、d≤128、batch 32–64、fp32）在 2 GB 内很宽松；
  d256 的 mean-pool 也够。
- ⚠️ **勉强**：d256 / 4L 级别（按 A800 上 d512·8L=12M 参数→6.65 GB 反推，约 0.8–1 GB）**估计可行但无余量**
  （估算，未实测）。
- ❌ **不能**：bf16/AMP 加速、大 batch、d512·8L 那类 12M 参数配置、任何 LLM 微调（权重本身 ≫2 GB）。
- 第二个瓶颈：WSL 侧 **4 核 / 8 GB**。数据读盘（jsonl/compact 解码）与引擎 rollout 比 GPU 更早饱和，
  8 GB 内存限制 precollate 池大小 —— 即使 GPU 有余量，**端到端 epoch 时间由 CPU/IO 决定**。

**对「提升博弈水平」的判定**：MX550 **不是**提强度的路径（网络不是瓶颈，D22/D24）。它的正确定位：
1. **CI/冒烟节点**：跑训练管线、自博弈循环、GPU 代码路径（AMP/`torch.compile` 正确性）的回归，避免 A800 窗口内才发现 bug；
2. **并行采集/评测节点**（4 核，分片累积，`train.collect` 断点续跑）；
3. 可选：小网络 GPU 训练，本机 CPU 饱和时用来分担。
训练节点归属维持现状：**默认本机 12 核 CPU，规模化等 A800 窗口**。

### 5.4 补测命令（在 WSL MX550 上跑，回填 PLAN §5.2）

```bash
# 设备对照（同池、同超参，cpu vs cuda 直接可比；含 d256 上限探边）
python -m scripts.bench_device --devices cpu cuda --d-models 64 128 256 \
    --epochs 12 --batch-size 32 --precollate --games 6
# 算子/带宽/dtype 与批推理（确认 fp32≫fp16/bf16 与 launch-bound 的结论）
python -m scripts.bench_ops --device cuda
# 注意力模型 d128/d256 × 2/4 层（显存与强度对照）
python -m scripts.bench_model --device cuda --d-model 256 --n-layers 4 --epochs 30
```

---

## 6. 立即执行队列（本地可跑，不依赖机会算力）

| 序 | 任务 | 产物/验收 |
|---|---|---|
| 1 | L2：AWR/CRR 重加权 vs BC 的配对 A/B（复用 `train/distill.py` 损失框架 + 已有 store） | `reports/train/` 配对 Δ + CI；floor 违规数 |
| 2 | L3：value 头标定（分布化 / `search_value` 目标）与 ECE 对比 | 标定报告；搜索重开前置是否达标 |
| 3 | L4a：worst-matchup 定向补采（5 套最差对手 × ≥200 种子/deck） | floor 违规数变化 |
| 4 | L4b：对手 archetype token + 多 deck 单模型试点 | 同上 |
| 5 | L5：自博弈循环脚本 + checkpoint 池 + 门禁（种子＝已打平专家的 3 套 deck） | 阶梯曲线（vs 锚点与上一代） |

---

## 7. 未决

1. 长尾 floor 的归因仍未定：**表示、动作抽象、还是数据覆盖**（`reports/datafirst_bc.md` §9.2）——L4b 与 L6 分别探一侧。
2. 稀疏回报下的离线 RL 权重（β、回报定义：胜率 / 骰子余量 / 回合数）未做敏感性。
3. WSL 上 `torch.compile` 对 cc 7.5 的收益未测（Turing 上多半无收益或有反效果）。
4. MX550 端到端 epoch 时间（含读盘）未测 —— 决定它能否当第 3 个训练节点。
