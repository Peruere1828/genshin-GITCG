# 2 天高强度实验计划：从 0.46 到「顶过专家 + 修 floor」（含 Kaggle GPU）

> 2026-10-16。排序原则（用户指定）：**效果优先，运行时长次要（上限 ≈2 天）**。
> 依据：`reports/datafirst_bc.md`、`reports/train/bc_scale_datafirst.md`、`reports/search/a4_search_budget_ladder.md`、PLAN §0 D22–D24。
> 结论前置：**真正能抬强度的是两件事**——① 在现有语料上引入**非模仿的改进算子**（Q/DMC 或优势重加权）；
> ② **价值头标定**（它是搜索与 RL 的共同前置）。③ floor 靠**定向数据 + 对手条件化**。
> 这三件都**不需要新的大算力**，只需要「很多 epoch」——这正是 Kaggle GPU 能廉价买到的。

---

## 1. Kaggle 可用性（已核实）

| 项 | 事实 | 来源 |
|---|---|---|
| CLI | 本机 `/home/ubuntuaaa/miniconda3/bin/kaggle` **2.2.4**，账号 `fmyh1828`，`~/.kaggle/kaggle.json` 已配 | 本机实测 |
| GPU 额度 | `kaggle quota` → **GPU 30.00h 剩余 / 30h，2026-10-17 刷新**（即 2 天内可用 ≈**60h**；TPU 20h） | `kaggle quota` |
| **T4×2 可脚本指定（解决「要手点网页」）** | `kaggle kernels push --accelerator NvidiaTeslaT4`，或 metadata `"machine_shape": "NvidiaTeslaT4"`（官方文档：`NvidiaTeslaT4` = GPU T4 ×2；另有 `NvidiaL4`、`TpuV5E8`） | kaggle-cli 2.2.4 `--help` + `docs/kernels_metadata.md` |
| **P100：不要指望** | ① Kaggle 已公告 **2026-09-15 停用 P100**；② 即便出现，Kaggle 现行 torch 只编到 **sm_70+**，P100 是 **sm_60** → 直接不可用（除非自装旧 torch，需联网+时间，不划算） | Kaggle 公告 + `Kaggle/docker-python#1546` |
| 单会话 | **12h 上限**，`kaggle kernels push -t <秒>` 可控；脚本 kernel 支持（`kernel_type: "script"`） | Kaggle 文档 |
| 机器 | 4 vCPU / 29 GB RAM / **16 GB 显存 ×2** / `/kaggle/working` 20 GB 自动保存 | Kaggle 文档 |
| 数据集 | **200 GB/数据集**、**200 GB 私有总量**（旧的 20 GB 说法已过时）；挂载在 `/kaggle/input/<slug>`（只读） | Kaggle 文档 |
| **网络** | Kaggle 的 **blob 上传端点是 `googleapis.com`**，本机直连超时（实测挂在 0 B/s）；**必须走本地代理**（本机 `127.0.0.1:7890`，实测上传 5–30 MB/s） | 本机实测（`~/.kaggle` 已有凭据；`kaggle datasets list` 直连 OK，上传不 OK） |

**判读：Kaggle 对我们是「免费的 epoch 预算」，不是「免费的对局引擎」。**
采集/评测/MCCFR 全在 V8/cffi（CPU），Kaggle 只有 4 vCPU（≈1/3 本机）→ **采集与认证留在本地或 WSL**。

### 1.1 训练路径已在本地验证「无引擎可跑」

`scripts/datafirst_bc_train.py` / `datafirst_bc_store.py` **不导入 gitcg**（`weights_only=False` 读 store）。
实测（本地，模拟 Kaggle 只读输入）：

```bash
GITCG_DATA_DIR=/tmp/k/data GITCG_REPORTS_DIR=/tmp/k/reports \
python -m scripts.datafirst_bc_train --store-cache data/store/double_geo_navia.pt \
  --d-model 64 --epochs 1 --train-limit 512 --val-limit 256 --out /tmp/k/ckpt_smoke.pt --report /tmp/k/reports/smoke.json
# [load] cache ... train=512 val=256 → epoch 0 → [done] train_seconds=0.2（10s 含 6GB 级 store 载入路径）
```

两个必须在 bundle 里带上/绕开的点：
- `envs/decks.py` 顶层 `from gitcg import Deck`（被 `envs.observation.default_encoder()` 间接引用）→ **把 11 MB 的 `gitcg-0.21.0-py3-none-manylinux_2_34_x86_64.whl` 一起传**（`njugit/vendor` 分支已有；也可 Kaggle 联网 `pip install gitcg==0.21.0`）。
- `agents/scripted/assets.py` 的缓存路径 **不吃** `GITCG_DATA_DIR`，硬编码 `repo_root/data/assets/...`（可用 `GITCG_ASSETS_CACHE` 覆盖）→ bundle 里带上 `data/assets` + `configs/`，或设该环境变量。

---

## 2. 实验设计（效果优先）

### 2.1 四个假设（每条都有独立 A/B 与门禁）

| 假设 | 内容 | 现状证据 | 本次判定门禁 |
|---|---|---|---|
| **H1 预算** | 100–200 epoch × 全量语料能继续抬强度 | 5→15 ep 显著 +0.076（D24 明确建议「优先把 epoch 放大到 100–200」） | pooled ≥0.50 且 **floor 违规数明显下降** |
| **H2 改进算子** | 非模仿目标（**Q/DMC 的 MC 回报** 或 **优势加权 BC**）能**超过同 deck 专家** | BC 天花板＝teacher（最好只打平） | 配对 Δ 的 **CI 下界 ≥ −0.02**（非劣）+ 点估计 >0 |
| **H3 价值标定** | 价值头标定（分布化/HL-Gauss + `search_value` 目标）→ 搜索 Δ ≥ 0 | 现在根价值区分度 ≈0、搜索 Δ 显著负（D23） | 标定 ECE 显著下降 **且** 搜索配对 Δ ≥ 0 |
| **H4 floor** | worst-matchup 定向数据 + 对手 archetype 条件化能修长尾 | floor 全 deck FAIL、与数据量无关（结构性） | floor 违规数 **6–20 → ≤5** |

### 2.2 待写代码（都在训练侧，本地可单测）

1. **`Q/DMC` 损失**：同一 CVPN trunk 换成「每个 option 出 Q 值」，目标＝**MC 回报**（episode 结果，可加余量/回合数 shaping）；推理＝`argmax Q`（先验/搜索接口不变）。这是唯一能**选到 teacher 不会选的合法动作**的改进算子（DouZero 蓝本：`refs/AI`）。
2. **优势加权 BC（AWR/CRR）**：`w = exp(β·(MC 回报 − V(s)))`，clip；`β=0` 精确退化为现 BC（回归测试）。
3. **价值头标定**：`--value-target {outcome, hlgauss, search_value}`；HL-Gauss 用 101 bins 的交叉熵替代 MSE；`search_value` 已由 S-C 写进 replay（`train.data.collect_search_samples`）。
4. **对手条件化**：观测追加「对手 archetype token」（对手 3 个角色是**公共信息**）+ 多 deck 单模型（`reports/datafirst_bc.md` §9.4/§9.5 未决）。
5. **Kaggle 侧胶水**：`kaggle/train_arm.py`（跑 run 列表、`CUDA_VISIBLE_DEVICES` 分卡、**幂等可续**：已存在的 checkpoint 跳过、逐 run 落 `summary.json`）+ `kernel-metadata.json` + `scripts/kaggle_push.sh` / `kaggle_pull.sh`。

---

## 3. 排产（Day 0 / 1 / 2）

### Day 0 — 本地准备（≈4–6h，含后台传输）

| # | 任务 | 产出 | 备注 |
|---|---|---|---|
| T0.1 | 打 Kaggle 数据集 `gitcg-train-v1`：repo 代码 tar（不含 data/reports）＋ gitcg wheel（11 MB）＋ `configs/`＋`data/assets`（3.3 MB）＋ store 语料（`store/sa_all.pt` 6.0 G、`store/dfbc_*.pt` 4.6 G，合计 ≈11 GB ≤ 20 GB 上限） | `kaggle datasets create -p bundle` | **先起后台传**，是整条链路最长的一环；若上行太慢，先只传单 deck store（6 G）跑通 |
| T0.2 | 写 `kaggle/` 胶水（kernel 脚本 + metadata + push/pull） | `kaggle kernels push --accelerator NvidiaTeslaT4` | 私有 kernel；`enable_internet: false`（自给自足） |
| T0.3 | **CPU kernel 冒烟**（不烧 GPU 额度）：挂载数据 → 装 wheel → 1 epoch → 回写 checkpoint | `kaggle kernels status/output` | 验证路径/权限/numpy-torch 版本兼容（pickle：torch≥2.6 默认 `weights_only=True`，本仓全用 `False`） |
| T0.4 | 实现 §2.2 的 1–3（训练侧）+ 单测 | `tests/test_offline_rl.py` | β=0 ≡ BC；Q 目标不引入隐藏信息到**输入**（只作标签，符合 §C10） |
| T0.5 | 启动 **worst-matchup 定向补采**（tmux，后台） | `data/replays/tgt_<deck>_<opp>` | 每 deck 取最差 3–5 套对手 × 200–300 种子；本机 ~4,000 局/h |

### Day 1 — Kaggle 会话 1（≈10–11h，2 卡并行）＋ 本地认证

| 卡 | 跑什么 | 配置 |
|---|---|---|
| 0 | **H1 预算轴**：BC d256，同数据同种子，**15 / 50 / 150 epoch** | `store/sa_all.pt` + `dfbc_sa.pt`；batch 512–1024、lr 1e-3→3e-4 cosine、fp16 AMP（T4 有 tensor core）+ `--precollate` |
| 1 | **H2 改进算子**：AWR β∈{0.5,1,2} × 100 ep；**Q/DMC** × 100 ep；对照＝同 seed 的 BC-100ep | 同上；每个 run 落 checkpoint + `summary.json` |
| 本地 | 拉回 checkpoint/report → `scripts/eval_protocol --contestant neural:<ckpt> --contestant-deck <deck> --seeds 20 --strict-ci --baseline expert:<deck>`；**只认证每组 top-2** | 800 局/组 ≈ 30–60 min（12 核） |
| 本地（并行） | 定向补采继续跑 | — |

**门禁 G1**：H1 出 pooled/floor 数字；H2 出「Δ 下界 ≥ −0.02 且点估计 >0」→ 若过，H2 的赢家即刻作为 **H5 自博弈**的种子策略。

### Day 2 — Kaggle 会话 2（2 卡）＋ 本地搜索消融

| 卡 | 跑什么 | 门禁 |
|---|---|---|
| 0 | **H3**：三个 value 目标（outcome / HL-Gauss / `search_value`）各训一版 → 出 ECE/AUC；把标定最好的 value 交给本地 fork 搜索（`ForkSearchPolicy` 的 `priors` / `--value-spec`）跑 A4 同协议的配对消融 | **G2**：搜索配对 Δ ≥ 0（否则正式关闭搜索分支，写进 §0） |
| 1 | **H4**：多 deck 单模型 + 对手 archetype token，训练集＝Day 0 的定向补采 + 现有语料 | **G3**：floor 违规数 ≤5；最差 matchup 净涨 |
| 本地 | 若 G1 过：**启动自博弈迭代 1**（用 H2 赢家对池 + 自我对弈，~10h/40k 局，`train.collect --teacher-spec neural:<ckpt> --opponent-spec ...`，保留 `expert:` 锚点防漂移） | 阶梯：net₁ vs 上一代/锚点 |
| Kaggle 尾部 | 自博弈数据到位则训一版（下一周额度仍在） | — |

### 资源账

- Kaggle GPU：Day1 ~11h + Day2 ~11h = **~22h ≤ 60h**（30h/周，10-17 刷新）。
- 本地 CPU：采集 6–8h、认证 4–6h、实现 4–6h；不与 GPU 抢。
- 上传：≈11 GB（上行决定，1.5–5h，后台跑）。

---

## 4. 风险与回退

| 风险 | 回退 |
|---|---|
| 上传太慢/超 20 GB | 只传单 deck store（6 G）+ 该 deck 的实验先跑；多 deck 语料下一版数据集 |
| numpy/torch pickle 版本不兼容（本机 numpy 2.4 / torch 2.14+cpu） | 用 compact replay 在 Kaggle 侧**现场重建 store**（`datafirst_bc_store.py`），或改用 `.npy` 落盘 |
| 12h 会话被砍 | `-t 39600` 提前优雅退出；`train_arm.py` 幂等（已有 checkpoint 跳过）→ 新版本 kernel 续跑 |
| Q/DMC 数据不足（仅 ~8k 局 / 372k 决策） | 期望值放低（几个 pp）；通过则说明「自博弈采集」值得投，这是本计划最大的分叉点 |
| 价值标定后搜索仍 Δ<0 | 正式归档搜索分支（写 §0），资源全投 H2+H5 |
| floor 修不动 | 说明是**表示**问题 → 下一轮押注分层动作头 + belief 作输入特征（§0.1 C10 前提不变） |

---

## 5. 「批准后立刻做的 3 件事」

1. T0.1 起后台传输数据集（11 GB，最长链路先跑）。
2. T0.2+T0.3 搭 `kaggle/` 胶水并做 **CPU 冒烟**（不消耗 GPU 额度，验证挂载/装包/回写）。
3. T0.4 实现 Q/DMC + AWR（本地单测），同时 T0.5 起定向补采。

> 说明：以上**不改验收口径**（仍是纯策略、逐 deck、≥20 seeds、strict CI；PLAN §6.3/D24）。只有当某个门禁（G1/G2/G3）真的翻转时，才建议在 PLAN §0 追加决策记录 + 更新 §10 的「下一步」。

---

## 6. 执行记录

### Day 0（2026-10-16）——已完成

| # | 项 | 状态 | 证据/产物 |
|---|---|---|---|
| T0.0 | Kaggle 可用性核实 | ✅ | `kaggle` CLI 2.2.4（`~/miniconda3/bin/kaggle`）、账号 `fmyh1828`、GPU 额度 **30h 剩余**（10-17 刷新）；**T4×2 可由 `--accelerator NvidiaTeslaT4` 指定**（不必手点网页）；**P100 停用且 sm_60 与现行 torch 不兼容** |
| T0.0b | 网络绕行 | ✅ | 上传必须经本地代理 `http://127.0.0.1:7890`（googleapis 直连超时）；已固化进 `scripts/kaggle_{push,pull}.sh`（`KAGGLE_PROXY` 可覆盖） |
| T0.1 | 数据集 | 🔄 | `fmyh1828/genshin-gitcg-code`（repo tar + gitcg wheel + assets，11.8 MB，**ready**）；`fmyh1828/genshin-gitcg-corpus`（`sa_all.pt` + 4 个 deck store，**11 GB**，上传中） |
| T0.2 | kernel 胶水 | ✅ | `kaggle/train_arm.py`（幂等、双卡并行、`.part.pt` 原子落盘、`mode.txt` 固定模式、`--max-seconds` 到点收工）、`kaggle/kernel-metadata.json`（script / 私有 / `NvidiaTeslaT4` / 挂两个数据集）、`scripts/kaggle_push.sh`、`scripts/kaggle_pull.sh`、`kaggle/README.md` |
| T0.3 | 冒烟（本地等价） | ✅ | 用 `GITCG_KAGGLE_{WORK,INPUT,DEVICE}` 在本地复现整套 kernel 流程：解 tar → 解 gitcg wheel → 建合成 store → 4 个 run（ce/awr/q + 真 store 探针）全部 `ok`，checkpoint/report/summary 落盘（`/tmp/opencode/kwork`）。**待推 Kaggle GPU 版复核 torch/numpy 版本与双卡** |
| T0.4 | 离线 RL 目标 | ✅ | `scripts/train_bc_scale.py` 加 `loss ∈ {ce,awr,q}`（+`awr_beta`/`awr_max_weight`/`q_target`/`q_gamma`）；`scripts/datafirst_bc_train.py` 暴露同名 CLI；`tests/test_offline_rl.py`（β=0 ≡ CE 逐梯度等价、AWR 提高败局里正回报动作的质量、Q 头在「专家走坏手」的状态上反转偏好、折扣回报 `γ^剩余步`）；全量测试 **384 passed** |
| T0.5 | 定向补采 | 🔄 | `kaggle/grids/run_targeted_collect.sh`（4 deck × 4 长尾对手 × 300 种子 = 4,800 局，`seed-start 1000`）已在 tmux/setsid 后台跑 |

### Day 1（2026-10-16/17）——结果（认证口径：`eval_protocol`，10 seeds，配对，`--strict-ci`）

| run | 训练数据 | epochs | val_acc | pooled | 配对 Δ vs 同 deck 专家 | floor 违规 |
|---|---|---|---|---|---|---|
| `orl_bc_e15`（本地 CPU） | 30k | 15 | 0.764 | 0.4763 | −0.033 [−0.090,+0.025] ns | 11/20 |
| `orl_awr_b1_e15` | 30k | 15 | 0.762 | 0.4750 | −0.034 [−0.092,+0.025] ns | 11/20 |
| `orl_awr_b2_e15` | 30k | 15 | 0.751 | 0.4713 | −0.038 [−0.096,+0.021] ns | 11/20 |
| `orl_q_disc_e15` | 30k | 15 | （acc 0.18 无意义） | **0.0675** | **−0.441 [−0.490,−0.392]** | 20/20 |
| `kg_bc_e15`（Kaggle T4） | **334k（全量）** | 15 | **0.851** | **0.4288** | **−0.080 [−0.131,−0.029] 显著负** | 12/20 |
| `kg_bc_e50`（Kaggle T4） | 334k | 50 | 0.872 | 0.4688 | −0.040 [−0.091,+0.011] ns | 12/20 |
| 基线 `expert:superconduct_aggro` | — | — | — | 0.5088 | — | **8/20** |

**判定（H1/H2 全线未过门禁）**：

- **AWR（优势加权 BC）无增益**：三条臂与 BC 在噪声内相同（±0.005），机制指纹也极小（赢局−输局一致率差仅 +0.005，BC 为 −0.004）。
- **Q/DMC（`--loss q`）灾难性失败**：pooled 0.068。诊断：argmax 分布并不退化（mean_rel_pos 0.43、前五 index 分散），但只监督「被走动作」的 Q 在未走动作上无约束 → 贪心选出让分动作。**在该数据规模/动作空间下（DouZero 式）动作价值不可用**。
- **模仿保真度 ≠ 强度**：全量 334k 的 15 ep 把一致率从 0.764 抬到 **0.851**，但 pooled 反而从 0.476 **降到 0.429**（显著低于专家）。50 ep 回到 0.469（ns）。→ 「更多数据/更长训练」不单调转化为对池胜率。
- **floor 是继承性的**：连**专家自己**都有 8/20 对手低于 0.45；我们的网多 3–4 个违规。但**同一张网在个别 matchup 上显著强于老师**（double_geo_navia +0.25、electrocharged_moon +0.20、trial_skirk +0.12）→ **按 matchup/对手条件化**是与门禁对齐的下一杠杆（H4）。

**Kaggle 侧（口径修正）**：上一版 kernel **不是被队列卡死**，而是真跑到 2× Tesla T4 并产出了 checkpoint；`timeout` 来自①我设的**单 run 45 min 上限**（150 ep 需要 ~90 min）②**会话 `-t 7200`（2 h）**。已修正为单 run 4 h / 会话 11 h，重推的 5×150 ep 网格正在跑（GPU 已计 0.41 h）。实测吞吐：全量 334k 样本 **≈36 s/epoch（~9.2k 样本/s）**，仍是 CPU collate 瓶颈（A800 上 precollate 相对它快 39×）→ **`--precollate` 是 Kaggle 上量前必做**。

### Day 1b（2026-10-17）——H3 价值头标定与搜索门禁、H4 混合语料

**H3a 价值头标定**（`scripts/eval_value_calibration.py`，`sa_all.pt` val 8k，`reports/train/value_calibration_*.json`）：

| 价值头 | pred | 实际 score | bias | Brier（skill） | ECE | AUC |
|---|---|---|---|---|---|---|
| 30k/15ep · identity+MSE（基线） | 0.577 | 0.480 | +0.097 | 0.1908（0.236） | 0.0972 | 0.807 |
| 30k/15ep · sigmoid+Brier | 0.598 | 0.480 | +0.118 | 0.1924（0.229） | 0.1180 | 0.811 |
| 30k/15ep · sigmoid+BCE | 0.622 | 0.480 | +0.142 | 0.1976（0.208） | 0.1419 | 0.815 |
| 全量 334k/15ep · identity+MSE | 0.494 | 0.480 | +0.014 | 0.1731（0.307） | 0.0569 | 0.825 |
| **全量 334k/50ep · identity+MSE** | 0.529 | 0.480 | +0.049 | **0.1671（0.330）** | **0.0606** | **0.841** |

结论：**价值头「会排序」**（AUC 0.81–0.84，D23 说的「区分度≈0」指的是 MCCR 根价值，不是这个学习头）；
**calibration 的杠杆是训练预算，不是损失函数**——proper scoring rule（Brier/BCE）在小预算下反而把乐观偏差放大
（+0.097→+0.118→+0.142），全量长训才把它压到 ECE 0.061 / bias +0.049。

**H3b 搜索门禁**（A4 同协议：6 对手 × 6 种子 = 72 配对局；`reports/search/h3_*.json`）：

| 档 | 基座 | 前沿价值 | expert（无搜索） | cfr（搜索） | 配对 Δ | 95% CI | 对照 A4 |
|---|---|---|---|---|---|---|---|
| i8 d1 c4 ms8 | expert | 全量 50ep（AUC 0.841） | 0.4306 | 0.3889 | −0.0417 | [−0.200,+0.116] | A4 同档 −0.042 |
| i8 d1 c4 ms8 | 网（base+prior） | 同上 | 0.4167 | 0.3472 | −0.0694 | [−0.226,+0.087] | A4 net 档 ≈ 同号 |
| i8 d2 c4 ms4 | expert | 同上 | 0.4306 | 0.5278 | +0.0972 | [−0.048,+0.242] | A4 D 档 +0.097（全池复测 −0.114） |

结论：**门禁未过**。d1 档结果与 A4 **逐值相同** → 换价值前沿在 depth-1 不改变选择；d2 档那个 +0.097 是
A4 已有的噪声级点估计（全池 −0.114），**不是标定带来的增益**。→ 「价值头标定」单独不足以重开搜索；
D23 另外两条前置（非任意候选裁剪、门控覆盖）仍是缺口。

**H4 混合语料（floor 方向）**：`data/store/mixed_four_decks.pt`（11,200 局 = 4 deck × 20 池 × 80 种子 + 4 个 deck × 4 长尾对手 × 300 种子的定向补采；542k/28.8k 样本）
→ 单模型 `data/checkpoints/h4_mixed_d256_e15.pt`（200k 样本、15 ep、val_acc 0.682）；逐 deck 认证（4 deck × 20 池 × 10 seeds，配对 vs 各自 expert）在跑。

### Day 2（2026-10-17）——Kaggle 150ep 网格 + H4 结果

**Kaggle 会话（2×T4，4.64 h/30 h 周额度，5/5 ok）**：`sa_bc_d256_e150` 96 min、`sa_awr_b05_e150` 98 min、
`sa_awr_b1_e150` 96 min、`sa_awr_b2_e150` 98 min、`sa_q_disc_e150` 83 min；全量 334k 样本，checkpoint 已回流
`data/checkpoints/kaggle/`。终端训练指标：

| run | val_acc | val_q_mse/val_ce | 备注 |
|---|---|---|---|
| bc_e150 | 0.8945 | 0.163（value） | — |
| awr_b05_e150 | 0.8827 | 权重上限 1.72 | — |
| awr_b1_e150 | 0.8980 | 权重上限 2.87 | — |
| awr_b2_e150 | 0.8730 | 权重上限 6.04 | — |
| q_disc_e150 | 0.1871（与老师一致率，无意义） | **0.0609（最好）** | **Q 回归拟合最好、策略最差** → 失败在策略抽取（对未走动作无约束），不是价值拟合 |

**价值头标定（150 ep，`reports/train/value_calibration_kg150.json`）**：`awr_b1_e150` **ECE 0.0274 / bias +0.012 / AUC 0.848**，
`bc_e150` 0.0449 / +0.031 / 0.845，`q_disc_e150` 0.0353 / +0.017 / 0.843
→ **训练预算把标定推到 ECE ≈ 0.03（50 ep 时 0.061），D23 的「标定过的价值头」这一条基本达标**；
但 H3b 已证「标定本身不能让搜索超过不搜索」。

**H4 混合语料单模型逐 deck 认证**（`h4_mixed_d256_e15.pt`，200k 样本/15 ep；10 seeds 配对）：

| deck | pooled | Wilson95 | 同 deck 专家 | 配对 Δ | 95% CI | floor 违规 | merge 门禁 |
|---|---|---|---|---|---|---|---|
| superconduct_aggro | 0.5075 | [0.459,0.556] | 0.5088 | −0.0013 | [−0.060,+0.057] | 15 | ✗ |
| dvalin_bonk | 0.6438 | [0.596,0.689] | 0.6863 | −0.0425 | [−0.092,+0.007] | 10 | ✗ |
| unyielding_geo | 0.7837 | [0.741,0.821] | 0.7987 | −0.0150 | [−0.060,+0.030] | 6 | ✗ |
| double_geo_navia | 0.7250 | [0.679,0.766] | 0.6813 | **+0.0437** | [−0.008,+0.096] | 8 | **✓ PASS** |

**首个 merge 门禁 PASS（double_geo_navia，非劣 +0.044）**；四个 deck **全部统计打平或更好**（无显著负），
且 SA 从单 deck BC 的 0.476/0.429 **升到 0.5075 ≈ 专家 0.5088**（Δ=−0.0013，逐 deck 首次追平专家）。
**floor 仍全 FAIL**（SA 15 违规 > 专家自己的 8）——长尾 matchup 依旧是唯一未过的门禁。

### Day 2b（2026-10-17）——150ep 全量认证：**四条线全部到顶**

`sa_all.pt`（334k 样本）150 epoch、2×T4，10 seeds 配对认证（`reports/explore/kg150_*.json`）：

| run | val_acc | pooled | 配对 Δ vs 专家(0.5088) | 95% CI | merge |
|---|---|---|---|---|---|
| `sa_bc_d256_e150` | 0.8945 | 0.453 | −0.056 | [−0.108,−0.004] | ✗ |
| `sa_awr_b05_e150` | 0.8827 | 0.431 | −0.077 | [−0.131,−0.024] | ✗ |
| `sa_awr_b1_e150` | 0.8980 | **0.352** | **−0.156** | [−0.210,−0.103] | ✗ |
| `sa_awr_b2_e150` | 0.8730 | 0.471 | −0.037 | [−0.091,+0.016] | ✗ |
| `sa_q_disc_e150` | 0.1871 | 0.275 | −0.234 | [−0.289,−0.179] | ✗ |

（对照：30k/15ep BC 0.476、30k/15ep AWR 0.475、30k/15ep Q 0.068。）

**三条判定**：

1. **模仿保真度与强度脱钩**：val_acc 随预算单调上升 0.764 → 0.851 → **0.895**，pooled 始终停在 0.43–0.48
   且**从未达到老师 0.509**；最大预算下 BC **显著劣于自己的老师**。
2. **离线 RL 在长预算下有害**：AWR 15 ep 中性、150 ep **显著负且随 β 单调恶化**（β=1 达 −0.156）；
   Q 从 0.068（15 ep）升到 0.275（150 ep）仍远低于专家。→ AWR/Q 归档，不作为主线。
3. **价值标定由预算解决，不由损失函数解决**：ECE 0.097（30k/15ep）→ 0.061（334k/50ep）→ **0.027–0.045**（150ep），
   AUC 0.807 → 0.848；Brier/BCE 在小预算下反而更差。

**唯一正增益来自数据构成**：H4 混合语料单模型在 4/4 deck 打平或更好（double_geo 首个 merge PASS），
SA 从单 deck BC 的 0.476/0.429 抬到 **0.5075 = 专家水平**。→ **主线收敛为「扩数据构成 + 按 matchup 攻坚 floor」**。

### 2026-10-18 —— floor 门禁的统计功率与 H5

**floor 的统计功率（测量口径）**：`--strict-ci` 的 floor 用「Wilson 下界 ≥ 0.45」，而 10 seeds = 每 matchup 20 局。
数值扫描（`Wilson 下界 = 0.45` 反解）：

| 每 matchup 局数 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|
| 过关所需真实胜率 | **0.668** | 0.604 | 0.576 | 0.548 |

即 10 seeds 下 floor 实际是「**每 matchup ≥ 67%**」的挑战级指标（D24「floor 全 FAIL」由此得到解释）。
**攻坚口径改用「rate < 0.45 的 matchup 数」**（H4 模型：SA 9、dvalin 3、unyielding 1、double_geo 1），
严格 Wilson 口径留作拉伸目标；两者分列报告，避免把统计功率不足误读成策略崩。

**H5 matchup 重加权**（`--matchup-weight {balance,deficit}`，`tests/test_matchup_weights.py`）：
按 `(deck, opponent)` 分组，`balance` 反频率、`deficit` 只抬「数据本身胜率 < floor」的 matchup
（权重 `1+strength·(floor−rate)`，归一化均值 1；`none` 与历史 BC 损失逐值相同）。
实测权重范围 0.84–1.22（多数 matchup 已在 0.45 以上，效应温和）。正在跑 `h5_mixed_deficit_e15`（混合语料 200k/15ep）
+ SA/dvalin 认证，并跑 **跨 deck 迁移对照**（SA-only 同预算 200k/15ep）判定 H4 的 +0.03 是「跨 deck 数据」还是「更多数据」。

**语料广度（新主线）**：`kaggle/grids/run_corpus_all_decks.sh` 补齐其余 16 deck × 20 池 × 80 种子（≈25,600 局 compact），
目标「单模型打全部 20 deck」（Web/decklab 也需要）。依据：目前唯一实测正增益来自数据构成/跨 deck 迁移。
靶向补采第 2 轮同时进行（`run_targeted_collect_round2.sh`：shenhe_double_swirl / double_geo_navia /
superconduct_mika_attack / dvalin_bonk / yelan_clorinde_ororon，seed 2000+）。

### 2026-10-19 —— H4 v2 / H5 / 迁移 A/B / 自博弈筛选：**首个显著超过脚本专家的策略**

| 实验 | deck | pooled | expert | 配对 Δ | 95% CI | rate<.45 | merge |
|---|---|---|---|---|---|---|---|
| **H4 v2（混合 542k，150ep）** | **dvalin_bonk** | **0.7575** | 0.6863 | **+0.0712** | **[+0.028,+0.115]** | 1 | **✓ PASS** |
| H4 v2（150ep） | superconduct_aggro | 0.5050 | 0.5088 | −0.0037 | [−0.057,+0.050] | 11 | ✗ |
| H4 v2（50ep） | dvalin_bonk | 0.7050 | 0.6863 | +0.0187 | [−0.027,+0.065] | 2 | ✗ |
| H4 v2（50ep） | superconduct_aggro | 0.3525 | 0.5088 | −0.1562 | [−0.210,−0.103] | 15 | ✗ |
| H5 deficit 重加权 | superconduct_aggro | 0.4175 | 0.5088 | −0.0912 | [−0.148,−0.034] | 11 | ✗ |
| H5 deficit 重加权 | dvalin_bonk | 0.6525 | 0.6863 | −0.0338 | [−0.082,+0.015] | 3 | ✗ |
| 迁移对照：SA-only 200k/15ep | superconduct_aggro | 0.4788 | 0.5088 | −0.0300 | [−0.082,+0.022] | 11 | ✗ |
| 自博弈筛选 BC（只学胜局） | superconduct_aggro | 0.3688 | 0.5088 | −0.1400 | [−0.200,−0.080] | 13 | ✗ |
| 自博弈筛选 BC | dvalin_bonk | 0.6663 | 0.6863 | −0.0200 | [−0.069,+0.029] | 4 | ✗ |

**判定**：

1. **H4 v2 在 dvalin_bonk 上显著超过脚本专家（+0.0712，CI 下界 +0.028）**——本战役**第一个**正增益；
   且 dvalin 的预算轴单调（v1 200k/15ep −0.043 → e50 +0.019 → e150 **+0.071**），floor 违规数 1（v1 为 3）。
   → 「**混合语料 + 长预算**」是已验证有效的配方（但只在 dvalin 上显著；SA 上仍是打平，e50 甚至会崩）。
2. **跨 deck 迁移方向性成立**：同预算 200k/15ep，混合 0.5075 vs SA-only 0.4788（+0.029，CI 含 0）——不是「更多数据」，而是**别的 deck 的对局**在帮忙。
3. **matchup 重加权（H5）负增益**（SA −0.091 显著劣）：与 AWR 同型失败（偏离模仿目标）。
4. **自博弈「只学胜局」负增益**（SA −0.14）：筛选按结果而非按技能，胜局多为对手发挥差的运气局（生存者偏差），无重要性修正时不成立。
5. **语料广度已就位**：`all_*` 16 deck × 20 池 × 80 种子 = **25,600 局**；全 deck store（20 deck × 20 池 × 40 种子）
   `data/store/all_decks_40s.pt` = **707,568/46,199 样本（15,200 局，12.2 GB）** → 单模型打全部 20 deck 的训练输入。

**当前唯一被验证的强度路线**：混合/跨 deck 语料 + 大预算（dvalin +7.1pp；SA 打平）。
所有「非模仿改进算子」（AWR / Q-DMC / 搜索 / 胜局筛选 / matchup 重加权）全部中性或有害。

### 2026-10-19b —— 换种子复验：**dvalin 的「首个显著正增益」不复现**

评估口径的教训：直接挑报告里最好的数字会被种子集骗。对「按 deck 选最优 checkpoint」做**新种子**复验
（seeds 50–59，与选型用的 0–9 完全不重叠，同样 10 seeds 配对）：

| deck | 选型臂 | 选型集 Δ (seeds 0–9) | **复验 Δ (seeds 50–59)** | 复验 CI | 结论 |
|---|---|---|---|---|---|
| superconduct_aggro | v1 200k/15ep | −0.0013 | **−0.005** | [−0.062,+0.052] | 打平（稳定） |
| double_geo_navia | v1 200k/15ep | +0.0437 | **+0.013** | [−0.038,+0.063] | 打平（原值缩水） |
| dvalin_bonk | v2 542k/150ep | **+0.0712** | **+0.003** | [−0.041,+0.046] | **不复现**（种子集运气） |
| unyielding_geo | v2 542k/150ep | +0.0238 | **−0.013** | [−0.053,+0.028] | 打平（原值缩水） |

**修正后的诚实结论**：四个 deck 在**两套互不重叠的种子**上都是 Δ ≈ 0 ± 0.04 —— 所有方法（BC / AWR / Q-DMC /
搜索 teacher / 胜局筛选 / matchup 重加权 / 混合语料 / 预算放大）最终都落在**与脚本专家打平**，
**没有任何方法能稳定超过 teacher**（此前 +0.0437 / +0.0712 / +0.0238 全部是选型种子上的运气）。
混合语料的价值是「把 BC 从 −0.03~−0.15 拉到 0」，这本身是实质进步，但不是超过。

**对下一步的含义**：
1. 交付层面：把「单模型打全部 20 deck + 与专家打平」作为模型主线的**阶段性交付**（all-deck 模型正在 Kaggle 上训练）。
2. 研究层面：要真正超过 teacher，只剩**在线 RL / 自博弈**这一族（我们试过的都是离线/BC 派生）。离线尝试的失败模式已清楚：
   优势估计偏差（AWR）、对未走动作无约束的 argmax（Q）、筛选的生存者偏差（胜局筛选）。
   在线版本（引擎在环、用真实对局回报更新、带对手池）尚未尝试 —— 这是唯一还没被否定的方向。
3. 口径教训（写进流程）：**任何 +Δ 必须在选型种子之外复验**；报告里同时给两个种子集的结果。

### 2026-10-20 —— all-deck 稀释证据 + 在线 RL 开工（导师指导落地）

**A. all-deck 单模型（20 deck，707k 样本，d256，150ep）8 deck 认证**：

| deck | pooled | 专家 | Δ | 95% CI |
|---|---|---|---|---|
| unyielding_geo | 0.7975 | 0.7987 | −0.0013 | [−0.043,+0.040] |
| superconduct_mika_attack | 0.5437 | 0.5675 | −0.0238 | [−0.078,+0.031] |
| aywolfskirk | 0.4550 | 0.5225 | −0.0675 | [−0.120,−0.015] |
| double_geo_navia | 0.5850 | 0.6813 | −0.0963 | [−0.149,−0.043] |
| klee_mavuika_mualani | 0.4050 | 0.5312 | −0.1263 | [−0.177,−0.075] |
| dvalin_bonk | 0.4863 | 0.6863 | −0.2000 | [−0.255,−0.145] |
| nahida_kokomi_raiden | 0.4313 | 0.6375 | −0.2062 | [−0.258,−0.154] |
| superconduct_aggro | 0.2838 | 0.5088 | −0.2250 | [−0.280,−0.170] |

→ **朴素「拼 20 deck 数据训一个 d256 池化模型」显著稀释**（8 个里 7 个变差，只有 unyielding 打平）。
与 4-deck 混合（打平）对比说明：**deck 数量 ↑ 而容量/每 matchup 数据不变 → 负迁移**。

**B. 导师指导 → 落点（按我们已有的证据排序）**：

| 导师方向 | 我们的落点 | 状态 |
|---|---|---|
| **(a) 在线 RL / 自博弈（引擎在环）** | 新增 `--loss pg`：REINFORCE + value baseline + entropy（on-policy；数据来自 `neural:<ckpt>#t=1.0` 的自博弈，`teacher_index` 即采样动作）。循环 `kaggle/grids/run_online_rl_round.sh`：采集 4 deck × 20 池 × 25 种子 → PG 2ep → 新种子认证 | **round 1 运行中** |
| **(d) 剥削 / 最佳响应（脚本是确定性环境）** | 把 PG 的目标从「对池平均」改成「对指定脚本最大化胜率」：先按 matchup 定向采集（已有 `tgt_*` 5k + `tgt2_*` 5k 局），再对最差 matchup 做 PG；若 (a) 有信号再上 best-response shaping | 待 (a) 结果 |
| **(b) 容量 / 多任务结构** | 证据支持：transformer 43ep 的 val_acc（0.737）已超池化 CNN 150ep（0.693）。落点＝deck-id embedding + 共享 trunk vs 纯混训（等参数量对照）、每 deck 独立策略头/MoE；**需 GPU，等 10-24 额度重置或 A800 窗口** | 待算力 |
| **(c) 搜索感知训练** | 已归档（标定价值头仍未带来增益）；若重开，按导师意见走 **Gumbel 式探索 + 显式搜索目标进损失**（而非仅把搜索结果当数据源） | 归档 |

**C. 评测纪律（导师意见，已写入所有新实验）**：
- **预注册**：每轮实验先写死「接受规则」（本次：新种子上配对 Δ 的 CI 下界 ≥ −0.02 且点估计 > 0）；
- **种子隔离**：第 r 轮采集用 4000+1000r、认证用 60+10r，绝不与历史种子重叠；
- **功效**：n=20/matchup 需真实胜率 ≥0.668 才过严格 floor → 结论一律看配对 Δ，floor 用「rate<0.45 计数」；
- **报告回归**：任何 deck 退化 >2pp 一并列出，不做挑选。

**D. Kaggle 额度与流程教训**：本周 **29.24/30 h 已用尽**（会话 7/8 无产出：transformer 再次撞 4h 单 run 上限；
`mixed_4deck_40s.pt` 因数据集版本未随 kernel 挂载而 skipping）。→ 下个窗口：先 `datasets status` 确认版本 ready
**再**推 kernel，并在日志里核对 `corpus stores` 列表；transformer 类慢模型按「每 epoch 实测分钟 × 目标 epoch」排，不要按 CNN 估。

### 2026-10-21 —— 在线 RL 三轮：**double_geo 的剥削增益三次复现**，其余 deck 退化

按导师 (a)/(d) 落地的 on-policy 策略梯度，三轮全部在**互不重叠的新种子**上认证：

| deck | r1e 共享策略（110–119） | rl_pd1 按 deck（130–139） | r2 按 deck + 4× 数据（140–149） | 结论 |
|---|---|---|---|---|
| superconduct_aggro | −0.146 | −0.058 | −0.078 | 退化（与老师有差距，PG 只放大噪声） |
| dvalin_bonk | −0.063 | −0.063 | **−0.128** | 迭代反而更差 → 停止对落后 deck 做 PG |
| unyielding_geo | +0.003 | −0.009 | −0.010 | 打平（floor 违规 0–2，最干净） |
| **double_geo_navia** | **+0.100** [+0.047,+0.153] | **+0.058** [+0.008,+0.107] | **+0.070** [+0.019,+0.121] | **三次复现的显著正增益** |

**判定与规则（写死）**：
1. **double_geo 的 +5.8~10pp 是本项目第一个可复现的「超过脚本专家」**（三个独立新种子集、CIs 全部排除 0）。
2. **PG 只在「BC 已经 ≥ 老师」的 deck 上有效**；对落后 deck（SA、dvalin）会把策略推离模仿目标、显著退化
   （dvalin 连续两轮：−0.063 → −0.128）。→ **默认流程：per-deck 评估基线强度，只在 parity+ 的 deck 上开 PG**。
3. 因此模型主线的交付形态改为「**按 deck 的 specialist 集合**」：每个 deck 用其认证最优臂（BC v1/v2 / PG 微调），
   而不是追求一个全 deck 单模型（后者已证 7/8 退化）。

**进行中**：double_geo 第三轮（种子 150–159，预注册同规则）→ 判断增益是否继续增长或饱和。
**待 GPU（10-24）**：(b) 容量/多任务——deck-id embedding / 独立策略头 / MoE 的等参数对照；
transformer 长训练（43ep val_acc 0.737 已超池化 CNN 150ep 0.693，容量是明确瓶颈）。

### 2026-10-22 —— double_geo 剥削阶梯：**+18.8pp（pooled 0.890）**，且同种子对照证明是「轮次带来的」

| 检查 | 种子 | pooled | 专家 | Δ | 95% CI |
|---|---|---|---|---|---|
| dgeo r3 | 150–159 | **0.890** | 0.7025 | **+0.1875** | [+0.139,+0.236] |
| **对照：r2 在同一套种子** | **150–159** | **0.800** | 0.7025 | **+0.098** | [+0.045,+0.150] |
| dgeo r3 复验 | 160–169 | **0.882** | 0.7240 | **+0.158** | [+0.116,+0.199] |

**三条硬结论**：
1. **同种子对照**（r2 vs r3 on 150–159：0.800 → 0.890）证明增益**来自第 3 轮更新本身**，不是种子集便宜。
2. **两套全新种子复现**（0.890 / 0.882），对脚本池胜率 **~88–89%**，比专家高 **+16~19pp**。
3. **floor 显著改善**：r3 只有 **1** 个 Wilson floor 违规（最差 matchup rate 0.65），而专家自己约 8 个。

→ **迭代式 on-policy 剥削（collect → PG → 认证，每轮换新种子）在 parity+ 的 deck 上可以把「打平」推到「压制」**。
这是本项目第一个**大幅、可复现**地超过固定脚本的结果，正对应导师指出的「脚本是确定性环境，应按最佳响应/剥削来做」。

**进行中（链式）**：dgeo 第 4 轮（170–179，看是否饱和或继续爬升）+ **unyielding 第 1/2 轮（180–189 / 190–199）**
——检验该配方能否迁移到另一个 parity 的 deck。

### 2026-10-24 —— dgeo 饱和（0.870 / 0 个 floor 违规）、unyielding 转正、dvalin 归因

| 检查 | 种子 | pooled | 专家 | Δ | 95% CI | rate<.45 | merge |
|---|---|---|---|---|---|---|---|
| dgeo r4 | 170–179 | **0.870** | 0.685 | **+0.185** | [+0.135,+0.235] | **0** | ✓ |
| unyielding r1 | 180–189 | 0.8275 | 0.8675 | −0.040 | [−0.081,+0.001] | 1 | ✗ |
| unyielding r2 | 190–199 | **0.8625** | 0.820 | **+0.0425** | [−0.001,+0.086] | **0** | ✓ |

**结论**：
1. **dgeo 已饱和**：r3 0.890/0.882 → r4 0.870（同一水平），且 r4 的 **floor 违规为 0**（20 个 matchup 全部 ≥45%）。
   dgeo specialist 冻结为交付件：**pooled ~0.87–0.89，比脚本专家高 16–19pp，无 floor 违规**。
2. **unyielding 需要两轮才转正**（r1 −0.040 → r2 +0.0425），说明 parity 附近的 deck 第一轮基本是噪声，
   要看第二轮趋势；r3 在跑。
3. **dvalin 的退化是「选错底座」而非「PG 有害」**：`h4m_bc_d256_e150`（h4m 4-deck 语料）在 dvalin 上是
   **+0.0712 [+0.028,+0.115]**，而此前 PG 用的是共享 `h4_mixed` 锚（在 dvalin 上本来就 −0.043）。
   → 流程修正：**每个 deck 的阶梯从该 deck 认证最优的臂起步**，不是从一个共享锚起步。

**Kaggle 额度已重置（30 h）**。已推 `cap` 网格（`kaggle/train_arm.py` 的 `CAPACITY_RUNS`），
用 4-deck/40s 密度对照 + 更大主干 + transformer 分离「deck 数量」与「每对手对局密度」两个混淆因素：
| run | 语料 | 主干 | epochs |
|---|---|---|---|
| cap_4deck40s_cvpn_d256_e150 | mixed_4deck_40s（4 deck，40 seeds/matchup） | cvpn d256 | 150 |
| cap_h4m_cvpn_d384_e120 | mixed_four_decks（4 deck，~250 seeds/matchup） | cvpn d384 | 120 |
| cap_alld_cvpn_d512_e100 | all_decks_40s（20 deck，40 seeds/matchup） | cvpn d512 | 100 |
| cap_alld_transformer_d256_l4_e60 | all_decks_40s | transformer 4L | 60 |

**排队中（本机）**：链条结束后自动跑 Group B——给最接近 parity 的四个 deck（SA −0.001、mika −0.024、
aywolfskirk −0.068、natlan −0.085）各采 2,000 局专家对局（密度对齐 Group A），再用 `alld_bc` 补认证
其余 12 个从未认证的 deck，把 specialist 集合补齐。

### 2026-10-24（续）—— unyielding 也爬到 0.90；dvalin 平台期；12 个未认证 deck 的基线摸底

| 检查 | 种子 | pooled | 专家 | Δ | 95% CI | rate<45% | merge |
|---|---|---|---|---|---|---|---|
| **unyielding r3** | 200–209 | **0.9025** | 0.7975 | **+0.105** | [+0.061,+0.149] | **0** | ✓ |
| dvalin r1（正确底座） | 210–219 | 0.7800 | 0.7425 | +0.0375 | [−0.009,+0.084] | 1 | ✓ |
| dvalin r2 | 220–229 | 0.6850 | 0.6550 | +0.0300 | [−0.021,+0.081] | 2 | ✗（差 0.001） |

**unyielding 阶梯**：r1 −0.040 → r2 +0.0425 → **r3 +0.105（0.9025）**。与 dgeo 同形：parity 附近前两轮是噪声，
第三轮才跳变。**两个 deck 都停在 ~0.87–0.90**，说明这不是单 deck 的偶然。
**dvalin 平台期**：从正确底座起步后 +0.0375 / +0.030，第二轮不涨 → 需要换机制（更深搜索/更大底座），
不是继续加轮次。

**未认证 deck 的摸底**（`alld_bc_d256_e150` 作底座，新种子 400+，10 seeds 配对）：

| deck | pooled | 专家 | Δ |
|---|---|---|---|
| dual_mualani_stacks | 0.4275 | 0.5575 | −0.130 |
| klee_mavuika_mualani | 0.3950 | 0.5525 | −0.158 |
| yelan_clorinde_ororon | 0.2700 | 0.4475 | −0.178 |
| nahida_kokomi_raiden | 0.4475 | 0.6550 | −0.208 |
| mualani_double_pyro | 0.3362 | 0.5637 | −0.228 |

→ **20-deck 单模型不能作为这些 deck 的底座**（全部 −13~−23pp）。它们必须先有「按 deck 的稠密语料 + 专用底座」，
这正是 Group B 采集（4 deck × 2,000 局）在做的事；其余 deck 需要同样的补采。

**Kaggle 事故与修复**：`cap` 网格第一次跑成了 `runs`（SA 网格）——`resolve_pinned_mode()` 的模式白名单
写死为 `("probe","smoke","runs","h4","alld")`，新加的 `cap` 被静默丢弃、回落到 auto→runs，**烧掉 4.5h/30h**。
已改为按 `MODE_RUNS` 表校验。副作用不全是坏：`sa_bc_d256_e150`（`sa_all.pt` 稠密语料，val_acc 0.886）
正好是 SA 需要的专用底座，已回传本机作为 SA 阶梯的起点。

**已重推 `wave1` 网格**（4 个 Group B 专用底座 + 原 4 个密度/容量对照），并新建语料数据集
`genshin-gitcg-corpus-gb`（4×2,000 局 + 4-deck/40-seed 密度对照）。

**本机排队**：`run_ladder_wave2.sh` —— SA/mika/aywolfskirk/natlan 各从专用底座起一轮 + dvalin r3，
全部用新种子（采集 9500–9800/8700，认证 520–569）。

### 2026-10-25 —— SA 阶梯即使换稠密底座也失败；dvalin 确认平台期；12 deck 摸底完成

| 检查 | 种子 | pooled | 专家 | Δ | 95% CI | merge |
|---|---|---|---|---|---|---|
| SA r1（底座 `sa_bc_d256_e150`，val_acc 0.886） | 520–529 | 0.4200 | 0.5050 | **−0.085** | [−0.140,−0.030] | ✗ |
| dvalin r3 | 560–569 | 0.7650 | 0.7325 | +0.033 | [−0.015,+0.080] | ✓ |

**SA 是唯一「换稠密底座也不行」的 deck**：底座模仿精度 0.886，但 SA 的脚本专家本身只有 ~0.50，
且 SA 在多组对局里是天然劣势方；PG 在胜率目标上只会把这些劣势放大。→ **SA 不做 PG 阶梯**，
留给「搜索/更优模仿」或直接放弃（记录为负结果）。
**dvalin 确认平台期**：+0.0375 / +0.030 / +0.033（三轮几乎不动）→ 阶梯机制在 dvalin 上饱和于 +3pp。

**12 个未认证 deck 摸底（`alld_bc_d256_e150` 底座，新种子 400–519）**：平均 Δ **−0.145**，无一可用。
其中 Skirk 系（`skirk_chasca_freeze` 0.085、`beidou_skirk_ayaka` 0.124、`ice_water_battleship` 0.148、
`skirk_ayaka_navia` 0.228）**连脚本专家自己都只有 0.19–0.38**——这些 deck 对全池是结构性劣势，
拿它们算「强度」没有意义，先排除在 specialist 目标之外。

**本轮决策**：
1. specialist 目标限定在**脚本专家 ≥0.55** 的 deck（有真实可赢的对局）：已覆盖 dgeo/unyielding/dvalin、
   Group B（SA 除外）/Group C 六个（klee 0.55、nahida 0.65、shenhe 0.72、electro_manifestation 0.66、
   mualani_double_pyro 0.56、dual_mualani 0.56）。
2. **SA 从阶梯名单剔除**（负结果，已记录）；dvalin 保持 +3pp 的 specialist，不再加轮次。
3. Group C 采集已启动（6 deck × 2,000 局，种子 11000–11599），随后 GPU 出底座 → 逐 deck 一轮阶梯。

### 2026-10-26 —— wave1 空跑事故（`--run-timeout` 漏进训练器）+ Group C 采集完成

`wave1` 网格 8 个 run **全部在 1 秒内失败**：`spec_to_argv()` 把 `run_timeout`（kernel 侧的
per-run 超时）当成训练器参数转发，`datafirst_bc_train` 报 `unrecognized arguments: --run-timeout`。
GPU 只花了 ~0.1h（不是大损失），但整晚没有产出。修复：`KERNEL_ONLY_KEYS = {"name","store","run_timeout"}`，
并加了本地断言（生成 argv 里不得出现 `run_timeout`）。

**Group C 采集完成**：klee / nahida / shenhe / electro_manifestation / mualani_double_pyro /
dual_mualani_stacks 各 **2,000 局**（种子 11000–11599），store 72k–102k 样本/deck。

**重排 `wave2` 网格**（一个 session，14 run，~7.6h）：Group C 6 个底座 + Group B 4 个底座
（上一轮没跑成）+ 4 个密度/容量对照；便宜的底座排前面，先解锁本机阶梯。

**同时本机补强**：对两个已达标的 specialist 在**全新种子**上做 20-seed 复验（dgeo 600–619、
unyielding 620–639，各 800 配对局），把交付件的置信区间收窄。

### 2026-10-26（续）—— **两个 specialist 在 20 seeds 上通过完整门禁**

| deck | 种子 | pooled | 专家 | Δ | 95% CI | 样本 | gates |
|---|---|---|---|---|---|---|---|
| **double_geo_navia** (rl_dgeo3) | 600–619 | **0.880** | 0.712 | **+0.168** | [+0.134,+0.201] | 800 配对 | pooled✓ floor✓ clean✓ **PASS** |
| **unyielding_geo** (rl_ungeo3) | 620–639 | **0.909** | 0.854 | **+0.055** | [+0.027,+0.083] | 800 配对 | pooled✓ floor✓ clean✓ **PASS** |

这是第一次有策略在**完整验收门禁**（pooled + 严格 Wilson floor + clean）上通过——10 seeds 时
floor 门槛因样本量不足必然标红（n=20 时 rate 0.5 的 Wilson 下界只有 ~0.30），20 seeds 后统计功效够了。
**交付件定稿**：dgeo 0.880（+16.8pp）、unyielding 0.909（+5.5pp，且专家在本种子集上高达 0.854）。

### 2026-10-26（夜）—— **稠密语料把 Group C 六个 deck 从 −0.13~−0.31 拉到 −0.08~+0.00**

每个 deck 2,000 局专家对局（100 种子 × 20 对手）训练专用 BC 底座，在**新种子**上认证：

| deck | 旧底座（alld）Δ | **稠密底座 Δ** | 提升 | 底座 val_acc |
|---|---|---|---|---|
| electro_manifestation_nuke | −0.314 | **+0.003**（0.660） | +0.32 | 0.820 |
| shenhe_double_swirl | −0.250 | **−0.015**（0.668） | +0.24 | 0.832 |
| mualani_double_pyro | −0.228 | −0.035（0.543） | +0.19 | 0.785 |
| dual_mualani_stacks | −0.130 | −0.070（0.485） | +0.06 | 0.802 |
| klee_mavuika_mualani | −0.126 | −0.068（0.458） | +0.06 | 0.874 |
| nahida_kokomi_raiden | −0.208 | −0.080（0.533） | +0.13 | 0.785 |

**结论**：20-deck 单模型的巨大缺口不是「任务太多无法共享」，而是**每个 matchup 的对局数太少**——
补到 100 种子/matchup 后，同一套 d256 网络把差距抹掉 6–32pp。两个 deck（electro、shenhe）已到 parity，
按预注册门槛（Δ≥−0.02 且 pooled≥0.45）**自动进入阶梯**（`run_ladder_wave3.sh`，新种子 820/810）。

剩余四个（mualani/dual/klee/nahida）仍落后 3.5–8pp：下一步是加轮次补采（同一 deck 再采 2,000 局）
或换更大主干，而不是再动任务结构。

### 下一步（2026-10-19 修订）

1. **all-deck 单模型（Kaggle 会话 5，正在跑）**：`all_decks_40s.pt`（20 deck × 20 池 × 40 种子，707k 样本）上
   BC 150ep / 50ep + 4 层 transformer 80ep（`workers=1` 串行）；产物回流后逐 deck 认证（先 4 个已知 deck，再抽 4 个）。
2. **铺开已验证配方**：H4 v2 配方（混合 542k + 150ep）在 dvalin 上 +0.071；在 unyielding / double_geo 上复现中
   （本地正在认证 unyielding），并做 4-deck × 150ep 与 v1 的对照。
3. **语料构成继续扩（低优先）**：全 deck 语料下一版可混入自博弈对局（8,000 局已有）与 round-2 靶向（5,000 局）；
   注意自博弈**筛选**路线已否，仅作混入数据才可能有用（未验证）。
4. **明确归档**：AWR / Q(DMC) / 搜索-as-teacher / 胜局筛选 / matchup 重加权 —— 全部中性或显著负，不再排产；
   保留 CLI 与报告作为可复现负结果。
5. **工程收口**：代码/报告已提交至 `9c43095`/`f0a9801`；`data/` 与 checkpoint 不入库（既有 gitignore）。
