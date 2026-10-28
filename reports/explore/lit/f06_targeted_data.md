# worst-matchup 靶向采集、对手条件化与数据选择

> 范围：只读本仓库代码/报告与 `refs/`，检索一手论文与官方代码；不跑训练/评测、不改代码、不写 `data/`。
> 结论口径：**纯策略**（LLM 不参与，PLAN §6.3）；强度只看引擎胜率，不看 store `val_acc`。
> 本文所有“本报告计算”的数字，均由 `reports/arena/20261008T163940_full_matrix_20x20.json` 与 `reports/explore/*.json` 只读重算，未新增对局。
> **brief 校正**：brief 给的 Mindermann 2022 `arXiv:2202.01365` 实为远程视障辅助的 3D 地图论文（*Feasibility of Interactive 3D Map for Remote Sighted Assistance*），正确是 `arXiv:2206.07137`；PCGrad 的 `arXiv:2005.10447` 实为 Lorentz 流形上的反问题论文，正确是 `arXiv:2001.06782`。两处均已按一手页面更正。

---

## 0. 一句话结论（可执行）

先把 floor 违规拆成**「老师也输」**与**「老师能赢、学生输」**两类：只有后者值得靶向补采。用**配对 Δ（同种子学生 vs 专家）+ Wilson 下界 + 经验贝叶斯收缩**选目标，确认后再采；采集层按 **25–35% 靶向 + 65–75% 均匀/锚点**混合，**不要**在损失层做 deficit 重加权（H5 已负）；对手条件化用**公开阵容（3 角色）的 FiLM**，它是零新增对局的第二杠杆，不是主杠杆。最小可判定实验：先做 1 deck 试点（1,400 新采集局 + 960 评测局，约 2–3 h），信号为正再扩到 3 deck（8,400 新采集局 + 5,840 评测局，约 8–13 h CPU）。

---

## 1. 与我们问题的对应关系

### 1.1 先分层：floor 违规里有多少是“可修的”

评测池 20 套脚本专家，`reports/SPECIALISTS.md` 的最优臂在弱 deck 上仍挂 6–20 个 floor。把 `reports/arena/20261008T163940_full_matrix_20x20.json`（20×20、50 seeds、双向、每对 100 局）当作**同 deck 专家对同对手的天花板**，再与学生最新认证臂的逐对手胜率对齐，可得到下面这张分层表（本报告重算）：

| deck（我方） | 学生 pooled（SPECIALISTS 最优臂） | 专家 pooled（20×20 矩阵） | 老师也输的对手数（expert Wilson low <45%） | 可修对手数（expert low ≥45% 且学生 point <45%） | 学生 floor 违规数（point <45%, n=20） |
|---|---:|---:|---:|---:|---:|
| natlan_battleship | 0.256 | 0.237 | 18 | 0 | 15 |
| beidou_skirk_ayaka | 0.124 | 0.192 | 20 | 0 | 19 |
| skirk_chasca_freeze | 0.085 | 0.175 | 20 | 0 | 19 |
| skirk_ayaka_navia | 0.228 | 0.181 | 20 | 0 | 16 |
| ice_water_battleship | 0.147 | 0.186 | 18 | 1 | 19 |
| yelan_clorinde_ororon | 0.270 | 0.489 | 14 | 1 | 15 |
| trial_skirk_ayaka_navia | 0.307 | 0.439 | 13 | 2 | 15 |
| klee_mavuika_mualani | 0.458 | 0.541 | 10 | 1 | 10 |
| superconduct_aggro | 0.505 | 0.529 | 12 | 2 | 11 |
| superconduct_mika_attack | 0.544 | 0.566 | 10 | 2 | 8 |
| nahida_kokomi_raiden | 0.532 | 0.641 | 7 | 1 | 8 |
| dual_mualani_stacks | 0.485 | 0.531 | 13 | 0 | 9 |
| mualani_double_pyro | 0.542 | 0.560 | 12 | 0 | 6 |
| aywolfskirk | 0.482 | 0.523 | 13 | 0 | 9 |
| electro_manifestation_nuke | 0.685 | 0.676 | 5 | 0 | 2 |
| shenhe_double_swirl | 0.703 | 0.700 | 5 | 0 | 2 |
| dvalin_bonk | 0.780 | 0.733 | 3 | 0 | 1 |
| unyielding_geo | 0.909 | 0.831 | 2 | 0 | 0 |
| double_geo_navia | 0.880 | 0.723 | 5 | 0 | 0 |

读法：

- **natlan 的 floor 不是数据问题**：专家自己只有 0.237（2,000 局矩阵）/ 0.285（10-seed 评测，`reports/SPECIALISTS.md`），18/20 个对手连专家都低于 45%。用 `expert:natlan_battleship` 做 teacher 补采，最多把学生拉到老师附近，**不可能把 floor 修到 0**。beidou / chasca / skirk_ayaka 同理（20/20 老师也输）。
- 19 个有认证臂的 deck 里，严格口径下只有 **10 个“可修”对手**（expert Wilson low ≥45% 且学生 point <45%），其中 expert Wilson low ≥50% 的只有 **8 个**。floor 违规 6–20 个的大部分是**继承自 deck 本身强度**，不是策略可学的部分。
- 因此，**验收应同时报“绝对 floor”和“相对老师的配对 Δ”**。只报绝对 floor 会把 deck 强度算到策略头上；只报 pooled 又会掩盖长尾。D24 的“逐 deck 报”应再加一列“老师 floor 违规数”，否则无法判断改进空间。

严格的“可修”清单（本报告重算，按 gap 排序；student 为 10-seed point，需 fresh seeds 复核）：

| 我方 deck | 对手 | 专家 rate（Wilson low） | 学生 rate | gap |
|---|---|---:|---:|---:|
| superconduct_mika_attack | mualani_double_pyro | 0.68 (0.58) | 0.25 | +0.43 |
| nahida_kokomi_raiden | superconduct_mika_attack | 0.72 (0.63) | 0.40 | +0.32 |
| klee_mavuika_mualani | dual_mualani_stacks | 0.65 (0.54) | 0.35 | +0.30 |
| yelan_clorinde_ororon | trial_skirk_ayaka_navia | 0.66 (0.56) | 0.40 | +0.26 |
| trial_skirk_ayaka_navia | klee_mavuika_mualani | 0.60 (0.50) | 0.35 | +0.25 |
| trial_skirk_ayaka_navia | trial_skirk_ayaka_navia | 0.60 (0.50) | 0.35 | +0.25 |
| superconduct_aggro | yelan_clorinde_ororon | 0.64 (0.54) | 0.40 | +0.24 |
| superconduct_aggro | superconduct_mika_attack | 0.56 (0.46) | 0.35 | +0.21 |
| superconduct_mika_attack | superconduct_mika_attack | 0.60 (0.50) | 0.40 | +0.20 |
| ice_water_battleship | skirk_chasca_freeze | 0.59 (0.48) | 0.40 | +0.19 |

这张表就是“靶向采集”的候选池。**natlan / beidou / chasca / skirk_ayaka 不应进入 BC 靶向采集**；它们需要更强的 teacher（在线 PG/自博弈）或换 deck，而不是更多专家示范。

### 1.2 为什么 H5 matchup 重加权失败（一个可复现的机制解释）

`scripts/train_bc_scale.py:308-349` 的 `matchup_weights_for_store` 用 `store.value[i]` 算每个 `(deck, opponent)` 的“胜率”，而 `store.value` 是**专家对局的结果**，不是学生对专家的胜率。`deficit` 模式因此会**抬高“老师也输”的 matchup 的权重**——正好把最不可学的标签放大。实测权重只有 0.84–1.22（多数 matchup 已在 0.45 以上），却在 SA（superconduct_aggro）上 −9.1pp、dvalin 上 −3.4pp（`reports/explore/plan_strength_2day_kaggle.md`）。这不是“偏离模仿目标”这么简单，而是**权重方向错了**：它奖励的是“老师输”，不是“学生可追的 gap”。

H5 的四个叠加问题：

1. **信号源错**：用 teacher outcome 而不是 student-vs-expert 的配对 Δ。
2. **没有新信息**：只是在旧语料上改损失权重，不能增加 target matchup 的状态/对手行为覆盖。
3. **组内样本太少**：每 matchup 约 20 局，deficit 估计本身噪声大；再归一化到均值 1，实际只微调了 0.84–1.22，效应弱且方向不稳。
4. **改变了有效学习率/校准**：mean-normalized 权重压低多数样本，等于在非目标 matchup 上少训练；对 BC 的校准是纯伤害。

改进方向不是“再调 strength”，而是：**把干预从损失层搬到采集层**（改变数据构成），并且只用 **student-vs-expert 的可修 gap** 定权。见 §3 R2。

### 1.3 对手条件化的现状：token 有，但等于没有

代码事实（只读核对）：

- `reps/opponent_identity.py` 的 `opponent_tags()` 只有 7 个粗标签（`active_population_slot` / `frozen_population` / `meta_population` / `baseline_random` / `baseline_heuristic` / `baseline_scripted` / `unknown`）。即使把 20 套脚本对手都标成 `baseline_scripted`，也不区分具体对手；而当前采集默认连这个都不传，落在 `unknown`。
- `reps/observation_encoder.py` 确实算了 `opponent_tag_id`、`opponent_entity_id`、`opponent_deck_id`，但 `train/data.py:183-185` 调 `encode_context(context, history=())` 时没有传 `opponent_*` 参数；`envs/match.py:431-454` 的 `_build` 也没有把 `PolicyPlayer.metadata` 传给 `build_decision_context`。本机抽查 `data/replays/gc_mualani_double_pyro/samples.jsonl`，每条样本的 `opponent_tag_id=6`（unknown）、`opponent_entity_id=0`、`opponent_deck_id=0`。也就是说，**当前 BC 语料里的对手身份特征是全零/unknown**。
- `opponent_deck_id` 用 65 桶的 `stable_hash_id`；本报告对 20 个 deck 名实算，只有 **18 个唯一 id**，`dual_mualani_stacks` / `electrocharged_moon` / `beidou_skirk_ayaka` 三套撞到同一个桶 57。即使把字段接上，**65 桶也不能做 20 路条件化**。
- 对手的 3 个角色是公共信息，且 `envs/decks.py` 的 `expert_deck_specs()` 显示 20 套 deck 只有 **19 个唯一阵容**（`skirk_ayaka_navia` 与 `trial_skirk_ayaka_navia` 共用 1115/1116/1608）。所以“按公开阵容条件化”几乎等价于“按 deck 条件化”，但不依赖非公开的 30 张牌表。
- `train/model.py` 的 CVPN 是 **mean pooling**：`global_vec = (projected * mask).sum(1) / denom`，然后 `option_scorer(cat([global_vec, option_vec]))`。全局 token 只占几十个 token 的 1/几十（实际观测约 40–100 个 token）；对手角色 token 也只占 6 个。**即使把对手 id 塞进全局 token，也会被平均池化稀释**。

结论：当前不是“要不要把对手作为 token”，而是 **token 输入存在但被稀释、显式 id 又是零/碰撞**。最小有效改动是**独立的公开阵容 embedding + FiLM**，把条件化变成乘性交互，而不是再加一个被平均掉的 token。对抗式 opponent model 对固定脚本没有必要：对手是确定性的、阵容公开、评估池已知；DRON 那类模型是为隐藏/非平稳策略设计的。

### 1.4 数据密度是主因，条件化是第二杠杆

`reports/explore/plan_strength_2day_kaggle.md`（2026-10-27）的对照很关键：同一 d256、同一批对手，只改语料密度/容量/任务数：

- 4 deck、250 seeds/matchup：val_acc 0.8377；
- 4 deck、40 seeds/matchup：0.7281；
- 20 deck、40 seeds/matchup：0.6932；
- 4 deck 加容量 d384：0.8326（不升）；
- 20 deck 加容量 d512：0.6753（不升）。

即 20-deck 崩塌里约 **3/4 来自每 matchup 密度**，1/4 来自任务数；容量不是瓶颈。20-deck 单模型的 8 个认证 deck 里 7 个显著变差，也是这个原因。**所以“对手条件化”最多处理那 1/4 的残留；主杠杆仍是每个 matchup 的对局数。** 这与 L4 的判断一致：靶向补采优先于条件化。

### 1.5 选目标的统计口径（直接回答 Q1）

三种统计量各管一件事，不能互相替代：

| 用途 | 统计量 | 为什么 | 最小样本 | 更新频率 |
|---|---|---|---|---|
| **认证绝对 floor** | Wilson 下界（per opponent） | 小 n 下 Wald 覆盖很差；Wilson 是项目现用口径 | 目标 0.55 → 100 局/对手；目标 0.50 → 385 局/对手（`eval_protocol.md`） | 每次完整认证（800 局/组）后 |
| **选 target / 诊断** | 配对 Δ（同 seed/side 学生−专家）+ per-opponent CI | 抵消种子/先手/对手运气，方差远小于两个独立 CI | 100 局/target 约 80% 功效抓 0.35→0.55；抓 0.40→0.50 需 ~400 | 每轮认证后；同一 target 需两套不重叠 seeds 复现 |
| **小 n 排序** | Beta-Binomial 收缩 `p̃=(w+κμ)/(n+κ)`，`κ=10–20`；按 `P(p<0.45)` 或收缩 gap 排序 | 直接取 raw min 会触发 optimizer’s curse（选中值偏极端）；Brown 2008 也显示 naive 当前平均是最差预测 | 三分类（40–60 局/matchup）用收缩；确认用 100 局/matchup | 每轮更新一次；冻结 target 集 4–6 周 |

**worst matchup 的定义**（本报告采用）：

```
T_o = 1[expert_wilson_low ≥ 0.45] · max(0, p_e − 0.45) · max(0, 0.45 − p_s) · 1[p_s < 0.45]
```

其中 `p_e`、`p_s` 先用上面的 Beta 收缩（或配对 Δ 的收缩）估；`T_o` 只对“老师能赢、学生可能输”的 matchup 非零。**选 target 时按 `T_o` 排序，不按 raw `p_s` 最小**；`T_o` 为 0 的 matchup（老师也输）不进入靶向采集。Stage 0 再用 50 fresh seeds（100 局/target）确认 `p_s<0.45` 且 `paired Δ<−0.05`，才算 target。

**噪声底线**：10 seeds（n=20/matchup）下，真实 0.50 的 matchup 有 **25.2%** 概率 point-rate <45%；20 个 matchup 期望 5 个假违规。Wilson 下界在 n=20 时要真实胜率 ~0.668 才能过关；n=100 时 ~0.548。**用 400 局/组（10 seeds）的 floor 违规数做实验判据，基本是在测噪声。**

---

## 2. Annotated bibliography（15 项）

> 每项按“问题 / 方法核心 / 证据 / 局限 / 链接 / 可迁移点”写。论文标题保留英文。

### 2.1 Training Region-based Object Detectors with Online Hard Example Mining（OHEM）

- **问题**：检测数据里 easy background 压倒 hard foreground，手工 fg/bg 比例是启发式。
- **方法核心**：对每个 RoI 算当前损失，按损失选 top-N 形成 minibatch；用 NMS（IoU 0.7）去重高度重叠的 RoI，避免相关损失重复计数；只对选中 RoI 反传，计算量与普通 SGD 同量级。不需要 fg/bg 比例。
- **证据**：PASCAL VOC07 上比手工 `bg_lo=0.1` 的 Fast R-CNN 高 **+2.4 mAP**（VGGM）；与多尺度等技巧合到 **78.9 / 76.3 mAP**（VOC07/12）。COCO 上数据越大、越难，收益越明显。
- **局限**：假设 hard example 的标签可靠；“hard”是损失高，不等于可学。后续 RHO-LOSS 明确指出高损失点常是噪声/离群。
- **链接**：[arXiv:1604.03540](https://arxiv.org/abs/1604.03540)，代码 [abhi2610/ohem](https://github.com/abhi2610/ohem)。
- **可迁移点**：① 只在“老师能赢”的 matchup 内做 hard 采样；② 同一局的高损失决策高度相关，必须按局/按状态去重（OHEM 的 NMS 思想），不要让一局长对局刷满 batch；③ 不要用 OHEM 跨 matchup 选“老师也输”的样本。

### 2.2 Prioritized Experience Replay（PER）

- **问题**：均匀 replay 把同样多的更新给罕见/高价值 transition 和冗余 transition。
- **方法核心**：`P(i)=p_i^α / Σ_k p_k^α`；`p_i=|δ_i|+ε`（proportional）或 `p_i=1/rank(i)`（rank-based）。用 sum-tree 做到 O(log N) 采样/更新；重要性采样权重 `w_i=(1/(N·P(i)))^β`，β 从初值退火到 1，并按 `1/max w_i` 归一化，只向下缩放。
- **证据**：Atari 49 游戏里 **41 个**优于 uniform DQN；学习速度在 41–47% 的训练时间内达到 Double DQN 的等效点。rank-based 对噪声尖峰更稳健。
- **局限**：TD-error 优先级在奖励随机/自举误差大时会追噪声；需要 IS 校正；对 BC（非 TD）不能直接套。
- **链接**：[arXiv:1511.05952](https://arxiv.org/abs/1511.05952)。
- **可迁移点**：① 用 **rank** 而不是原始 deficit 大小给 matchup 排序（抗离群）；② 保留所有 matchup 的非零采样概率（uniform 混合项）；③ 如果采样分布偏离评估分布，必须做校正或在评估时用均匀集；④ BC 场景更适合“数据混合权重”，不是 PER 的 TD 优先级。

### 2.3 Active Learning Literature Survey（Settles 2009）

- **问题**：标注预算有限时，如何选最有信息量的未标注样本。
- **方法核心**：不确定性采样（least confident / margin / entropy）、query-by-committee、expected model change、expected error reduction、density-weighted 方法；batch-mode 主动学习；停止准则。
- **证据**：综述性技术报告，给出各方法的适用条件与失效模式；没有单一“最佳”方法，取决于标签成本、模型类型、数据冗余度。
- **局限**：核心假设是**标注贵**；本项目标签由引擎/脚本自动产生，真正的成本是**对局数**和**matchup 覆盖**，不是标注。直接把 AL 的不确定性采样搬过来会选到离群/噪声状态。
- **链接**：[Settles, Active Learning Literature Survey, UW–Madison TR1648](https://burrsettles.com/pub/settles.activelearning.pdf)（本次未抓取全文，按经典综述引用；可信度：技术报告）。
- **可迁移点**：① 把“不确定性”用在**状态多样性**（同一 matchup 内选不同局面），不是跨 matchup 选目标；② 用 committee/ensemble 的方差做“这个 matchup 是否可学”的诊断；③ 不要用 AL 的 label-cost 逻辑替代对局预算逻辑。

### 2.4 Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt（RHO-LOSS）

- **问题**：高损失点可能是噪声/离群，硬采样会放大它们；课程学习又可能反复训练已学会的点。
- **方法核心**：可约损失 `S(x,y)=L[y|x;D_t]−L[y|x;D_ho,D_t]`，即“在当前模型上学这个点，能让 holdout 损失降多少”。用 holdout 模型估第二项；选可约损失最大的点。只有“可学、值得学、还没学会”的点被选中。
- **证据**：Clothing-1M 上比 uniform 少 **18×** 训练步、最终精度 **+2%**；跨 MLP/CNN/BERT、7 个数据集一致优于高损失/高梯度范数选择。论文明确指出高损失/高梯度选到的多是噪声标签点。
- **局限**：需要 holdout 模型与额外前向；holdout 分布要与目标一致；对“标签本身错”的组只能降到低权重，不能纠正。
- **链接**：[arXiv:2206.07137](https://arxiv.org/abs/2206.07137)，代码 [OATML/RHO-Loss](https://github.com/OATML/RHO-Loss)。
- **可迁移点**：这是“老师也输的 matchup 不该补采”的理论依据。把 matchup 当“点”：`reducible_gap = L_student(o) − L_expert/reference(o)`；如果 expert 自己也高损失（打不赢），可约损失小，**不要 target**。与本文 §1.1 的 teacher-limited 过滤同构。

### 2.5 Curriculum Learning（Bengio et al. 2009）

- **问题**：非凸训练里，样本呈现顺序是否影响收敛速度与局部极小质量。
- **方法核心**：先易后难；在 shape recognition 里先在 BasicShapes（正方形/圆/等边三角形）训练 k epoch，再切到 GeomShapes（更多变化）；在 language modeling 里逐步扩大词表/句子难度。
- **证据**：课程学习显著改善泛化；**只用 easy 数据效果差**；**把 easy+hard 混在一起、无课程，也不如分阶段课程**（论文明确做了这两个对照）。作者把课程解释为 continuation method/正则。
- **局限**：“难度”依赖任务定义；如果 easy 分布与目标分布差异大，课程会引入偏置；没有通用的难度度量。
- **链接**：[DOI:10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)，[ICML 2009 PDF](https://icml.cc/2009/papers/119.pdf)。
- **可迁移点**：靶向采集不能“只采失败”也不能“只采容易”：先均匀 warmup，再逐步加 target；保留 anchor 防遗忘。E1 的 65–75% anchor 就是“不要只采 hard”的工程化。

### 2.6 Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics

- **问题**：数据质量参差，但只看最终精度无法区分“难学”与“噪声/歧义”。
- **方法核心**：一次训练中记录每个样本的 confidence（真实类概率均值）、variability（跨 epoch 标准差）、correctness；画出 data map。三个区域：easy-to-learn（低 variability、高 confidence）、hard-to-learn（低 variability、低 confidence，常对应标注错误）、ambiguous（高 variability）。
- **证据**：在 SNLI 等 4 个数据集上，用 33% 子集训练；论文报告 ambiguous 区域对 OOD 泛化贡献最大，hard-to-learn 区域“often correspond to labeling errors”，easy-to-learn 区域主要服务优化。数据地图选择可优于随机/高置信子集，但“最优子集”取决于任务。
- **局限**：需要多次 epoch 的训练动态；对 RL/策略数据没有直接定义；ambiguous 不一定等于“该多采样”。
- **链接**：[arXiv:2009.10795](https://arxiv.org/abs/2009.10795)，代码 [allenai/cartography](https://github.com/allenai/cartography)。
- **可迁移点**：在决定 target 前，先看该 matchup 的训练动态：如果模型**一直低置信且低方差**，更像“学不动/标签有问题”（对应老师也输），不要加数据；如果**高方差**，更像“歧义但可学”，可以加数据/加容量。这是比“loss 高就采”更稳的筛查。

### 2.7 Prioritized Level Replay（PLR）

- **问题**：RL 在程序生成环境里均匀采 level，浪费在已会/太难的 level 上；需要按“学习潜力”采样。
- **方法核心**：把 level 当采样单元。`P_replay=(1−ρ)P_S+ρP_C`；`P_S ∝ h(S_i)^{1/β}`，`h(S_i)=1/rank(S_i)`（rank 比 proportional/greedy 稳）；`P_C ∝ (c−C_i)/Σ(c−C_j)` 是 staleness 分布；每局先 Bernoulli 决定 replay 还是采新 level。分数用 TD-error 估计“重访该 level 的未来学习潜力”。
- **证据**：Procgen 上匹配此前 SOTA，并与前 SOTA 结合把 test return 提升到 **>76%**（相对标准 RL baseline）；rank prioritization 优于 proportional/greedy。
- **局限**：需要可识别的 level 与可重置环境；TD-error 在稀疏回报/长局下噪声大；staleness/温度需要调。
- **链接**：[arXiv:2010.03934](https://arxiv.org/abs/2010.03934)，代码 [facebookresearch/level-replay](https://github.com/facebookresearch/level-replay)。
- **可迁移点**：**这是本文最接近的算法模板**：把 20 个对手当 20 个 level；按“可修 gap/学习潜力”排序；rank + 温度 β；用 staleness 保证每个对手都被回访；用 Bernoulli 的 uniform 项保证覆盖。E1 的采样矩阵就是 PLR 的离散化版本。

### 2.8 DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

- **问题**：多域预训练的混合比例靠手工/下游调参；需要无下游任务的最坏域优化。
- **方法核心**：先训 reference model（均匀混合）；再训小 proxy，做 group DRO：`min_θ max_{α∈Δ} Σ_i α_i [L_i(θ)−L_ref,i]`，即**最坏域的 excess loss**（不是原始 loss，避免把高熵/噪声域抬上去）；取训练过程中 α 的平均作为域权重；最后按权重重采样训大模型。
- **证据**：280M proxy 给 8B 模型（30×）定权重；The Pile 上平均 few-shot **+6.5pp**，达到 baseline 精度少 **2.6×** 训练步；GLaM 上甚至匹配下游调权。
- **局限**：需要 reference/proxy 两轮训练；域定义要合理；proxy 与主模型规模差异大时权重可能漂移；对“域本身不可学”只能降权，不能创造信号。
- **链接**：[arXiv:2305.10429](https://arxiv.org/abs/2305.10429)，代码 [sangmichaelxie/doremi](https://github.com/sangmichaelxie/doremi)。
- **可迁移点**：把 matchup 当 domain；**只在可修 matchup 上做 DoReMi**（否则 minimax 会把 natlan 这种“老师也输”的域抬上去，重演 H5）。用 d128 proxy 定 20 个 matchup 的采集/训练混合比例，主模型 d256；这比 loss 重加权安全，因为它改的是数据构成，不是标签分布。

### 2.9 Massively Multilingual Neural Machine Translation in the Wild（temperature sampling）

- **问题**：多语言语料极度不均衡，直接按自然比例采样会饿死低资源语言。
- **方法核心**：按 `p_l^{1/T}` 采样，`p_l` 是语言 l 的语料占比；`T=1` 为自然分布，`T=100` 近似均匀，论文用 `T=5` 做平衡采样。
- **证据**：`T=5` 提升高资源语言，同时保留低资源迁移；`T=100` 过度采样低资源会伤高资源。
- **局限**：温度是全局的，不能表达“某几个 matchup 特别重要”；对 RL/BC 的 match 分布没有直接理论保证。
- **链接**：[arXiv:1907.05019](https://arxiv.org/abs/1907.05019)。
- **可迁移点**：如果不想手写 300/700 的配额，可用 `q_o ∝ (n_o or deficit_o)^{1/T}`，`T=2–5`，并加 uniform 下限与 3× 上限。温度采样比硬配额平滑，适合多轮迭代。

### 2.10 Opponent Modeling in Deep Reinforcement Learning（He et al. 2016）

- **问题**：多智能体里对手策略未知且非平稳；需要在不显式分类对手的情况下自适应。
- **方法核心**：DRON。把对手的观测/动作历史编码成 opponent representation，再与 state representation 结合：DRON-concat 直接拼接；DRON-MoE 用 gating 网络对 K 个 Q 专家加权。可选用多任务监督预测对手动作/策略。
- **证据**：模拟足球与 quiz bowl 两个任务上优于 DQN 及其变体；能自动发现不同对手策略模式；对非平稳对手更稳。
- **局限**：对手是**未知/非平稳**时才有价值；如果对手是固定脚本、阵容公开，学习一个 opponent model 是多余的，且增加方差。
- **链接**：[PMLR v48 he16](https://proceedings.mlr.press/v48/he16.html)，[arXiv:1609.05559](https://arxiv.org/abs/1609.05559)，代码 [hhexiy/opponent](https://github.com/hhexiy/opponent)。
- **可迁移点**：我们的 20 套脚本是固定、确定性的，**不需要 DRON 式隐变量建模**；需要的是把公开阵容显式喂进去（§3 R3）。如果将来打人类/league，再考虑 DRON。

### 2.11 FiLM: Visual Reasoning with a General Conditioning Layer

- **问题**：如何用条件信息（问题、任务、对手）调制网络特征，而不是简单拼接。
- **方法核心**：`FiLM(F_i | γ_i, β_i)=γ_i F_i+β_i`；由条件网络输出每个 feature 的 scale/shift，做 feature-wise affine。比 concatenation 更强，参数少，能插入任意层。
- **证据**：CLEVR 上把 SOTA error **减半**；对消融/架构改动稳健；few-shot/zero-shot 泛化好。
- **局限**：条件必须是离散/低维或由小网络编码；条件过强可能过拟合（A2 的 id embedding 就是教训）；需要正确的 zero-init 以保持底座。
- **链接**：[arXiv:1709.07871](https://arxiv.org/abs/1709.07871)，代码 [ethanjperez/film](https://github.com/ethanjperez/film)。
- **可迁移点**：在 CVPN 的 `global_vec` 上加 `(1+scale)*global_vec+shift`，条件用**公开阵容**的 19 类 embedding；zero-init 使初始为恒等，再从 BC 底座 fine-tune。

### 2.12 Gradient Surgery for Multi-Task Learning（PCGrad）与 GradNorm

- **问题**：多任务共享 trunk 时，任务梯度方向冲突/量级不平衡，导致负迁移。
- **方法核心**：PCGrad：若 `g_i·g_j<0`，把 `g_i` 投影到 `g_j` 的法平面：`g_i ← g_i − (g_i·g_j/||g_j||²) g_j`；随机任务顺序。GradNorm：学习任务权重 `w_i`，使梯度范数匹配 `Ḡ·r_i^α`，`r_i` 是相对逆训练率；只有一个 α。
- **证据**：PCGrad 在多任务 RL 上 **>30% 绝对提升**，多任务监督也有效；GradNorm 在 NYUv2 只增加 **~5%** 训练时间，匹配/超过 grid search。
- **局限**：PCGrad 对 T 个任务要 O(T²) 投影/多次反传；GradNorm 需要每任务 loss 与额外梯度计算，任务数多时不稳定。两者都假设“任务冲突是主要瓶颈”。
- **链接**：[PCGrad arXiv:2001.06782](https://arxiv.org/abs/2001.06782)，代码 [tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad)；[GradNorm arXiv:1711.02257](https://arxiv.org/abs/1711.02257)，代码 [hosseinshn/GradNorm](https://github.com/hosseinshn/GradNorm)。
- **可迁移点**：先做**梯度余弦诊断**：同一 shared trunk 上，按 matchup 取 batch，算 `cos(g_i,g_j)`。若大量负值，再考虑 PCGrad/每 deck adapter；不要一上来就上。现有证据（SA 与 dgeo 的 PG 更新方向相反：SA −0.146 vs dgeo +0.100）说明冲突真实存在，但 20-deck 崩塌的主因是密度（§1.4），PCGrad 是第二/第三杠杆。

### 2.13 Distributionally Robust Neural Networks for Group Shifts（Group DRO）

- **问题**：平均精度高但最差组崩；需要最小化最坏组损失。
- **方法核心**：`min_θ max_{α∈Δ} Σ_g α_g L_g(θ)`；用 exponentiated gradient 更新组权重。关键发现：在过参数化网络里，**naive group DRO 会失败**（训练损失都为零时最坏组也零），必须配强 L2/early stopping 等正则，才能把 worst-group 精度抬 **10–40pp**。
- **证据**：Waterbirds/CelebA/MultiNLI 上，正则化 group DRO 显著优于 ERM；无正则的 group DRO 不比 ERM 好。
- **局限**：组标签要可靠；小组样本少时组权重噪声大；需要额外正则/早停。
- **链接**：[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)，代码 [kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)。
- **可迁移点**：floor 就是 worst-group 目标，Group DRO 是“正确目标”的 principled 版本；但 H5 是一个**无正则、组定义/信号源都错**的粗糙重加权，失败与 Sagawa 的警告一致。若要做，必须在**新采集的可修 matchup 数据**上做，并加正则/早停，且组权上限 2–3×。

### 2.14 Grandmaster level in StarCraft II using multi-agent reinforcement learning（AlphaStar League / PFSP）

- **问题**：纯 self-play 会循环/遗忘；需要既强又不易被剥削的 population。
- **方法核心**：League 三类 agent：main agents 用 **PFSP** 选对手，`P(B) ∝ f(P(A beats B))`；`f_hard(x)=(1−x)^p` 聚焦最难对手，`f_var(x)=x(1−x)` 聚焦同水平对手（给挣扎中的 main agent/exploiter 用）；main exploiters 专打当前 main agent，league exploiters 找全联盟盲点，定期重置。主 agent 混合约为 35% self-play + 50% PFSP + 15% forgotten/exploiters（据 Methods 摘录，**未核实**：本次未逐字核对 Nature PDF）。
- **证据**：AlphaStar 在三族上达到 Grandmaster，高于 99.8% 官方排名玩家；消融显示 PFSP 在 relative population performance、可剥削性、最终强度上都优于 FSP（Extended Data Fig. 5）。FightLadder 在格斗游戏里用 `C ∝ 1−p(win)` 的对手课程，12 角色 20 epoch 后全游戏胜率 >0.6。
- **局限**：PFSP 是 self-play league 的对手选择，不是“专家 BC 的补采”；纯 `f_hard` 会把弱 deck 送到永远赢不了的对手前（论文自己用 `f_var` 兜底）；需要大量对局与 population。
- **链接**：[Nature 575, 350–354 (2019)](https://www.nature.com/articles/s41586-019-1724-z)；[FightLadder arXiv:2406.02081](https://arxiv.org/abs/2406.02081)，代码 [wenzhe-li/FightLadder](https://github.com/wenzhe-li/FightLadder)。
- **可迁移点**：**不要只采 worst**。用 `f_hard` 只在“可修且学生接近”的 matchup；对 natlan 这种老师也输的 deck 用 `f_var`（同水平/可赢的少数对手）或直接不训；保留 10–15% 的 forgotten/mastered 数据防遗忘。E1 的 65–75% anchor 就是 `f_var`+forgotten 的工程版。

### 2.15 OpenAI Five 的 surgery 与 Suphx 的 oracle guiding

- **问题**：① 训练中途改观测/网络/动作空间，如何不毁掉已学技能；② 不完全信息游戏里，如何利用训练期可见的隐藏信息，但部署时只用公共信息。
- **方法核心**：OpenAI Five 的 surgery：加宽层时 `W1'=[W1; R()]`、`W2'=[W2 0]`，新维度随机、输出权重置零，保证新模型在初始时刻与原模型**函数等价**；零权重随后由梯度接管。整个项目约每两周一次 surgery。Suphx 的 oracle guiding：训练时把完美信息特征 `x_o` 与正常特征 `x_n` 一起输入，用 Bernoulli dropout `δ_t`（`P(δ=1)=γ_t`）逐步把 γ 从 1 降到 0；γ=0 后继续训练，并把 lr 降到 1/10、裁掉重要性权重过大的样本。
- **证据**：OpenAI Five 在 7,000+ 场对社区比赛中赢 **99.4%**；surgery 让 10 个月训练不必从头。Suphx 在 Tenhou 达到稳定段位 **8.74 dan**、高于 **99.99%** 官方排名玩家。
- **局限**：surgery 需要精确的形状/函数保持，零初始化可能引入对称性（论文说只零最小集合）；Suphx 的 oracle 训练依赖额外特征，部署必须完全 drop，且 info-leak 门禁要单独验证。
- **链接**：[OpenAI Five arXiv:1912.06680](https://arxiv.org/abs/1912.06680)，[OpenAI blog](https://openai.com/index/how-to-train-your-openai-five/)；[Suphx arXiv:2003.13590](https://arxiv.org/abs/2003.13590)。
- **可迁移点**：① 加 FiLM/对手 embedding 时**zero-init 输出**，让初始策略与 BC 底座逐值相同（仓库已有 `train/model.py:220` 的 `zero_init_identity_branch` 模式）；② 如果将来用 privileged 信息做 belief/对手模型，用 Suphx 式退火，部署前完全 drop，并加 leak test。两者都不是当前 floor 的第一杠杆。

### 2.16 统计工具（不计入 15 篇正文）

- **Wilson 1927**：二项比例的 score interval；小 n 比 Wald 稳。[DOI:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)。
- **Brown, Cai & DasGupta 2001**：系统比较二项区间，推荐小 n 用 Wilson 或 Jeffreys，Wald 的覆盖性质很差。[DOI:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)。
- **Brown 2008**：用赛季前段打率预测后段，**naive 当前平均是最差预测**；empirical Bayes/hierarchical Bayes 显著更好。[arXiv:0803.3697](https://arxiv.org/abs/0803.3697)，[DOI:10.1214/07-AOAS138](https://doi.org/10.1214/07-AOAS138)。
- **Smith & Winkler 2006**：optimizer’s curse——从多个带噪声估计里选最大/最小，选中值的估计会系统性偏乐观/偏极端；用 Bayesian 收缩校正。[DOI:10.1287/mnsc.1050.0451](https://doi.org/10.1287/mnsc.1050.0451)。

---

## 3. 最小可复现 recipe（按性价比排序）

### R0（0 新对局）：先做“老师天花板 vs 学生 gap”分层 + 现有 arm 的 honest router

- **输入**：`reports/arena/20261008T163940_full_matrix_20x20.json`、各 deck 的 `reports/explore/*.json`、`data/eval/*.jsonl`。
- **输出**：§1.1 的两张表；每个 deck 的 `teacher_limited` / `reducible` 计数；一个 `targets.json`（deck, opponent, expert_rate, student_rate, gap）。
- **做法**：
  1. 对每个 `(deck, opponent)` 取 20×20 矩阵的 expert rate 与 Wilson low（n=100）；
  2. 取学生最新认证臂的 per-opponent rate（n=20），算 point 与 Wilson；
  3. 标 `teacher_limited = expert_wilson_low < 0.45`；`reducible = expert_wilson_low ≥ 0.45 and student_point < 0.45`；按 gap 排序；
  4. 用 `scripts/matchup_router.py` 在现有 arm 上做 per-opponent 路由（even seeds 选、odd seeds 打分），看 routed floor 违规数是否已经 ≤2。若是，优先做 portfolio，不要补采。
- **代码**：`scripts/matchup_router.py`、`eval/scoreboard.py`、`reports/SPECIALISTS.md`。
- **成本**：0 新对局；1–2 h 读/算。
- **信号**：可修集合大小；router 的 held-out floor 违规数。
- **失败判据**：如果 19 个 deck 里可修 <5，或 router 已把 floor 违规压到 ≤2，则靶向数据的期望收益很低，转向 portfolio/PG。

### R1（0 新对局）：把“选目标”的统计口径修对

- **输入**：同 R0，外加每个 deck 的 fresh-seed 评测 JSONL。
- **输出**：每个 matchup 的 `paired Δ vs expert`（同 seed/side）、`Wilson low/high`、`Beta 收缩后的 p̃`、`P(p<0.45)`。
- **公式**：
  - 配对：`d_i = score_student,i − score_expert,i`；`Δ̄ ± z·sd(d)/√n`；
  - 收缩：`p̃_o=(w_o+κμ)/(n_o+κ)`，`κ=10–20` 局，`μ` 用 pool mean 或 deck mean；
  - 选择分：`T_o = 1[expert_low≥0.45] · max(0, p_e−0.45) · max(0, 0.45−p_s) · 1[p_s<0.45]`（`p_e` 专家 rate，`p_s` 学生 rate），按 `T_o` 排序，**不要按 raw min**。
- **代码改动（建议，不在本次执行）**：`eval/scoreboard.py` 的 `paired_difference.per_opponent` 目前只有 delta、没有 CI；加 per-opponent bootstrap CI（复用 `cluster_bootstrap_ci`），输出 `n/delta/ci_low/ci_high`。`scripts/eval_protocol.py` 的 `--baseline expert:<deck>` 已经能生成配对数据，只差聚合。
- **成本**：0 新对局；0.5–1 h 计算。
- **信号**：target 排序稳定性（两套不重叠 seeds 的 rank correlation）。
- **失败判据**：同一 matchup 在两套 seeds 上排名差异 >5 位，说明 20 局/ matchup 的排序不可靠；先补到 40–60 局再选。

### R2（主实验）：采集层混合，而不是损失层重加权

- **输入**：每个 deck 的现有 dense BC store（约 2,000 局专家对局）、`targets.json`、fresh seeds。
- **输出**：两个语料 arm（control: uniform 1,400 局；treatment: target 300×k + uniform 补足 1,400 局），各训一个 checkpoint。
- **采样矩阵（E1 用离散版）**：
  - 只对 `reducible` 且 fresh-seed 复核 `student<0.45`、`Δ<−0.05` 的 matchup 设 300 局；
  - 其余 20 个对手按 uniform 补足；每个 deck 总 1,400 局；
  - 单 matchup 不超过总语料的 25%（300/1,400≈21%），避免一个 matchup 压倒；
  - 控制 arm 同总预算，全 uniform（每对手 70 seeds）。
- **一般公式（多轮迭代用）**：在可修集合内 `q_o ∝ 1/rank(T_o)^{1/β}`，`β=0.5–1.0`；混 `ε/20` uniform（`ε=0.6–0.7`）；cap `q_o ≤ 3/20`；每轮按最新 800 局认证更新 `T_o`。
- **训练**：先用 `scripts/datafirst_bc_store.py` 把 base + 新语料合成 store；再 `scripts/datafirst_bc_train.py --store-cache ... --init-from <base> --epochs 5`。treatment/control 同 epoch、同 seed、同 base；**不做** `--matchup-weight deficit`。
- **代码/命令（无需改采集代码）**：
  ```bash
  # treatment: 每个 target 单独采 300 seeds，再补 uniform
  python -m train.collect --deck D --opponents OPP --seed-start 12000 --seed-count 300 \
      --format compact --tag tgt_D_OPP --workers 12
  python -m train.collect --deck D --all-opponents --seed-start 13000 --seed-count 40 \
      --format compact --tag anchor_D --workers 12
  # control: 全 uniform 1,400 局（20 对手 × 70 seeds）
  python -m train.collect --deck D --all-opponents --seed-start 14000 --seed-count 70 \
      --format compact --tag ctrl_D --workers 12
  # 合并 store
  python -m scripts.datafirst_bc_store --replays data/replays/tgt_D_OPP data/replays/anchor_D \
      --store-cache data/store/e1_treat_D.pt --val-every 10
  # 从底座 fine-tune
  python -m scripts.datafirst_bc_train --store-cache data/store/e1_treat_D.pt \
      --init-from data/checkpoints/<deck>_base.pt --epochs 5 --out data/checkpoints/e1_treat_D.pt
  ```
- **需要改的仓库文件**：严格说**不需要改采集**；若要自动按 `targets.json` 生成配额，加一个 `scripts/collect_targeted.py`（读 JSON、循环调用 `train.collect`）即可。训练复用 `scripts/datafirst_bc_train.py`（`--store-cache/--init-from`）与 `scripts/datafirst_bc_store.py`（`--replays` 合并）。
- **成本（E1 full，3 deck）**：8,400 新采集局（约 1.7–2.1 h @4,000–4,900 局/h；compact 约 5.5 GB，`--no-privileged-state` 约 2.5 GB）；训练 6 runs × 64k 新样本 + 20k replay × 5 epoch ≈ 3–7 h CPU（A800 <1 GPU·h）；评测 5,840 局 ≈ 3.9 h。总计约 8–13 h。
- **预期信号**：通过确认的 target 里 ≥2 个 paired Δ ≥ +0.10（CI low >0）；3 个 deck 的 floor 违规数合计 −≥2；非 target 不出现 >5pp 的成片退化。
- **失败判据**：<1 个 target 改善，或任一非 target 退化 >5pp 且重复出现，或 pooled CI low < −0.02。则停止 BC 靶向路线，转 PG。

### R3（第二杠杆）：公开阵容 FiLM，零新增对局

- **输入**：现有 store（含 token/option/keys）、每个样本的对手 3 角色 id（从 `player_view` 的 opponent characters 取，公共信息）。
- **输出**：在 CVPN `global_vec` 上加 FiLM 的 checkpoint；与同数据、同 epoch 的 base 做 A/B。
- **做法**：
  1. 在 `reps/observation_encoder.py` 的 `EncodedObservation` 加 `opponent_lineup: tuple[int,int,int]`（从 `context.player_view.players[1-acting].characters` 的 `definition_id` 排序得到；不要从 `full_state` 取）；
  2. `train/data.py` 把它写进 sample payload（缺省 `(0,0,0)`，向后兼容）；
  3. `scripts/train_bc_scale.py` 的 `Store` 加 `lineup_ids`，`_collate` 加一个 `(B,)` long tensor；从 `key[1]` 或 payload 读；
  4. `train/model.py` 加 `lineup_emb`（19–20 类 + unknown）与 `film = nn.Linear(d_emb, 2*d)`；`forward_all(..., lineup_ids)` 里 `scale, shift = film(emb).chunk(2,-1); global_vec = global_vec*(1+scale)+shift`；`zero_init` 最后一层，使初始恒等（复用 `zero_init_identity_branch` 的写法）；
  5. `agents/neural.py` 推理时从 `observation.opponent_lineup` 取 id；
  6. 加 leak test：两份 state 只差对手手牌/牌堆，lineup 相同，编码输出必须一致；再加一个 store/engine 一致性测试（同 `(deck,opponent,seed,step)` 的 lineup id 相同）。
- **关键超参**：`d_lineup=32`、`d_model=256`、5 epoch、lr 3e-5；zero-init；只对 `global_vec` 做 FiLM，不动 option encoder。
- **代码**：`train/model.py`、`reps/observation_encoder.py`、`train/data.py`、`scripts/train_bc_scale.py`、`agents/neural.py`、`tests/test_info_leak.py`。
- **成本**：0 新采集局；2 deck × 2 arm × 5 epoch × 92k ≈ 3–5 h CPU（A800 <0.1 GPU·h）；评测 1,600 局 ≈ 1.1 h。
- **预期信号**：2 个 deck 里 ≥1 个 floor 违规 −≥2，pooled 不降；target matchup 的 paired Δ 有正号。
- **失败判据**：两个 deck 都无变化或变差；或 val_acc 涨、引擎胜率不涨（A2 重演）→ 停止条件化，回到数据/PG。

### R4（可选，最后）：DoReMi/Group DRO 的 mixture-level 版本

- **输入**：20 个 matchup 的已有语料 + 一个 reference BC（均匀）。
- **输出**：20 维采样权重 `α`（用于采集或训练 batch 的 matchup 采样），不是 loss 权重。
- **做法**：d128 proxy 做 group DRO on excess loss `L_proxy(o)−L_ref(o)`；只对 `reducible` 组开放权重，teacher-limited 组固定为 0 或极低；取训练过程 α 的平均；主模型按 α 重采样。加 L2/early stopping（Sagawa 的关键条件）。
- **成本**：proxy 约主模型 1/4 计算；无新对局（复用 store）；2–4 h CPU。
- **信号**：最差 matchup 的 CE/胜率改善，且平均不降。
- **失败判据**：α 把权重集中到 1–2 个 matchup（>50%），或 worst-group 胜率不升；说明组太小/噪声大，停止。

---

## 4. 预注册实验

> 本文即预注册。实验前不改判据；所有新种子必须与历史 `data/eval` 不重叠；评测只跑纯策略，`--workers 12`，引擎 lock 固定。

### E1（主实验）：可修 matchup 的采集层混合 vs 等预算 uniform

**假设**：在 `reducible` matchup 上把 300 局从 uniform 配额里“换”给 target，能把通过 Stage 0 的 target 的 paired Δ 提高 ≥10pp，并让 3 个 deck 的 floor 违规数合计下降 ≥2，而 pooled 不劣于 −2pp。

**目标 deck / target（4 个，均为 §1.1 的可修项）**：

| deck | target opponents（每个 300 局） | expert rate (Wilson low) | student rate（10-seed，待 fresh 复核） |
|---|---|---:|---:|
| nahida_kokomi_raiden | superconduct_mika_attack | 0.72 (0.63) | 0.40 |
| superconduct_mika_attack | mualani_double_pyro | 0.68 (0.58) | 0.25 |
| superconduct_mika_attack | superconduct_mika_attack | 0.60 (0.50) | 0.40 |
| superconduct_aggro | yelan_clorinde_ororon | 0.64 (0.54) | 0.40 |

**Stage 0（确认，必须）**：用 50 fresh seeds × 2 sides（100 局/target）跑 `eval_protocol --opponents <targets> --baseline expert:<deck>`。保留条件（point estimate）：`student_rate < 0.45` 且 `paired Δ < −0.05` 且 `expert Wilson low ≥ 0.45`。不满足的 target 删除，不补新 target（避免挑噪声）；被删除的 target 配额用 uniform 补足，保证两 arm 等总量。成本 400 局，约 16 min。

**种子隔离**：Stage 0 用 1100–1149（50 seeds）；focused eval 用 1200–1249（target 50 seeds）与 1250–1269（非 target 10 seeds）；认证用 1300–1319（20 seeds）；采集用 12000+/13000+/14000+。pilot 用自己的 1050–1099（confirm）与 1150–1199（eval）。历史已用过 0–9、50–59、60+10r、810–860、900–909；启动前用 `data/eval/*.meta.json` 核对不重叠，不满足就顺延到 2000+。

**Arm 设计（每 deck 总 1,400 局，两 arm 等总量）**：

| deck | treatment | control |
|---|---|---|
| nahida_kokomi_raiden | 1 target × 300 + 20 对手 × 55 seeds（1,100） | 20 × 70（1,400） |
| superconduct_mika_attack | 2 targets × 300 + 20 × 40（800） | 20 × 70（1,400） |
| superconduct_aggro | 1 target × 300 + 20 × 55（1,100） | 20 × 70（1,400） |

- 采集：3 deck × 2 arm × 1,400 = **8,400 新局**；target 共 1,200 局；treatment 的 uniform 为 3,000 局，control 为 4,200 局（两 arm 总量相等）。
- 训练：从各 deck 的 BC 底座 `--init-from`（`scripts/datafirst_bc_train.py`），5 epoch、同 lr/batch/seed；treatment/control 只差数据构成；**不做** `--matchup-weight`。
- 评测（focused）：每个 target 50 seeds × 2 sides = 100 局；每个非 target 对手 10 seeds × 2 sides = 20 局。每 deck ≈ 480–560 局；3 deck × 2 arm ≈ **3,040 局**。
- 认证（仅胜出 arm）：`--opponents` 全 20、`--seeds 20`、`--strict-ci`，3 deck × 800 = **2,400 局**。

**主指标**：通过 Stage 0 的 target 的 paired Δ（student − expert，同 seed/side），以及“Δ ≥ +0.10 且 CI low >0”的 target 数。
**次指标**：3 deck 的 floor 违规数（point rate <45%）；Wilson floor 只在认证阶段看；非 target 的逐对手退化。
**Guard**：pooled 的 paired Δ CI low ≥ −0.02；`fallback_rate=0`、`error_rate=0`；无 target 退化 >5pp。
**判定**：
- **成功**：≥2 个通过确认的 target 达到 +0.10 且 CI low >0；floor 违规合计 −≥2；guard 全过。→ 扩到 §1.1 的 8 个高置信可修 matchup，再按同一配方跑第二轮。
- **部分成功**：1 个 target 改善且 guard 过。→ 只在该 matchup 复现一次（新 seeds），不要外推。
- **失败**：0 个改善，或任一 guard 失败。→ 停止 expert BC 靶向；对 teacher-limited deck 走在线 PG（`--loss pg`，现有 `run_online_rl_round.sh`）或换 deck。
- **停止**：Stage 0 后 <3 个 target 通过确认 → 不进入采集；直接报告“可修集合太小”。

**功效与样本量**：100 局/target/arm 对 0.35→0.55 约 80% 功效（α=0.05 双侧，独立两比例；配对只会更好）；对 0.40→0.50 只有 ~30–40%，所以本实验是**筛选级**，不是认证级。floor 违规数在 20 seeds（n=40/matchup）下噪声很大：真实 0.50 的 matchup 有 21.5% 概率被 point-rate 误判 <45%，不要用“违规数变化 1–2 个”当唯一判据。

**预算**：8,400 采集局（1.7–2.1 h；compact 5.5 GB / 无 privileged 2.5 GB）+ 400 确认局 + 3,040 focused + 2,400 认证 = **14,240 新局**；训练 3–7 h CPU（A800 <1 GPU·h）；评测 3.9 h；总计 **8–13 h**，0 GPU 必需。

**最小 pilot（先跑这个）**：只做 nahida × 1 target。treatment 300 + 400 uniform（20×20）= 700；control 700 uniform（20×35）；共 1,400 采集局；focused eval 100 + 19×20 = 480/arm，共 960（用 1150–1199，避免与 full E1 的 1200–1249 重叠）；训练 2 runs × 5 epoch ≈ 1–1.5 h；总 **~2–3 h**。判据：target Δ ≥ +0.10 且 CI low >0 才扩到 full E1；否则停止。

### E2（第二实验）：公开阵容 FiLM vs 纯 mean-pool（固定数据）

**假设**：在相同语料/epoch 下，公开阵容 FiLM 能把 2 个 deck 中至少 1 个的 floor 违规数降 ≥2，且 pooled 不降；如果两个 deck 都无变化，说明条件化不是当前瓶颈。

**Arm**：base CVPN（现有）vs base + lineup FiLM（zero-init）；2 deck：`nahida_kokomi_raiden`、`superconduct_mika_attack`；同一 store、同一 5 epoch、同一 seed、同一 `--init-from`（`scripts/datafirst_bc_train.py`）。
**数据**：0 新采集局；用两个 deck 的现有 dense BC store。
**评测**：每 deck 20 seeds × 20 对手 × 2 sides = 800；2 deck = **1,600 局**；用 1400–1419（与 E1 不重叠）；target matchup 额外看 per-opponent Δ（n=40，筛选级）。
**主指标**：floor 违规数（point <45%）与 target 的 paired Δ；**不看 val_acc**。
**判定**：≥1/2 deck 的 floor 违规 −≥2 且 pooled CI low ≥ −0.02 → 保留 FiLM，进入 E1 的 2×2 组合（targeted data × FiLM）；否则停止条件化，把预算给数据/PG。E2 是筛选级（20 seeds，n=40/matchup）；若 floor 违规只差 1 个，不据此下结论，需 50 seeds 复核。
**成本**：0 新局；训练 4 runs × 5 epoch × 92k ≈ 3–5 h CPU（A800 <0.1 GPU·h）；评测 1.6 h；总 **4–6 h**。
**护栏**：info-leak 测试必须 PASS；lineup 只从 `player_view` 的 opponent characters 取；不要用 65 桶 `opponent_deck_id`（3 路碰撞）；不要用 deck slug 的 30 张牌表信息。

---

## 5. 与现有负结果的关系（为什么这次不会重演）

| 已有负结果 | 我们的处理 | 为什么不同 |
|---|---|---|
| H5 matchup 重加权（SA −9.1pp） | 不做 loss 重加权；改采集层混合 | H5 用 teacher outcome 定权、抬高老师也输的 matchup；我们用 student-vs-expert 可修 gap，且只加新数据 |
| AWR / Q-DMC / 搜索 teacher / 胜局筛选 | 全部不用 | 这些是“非模仿改进算子”或软目标；E1 仍是硬标签 BC，只改数据构成 |
| 20-deck 单模型（7/8 变差） | 继续 per-deck specialist；不合并 20 deck | 密度是主因（3/4），合议会稀释每 matchup；E1 只在单 deck 内加密度 |
| BC 数据量 8k→48k 不显著 | 不加 uniform 总量；control 与 treatment 等总量 | E1 对比的是**构成**，不是总量；同 epoch、同 base |
| dvalin +7.1pp 不复现 | fresh seeds + 预注册 + 不看选型种子 | Stage 0 与最终评测都用未参与选型的 seeds；不挑最好 checkpoint |
| A2 id embedding val_acc 假阳性 | 只看引擎胜率；FiLM zero-init；不做大 id 表 | A2 是 1,024 桶的长尾卡 id + 随机初始化；FiLM 是 19 类公开阵容 + 恒等初始化 |
| H4 混合语料只在 dvalin 显著 | 把“混合”作为 anchor，不作为主假设 | E1 的 target 是逐 matchup 可修 gap，不是“多 deck 混训” |

**关键区别**：H5 是“在旧数据上改权重”，E1 是“在新数据上改构成”；前者没有新信息，后者直接增加 target matchup 的状态/对手行为覆盖。RHO-LOSS/DoReMi 都支持“按可约损失/最坏 excess loss 选数据”，而不是按原始 loss/teacher 胜负选。

---

## 6. 不要做什么（negative guidance）

1. **不要用 10 seeds（n=20/matchup）的 point-rate 选 target**。真实 0.50 的 matchup 有 25.2% 概率显示 <45%；20 个 matchup 里期望 5 个“假 floor 违规”。先补到 40–60 局，或用收缩后的 `P(p<0.45)`。
2. **不要把 natlan/beidou/chasca/skirk_ayaka 放进 expert BC 靶向采集**。专家自己 0.17–0.29，18–20/20 个对手低于 45%；补数据只能把学生拉到老师附近，floor 仍 FAIL。
3. **不要只采失败 matchup**。AlphaStar 的 `f_hard(1)=0` 是“不再打已赢的”，但它同时保留 35% self-play + 15% forgotten/exploiters（据 Methods 摘录）；Bengio 的“只 easy”与“easy+hard 混合无课程”都差。target-only 会饿死其余 16–19 个 matchup，重演 20-deck 稀释。
4. **不要用 `--matchup-weight deficit` 的原实现**。它的 rate 来自 `store.value`（teacher 胜负），会抬高老师也输的组；要改就改成 student-vs-expert 的 gap，并且优先改采集而不是 loss。
5. **不要把 focal loss 当第一杠杆**。Focal loss 在 detection 的 class imbalance 上有效（FL 36.0 AP vs OHEM 32.8），但 policy CE 没有 background 类，且“hard”决策可能是 ambiguous/noisy；先做数据构成。
6. **不要用 65 桶的 `opponent_deck_id`**。本报告实算 20 个 deck 只有 18 个唯一 id，三套撞桶 57；用 19 类公开阵容或直接 20 路 index。
7. **不要用 deck slug 的完整牌表信息做条件化**。对手 3 角色是公共的；30 张牌表不是。用 `player_view` 的 3 个角色 id（19/20 唯一），不要从 `full_state` 取。
8. **不要对固定脚本上 adversarial opponent model**。对手确定性、阵容公开、池固定；DRON 是给隐藏/非平稳策略的。需要的是显式条件化，不是隐变量建模。
9. **不要先上 PCGrad/GradNorm**。20-deck 崩塌的 3/4 是密度，1/4 是任务数；先补密度/条件化，再做梯度余弦诊断，最后才考虑 gradient surgery。
10. **不要用 val_acc 做 gate**。A2 的 val_acc 0.866 vs 0.838，引擎 dgeo 0.266 vs 0.720；store→engine 不一致是硬教训。
11. **不要在同一批 seeds 上选 target 又报告增益**。optimizer’s curse：选最小/最大噪声估计会系统性偏极端；选型/确认/认证必须用不重叠 seeds。
12. **不要在没有 fresh-seed 复核的情况下把 10-seed 的“可修”当事实**。§1.1 的 10 个候选里，至少要用 50 seeds 复核；复核不过的删掉，不补新 target。

---

## 7. 参考（链接、版本/日期、可信度）

**数据选择 / 课程 / 主动学习**

- Shrivastava, Gupta, Girshick. *Training Region-based Object Detectors with Online Hard Example Mining.* CVPR 2016. [arXiv:1604.03540](https://arxiv.org/abs/1604.03540) · [code](https://github.com/abhi2610/ohem)。可信度：同行评审 + 官方代码。
- Schaul, Quan, Antonoglou, Silver. *Prioritized Experience Replay.* ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)。可信度：同行评审。
- Settles. *Active Learning Literature Survey.* UW–Madison CS TR1648, 2009（后修订）。[PDF](https://burrsettles.com/pub/settles.activelearning.pdf)。本次未抓取全文；按经典综述引用。
- Mindermann et al. *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.* ICML 2022. [arXiv:2206.07137](https://arxiv.org/abs/2206.07137) · [code](https://github.com/OATML/RHO-Loss)。**brief 的 2202.01365 有误**。
- Bengio, Louradour, Collobert, Weston. *Curriculum Learning.* ICML 2009. [DOI](https://doi.org/10.1145/1553374.1553380) · [PDF](https://icml.cc/2009/papers/119.pdf)。
- Swayamdipta et al. *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.* EMNLP 2020. [arXiv:2009.10795](https://arxiv.org/abs/2009.10795) · [code](https://github.com/allenai/cartography)。
- Jiang, Grefenstette, Rocktäschel. *Prioritized Level Replay.* ICML 2021. [arXiv:2010.03934](https://arxiv.org/abs/2010.03934) · [code](https://github.com/facebookresearch/level-replay)。
- Xie et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) · [code](https://github.com/sangmichaelxie/doremi)。
- Arivazhagan et al. *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges.* 2019. [arXiv:1907.05019](https://arxiv.org/abs/1907.05019)。

**对手建模 / 条件化 / 多任务**

- He, Boyd-Graber, Kwok, Daumé III. *Opponent Modeling in Deep Reinforcement Learning.* ICML 2016. [PMLR](https://proceedings.mlr.press/v48/he16.html) · [arXiv:1609.05559](https://arxiv.org/abs/1609.05559) · [code](https://github.com/hhexiy/opponent)。
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871) · [code](https://github.com/ethanjperez/film)。
- Yu et al. *Gradient Surgery for Multi-Task Learning.* NeurIPS 2020. [arXiv:2001.06782](https://arxiv.org/abs/2001.06782) · [code](https://github.com/tianheyu927/PCGrad)。**brief 的 2005.10447 有误**。
- Chen et al. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* ICML 2018. [arXiv:1711.02257](https://arxiv.org/abs/1711.02257) · [code](https://github.com/hosseinshn/GradNorm)。
- Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) · [code](https://github.com/kohpangwei/group_DRO)。
- Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)（用于说明“hard mining 在 class imbalance 上的证据”，不直接迁移到 policy CE）。

**游戏 AI**

- Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575, 350–354 (2019). [Nature](https://www.nature.com/articles/s41586-019-1724-z)。PFSP 公式与 `f_var` 来自 Methods 摘录；35/50/15 混合比例**未逐字核对 PDF**。
- Berner et al. *Dota 2 with Large Scale Deep Reinforcement Learning.* 2019. [arXiv:1912.06680](https://arxiv.org/abs/1912.06680) · [OpenAI blog](https://openai.com/index/how-to-train-your-openai-five/)。
- Li et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning.* 2020. [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)。
- Li et al. *FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning.* ICML 2024. [arXiv:2406.02081](https://arxiv.org/abs/2406.02081) · [code](https://github.com/wenzhe-li/FightLadder)。

**统计**

- Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22(158):209–212, 1927. [DOI](https://doi.org/10.1080/01621459.1927.10502953)。
- Brown, Cai, DasGupta. *Interval Estimation for a Binomial Proportion.* Statistical Science 16(2):101–133, 2001. [DOI](https://doi.org/10.1214/ss/1009213286)。
- Brown. *In-season prediction of batting averages: A field test of empirical Bayes and Bayes methodologies.* Annals of Applied Statistics 2(1):113–152, 2008. [arXiv:0803.3697](https://arxiv.org/abs/0803.3697) · [DOI](https://doi.org/10.1214/07-AOAS138)。
- Smith, Winkler. *The Optimizer’s Curse: Skepticism and Postdecision Surprise in Decision Analysis.* Management Science 52(3):311–322, 2006. [DOI](https://doi.org/10.1287/mnsc.1050.0451)。

**本仓库证据**

- `reports/arena/20261008T163940_full_matrix_20x20.json`：20×20 专家矩阵，50 seeds、双向、40,000 局；本文 expert rate/Wilson 的来源。
- `reports/SPECIALISTS.md`：19 个 deck 的最优认证臂与 `matchups <45%`。
- `reports/explore/plan_strength_2day_kaggle.md`：H4/H5、dvalin 复验、密度 vs 容量、20-deck 稀释。
- `reports/explore/a2_id_screening.md`：A2 val_acc 假阳性与引擎胜率。
- `reports/explore/eval_protocol.md`：Wilson/paired Δ/floor 的样本量表与门禁定义。
- `scripts/train_bc_scale.py:308-349`：H5 的 teacher-outcome 权重实现。
- `reps/opponent_identity.py`、`reps/observation_encoder.py`、`envs/match.py`、`train/model.py`：对手 id / metadata / mean-pool / zero-init 的代码事实。

---

## 附：本报告用到的计算口径

- **student rate**：`reports/SPECIALISTS.md` 每个 deck 的最优认证臂；400 局（10 seeds × 2 sides × 20 对手），per-opponent n=20。选型种子偏差存在，故只用于分层，不作为效应量。
- **expert rate**：20×20 矩阵每个 `(deck, opponent)` 的 100 局（50 seeds × 2 sides），per-opponent n=100；Wilson low 用 z=1.96。
- **teacher-limited**：`expert_wilson_low < 0.45`。
- **reducible**：`expert_wilson_low ≥ 0.45` 且 `student_point < 0.45`；这是**筛选**口径，不是确认口径。
- **floor 违规**：`student_point < 0.45`，per-opponent n=20；仅描述，不作为统计判据。
- **点估计假阳性**：`P(Binom(n,0.5) < 0.45n)`：n=20 为 25.2%，n=40 为 21.5%，n=60 为 18.3%，n=100 为 13.6%（本报告用精确二项分布计算）。
- **Wilson 过关所需真实胜率**（项目 `eval_protocol.md` / `plan_strength_2day_kaggle.md`）：n=20 → 0.668；n=40 → 0.604；n=60 → 0.576；n=100 → 0.548。
- **检测功效**（本报告计算，单侧 α=0.10，精确二项）：n=20 对 0.35 vs 0.45 约 25%；n=40 约 44%；n=60 约 56%；n=100 约 77%。对 0.30 vs 0.45，n=60 约 84%。这就是“20 局只能抓严重失败”的由来。

---

*文件：`reports/explore/lit/f06_targeted_data.md`。只读调研，未改代码、未跑训练/评测、未写 `data/`。*
