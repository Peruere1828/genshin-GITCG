# train/eval 一致性、feature consistency、calibration 与可预测评测口径

> 2026-10-27。只读调研：读代码与 `refs/`、读现有 store/replay（只读）、跑 1 局引擎探针（非训练/非评测），
> 未改代码、未写 `data/`。论文结论与我们的测量分开标注；不确定处标「未核实」。
> 与 `reports/explore/a2_id_screening.md` 配套阅读。

---

## 0. 一句话结论（可执行）

1. **A2 的根因已经在特征层被实锤，不是「疑似」**：compact 语料默认 `float16` 存 `token_features/option_features`，
   而 definition 列（token 42 / option 30）是 `id/400000`。fp16 把这张 id 压成约 `id/2048` 的精度，
   于是训练侧读到的 card id 与引擎侧（float32）读到的**不是同一个 id**。
   单点探针（1 seed × 1 matchup × 48 decisions）：card 实体行 **1267/1277 = 99.2% 失配**，
   char 实体行 **0/348 失配**；失配处 id bucket **100% 不同**。A2 的 id embedding 训练与服务读的是两套索引。
2. **先过 parity 门禁，再谈结构**：parity harness ≈ 0.3 CPU·h 建设、< 5 min 运行；重采一份干净语料
   （4 deck × 20 对手 × 20 seeds = 1,600 局）≈ 20 min 引擎时间。比任何一次「加结构重训」便宜两个数量级。
3. **val_acc 不能做 model selection**：A0/A0i/A0i_noid 三点上 val_acc 排序（0.866 > 0.838 > 0.397）与引擎 dgeo
   排序（0.266 < 0.720 < 0.741）**完全反向**。选择口径换成「配对引擎小样本 + per-deck floor + 冻结 cert 种子」，
   并把 val_acc 降级为 sanity check。

---

## 1. 与我们问题的对应关系

### 1.1 A2 是三层问题叠加，不是一个 bug

| 层 | 机制 | 证据 | 文献对应 |
|---|---|---|---|
| L1 特征一致性 | fp16 破坏 definition 列，id embedding 训练/服务索引不同 | 本报告 §3.1 探针；`train/data.py:350,474`；`reps/observation_encoder.py:54,57-76` | Sculley 2015；Breck 2017；Google Rule #29 |
| L2 状态分布偏移 | store 是 teacher 诱导的状态，eval 是模型自己诱导的状态；BC 误差复合 | A2 store val 0.866 vs 引擎 0.266；A0i_noid val 0.397 但 dgeo 0.741 | Ross 2011（DAgger）；Fujimoto 2019；Levine 2020 |
| L3 选择口径 | val_acc 奖励「像 teacher」，不奖励「赢」；且只保存最后一个 epoch（已过拟合） | `a2_id_screening.md` §1：ep70 best 0.884，ep149 0.866，最终 ckpt 被认证 | Henderson 2018；Agarwal 2021；Zhang 2018 |

L1 修好不等于强度回来（A0i_noid 的 dvalin 0.303 无法用 fp16 解释），但**不修 L1，任何 id/实体特征实验都无法解释**。

### 1.2 五个问题的答案（细节见 §3 对应 recipe）

| 问题 | 结论 | 位置 |
|---|---|---|
| Q1 逐元素 parity harness | 三层（确定性 / 同状态双路编码 / 序列化+跨进程）+ 分列容差；definition 列比 **id 整数**不比浮点；先比 state digest 定位上游 | R1 |
| Q2 分布级监控 | token/option 计数、hand token、opponent token 占比、id 词表外比例、request_type 份额、PSI/SMD、value AUC/Brier；阈值锚在 train/val 零分布 | R2 |
| Q3 checkpoint 选择 | val_acc 与胜率脱钩；改用配对引擎小样本 + per-deck floor + val_ce early stop + 冻结 cert 块 | R3 |
| Q4 评测口径 | 20 seeds/800 局仅能判 ~6pp（π_d=0.35）；5pp 需 ~28–30 seeds；固定 n、不看中间结果、null-pair 校准、split-half 复现 | R4 |
| Q5 最小诊断顺序 | 确定性 → 双路编码 → definition 列 → 分布 → 选择散点 → 认证；合计 ~3.5 CPU·h 出结论 | R5 |

---

## 2. Annotated bibliography

> 只列与「我们能改的那一行代码」有关的点。效应量只写论文里能核到的；核不到的写「未核实」。

### 2.1 分布偏移与离线评估

**Off-Policy Deep Reinforcement Learning without Exploration**（Fujimoto, Meger, Precup, ICML 2019）
- 问题：固定 batch 上，标准 off-policy 算法为什么失效。
- 方法：把失败归因于 **extrapolation error**——(a) 数据里没有的 (s,a) 被函数近似"自信地"外推，(b) bootstrapping 把误差放大；提出 batch-constrained 学习，把动作限制在数据支撑附近。
- 证据：连续控制多个任务上，DQN/DDPG 在 uncorrelated data 上崩，BCQ 能学；论文以曲线与回报表为主，无单一效应量。
- 局限：连续控制、单任务；对离散组合动作（我们的 256 选项）没有直接结论。
- 链接：https://arxiv.org/abs/1812.02900
- 可迁移：store 里 teacher 的状态分布就是"数据支撑"。模型走出这个支撑后，**离线指标没有定义**——这正是 val_acc 失效的形式上的理由。

**Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**（Levine, Kumar, Tucker, Fu, 2020）
- 问题：离线 RL 的挑战清单。
- 方法：把分布偏移拆成 distributional shift、extrapolation error、bootstrapping 误差、trajectory-level 的 exposure bias。
- 证据：综述，无新实验；给出「离线评估必须报告不确定性」的立场。
- 局限：面向连续控制/机器人，游戏场景的例子少。
- 链接：https://arxiv.org/abs/2005.01643
- 可迁移：把「离线指标涨、在线跌」当作**默认预期**而不是异常；任何离线指标都必须先证明与引擎胜率同向（我们的 R3 散点实验）。

**Improving predictive inference under covariate shift by weighting the log-likelihood function**（Shimodaira, J. Stat. Plan. Inference 90(2), 2000）
- 问题：`p(y|x)` 不变、`p(x)` 变了（covariate shift）时，MLE 有偏。
- 方法：重要性加权 `w(x)=p_test(x)/p_train(x)` 修正 log-likelihood；权重估计误差会引入方差。
- 证据：理论 + 模拟；后续成为 covariate shift 的标准处理。
- 局限：要求 support overlap；我们这里 `p_train` 是 teacher 状态、`p_test` 是模型状态，**overlap 在后期对局会很小**，重要性加权只能做诊断不能做训练。
- 链接：https://www.sciencedirect.com/science/article/abs/pii/S0378375800001154（DOI 10.1016/S0378-3758(00)00115-4）
- 可迁移：估计 `w(x)` 的**有效样本量**（ESS）就能量化「eval 有多少状态是 store 没覆盖的」，比看 loss 直观。

**Dataset Shift in Machine Learning**（Quionero-Candela, Sugiyama, Schwaighofer, Lawrence 编, MIT Press 2009）
- 问题：把 dataset shift 分类：covariate shift、prior shift、concept shift。
- 方法：统一框架 + 各章方法（importance weighting、模型选择下的 shift）。
- 证据：编著，无单一实验；是术语与分类的权威来源。
- 局限：2010 年前的深度模型，无现代大模型案例。
- 链接：https://direct.mit.edu/books/edited-volume/3841/Dataset-Shift-in-Machine-Learning
- 可迁移：我们的 A2 同时有 covariate shift（状态）与 concept shift（teacher 的「好动作」在模型状态下不再最优）；两者要分开测。

**A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning（DAgger）**（Ross, Gordon, Bagnell, AISTATS 2011）
- 问题：监督式模仿在 learner 自己的状态分布上误差复合。
- 方法：每轮用当前策略跑、让专家标注访问到的状态，聚合成数据集；用 no-regret 在线学习给出保证。
- 证据：论文指出纯监督模仿的代价可随 horizon `T` 二次放大，DAgger 在 0-1 损失下做到近线性（引 Ross & Bagnell 2010 的结论）。
- 局限：需要交互式专家；我们的专家是规则引擎，标注几乎免费，**这一点上我们比论文场景更有利**。
- 链接：https://arxiv.org/abs/1011.0686
- 可迁移：这是 val_acc 失效的第一性解释；也是 R3「on-policy teacher agreement」的直接依据。

**Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning**（Voloshin, Le, Jiang, Yue, NeurIPS 2019）
- 问题：离线策略评估（OPE）估计器在什么条件下可信。
- 方法：统一 benchmark（COBS），跨 domain、behavior policy、trajectory 长度系统比较 IPS/WIS/DR/FQE/model-based。
- 证据：实践上 **weighted importance sampling 比无偏 IPS 更准、更省数据**；value-based 类里 FQE 是基准；估计器排名随环境因素变化。
- 局限：离散动作、低维状态；没有"策略强度"这种终局指标。
- 链接：https://arxiv.org/abs/1911.06854
- 可迁移：如果要做「离线胜率预估」，别用朴素 IPS（方差爆炸）；但更稳的路线是**不做 OPE，直接小样本引擎配对**（我们的成本允许）。

**Offline A/B testing for Recommender Systems**（Gilotte et al., WSDM 2018）
- 问题：离线估计能不能替代在线 A/B。
- 方法：比较 IPS/DR 等，提出 NCIS（normalized, clipped）估计器，并与线上业务指标对照。
- 证据：文中给出反例——NCIS 估 1.8、真实 2.1、baseline 1.9，**离线结论会把更好的策略判成更差**（§ 例子）。
- 局限：推荐系统、有随机曝光；与零和游戏不同构。
- 链接：https://arxiv.org/abs/1801.07030
- 可迁移：这是「离线指标涨、在线跌」最干净的公开案例；诊断方法是**离线估计器之间的分歧**（DM vs IPS 差得远 = 模型外推区）。

### 2.2 校准

**On Calibration of Modern Neural Networks**（Guo, Pleiss, Sun, Weinberger, ICML 2017）
- 问题：现代网络过度自信。
- 方法：ECE（M=15 等宽 bin）+ reliability diagram；temperature scaling（单参数、在 val 上最小化 NLL）。
  `ECE = Σ_m (|B_m|/n) · |acc(B_m) − conf(B_m)|`。
- 证据：ECE 常在 4–10%；temperature scaling **不改变 argmax/accuracy**，只改概率。
- 局限：分类；ECE 对 bin 敏感、对少数类不友好。
- 链接：https://arxiv.org/abs/1706.04599
- 可迁移：**对贪心策略做 temperature scaling 不会改变强度**——它只影响采样/探索。别把它当 strength lever；它只对 value head / search 有意义。

**Accurate Uncertainties for Deep Learning Using Calibrated Regression**（Kuleshov, Fenner, Ermon, NeurIPS 2018）
- 问题：回归/概率预测的区间不可信。
- 方法：在 held-out 上拟合 isotonic regression，把预测 CDF 映射到经验 CDF，再做单调重排；给出高概率校准界（具体常数未核实）。
- 证据：跨多个回归数据集把 calibration error 显著压低；对 Bayesian NN 也有效。
- 局限：需要足够的 held-out 数据；高维输出要逐维处理。
- 链接：https://arxiv.org/abs/1807.00263
- 可迁移：value head 若输出分布，可用 isotonic 做后处理；但我们更该先看 **AUC（区分度）**——D23 已经显示根价值区分度≈0，校准救不了没有区分度的头。

**Improving Regression Performance with Distributional Losses**（Imani & White, ICML 2018）
- 问题：MSE 回归在噪声目标上不稳。
- 方法：Histogram Loss（HL）——把标量目标变成 support 上的分布，最小化 KL；HL-Gauss 用截断高斯做软目标。
- 证据：多组回归/RL 任务上优于 MSE，且对超参更稳。
- 局限：要选 bin 数与方差；高动态范围目标需要 symlog。
- 链接：https://arxiv.org/abs/1806.04613；https://proceedings.mlr.press/v80/imani18a.html
- 可迁移：value head 从 MSE 换成 HL-Gauss 是**低风险**改动（`train/model.py` value head + loss），但要先解决 L1/L2，否则又是一次「离线指标好看」。

**Stop Regressing: Training Value Functions via Classification for Scalable Deep RL**（Farebrother et al., 2024）
- 问题：大网络下 MSE value 不 scale。
- 方法：把 value 目标转成 HL-Gauss 分类，交叉熵训练；配 symlog。
- 证据：MoE 单任务 Atari 上「30% better」，多任务 1.8–2.1×（论文原文数字）；ResNet/Transformer 一致。
- 局限：作者来自 DeepMind，算力规模远超我们；小网络收益可能小。
- 链接：https://arxiv.org/abs/2403.03950
- 可迁移：如果我们以后重开 search（D23 门禁），value 头的分类化是前置条件之一，不是强度捷径。

**Verification of forecasts expressed in terms of probability**（Brier, Monthly Weather Review 78:1–3, 1950）
- 问题：概率预报怎么打分。
- 方法：Brier score `(1/n)Σ(p_i−y_i)²`；严格 proper scoring rule。
- 证据：原始定义；后世标准。
- 局限：对 base rate 敏感；需要分解才知道是「偏」还是「不分」。
- 链接：https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml
- 可迁移：我们的 `scripts/eval_value_calibration.py` 已经算了 Brier/ECE/AUC；加上 Murphy 分解即可区分 reliability vs resolution。

**A New Vector Partition of the Probability Score**（Murphy, J. Appl. Meteor. 12:595–600, 1973）
- 问题：Brier 分数由什么构成。
- 方法：分解为 reliability（校准）+ resolution（区分）+ uncertainty（base rate）。
- 证据：经典分解；`scores` 等库有实现。
- 局限：bin 依赖。
- 链接：https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2
- 可迁移：D23「根价值区分度≈0」= resolution≈0；这种情况下再校准（reliability）也不会让搜索变好。分解能一句话说清。

**Obtaining Well Calibrated Probabilities Using Bayesian Binning**（Naeini, Cooper, Hauskrecht, AAAI 2015）
- 问题：ECE 的 bin 选择敏感。
- 方法：Bayesian binning into quantiles（BBQ），自适应分箱 + 贝叶斯先验。
- 证据：在多个数据集上比 equal-width/equal-count 稳。
- 局限：仍是标量校准指标。
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/9602
- 可迁移：15 等宽 bin 在 n<5k 时噪声大；我们的 `eval_value_calibration` 用等频 15 bin，够用，但报告 ECE 时要带 bootstrap CI。

### 2.3 评测统计

**Deep Reinforcement Learning that Matters**（Henderson, Islam, Bachman, Pineau, Precup, Erhan, Meger, AAAI 2018）
- 问题：RL 结果不可复现。
- 方法：系统改变 seed、超参、环境、codebase，看结论怎么翻。
- 证据：**同一超参、只换 seed** 就能让算法 A/B 的胜负翻转（Fig. 5 及正文）；建议报告显著性。
- 局限：连续控制、2017 年前的算法。
- 链接：https://arxiv.org/abs/1709.06560
- 可迁移：我们「换种子就显著」不是运气差，是默认现象；固定种子块 + 配对是唯一低成本解法。

**A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms**（Colas, Sigaud, Oudeyer, 2019）
- 问题：小样本、非正态、异方差下该用哪个检验。
- 方法：比较 t-test/Welch/Wilcoxon/bootstrap/permutation 的假阳性率与功效。
- 证据：分布非正态时假阳性率高于名义 α；**拿不准时用 α=0.01**；seed 数由相对效应量决定（power 0.8）。
- 局限：单任务两算法；我们是多对手分层。
- 链接：https://arxiv.org/abs/1904.06979
- 可迁移：分层（对手）比较优先 Welch t / bootstrap；Wilcoxon 检的是中位数，与我们的均值型指标不一致，且在小样本下并不自动更稳（论文自己就推翻了「非参更安全」的直觉）。

**Deep Reinforcement Learning at the Edge of the Statistical Precipice**（Agarwal, Schwarzer, Castro, Courville, Bellemare, NeurIPS 2021）
- 问题：3–10 runs 的少量种子下怎么比较。
- 方法：stratified bootstrap CI、IQM、performance profile、optimality gap、probability of improvement。
- 证据：逐任务 mean/median 在重尾分布下误导；IQM 更稳且更有效。
- 局限：面向多任务 benchmark；我们的"task"是对手，只有 20 个 cluster。
- 链接：https://arxiv.org/abs/2108.13264；代码 https://github.com/google-research/rliable
- 可迁移：`eval/scoreboard.py` 已有 opponent-cluster t-CI + bootstrap；再加 **IQM across opponents** 与 performance profile，能把「20 个对手里被 2 个拖死」显式化。

**Accounting for Variance in Machine Learning Benchmarks**（Bouthillier et al., MLSys 2021）
- 问题：benchmark 的方差来源（数据采样、初始化、超参）被低估。
- 方法：把整个 benchmarking 过程建模；分析常见比较方法。
- 证据：反直觉结论——在估计器不完美时，**增加变化来源反而更接近理想估计器**，计算成本降 51×。
- 局限：深度学习 benchmark，不是 RL 对局。
- 链接：https://arxiv.org/abs/2103.03098
- 可迁移：不要用「单次 run 的最优 checkpoint」当结论；要么多种子，要么明确写成 descriptive。

**Always Valid Inference: Bringing Sequential Analysis to A/B Testing**（Johari, Pekelis, Walsh, 2015/2017）
- 问题：连续监控 p 值会让假阳性爆炸。
- 方法：always-valid p 值/置信序列，任意停时有效；代价是样本量/功效的折衷。
- 证据：理论 + 工业 A/B 实验。
- 局限：需要预先定义度量与停时结构。
- 链接：https://arxiv.org/abs/1512.04922
- 可迁移：长跑认证如果必须中途看，用 always-valid CI 或 alpha spending，不要每次都用 1.96。

**A multiple testing procedure for clinical trials**（O'Brien & Fleming, Biometrics 35:549–556, 1979）
- 问题：多次 interim analysis 的 α 控制。
- 方法：group-sequential 边界（早期极严、后期接近 1.96）。
- 证据：临床试验标准；后续 Lan-DeMets α spending 推广。
- 局限：需要预设 look 次数。
- 链接：https://pubmed.ncbi.nlm.nih.gov/497341/
- 可迁移：认证跑 4 次 interim（25/50/75/100%）就够；具体边界查表，别手算（本报告不抄数字，避免记错）。

**The preregistration revolution**（Nosek, Ebersole, DeHaven, Mellor, PNAS 2018）
- 问题：事后选指标/选样本量导致不可复现。
- 方法：预注册假设、样本量、分析计划、停时。
- 证据：心理学大规模实践；ML 里是类比迁移。
- 局限：ML 的探索性强，预注册要留「探索性」标签。
- 链接：https://www.pnas.org/doi/10.1073/pnas.1708274114
- 可迁移：每个强度声明前，在 PLAN §0 写 6 行：假设/指标/种子/样本量/阈值/停时；没写的算探索性，不得进主线。

### 2.4 工程上的 train-serving skew / feature consistency

**Hidden Technical Debt in Machine Learning Systems**（Sculley et al., NeurIPS 2015）
- 问题：ML 系统的隐性债务。
- 方法：分类：glue code、pipeline jungles、dead experimental codepaths、configuration debt、undeclared consumers、**CACE（Changing Anything Changes Everything）**、data dependencies。
- 证据：工业案例综述。
- 局限：无定量。
- 链接：https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- 可迁移：我们的 `collate` 同时被训练和 `NeuralPolicy` 调用，是**唯一正确**的做法（one code path）；而 `collect_samples` 与 `NeuralPolicy.choose` 各自传 `history=()/include_training_targets=...`，属于 configuration debt，必须用断言锁死。

**The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction**（Breck, Cai, Nielsen, Salib, Sculley, IEEE Big Data 2017）
- 问题：测试到什么程度算够。
- 方法：28 项测试，4 类（data / model / ML infrastructure / monitoring）；分数取最弱项。
- 证据：Google 内部多个团队。
- 局限：面向在线服务；RL 对局需要改写。
- 链接：https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- 可迁移：把 parity 测试放进 CI 的 "data tests"，把分布监控放进 "monitoring tests"；分数取 min 正好对应我们的「任一 deck 崩即 FAIL」。

**Rules of Machine Learning**（Zinkevich, Google，持续更新）
- 问题：工程实践清单。
- 方法：**Rule #29**："The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time." **Rule #32**：训练与服务尽量复用同一份代码。Rule #29 还交叉引用 Rule #37「验证 serving 与 training 的一致性」。
- 证据：经验规则，非实验。
- 局限：面向推荐/分类。
- 链接：https://developers.google.com/machine-learning/guides/rules-of-ml
- 可迁移：我们的 store 应该保存**服务时实际算出的特征**（而不是另跑一遍 encoder）；现在 `train.collect` 正是这么做的——问题只出在落盘 dtype。

**Feast: point-in-time joins**（官方文档）
- 问题：训练集 join 到「未来」的特征 → 泄漏。
- 方法：point-in-time correct join，按事件时间取当时可见的特征值；离线/在线同一份定义。
- 证据：工程文档，无实验。
- 局限：面向特征库，不是 RL。
- 链接：https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- 可迁移：我们的「point-in-time」就是 `Game.is_resumable()` 边界快照（`envs/snapshot.py`）；重建 context 必须用**当时**的 state，不能用终局 state 反推。

**Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO**（Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry, ICLR 2020）
- 问题：实现细节 vs 算法本身。
- 方法：逐个打开/关闭 code-level optimizations。
- 证据：PPO 相对 TRPO 的收益**主要来自实现细节**；细节还改变算法的定性行为。
- 局限：策略梯度。
- 链接：https://arxiv.org/abs/2005.12729
- 可迁移：dtype、mask、option 顺序这些"细节"就是我们的 algorithm；不要指望"结构创新"能盖过它们。

### 2.5 过拟合与泛化

**A Study on Overfitting in Deep Reinforcement Learning**（Zhang, Vinyals, Munos, Bengio, 2018）
- 问题：RL agent 会不会过拟合。
- 方法：在训练环境/种子上过拟合，测 held-out 表现；加随机化。
- 证据：agent 会以多种方式过拟合；随机化环境能缓解（摘要级结论）。
- 局限：Atari/连续控制；未给出通用阈值。
- 链接：https://arxiv.org/abs/1804.06893
- 可迁移：A2 的 ep70→ep149 是同一现象的小规模版本；「训练更久 = 更强」不成立。

**Assessing Generalization in Deep Reinforcement Learning**（Packer et al., 2018）
- 问题：没有统一的 train/test 环境划分。
- 方法：按「训练关卡/测试关卡」拆分，系统评估泛化。
- 证据：标准 agent 在未见关卡上显著退化。
- 局限：网格世界/导航。
- 链接：https://arxiv.org/abs/1810.12282
- 可迁移：我们的「未见」= 新种子 + 模型自己诱导的状态，比新关卡更苛刻；cert 种子必须与采集种子不相交。

**Shortcut Learning in Deep Neural Networks**（Geirhos et al., Nature Machine Intelligence 2020）
- 问题：模型学「捷径」而非目标特征。
- 方法：跨视觉/NLP 的案例归纳；定义 shortcut 的判据（分布外失效、反事实测试）。
- 证据：大量案例；无统一度量。
- 局限：感知任务。
- 链接：https://arxiv.org/abs/2004.07780
- 可迁移：id embedding 是最典型的 shortcut——它能完美区分训练集 deck，但在引擎上换个 id 映射就失效。A2 的 dgeo 0.266 是 shortcut 的教科书案例。

---

## 3. 最小可复现 recipe

性价比排序：**R1**（0.3 h 排除一整类假设）> **R5**（3.5 h 出「能不能信 val_acc」的结论）> **R3**（改选择口径，不动训练）>
**R2**（长期监控，防回归）> **R4**（协议已成型，只补功效与停时）。每个 recipe 给：输入/输出、关键参数、伪代码、改动文件、成本、判据。

### R1. store ↔ engine 逐元素 parity harness（Q1）

**目标**：对同一个逻辑决策点，证明「训练语料里的特征」与「服务时算出的特征」是同一个向量；
若不是，定位到第一个 (row, column/option) 并给出原因。

**连接键**：`(deck, opponent, seed, step_index)`。`Store.keys` 正是这个四元组（`scripts/train_bc_scale.py:79`），
replay 的 task record 也带 `deck/opponent/seed`（`train/collect.py`）。`step_index` 是**每个座位各自**的决策序号
（`PolicyPlayer._step_index`），不是全局步数。

**三层，按代价从低到高**：

| 层 | 做法 | 判据 | 代价 |
|---|---|---|---|
| P0 确定性 | 同 `(deck,opp,seed)` 跑两遍（同进程 + 两个子进程），比动作序列与 `sha256(player_view)` | 完全一致 | 2 局 ×2 ≈ 30 s |
| P1 同状态双路编码 | 对捕获的 `built.context` 分别用 `include_training_targets=True/False` 编码 | token/option 特征**逐位相同**（该 flag 只影响 privileged/belief，不进模型） | < 10 s |
| P2 序列化 + 跨进程 | 从 replay 读回样本 → 与**新进程**里按 checkpoint 的 encoder config 重新编码的结果比 | 分列容差（下表） | 20 局 ≈ 2 min |
| P3 服务路径 | 用 `neural:<ckpt>` 真跑，捕获它自己的 `built.context`，与「把该 state 的 JSON 丢给新进程重编码」比 | 同 P2 | 20 局 ≈ 3 min |

P3 是关键：它证明「给定同一个 state，服务侧算出的特征」与「离线重算」一致，从而把
「特征函数」和「状态分布」两个问题分开。

**分列容差**（不要用一个全局 `atol`）：

| 列 | 容差 | 理由 |
|---|---|---|
| one-hot / 指示位（0/1） | 精确 | 任何 1e-3 偏差都是逻辑错 |
| **definition 列（token 42 / option 30）** | **先 `round(x*400000)` 转 int，再精确比** | 浮点 1e-4 在这里 = 40 个 id |
| 其余标量（[-1,1]） | P1: 精确；P2 fp32: 1e-6；P2 fp16: 1e-3 | fp16 在 1.0 附近的分辨率 4.9e-4 |
| mask | 不做等价性判据 | `token_mask` 恒 True；`option_mask` 由 `encode_legal_actions` 置全 True（`reps/action_hierarchy.py:177`），`collate` 又只按长度重建（`train/model.py:449`）——它不携带信息 |
| 行数 / 选项数 | 精确 | 见下 |

补一句 semantics：option 第 30 列是 `subject_definition_id`（打出的牌/用的技能/调谐的对象），
不是「这个选项的全部身份」。实测只有 **30.8%** 的 option 行非零，其余（declare_end、switch_active 等）是 0；
所以拿它做 option id embedding 会有一半以上落在 padding。要做 A4 的 option 表示，得自己组合
`(kind, subject, target_slots, used_dice)`，而不是复用这一列。

**必须断言的硬不变量**（任一失败 = parity fail，不是 warning）：
1. `len(option_features) == len(legal_low_level_codes) == len(legal_low_level_specs)`；
2. `all(code >= 0 for code in low_to_high_codes)`（`-1` = 高阶层映射漏了）；
3. `all(isfinite(x))`，且除 definition 列外 `|x| <= 1 + 1e-6`；
4. definition id `< 400000`（当前最大 333029，余量 17%；新卡池可能越界 → 越界会被 `_safe_ratio` 夹到 1.0，所有大 id 塌成一个）；
5. **语料 dtype 下的 id 可逆性**：`round(round_fp16(id/S)*S) == id`。对当前 20 套专家池的 152 张卡，
   fp16 下 **150 张不满足**（实测）；fp32 下 152/152 满足。

**第一处分歧怎么定位**：
1. 先比 `sha256(json.dumps(asdict(player_view), sort_keys=True))`。**digest 不同 = 上游状态不同**（seed/洗牌/策略），
   此时比特征没有意义；先修确定性。
2. digest 相同 → 按列块定位：token 的块边界是
   `[0:10] type / [10:13] owner / [13:19] phase / [19:25] request / [25:34] kind / [34:41] tag / [41:65] scalars(24)`；
   option 的块是 `[0:9] kind / [9:17] used dice / [17:25] auto dice / [25:35] 10 scalars / ...`。
   报「第一个不同的 (row, block, col) + 该 block 的失配计数」，不要报 65 维的最大值（会被定义列淹没）。
3. 选项按 **semantic key** 对齐（`reps.action_hierarchy.semantic_action_key_for_spec`），不要按 index；
   当前架构 option scorer 是逐项独立的，permutation 等价，但 index 决定 tie-break 与 `codes[index]` 映射。

**顺序无关性**：CVPN 是 mask-normalized mean-pool（`train/model.py:132-140`），对 token 顺序不变；
所以 token 应同时比「原始序列」与「按内容排序后的多重集」——前者抓流水线漂移，后者是模型行为等价类。
option 同理：先比多重集 `(semantic_key, vector)`，再单独比 `codes` 的 index 顺序。
**如果以后换成带位置编码的 Transformer，顺序就从"提示"变成"硬约束"**，这条要写进注释。

**落地的文件改动**（最小集）：
- 新增 `envs/parity.py`：`ParityRecorder`（包住 policy，记录 `built.context` + `player_view` digest + 编码结果）；
  `compare_samples(a, b)` 返回第一个分歧与分块统计。
- `envs/match.py`：`run_match(..., parity_recorder=None)`，在 `_decide` 里 `recorder.record(built, code)`（不动现有逻辑）。
- 新增 `tests/test_feature_parity.py`：2 decks × 2 seeds × 全决策，P1 精确 + id 可逆 + 不变量；目标 < 60 s。
- 新增 `scripts/parity_audit.py --store <pt> --seeds 20`：P2/P3，输出 `reports/parity/<tag>.json`（只聚合，不入库原始特征）。
- （可选）`train/data.py::_pack_dense` 支持 per-field dtype，让 definition 列走 fp32。

**伪代码**：
```
for deck, opp, seed in tasks(4 decks, 1 opp, 5 seeds):
    store_rec  = load_replay(deck, opp, seed)          # 原始 payload，不经过 Store
    items      = replay_match(deck, opp, seed)         # 捕获 built.context，48–90 条
    digest     = sha256(player_view_json)              # P0
    assert digest == store_rec.digest
    for k, built in items:
        a = enc.encode(built, include_training_targets=True)    # 采集路径
        b = enc.encode(built, include_training_targets=False)   # 服务路径  (P1)
        assert a.token == b.token and a.option == b.option
        s = store_rec.samples[k]                                # fp16 读回 (P2)
        assert len(a.option) == len(s.option) == len(built.codes)
        for row: assert recover_id(a,42) == recover_id(s,42)
        for opt: compare_by_semantic_key(a.option, s.option)
```

**成本**：建设 0.3 CPU·h；CI 测试 < 1 min；20 局审计 ~5 min。**没有一个训练 run 比它便宜。**
**失败判据**：任一 id 失配、任一列超容差、任一不变量失败、任一 digest 不一致（后者先归因于确定性）。

### R2. 分布级监控（Q2）

只做逐元素不够：逐元素能证明「函数一致」，不能证明「输入分布一致」。第二类指标按 **训练集 train/val 的差** 当零分布，
而不是拍脑袋定阈值——先量出噪声底，再定报警线。

实测基线（`data/store/double_geo_navia.pt`，20k train / 2.8k val，dgeo vs 20 对手池）：

| 指标 | 基线 | 报警 | 硬失败 |
|---|---|---|---|
| token 数 T（无 history） | mean 28.8–29.3 / p50 30 / p90 37 / p99 42 / max 47 | 均值漂移 > 10%，或 p99 > 50 | T = 0；或 T > 73（说明 history 被打开；> 105 会被 `max_tokens` 静默截断） |
| option 数 K | mean 23–29（切片敏感）/ p50 13–14 / p90 96 / p99 192 / max 256 | 均值 ±20%，或 p99 > 288（1.5×） | K = 0，或 K ≠ len(codes) |
| P(K=1) | 9.2% | > 15% | — |
| hand_card token/决策 | 7.05，零值 0.1% | < 5.5 或零值 > 2% | 零值 > 5%（= 手牌被掩码，典型 view bug） |
| opponent token 占比 | 0.292 | 落在 [0.24, 0.35] 外 | 0（owner 映射坏了） |
| definition id 词表外比例 | 干净语料应 ≈0；**当前 fp16 语料实测 63.8%**（见 §3.1） | > 0.5%（且非新卡池） | > 5%（若确认是新卡池，重设基线而不算 bug） |
| request_type 份额 | 需 replay 层统计（Store 未存） | 任一类型 ±5pp（n ≥ 2,000 决策） | 出现未知类型 |
| fallback / illegal / `low_to_high=-1` | 0 | 任何 > 0 | 任何 > 0 |
| 重复 semantic key | 0（dedupe 后） | > 0.1% | > 1% |
| value 预测均值 vs 引擎实测胜率 | 同种子块上应一致 | \|Δ\| > 0.05 | — |
| value AUC | —（D23 报 ≈0.5） | < 0.6 | < 0.55（无区分度，搜索无意义） |
| 每列 PSI（10 等频 bin） | 由 **train vs val** 估计噪声底 | 超过基线 + 0.1 | > 0.25 |

说明：
- PSI 的 0.1/0.25 是**信用评分行业惯例**（见 Minitab 文档），不是定理；用它做「看板报警」可以，做「验收门禁」不够。
  `PSI = Σ_b (p_b − q_b) · ln(p_b / q_b)`，p=当前、q=基线；门禁用上面的硬不变量 + id 可逆性。
- `T > 50`：在**当前无 history** 的语料里实测 max=47，**从未发生**。A2 报告里「T>50 的 token」这个怀疑点可以结案；
  真正的风险在 `max_tokens=105`（含 32 个 history token）——一旦有人给 serving 打开 history 而采集没开，T 会跳变且被静默截断。
  建议把 `history_len` 写进 encoder config 并在加载时断言。
- `Store` 现在**不保存 request_type**（`scripts/train_bc_scale.py:65-79`），request_type 分布只能从 replay JSONL 统计。
  要么在 Store 加一列 `request_type: int`，要么把监控放在采集端（`train/collect.py` 的 `sample_stats` 已有 request_type 集合，但没有直方图）。
- `token_mask`/`option_mask` 全 True 是设计事实；如果以后有人用 `option_mask` 过滤非法动作，会得到一个**永远不过滤**的静默 bug。
  建议在 `collate` 里加一句注释或直接删掉这两个字段。

**成本**：监控代码 0.5 CPU·h；单次统计 20k 样本 ≈ 1 min（store 载入占大头）；每次认证顺带跑 ≈ 2 min。
**失败判据**：任一硬失败列；或模型输入列 PSI > 0.25 且不是新卡池/新对手。

### R3. checkpoint 选择（Q3）

**为什么 val_acc 会与引擎胜率脱钩**（四个机制，按我们证据强弱排序）：
1. **状态分布变了**（covariate shift + 误差复合）。val 是 teacher 走出来的状态；eval 是模型自己走出来的状态。
   Ross et al. 2011 的结论：监督式模仿的误差可随 horizon 二次放大；我们的对局 ~46–90 个决策，足够放大。
2. **目标函数不匹配**。accuracy 给每个决策同等权重；胜率只关心少数关键决策（斩杀、留骰、切人）。
   99% 的「declare end」猜对，抵不过 1% 的斩杀失误。
3. **teacher 标签不是 ground truth**。expert-vs-expert 语料里，标签是某个规则引擎的选择；拟合得更像它 ≠ 更强。
4. **shortcut**。A2 实测：val_acc 排序 0.866(A0i) > 0.838(A0) > 0.397(A0i_noid)，
   引擎 dgeo 排序 0.266 < 0.720 < 0.741，**完全反向**；再叠加 fp16 让 id 分支在 store 里"自洽"、在引擎上"换索引"。
   3 个点不足以做统计，但足以否定「val_acc 高 ⇒ 强」这个假设。

**选择栈（从便宜到贵，只把最贵的一步留给最终认证）**：

| 层 | 指标 | 样本量 | 用途 |
|---|---|---|---|
| L0 | `val_ce`（不是 acc） | 全 val | early stop / `--save-best`；proper scoring rule，对 id shortcut 没那么敏感 |
| L1 | per-deck val_acc + ACTION-only acc + top-2 acc + 平均 logit margin | 全 val | 发现「某个 deck 崩了」；pooled 数字会掩盖 |
| L2 | **on-policy teacher agreement**（模型自己走，每个状态问规则专家） | 100–200 局 | 去掉状态分布那一半偏移；比 off-policy val_acc 更接近真实 |
| L3 | **配对引擎小样本**（vs 固定 baseline，同种子双向） | 200 局/deck | 唯一因果对齐的指标；可检出 ≥10–13pp |
| L4 | 认证（冻结 cert 种子） | 800 局/deck | 只能跑一次，不能用来选 |

L2 是本报告最推荐的「新离线指标」：不需要训练、不需要新数据，只需要在 `agents/neural.py` 外面包一层
`CapturePolicy`，每条决策顺带调 `expert_policy(deck).choose(built)` 记一个标签。成本 100 局 ≈ 4 min。
它仍然是「像不像专家」，但至少在**模型自己的状态分布**上测，避开了 L1 的最大偏差。

**early stop 不泄漏评测集**：
- 三个不相交种子块：**train**（采集，0–79）、**val**（epoch 选择，`val_every=25` 已经在做，`scripts/datafirst_bc_store.py:58`）、
  **cert**（900–919，冻结，一个模型族只开一次）。
- 规则：`val_ce` 连续 15–20 epoch 不降就停；保存 best 而非 last（A2 的最终 ckpt 是 ep149，best 是 ep70）。
- 预算先验固定（如 100 epoch），val 只用来确认，不用来「练到 val 峰值」——后者在足够多 epoch 下本身就会过拟合 val。
- 如果要扫 k 个变体：用 successive halving 在 val 上砍，只有 top-1 进 cert；并在预注册里写明 k。
  否则「k 个噪声估计取 max」天然向上偏（Bouthillier 2021 / Agarwal 2021 的 selection bias）。

**伪代码**：
```
best_ce, best_ep = inf, -1
for ep in 0..E:
    train_one_epoch()
    if ep % 5 == 0:
        ce = val_ce(ckpt(ep))            # L0，不是 val_acc
        save_if_best(ckpt(ep))
        if ep - best_ep >= 15: break     # early stop
shortlist = [ckpt(best_ep), ckpt(last_ep)]           # 只允许 2 个
for c in shortlist:
    delta[c] = engine_paired(c, baseline, seeds=100..119)   # L3，800 局
winner = argmax(delta)                                # 不能用 cert 种子
cert(winner, seeds=900..919)                          # L4，一次性
```

**成本**：L2 100 局 ≈ 4 min；L3 200 局/deck ≈ 8 min；L4 800 局/deck ≈ 32 min。**没有一项需要重新训练。**
**失败判据**：若 L2 与 L3 的符号不一致（模型更「像专家」但更弱），说明 teacher 本身在模型状态下不是好策略 →
不要再优化 agreement，转 DAgger 式重标注（用模型状态、专家标签）。

### R4. 评测口径（Q4）

现有 `scripts/eval_protocol.py` 已经做对的事：双向、配对种子、Wilson + opponent-cluster CI + bootstrap、
per-opponent floor、`strict-ci`、非劣性容差 0.02、merge gate、流式落盘 + 指纹护栏。**不要重写，只补三件事。**

**功效**。仓库的 `detection_sample_size`（`eval/scoreboard.py:640`）是**非配对**公式
`n/arm = (z_{1−α/2}+z_β)² · 2p(1−p) / δ²`，p=0.5 时 = `3.92/δ²`，δ=5pp → 1,568/arm。这是保守上界。
配对设计的正确公式是 McNemar 型：
`n_pairs ≈ (z_{1−α/2}+z_β)² · π_d / δ²`，`π_d` = 两模型结论不同的比例，`δ = p01 − p10`。

| δ（绝对胜率差） | π_d=0.15 | π_d=0.25 | π_d=0.35 |
|---|---:|---:|---:|
| 3pp | 1,308 | 2,180 | 3,052 |
| 5pp | 471 | 785 | 1,099 |
| 8pp | 184 | 307 | 429 |
| 10pp | 118 | 196 | 275 |

换算成我们的协议（每个 seed = 20 对手 × 2 边 = 40 局/deck）：
- **20 seeds（800 局）**：π_d=0.25 时可检出 ≈5pp；π_d=0.35 时 ≈5.9pp；π_d=0.5 时 ≈7pp。
- **10 seeds（400 局）**：只能检出 ≈7–10pp，**只能当筛选，不能当结论**。
- 要稳定判 5pp，按 π_d∈[0.25,0.35] 需要 **20–28 seeds**；判 3pp 需要 33–76 seeds（不现实）。
- 以上都是 pooled 配对检验；**cluster gate 更严**（对手间异质 + 19 df 的 t），实际可检出的 δ 还要再大 10–20%。

**结论**：预注册时先写「我要判多大的 δ」。判不了就写 descriptive，不写「显著优于」。
A2 的 45pp 是例外（巨大效应），所以 10 seeds 也能判；但 A0 vs A0i_noid 的 2pp 级别差异，400 局什么都说明不了。

**避免「换种子就显著」的四条**：
1. **固定种子块 + 预注册**：screening 0–9、cert 900–919。cert 只用一次；用过就换 920–939 并声明。
2. **null-pair 校准**：用同一个 checkpoint 跑 A vs A（两个进程/两个 tag），在 20 个新种子上看 Δ 的分布。
   它应在 0 附近，SD ≈ 理论 SE。如果偏了，说明协议有系统性偏差（种子/座位/引擎锁），先修协议。
   成本：2 × 400 局 ≈ 32 min，一次性。
3. **split-half 复现**：把 20 seeds 分成两半，要求 Δ 的**符号在两半一致**；不一致就不算赢。
4. **不看中间结果**；必须看就用 always-valid CI（Johari）或 4 次 interim 的 O'Brien–Fleming 边界。

**多重比较**：20 个对手各报 95% CI，期望 ~1 个假阳性。floor 违规当**描述性护栏**（point estimate），
不要逐个做检验后宣称「显著变差」；真要检验就 Holm/BH。

**报告格式**（建议进 `reports/`）：
pooled 配对 Δ + 95% CI、opponent-cluster Δ CI、**IQM across opponents**、最差 3 个对手、floor 违规数、
fallback/error/truncation、以及「本次能检出的最小 δ」——最后一项是诚实性声明。

**成本**：改 `eval/scoreboard.py` 加 IQM/split-half/最小可检 δ 输出 ≈ 0.5 CPU·h；null-pair 2 × 400 局 ≈ 32 min（一次性）。
**失败判据**：null-pair 的 Δ 均值偏离 0 超过 1 个 SE，或 split-half 符号不一致 → 协议只能写 descriptive，不能写「显著优于」。

### R5. 最小诊断顺序（Q5）

| 步 | 检查 | 判据 | 成本（CPU·h） | 失败后 |
|---|---|---|---|---|
| 1 | 确定性：同 seed 两进程动作序列 + state digest | 完全一致 | 0.05 | 修 seed/洗牌/subprocess；其余全部暂停 |
| 2 | P1 双路编码（train flag vs serve flag） | 逐位相同 | 0.02 | 改 encoder 分支 |
| 3 | definition 列 id 可逆 + 语料 dtype | 100% 可逆；词表外 ≈0 | 0.1 | 换 fp32 / 加守卫；**当前已失败** |
| 4 | 模型自诱导状态的分布对比（T/K/hand/opp share/value） | 见 R2 阈值 | 0.3 | 定位 view/masking/option 枚举 |
| 5 | val_acc vs 引擎胜率散点（6–8 个 checkpoint） | 若 Spearman ≤ 0 → val_acc 不可用于选择 | 1.1（8×200 局） | 改用 L2/L3 选择 |
| 6 | 认证（3 deck × 800 局） | 预注册阈值 | 1.6 | — |
| 合计 | 到「能不能信 val_acc」的结论 | | **≈3.5** | |

第 3 步在本仓库**已经实测失败**（见 §3.1 探针），所以当前不应进入第 5 步之后的任何结构实验。

---

### 3.1 一手测量：A2 语料的 definition 列已经坏了（只读探针）

方法：取 `data/replays/dfbc_double_geo_navia_c80` 中 `(deck=double_geo_navia, opponent=aywolfskirk, seed=1)` 的一场；
用 `run_match` 按同样 seed/策略重放，捕获全部 48 个决策；每个决策用 `default_encoder()` 重新编码，
与 replay 里 fp16 存下的向量逐元素比较。**这是诊断探针，不是评测**（1 局）。

| 项 | 结果 |
|---|---|
| 轨迹可复现性 | 48/48 决策；T、K 全等；`low_level_action_codes` 顺序全等；teacher 动作全等 |
| 浮点差（非 id 列） | token 最大 2.3e-4，option 最大 1.7e-4（= fp16 量化） |
| token definition 列 | 1,100 个实体行中 **804 (73.1%)** 与引擎侧不同 |
| option definition 列 | 525 个实体行中 **463 (88.2%)** 不同（38/48 个决策受影响） |
| 按 id 规模 | char 行（<10k）**0/348**；card 行（≥10k）**1,267/1,277 (99.2%)**，37/38 个不同 card id 被破坏 |
| id bucket（1024） | 所有失配处 **bucket 100% 不同**（训练读的 embedding 行 ≠ 服务读的行） |
| 语料级信号 | store 前 3,000 样本中 63.8% 的实体 id 不在 20 套专家词表内（含少量合法的非卡 id，故只作旁证） |

机制：`DEFINITION_SCALE=400000`，fp16 的相对精度 2^-11 ⇒ **id 误差 ≈ id/2048**。
card id 在 111k–333k，误差 54–163；char id 1.1k–2.6k，误差 ≤1.3（实测 0）。
所以「character token 没事、card token 全坏」不是巧合。

影响面：
- **A0 不受影响**（没有 id 分支，2e-4 的输入扰动可忽略）；
- **A0i 受影响**（`id_buckets=1024`，训练/服务两套索引）；
- **A0i_noid 不受 fp16-id 影响**，所以它的 dvalin 0.303 另有原因（L2/L3），fp16 只是 A2 的一部分。

修复（三选一，推荐前两个）：
1. `train/collect.py --compact-dtype float32`（推荐；文件 ~1.2 MB/局，仍远小于 JSONL）；
2. `_pack_dense` 支持 per-field dtype：token/option 用 fp32，belief/privileged 继续 fp16；
3. 把 definition 拆成 int32 旁路（代码多，收益小）。

**注意**：已经写坏的语料**无法事后修复**（信息已丢）。要 id 特征就得重采：
4 deck × 20 对手 × 20 seeds = 1,600 局 ≈ 20 min（12 核，4,900 局/h），存储 ~2 GB。

---

## 4. 预注册实验

### E-PARITY-1（门禁；不是强度声明）

- **假设**：A0i 的 dgeo/ungeo 崩塌中包含一个可消除的 feature-skew 成分；修掉 definition 列后，
  store 与引擎的输入在**模型读到的每一列**上逐位一致。
- **对照**：同一批 `(deck, opponent, seed)` 的 store 样本 vs 引擎重放（P2/P3）；外加 A0（无 id）作阴性对照。
- **指标**：definition 列 id 失配数、其它列最大绝对差、T/K/codes 不一致数、state digest 不一致数。
- **种子与功效**：4 decks × 5 seeds = 20 局（store 侧）+ 20 局（P3 服务侧）；不做统计检验（这是确定性检查，n 不适用）。
- **预算**：0.3 CPU·h 建设 + 0.5 CPU·h 跑完（40 局 × 2 遍 + 2 次 store 载入）+ 重采 1,600 局（20 min 墙钟，12 核）；无 GPU。
- **判定**：`id 失配 = 0` 且 `其它列 max|Δ| = 0`（P1）/ `≤1e-6`（fp32 P2）→ PASS；
  任一 `state digest` 不一致 → 先修确定性再重跑。
- **停止**：PASS 后本实验永久进 CI（`tests/test_feature_parity.py`）；FAIL 则**冻结所有新结构训练**
  （A3/A4/A5、search、belief head），直到修复并通过。
- **反例记录**：若修复后 A0i 仍然 dgeo < 0.5，则 L1 不是主因，把预算转到 L2/L3（E-SELECT-1），
  并在 PLAN §0 记一条「fp16-id 已排除」。

### E-SELECT-1（val_acc 能不能用来选 checkpoint）

- **假设**：在 A0i 的训练轨迹上，`val_ce` 最优的 checkpoint（≈ep70）比最终 checkpoint（ep150）在引擎上强 **≥6pp**（dgeo），
  而两者的 val_acc 差异（0.884 vs 0.866）无法预测这个差距。
- **对照**：主对照 **ep70 vs ep150**；另 6 个等间隔 checkpoint（ep10…150）只用于画 val_acc–胜率散点（R5 第 5 步），不参与判定。
- **指标**：dgeo 配对 Δ（ep70 − ep150）、95% 配对 CI；附带 ungeo 作独立复现；per-deck floor。
- **种子与功效**：选择块 100–119（与采集 0–79 不相交），20 seeds × 20 对手 × 2 边 = 800 局/ckpt；
  π_d=0.35 时 80% 功效 ≈5.9pp，正好卡在 6pp 假设上。若只看 2 个 ckpt，1,600 局 ≈1.1 h。
- **预算**：重采 1,600 局（20 min，fp32）→ 重训 150 epoch ≈75 min（A2 实测，cuda）→ 评测 1,600 局 ≈1.1 h（1,500 局/h）→ 分析 <10 min。
- **判定**：Δ ≥ 6pp 且 95% CI 下界 > 0 → **「last-epoch 认证」作废**，强制 `--save-best` + `val_ce` early stop；
  Δ < 6pp → 「过拟合尾巴」不是主因，删掉这条假设，把预算给 L2（on-policy agreement）。
- **停止**：800 局/ckpt 后一次性判定；**不看中间结果**。若 400 局时配对 CI 已经完全落在 ±6pp 内（futility），提前停并记「无判定力」。

**门禁条款（写进 PLAN §0）**：E-PARITY-1 未 PASS 之前，不启动任何新结构训练（含 A3 决策记忆、A4 option attention、
search 重开、belief head 扩维）；不把 compact fp16 用于任何带 id/实体词表特征的语料。

---

## 5. 与现有负结果的关系（为什么这次不会重演）

| 已有负结果 | 本报告的解释 | 这次怎么不同 |
|---|---|---|
| 20-deck 单模型负迁移（8 个里 7 个变差） | deck 条件先验被 id/对手特征 shortcut 化；pooled val 掩盖单 deck 崩 | 先做 per-deck 分布监控与 floor，再谈多 deck |
| A2 val 0.866 / 引擎 0.266 | L1（fp16 id）+ L2（状态偏移）+ L3（选 last epoch） | L1 已定位且可修；L3 换选择口径 |
| 数据量 8k→48k 不显著 | 瓶颈是 epoch/优化预算，不是数据量；val_acc 又看不出强度 | 选择指标改成引擎配对 Δ，不再用 val_acc 做结论 |
| AWR/Q/DMC 负结果 | 离线 RL 在无支撑区外推（Fujimoto）；我们的数据是 teacher 轨迹，覆盖不到模型自己的状态 | 先做 on-policy agreement/DAgger-lite，再谈离线 RL |
| search-as-teacher 显著为负 | value 头 resolution≈0（D23），搜索用了没有区分度的价值 | R2 的 AUC 门禁；E-PARITY 通过前不重开 |

---

## 6. 不要做什么（negative guidance）

1. **不要用 pooled val_acc 做 model selection**，也不要用「最后一个 epoch」。A2 的引擎差 45pp，而它的 val_acc 反而更高。
2. **不要把 fp16 用在含 definition/id 列的稠密特征上**；也不要相信「反正是浮点，差 1e-4 无所谓」——那 1e-4 是 40 个 id。
3. **不要对贪心策略做 temperature scaling 然后宣称「校准提升强度」**。它不改 argmax（Guo 2017 明确说 accuracy 不变）。
4. **不要在 400 局上宣布 3–5pp 的胜利**。那不是效应，是噪声。
5. **不要用 cert 种子调参**，也不要在同一批种子上比较 6 个模型再取 max。
6. **不要在 state digest 不一致时比特征**——先修确定性。
7. **不要把 `option_mask` 当非法动作过滤器**：它恒为 True，`collate` 也不看它。
8. **不要用「加一个 Transformer / 加一个注意力头」替代 parity 修复**。Engstrom 2020 的结论反过来也成立：
   实现细节能吞掉结构收益。
9. **不要在没有 on-policy 评估的情况下重开 search**：value 头 AUC < 0.6 时，搜索只是在放大噪声。

---

## 7. 参考（链接、版本/日期、可信度）

**我们的一手材料**
- `reports/explore/a2_id_screening.md`（2026-10-27，本项目，高可信——原始数据在本机）
- `data/replays/dfbc_double_geo_navia_c80/samples.jsonl`（compact fp16，实测）
- `data/store/double_geo_navia.pt`（42,856 train / 2,801 val，实测统计）
- 代码：`reps/observation_encoder.py:54,57-76`；`reps/action_hierarchy.py:140-184,482-497`；`train/data.py:350,474-475`；
  `train/model.py:132-140,432-452`；`agents/neural.py:38-59`；`scripts/train_bc_scale.py:65-79`；
  `scripts/datafirst_bc_store.py:58-61`；`eval/scoreboard.py:251-375,640-645`；`scripts/eval_value_calibration.py`

**论文/文档**（链接于 2026-10-27 核对；部分出版社页面对自动抓取返回 403，仅作引用）
- Fujimoto et al. 2019, https://arxiv.org/abs/1812.02900（ICML 2019，高）
- Levine et al. 2020, https://arxiv.org/abs/2005.01643（arXiv v3，高）
- Shimodaira 2000, https://www.sciencedirect.com/science/article/abs/pii/S0378375800001154（JSPI 90(2)，高；正文未取，仅摘要/书目）
- Quionero-Candela et al. 2009, https://direct.mit.edu/books/edited-volume/3841/Dataset-Shift-in-Machine-Learning（MIT Press，高）
- Ross et al. 2011, https://arxiv.org/abs/1011.0686（AISTATS，高）
- Voloshin et al. 2019, https://arxiv.org/abs/1911.06854（NeurIPS 2021 版，高）
- Gilotte et al. 2018, https://arxiv.org/abs/1801.07030（WSDM 2018，高）
- Guo et al. 2017, https://arxiv.org/abs/1706.04599（ICML 2017，高；15 bins / 4–10% / accuracy 不变均已核对原文）
- Kuleshov et al. 2018, https://arxiv.org/abs/1807.00263（NeurIPS 2018，中高；保证的具体常数未核实）
- Imani & White 2018, https://arxiv.org/abs/1806.04613（ICML 2018，高）
- Farebrother et al. 2024, https://arxiv.org/abs/2403.03950（arXiv，中高；30% / 1.8–2.1× 为原文数字）
- Brier 1950, https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml（高）
- Murphy 1973, https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2（高）
- Naeini et al. 2015, https://ojs.aaai.org/index.php/AAAI/article/view/9602（AAAI，高）
- Henderson et al. 2018, https://arxiv.org/abs/1709.06560（AAAI 2018，高；seed 翻转结论已核对原文）
- Colas et al. 2019, https://arxiv.org/abs/1904.06979（arXiv，高；α=0.01 建议、power 0.8 已核对）
- Agarwal et al. 2021, https://arxiv.org/abs/2108.13264（NeurIPS 2021，高；3–10 runs、IQM、stratified bootstrap 已核对）
- rliable, https://github.com/google-research/rliable（实现，高）
- Bouthillier et al. 2021, https://arxiv.org/abs/2103.03098（MLSys 2021，高；51× 为摘要原文）
- Johari et al. 2015, https://arxiv.org/abs/1512.04922（arXiv，高）
- O'Brien & Fleming 1979, https://pubmed.ncbi.nlm.nih.gov/497341/（Biometrics，高；边界数值未抄）
- Nosek et al. 2018, https://www.pnas.org/doi/10.1073/pnas.1708274114（PNAS，高）
- Sculley et al. 2015, https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html（NeurIPS，高）
- Breck et al. 2017, https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/（IEEE Big Data，高；28 项/4 类已核对）
- Zinkevich, Rules of ML, https://developers.google.com/machine-learning/guides/rules-of-ml（文档，高；Rule #29/#32 原文已核对镜像）
- Feast docs, https://docs.feast.dev/getting-started/concepts/point-in-time-joins（文档，中；页面为 JS 渲染，仅取首页描述）
- Engstrom et al. 2020, https://arxiv.org/abs/2005.12729（ICLR，高）
- Zhang et al. 2018, https://arxiv.org/abs/1804.06893（arXiv，中高；仅摘要级）
- Packer et al. 2018, https://arxiv.org/abs/1810.12282（arXiv，中高）
- Geirhos et al. 2020, https://arxiv.org/abs/2004.07780（Nature MI，高）
- PSI 阈值 0.1/0.25：Minitab 文档 https://support.minitab.com/en-us/model-ops/monitor-deployed-models/details-about-the-population-stability-index-psi/（**行业惯例，非论文结论**）

**未找到 / 未核实**
- 公开的「棋牌/TCG 领域 sim-vs-dataset parity harness」论文：未找到（未核实，不排除闭源工业实现）。
  本报告的 parity 设计是从生产 ML 的 train-serving skew 实践（Rule #29/#32、Feast、ML Test Score）迁移的。
- O'Brien–Fleming 的具体边界数值：未抄录，避免记错；用前查表。
- Kuleshov 校准保证的精确常数：未核实。
