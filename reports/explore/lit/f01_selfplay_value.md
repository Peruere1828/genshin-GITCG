# 卡牌/不完全信息自博弈与价值校准（DouZero / Suphx / ReBeL / SoG）

> 2026-10-27。调研方式：读一手论文（arXiv HTML 全文，优先算法伪代码、附录超参表、实验数字）+ 读官方/参考实现
> （`kwai/DouZero` 的 `dmc.py`/`models.py`/`utils.py`、`facebookresearch/ReBeL` 的 `cfvpy/selfplay.py`）+
> 读本仓代码（`reps/belief.py`、`reps/belief_head.py`、`train/cfr_solver.py`、`envs/cfr_search.py`、
> `envs/cfr_probe.py`、`train/distill.py`、`train/model.py`、`train/data.py`、
> `refs/Rebel_base_RL/.../gitcg_world_model/*`）。**未改代码、未跑训练/评测、未写 `data/`。**
> 标「未核实」的是没有拿到一手材料直接确认的；标「本仓推断」的是我读代码得出的判断，可被一个便宜实验证伪。

---

## 0. 一句话结论（可执行）

**我们的问题不是「价值头没标定」，而是三种价值被当成了一种用**：`V^π(s)`（outcome 回归，150ep 头在 `sa_all` val 上 AUC 0.845 / ECE 0.045）、
`V^σ(β)`（搜索根价值）、`Q(s,a)`（候选动作价值）在代码里共用同一个标量头、同一个 `frontier_p0` 字段、
同一条 MSE 目标。ReBeL/SoG 的做法不是把 MSE 换成 HL-Gauss，而是**把不同语义的价值拆成不同的头/不同的 target**，
并且**只在 on-policy 的 PBS 上训练搜索价值**。
落到本仓，顺序应该是：

1. **先修两个消费端语义 bug（0 训练，1–2 h）**：identity 激活的价值头被当成概率并 `max(0,·)` 截断
   （`envs/cfr_probe.py:91-92`），以及"acting-player 视角"被当成"player-0 视角"
   （`train/resolver.py:95-105` vs `cfr_probe.py:146` / `envs/cfr_search.py:207-214`）。
   这两个 bug 足以单独制造"根价值区分度≈0"和"搜索覆盖专家却更常输"的测量结果。
2. **depth-1 + 冻结 expert 基座下，frontier 的正确 estimand 近似就是 `V^expert`，也就是我们已有的 outcome 头**——
   偏差只来自后续最多 `max_searches` 次搜索覆盖；不需要 search value，也不需要分布化。
   搜索价值 `V^σ` 只在 **depth≥2** 或 **基座是网络** 时才成为必需。
3. **搜索的动作选择只依赖同一状态内兄弟叶子的价值差**，不依赖绝对标定。所以 ECE/Brier 不是搜索的门禁；
   门禁应该是"同状态内 action-value spread"和最终引擎胜率。HL-Gauss 是低风险 drop-in，但论文自己说
   确定性动力学下与 MSE 相当，别指望它单独救搜索。
4. **自博弈采样**：评测分布是 20 套固定脚本，纯 self-play 优化的是另一个目标。可执行分布 =
   **池 60% + 当前策略自博弈 20% + 专家 anchor 10% + 历史 checkpoint 10%**；deck 未达 parity（Δ<−2pp）时
   把自博弈/历史降到 0，先做 100% 池 + anchor。
5. 预注册实验只做两个：**E1 修语义 + 门控搜索**（3,200 局、0 GPU·h）、**E2 采样分布 A/B**（6,000 采集局 + 1,200–1,600 评测局、GPU <0.1 h）。
   两个都用**引擎胜率**判定；E1 失败就正式关闭搜索分支。

### 四个老师问题的直接回答（详版见对应小节）

| # | 问题 | 一句话回答 |
|---|---|---|
| 1 | 怎么做 value calibration？最小改动？ | 先拆语义（outcome 头 / search 头 / oracle 头）；搜索头用**搜索值或 N 次 rollout 的均值**做 target + Huber，分布化（HL-Gauss/101 bins）只用在**连续值的 search 头**上。最小改动是修 `cfr_probe` 的符号/截断（0 训练）；加 search value 头只在 depth≥2 或网络基座时才必需。 |
| 2 | public belief 当策略输入还是辅助头？ | **两者都不是"全输入"**：SoG 参考实现是"主策略只看公共态 + 对手身份 embedding；belief summary 只进**独立的 search 头**"；ReBeL 的 PBS 是状态本身（定义使然）。本仓应保持 belief 为辅助头 + 只喂 search，且先补 **dice 分布**（现在 `_sample_hidden_dice` 是确定性轮转，不是采样）。 |
| 3 | 固定脚本池下 self-play 采样谁？ | 以**当前策略 vs 池**为主（≥60%），自博弈 20% 做探索/正则，专家 anchor 10% 防漂移，历史 checkpoint 10% 防循环；池内按 PFSP 式 deficit 权重采样。纯 self-play 不作为主数据源。 |
| 4 | 为什么 DouZero 式 DMC 此前是负结果？ | 最可能是**off-support argmax**（只监督被走动作）+ **目标尺度被深度污染**（γ^k 缩放 terminal-only ±1）+ 规模差 3–4 个数量级；不是"DMC 这个算法在卡牌里不行"。判别实验见 §1.4。 |

---

## 1. 与我们问题的对应关系

### 1.1 三种价值：我们混用了哪三个量

| 量 | 定义 | 我们现在的载体 | 正确用途 | 文献里的对应 |
|---|---|---|---|---|
| `V^π(s)` | 在行为策略 π（我们=脚本专家）下从 s 出发的期望胜负 | `train/model.py` 的 `value_head`；target 在 `train/data.py:176,192` 写为 `outcome_score(winner,0)`；MSE | PG/GAE 的 baseline；**depth-1 + expert 基座的 frontier** | AlphaZero 的 value；Suphx 的 value；SoG 的 `partial_value` |
| `V^σ(β)` | 在搜索策略 σ 下、公共信念 β 处的反事实价值 | **没有独立头**；`train/distill.py:117-126` 在 `search_value` 存在时把**同一个头**的 target 硬切成它（其余样本仍是 outcome），MSE | depth≥2 / 网络基座时的 frontier；search-as-teacher 的 value target | ReBeL 的 infostate value；SoG 的 CVPN value / `search_state_value` |
| `Q(s,a)` | 候选动作价值 | 没有独立头；DMC 实验把 policy logits 当 Q（`scripts/train_bc_scale.py:489-492`），只监督被走动作 | 候选打分/裁剪；DMC | DouZero 的 Q(s,a)（每个候选单独编码） |

**关键点 1（本仓推断，30 分钟可判定）**：`V^π` 与 `V^σ` 在"depth-1 + 冻结 expert 基座"下**近似是同一个量**。
因为根动作之后，搜索并不改变后续策略（`EngineParticleGame.transition` 只回放前缀，叶子之后仍由 expert base 继续），
所以叶子的正确价值近似是 `V^expert(s')`。偏差只来自后续最多 `max_searches` 次搜索覆盖（A4 是 8 次/局）。
我们现在用 outcome 头做 frontier，**estimand 近似是对的**。
`search_value` 在 depth≥2（内部节点由搜索策略接管）或基座换成网络时才显著不同。

**文献的硬数字**：SoG 的 Table S2 里，两个不完全信息游戏（HUNL、Scotland Yard）的 `p_td1` 都是 **0**——
value 头的 target **100% 来自 GT-CFR 的 counterfactual value，0% 来自 game outcome**；
同一张表里 Chess 是 0、Go 是 0.2。也就是说，在"不完全信息 + 有搜索"的设定里，SoG 压根不拿 outcome 训 value 头。
这不是"把 outcome 标定得更好"，而是**换 target**。但注意它的前提：搜索本身是 sound 的、且 query 由当前策略产生。
我们现在的搜索不满足这个前提，所以不能直接照搬 `p_td1=0`（见 §5）。

**关键点 2**：所以 D23 的"根价值区分度≈0 → 价值不校准 → MCCFR 的 regret 是噪声"这条因果链，
**可能建立在一个坏掉的测量上**（见 §1.3 D1/D2），而不是价值学习本身。H3b 已经反证了"标定本身不够"：
把 50ep、AUC 0.841、ECE 0.061 的价值头接到 d1 搜索，配对 Δ 与无价值前沿**逐值相同**（−0.0417 vs −0.042）——
如果价值头真的在起作用，换前沿至少应该改变选招；逐值相同更像是**所有叶子价值都被截断成同一个值**（见 D1）。

### 1.2 本仓 vs SoG/ReBeL 的组件对照（已实现 / 缺失）

| SoG/ReBeL 组件 | 本仓实现 | 状态 |
|---|---|---|
| 公共信念根（PBS determinization） | `reps/belief.build_public_belief_state` / `sample_belief_roots` | ✅ 有；但 dice 是假采样、粒子权重是启发式、学到的 belief 没接进来 |
| 只用公共信息的 infoset key | `train/cfr_solver.public_infoset_key` | ✅ |
| regret solver（external-sampling MCCFR） | `ExternalSamplingMCCFR.solve` → root strategy + root value + action values | ✅ |
| depth-limited frontier value | `envs/cfr_search.EngineParticleGame.frontier_value` | ⚠️ 符号/截断 bug；没有独立 search 头；没有 particle 间 uncertainty |
| 候选裁剪 | `envs/cfr_search.py:169-185`，无 prior 时按引擎选项序 | ⚠️ A4 自己已判定"任意序"是缺陷 |
| teacher 导出（policy/value） | `CFRSearchDecision.teacher_policy/teacher_value` | ✅ |
| 蒸馏损失 | `train/distill.py`：soft CE + value MSE + belief loss | ✅ 但 search value 无权重、无 mixture、无 Huber/HL-Gauss |
| belief 头 | `train/belief_head.py`：group marginals / card posterior / hand size / burst | ✅ 但**只当标签**，search 不用；且**没有 dice 头** |
| 对手身份条件化 | `reps/opponent_identity.py` + `reps/observation_encoder.py:207` 的 `opponent_tag` | ⚠️ 通道存在，但 `train/data.py:183` 从不传 → 永远是 `"unknown"` |
| oracle（特权信息）价值 | `reps/observation_encoder.py:381` 已产出 `privileged_state` 标签 | ❌ CVPN 没有 oracle 头/蒸馏（SoG 参考实现有） |
| 两个/三个 value 头 | 只有一个 `value_head` | ❌ |
| particle 间 `mean − λ·std` | 无 | ❌（SoG 参考实现有 `uncertainty_penalty`） |
| 门控覆盖 | `best_action()` 无条件覆盖专家 | ❌ |
| 历史/league 对手 | collector 只支持固定对手/单 ckpt | ❌ |

**消融证据的诚实状态**：ReBeL / SoG **都没有做**"belief 作为策略输入 vs 辅助头"的消融——PBS 是它们算法定义的一部分，
不是可选特征。能引用的只有两条间接证据：(i) ReBeL Fig. 2：用随机 PBS 训练的价值网学不到东西（**训练分布**重要，
不是"输入 vs 头"）；(ii) SoG 参考实现把 belief summary 只喂 `search_*` 头、主策略只看公共态
（`policy_value_model.py:289-320`，工程选择，不是论文消融）。所以本条只能靠本仓自己的消融回答（见 §3 R4），
不能假装文献已经给出答案。

### 1.3 代码级诊断（本仓，按"最可能解释根价值≈0"排序）

#### D1（事实，最优先）：identity 价值头被当成概率并截断到 0

- `train/model.py:44` 默认 `value_activation="identity"`；150ep 的 `bc_e150` 也是 identity
  （`reports/train/value_calibration_kg150.json` 的第一分箱 `mean_p=-0.0616`，说明原始输出会 <0）。
- `envs/cfr_probe.py:91-92`：`value = resolver.value(built); return min(1.0, max(0.0, value))`。
  **所有负的预测被压成 0**，于是 `frontier_p0=0` → `centred=-1`。
- 后果：所有"我方劣势"的叶子价值恒等于 −1，**兄弟叶子之间的排序被抹平**；如果一批候选的叶子都落在负区，
  action values 全相等 → regret matching 全 0 → 回退到 prior/uniform → `best_action()` 按 tie-break 取字典序**最大**的 key
  （`RootSolution.best_action` 的 `max(..., key=(prob, key))`）。
  这能同时解释 A4 的"netH 根价值 −0.718"（大量 −1）和 H3b 的"换价值前沿逐值相同"（全 1 个值，选招不变）。
- 判据（10 分钟）：对 150ep 头跑 200 个 val 状态，统计原始值 <0 的比例；若 >20%，D1 成立。
- 修复：identity → `p=(v+1)/2`；sigmoid → `p=v`；**不要截断**（截断放到最后一步做，并保留负值用于比较）。

#### D2（事实）：`frontier_p0` 在不同实现里有两种语义

- `train/resolver.py:84-105` 的 docstring 写的是"traverser side"，实现是 `encode_context(built.context)`，
  而 `context.player_view` 永远是**当前行动者**的掩码视图（`reps/observation_encoder.py:213-218`）。
  所以学到的价值是 **P(当前行动者赢)**。
- `envs/cfr_probe.py:47-75` 的启发式用的是 `state.players[0]`（座位 0），返回的是 **P(player 0 赢)**。
- `envs/cfr_probe.py:146` 把两者都写进 `frontier_p0`；`envs/cfr_search.py:207-214` 按 player-0 解释并翻转。
- 后果：在"行动者=玩家 1"的叶子上，价值符号**反了**。Genshin TCG 一个回合里同一玩家会连续做多个决策
  （出牌/选骰/选目标），所以叶子常常仍是玩家 0 → 大部分时候正确；但**恰好在回合交接处（对手开始行动）翻转**，
  而那里正是最需要判断"我这步交给对手后是优是劣"的地方。
- 判据：日志里按 `acting_player` 分组，比较"赢局/输局的平均 frontier 值"；若 acting=1 的组符号与直觉相反，D2 成立。
- 修复：`probe` 里按 `who` 转成 p0（`p0 = p_acting if who==0 else 1-p_acting`），字段改名 `frontier_p0` 保持；
  单测：fake resolver 返回 0.8 @ who=1 → p0=0.2。

#### D3（事实）：对手 dice 的"采样"是确定性轮转

- `reps/belief.py:97-100`：`return tuple(_DICE_TYPES[(sample_index + offset) % 8] for offset in range(count))`。
  这不是从分布采样，而是每个粒子给一组**互不相同**的骰子；真实骰池是独立随机掷骰，**重复元素很常见**
  （具体边缘分布需从 `full_state` 日志估计；本报告未核实）。
- dice 是本游戏**最影响对手合法动作**的隐藏量（能不能打出某张反应牌/支付费用）。粒子在这一维上系统性错误，
  会让搜索的"对手能做什么"整体失真，比手牌直方图的误差更致命。
- 修复（20 行）：按引擎掷骰分布 i.i.d. 采样（至少 uniform over 8 types，按 count 采样多重集），
  权重加 `P(multiset)`；分布可从 `full_state` 日志直接估计（collector 已有 god view）。

#### D4（事实）：学到的 belief 没有接进 search

- `reps/belief.build_public_belief_state` 已经接受 `belief_histogram` / `belief_remaining_deck_histogram` /
  `belief_group_hand_probs` / `belief_group_deck_probs`（`reps/belief.py:337-349`）。
- 但 `train/cfr_solver.belief_root_particles`（`:489-541`）不传这些参数；`envs/cfr_search._root_particles`（`:399-412`）也不传。
- 结果：belief 头训练了，但搜索根用的是"decklist − revealed + 均匀/组先验"的启发式粒子。
  这是一个**已存在接口、只差接线**的低成本改进。

#### D5（事实）：对手身份通道存在但从未使用

- `reps/observation_encoder.py:207-210` 支持 `opponent_tag` / `opponent_deck_name` / `opponent_entity_key`；
  `:533-534` 把 `opponent_tag` 编成 one-hot 进 global token。
- `train/data.py:183` 的调用是 `encode_context(context, history=(), include_training_targets=...)`
  → 全部取默认 `"unknown"`。`agents/neural.py` 的在线编码也没有传。
- 对本项目这是**公共信息**（对手的出战角色/阵容是公开的；完整 30 张牌组在我们的固定 deck 评测里也是已知的），
  且"同一公共局面、不同对手"的价值确实不同——
  这是比"手牌 belief"更值得进策略的隐藏变量。SoG 参考实现用 `opponent_tag_id`/`opponent_entity_id`/`opponent_deck_id`
  三个 embedding 做这件事（`policy_value_model.py:242-256`）。

#### D6（事实）：Q 的两个 confound 从未被消融

- `scripts/train_bc_scale.py:489-492` 只监督**被走动作**的 Q；未走动作无约束 → 离线 argmax 是 off-support。
- 所有 Q 臂都跑 `--q-target discount`（`kaggle/grids/run_offline_rl_grid.sh:37`、`kaggle/train_arm.py:110`），
  `q_scale = γ^(剩余步数)`（`train_bc_scale.py:377-390`）。对 terminal-only ±1，同一个游戏里**越早的决策 target 越接近 0**
  （无论最终胜负），只有最后一两步接近 ±1。同一状态内所有动作共享这个因子，所以 argmax 不变；
  但跨状态共享的 Q 头必须把"这个状态离终局还有多远"和"这个状态值多少"混进一个标量，
  早期决策的信噪比被系统性压低。DouZero 用 γ=1 正是为了避免这个（`dmc.py:target_buf` 对所有 step 都写终局回报）。
- 两个 confound 混在一起，不能从"Q 臂失败"推出"DMC 不适合本游戏"。

#### D7（事实，A4 已自陈）：候选裁剪是任意序

- `envs/cfr_search.py:176-184`：`candidate_limit` 在根节点按 `(-prior, index)` 取前 C 个；无 prior 时 prior≡0
  → 取引擎选项序前 C 个。A4 已把这个列为重开搜索的前置缺陷（`reports/search/a4_search_budget_ladder.md:190`）。
- 在"任意 4 个候选"上做 max，等于对 4 个随机叶子取最大值 → **乐观偏差**，且与动作质量无关。

### 1.4 为什么 DouZero 式 DMC 此前是负结果：可检验归因

先对齐事实：DouZero 的 Q 目标是**整局回报**（γ=1，terminal-only，±1/ADP），每个候选动作单独编码，
LSTM 看最近 5 个状态，ε=0.01 在线自博弈，45 actors × 4×1080Ti × 30 天；论文明确报告 TD/SARSA 无增益、
actor-critic 失败（`arXiv:2106.06135` §5.5）。DanZero 在 4 人 GuanDan 上用同样配方跑 **160 CPUs × 30 天**。

我们的 Q 实验：**离线**专家语料（8k–30k 局）、**只监督被走动作**、**γ^k 折扣**、**单头共享 option scorer**、
**无 history（`history=()`）**、训练 15–150 epoch。

| # | 可能原因 | 支持证据 | 判别实验（便宜） | 若成立的含义 |
|---|---|---|---|---|
| C1 | off-support argmax | Q 的 val MSE 最好（150ep 0.0609）但策略最差（15ep pooled 0.068；150ep 0.275）；DouZero 靠 ε 在线覆盖 | 推理时把 argmax 限制在专家 prior top-3 / 或在被走动作集合内 | 不是"Q 不行"，是"离线 Q 的 argmax 不行"；需 on-policy 覆盖 |
| C2 | γ^k 缩放 terminal-only ±1 | 所有 Q 臂都是 `discount`；同一状态内所有动作共享同一个 γ^k（argmax 不变），但**不同深度**的样本被缩放到不同尺度，一个跨状态共享的 Q 头必须把"深度"和"价值"混在一个标量里；DouZero γ=1 | 同数据同种子，γ=1 vs 0.97 两臂（1 次训练，~0.3 GPU·h） | 目标尺度被深度污染，Q 不可跨状态比较；早期决策的信噪比被压低 |
| C3 | 规模差 3–4 个数量级 | DouZero 30 天×4 GPU；DanZero 30 天×160 CPU；我们 1–2 GPU·h | 单 deck、20k 局 on-policy self-play（~5 h 采集）再训 Q | 数据量是硬门槛；DMC 不在本机预算内 |
| C4 | 表示/历史不足 | DouZero 4×15 动作矩阵 + 5 帧 LSTM；我们单帧 pooled state | 同数据加 history（`reps` 已支持 `history=`）训一版 | 需要序列表示，不是换损失 |
| C5 | matchup 方差主导 | 固定 deck 下胜负主要由对位决定；value AUC 高但可能是 matchup prior | **分层 AUC**：按 (deck,opponent) 分组算 within-group AUC（30 分钟，只读 store+ckpt） | 若 within AUC≈0.5，价值头只学到对位先验；需要 counterfactual target |
| C6 | 目标语义错 | 单一 value 头既当 baseline 又当 frontier 又当 Q | 见 §3 R1/R3 | 拆头即可，不必重跑 DMC |

**结论**：C1 最可能、C2 次之；C3 决定"即使修好也不该在本机重开 DMC"。
所以本报告**不建议**把 DMC 作为主线重跑；只建议把 C1/C2/C5 当作诊断（每个 <1 GPU·h），
用来解释历史负结果、并决定 search 头的 target 怎么造。

---

## 2. annotated bibliography

> 每条：问题 / 方法核心 / 证据 / 局限 / 代码或复现 / 对我们的可迁移点。论文标题保留 English。

### B1. DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning（ICML 2021）
- **问题**：3 人斗地主，动作空间最大 10^4 且随手牌变化，奖励只在终局。
- **方法**：Deep Monte-Carlo。状态编码成 4×15 one-hot 牌矩阵（手牌/已知他方牌/最近出牌/剩余张数/炸弹数），
  动作也编码成 4×15；LSTM(162→128) 看最近 5 个状态，拼上 319/430 维状态 + 54 维动作 → 6 层 512 MLP → 一个标量 Q。
  **每个候选动作单独过一次网**，argmax 选招；训练目标 = 整局回报（γ=1），loss=MSE；ε=0.01 探索；三个位置三张网。
- **证据**：4×1080Ti、45 actors、30 天；对 DeltaDou WP 0.586 / ADP 0.258；对 SL 0.659；SL 用同样网络、5×10^7 人类样本仍输。
  消融：TD/SARSA 无清晰增益；actor-critic 失败；动作编码让未见动作可泛化。
- **局限**：百万级 self-play 局；无搜索；3 人非零和，无 CFR 保证；依赖大量 on-policy 覆盖。
- **代码**：<https://github.com/kwai/DouZero>；关键文件
  [`douzero/dmc/dmc.py`](https://github.com/kwai/DouZero/blob/main/douzero/dmc/dmc.py)（`compute_loss` 就是 MSE）、
  [`models.py`](https://github.com/kwai/DouZero/blob/main/douzero/dmc/models.py)（LSTM + 6×512）、
  [`utils.py`](https://github.com/kwai/DouZero/blob/main/douzero/dmc/utils.py)（`target_buf` 对所有 step 写终局回报）。
- **可迁移**：(i) "每个候选单独打分"正是我们的 option 接口，不需要改架构；
  (ii) terminal-only 奖励必须 γ=1；
  (iii) argmax 之前必须有 on-policy 覆盖（ε 或池采样），否则 Q 在未走动作上无定义；
  (iv) 动作特征共享让未见动作可泛化——我们的 `option_features` 已经是这个思路。

### B2. DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning（2022）
- **问题**：DouZero 不看对手风格、学习慢。
- **方法**：给 DouZero 加对手建模（对手手牌/出牌历史特征）+ **coach network**（用完美信息/更强教师的输出来指导），
  加速训练并提升强度。
- **证据**：Botzone 400+ agents 排名第一（论文声称，未核实具体 Elo）。
- **局限**：coach 的完美信息只用于训练；实现细节/代码未核实。
- **链接**：<https://arxiv.org/abs/2204.02558>
- **可迁移**：与 Suphx oracle guiding、PerfectDou 同族——**特权信息只在训练期做教师**，执行期仍是掩码视图。
  我们的 `privileged_state` 已经每样本都存了，这是零采集成本的 oracle 头来源。

### B3. DanZero: Mastering GuanDan Game with Reinforcement Learning（2022）
- **问题**：4 人 GuanDan，长回合、人数可变、动作空间大。
- **方法**：分布式 DMC + 手工状态特征，actor 自博弈产样本，learner 用 DMC 更新。
- **证据**：160 CPUs + 1 GPU × 30 天；胜 8 个启发式基线；与人类玩家表现相当。
- **局限**：无搜索、无价值标定；纯 DMC；规模远大于我们。
- **链接**：<https://arxiv.org/abs/2210.17087>；续作 DanZero+（policy-based + 预训练模型）<https://arxiv.org/abs/2312.02561>
- **可迁移**：DMC 在"多轮卡牌 + 固定规则"里确实能工作，但门槛是 **10^2 CPU × 10 天**量级；
  这直接支持"我们此前的负结果更可能是规模/覆盖问题，而不是算法类别问题"。

### B4. PerfectDou: Dominating DouDizhu with Perfect Information Distillation（NeurIPS 2022）
- **问题**：不完全信息下策略学习慢、上限低。
- **方法**："perfect-training-imperfect-execution"：训练时用全局信息（所有人手牌）做特征，执行时只用私有信息；
  actor-critic（PPO + GAE）+ 完美信息蒸馏。
- **证据**：论文声称全面超过 DouZero/DeltaDou 等（NeurIPS 2022；具体胜率未逐表核实）。
- **局限**：训练期需要 god view（我们有 `full_state`，满足）；PPO 调参敏感。
- **链接**：<https://arxiv.org/abs/2203.16406>
- **可迁移**：**这是 Suphx oracle guiding 的另一种实现**，而且是在牌类游戏上。
  我们的 C10 只约束"观测不能含隐藏信息"，不约束"标签/教师可以用 god view"（`train/collect.py` 已经这么做）。
  → oracle value 头（输入 `privileged_state`，输出胜负）是合规且零新采集的。

### B5. Suphx: Mastering Mahjong with Deep Reinforcement Learning（2020）
- **问题**：4 人立直麻将，巨大隐藏信息、多轮计分、奖励延迟 8–12 轮。
- **方法**：
  1. **Global Reward Prediction (GRP)**：GRU 读每轮特征，预测**最终游戏奖励**；第 k 轮 reward = Φ(x^k) − Φ(x^{k−1})。
     训练目标 MSE（式 4）。
  2. **Oracle guiding**：先训一个能看所有人手牌/牌山的 oracle agent；然后以概率 γ_t 逐步 dropout 完美特征，
     γ_t: 1→0；之后再用 lr/10 + importance-weight 截断继续训。
  3. **pMCPA**：每轮开始固定自己手牌，采样 K 条对手手牌/牌山，rollout + 策略梯度微调，再用于本轮。
- **证据**：训练 1.5M 局/agent、44 GPUs × 2 天；离线评测 1M 局；RL-2（GRP+oracle）> RL-1（GRP）> RL-basic > SL；
  在线 Tenhou 5,760 局、stable rank 8.74（top human 7.46）；pMCPA 对非适应版胜率 66%（数百个初始轮）。
- **局限**：4 人非零和；算力极大；**无官方代码**（截至 2026-10-27 未找到；未核实）。
- **链接**：<https://arxiv.org/abs/2003.13590>
- **可迁移**：
  - GRP 对我们**不需要**：我们一局就是一个 match，outcome 就是 game reward，没有"多轮归属"问题。
    但"从当前局面预测最终奖励"正是 value head 在做的事——Suphx 的证据说明它**可以**学，只是需要大量数据。
  - **oracle guiding 直接可搬**：`privileged_state` 已在样本里；加 oracle 头 + 蒸馏即可（SoG 参考实现已有同样结构）。
  - pMCPA 对我们是"每个 matchup 开局前用已知对手脚本做 K 次 rollout 微调"——贵，但对手是确定性脚本，
    理论上可以**精确克隆对手**后做 best response，比 pMCPA 更省。这条与 E2 的"池内 exploit"方向一致。

### B6. ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games（NeurIPS 2020）
- **问题**：不完全信息下 RL+Search 不 sound；AlphaZero 式价值在 PBS 上无定义。
- **方法**：
  - 状态 = **public belief state** β（公共观测 + 双方 range）。
  - 搜索 = depth-limited CFR-D；叶子价值 = 价值网在叶子 PBS 的 **infostate value**（PBS 价值的 supergradient）。
  - **value target = 一次子博弈求解中 T 次 CFR 迭代的根 infostate value 的平均**
    （`v(β_r) = (1/T) Σ_t v^{π^t}(β_r)`）；Huber loss；prioritized replay。
  - self-play：从 β_r 解子博弈 → 记录 (β_r, v) → 按随机迭代 t 采一个叶子 PBS 作为下一个根 → 重复到终局。
  - 参考模型每 `network_sync_epochs` 同步一次；policy net 可选（warm start CFR）。
- **证据**：HUNL 对 top human +165 ± 69 mbb/g；Liar's Dice exploitability 接近 tabular CFR；
  **"random PBS 训练的价值网学不到东西"**（Fig. 2）——训练分布必须是实际对局会遇到的 PBS。
  附录说 MSE 比 Huber 差（"In preliminary experiments we found MSE for the value network ... did worse"）。
- **局限**：PBS 输入维度随 infostate 数线性增长；只证 2p0s；数据生成是瓶颈（128 台 × 8 GPU）。
- **代码**：<https://github.com/facebookresearch/ReBeL>；
  [`cfvpy/selfplay.py`](https://github.com/facebookresearch/ReBeL/blob/main/cfvpy/selfplay.py)：
  `loss_func`（huber/mse）、`_compute_loss_dict`（`values` 来自 datagen，不是 outcome）、
  `ValuePrioritizedReplay(capacity=2**20, alpha=1.0, beta=0.4)`。
- **可迁移**：
  (i) search value target 要**对迭代平均**，不是取最后一次；
  (ii) Huber 优于 MSE（与我们 H3 的"损失函数不是杠杆"不矛盾：H3 比的是 Brier/BCE，不是 Huber）；
  (iii) **不要在随机 determinization 上训 PBS 价值**——我们如果做 search 头，训练状态必须来自真实对局/搜索查询；
  (iv) 优先回放有价值的 PBS（我们可以按 root value 的绝对值/访问数加权）。

### B7. Student of Games: A unified learning algorithm for both perfect and imperfect information games（Science Advances 2023）
- **问题**：统一 perfect/imperfect info 的 search+learning，且 sound。
- **方法**：GT-CFR（growing-tree CFR）+ CVPN `f_θ(β)=(v,p)`（每个 infostate 一个 counterfactual value + 一个 prior policy）；
  **sound self-play**：价值 target 有两种来源——
  1. 对局主线上的 **TD(1) outcome**（以概率 p_td1）；
  2. **对搜索查询再做一次 GT-CFR，取 T 次迭代后的最终 counterfactual values**（bootstrap）。
  另外，solver 自己产生的 recursive queries 以概率 q_recursive 入 buffer；policy 用 KL/CE；value 用 Huber。
- **证据**（论文 + 附录 Table S2，列序 Chess / Go / Scotland Yard / HUNL）：
  - **HUNL：p_td1 = 0**（100% search value，0% outcome），q_search = 0.9，q_recursive = 0.1；
    **Scotland Yard：p_td1 = 0**，q_search = 0.3。**两个不完全信息游戏都不用 outcome 做 value target。**
  - Go：p_td1 = 0.2；Chess：0。
  - HUNL 对 Slumbot2019 +7 ± 3 mbb/g（3.1M 局），LBR 找不到 exploit；Leduc exploitability 随训练下降。
  - value head weight：Chess 0.25 / Go 0.5 / SY 1 / HUNL 1；policy head weight：1 / 1 / 0.05 / 0.01
    → **在不完全信息游戏里价值损失远重于策略损失**。
- **局限**：需要枚举每个 public state 的 infostates；TPU 级算力；poker 用了 bet abstraction。
- **链接**：<https://arxiv.org/abs/2112.03178>；Science Advances 9, eadg3256 (2023)
  <https://doi.org/10.1126/sciadv.adg3256>
- **本仓蓝本对照**（`refs/Rebel_base_RL/research/world_model/src/gitcg_world_model/`）：
  - `policy_value_model.py:201-205`：**三个头**——`search_value_head`、`partial_value_head`、`oracle_value_head`（+ `search_option_head`）。
  - `cvpn_training.py:1366-1414`：`outcome_value_loss`（MSE）、`teacher_value_loss`（MSE 到 search target）、
    `search_value_loss`（search 头 → search target）、`oracle_loss`（oracle 头 → outcome）、
    `oracle_distill_loss`（主 value → oracle.detach()）。
  - `continual_resolving.py:1570-1607`：根候选对**每个对手粒子**求 value，取 `mean` 和 `std`，
    选招分数 = `policy_log_prob + mean − uncertainty_penalty * std`；根价值 = 策略加权后的均值。
  - `sog_pipeline.py:1665-1741`：每轮 = 当前 ckpt 自博弈 → 抽 frontier queries → 解 query 得 target →
    训练（自博弈轨迹 + query targets + 最近 3 轮 replay，≤1024 局）。
- **可迁移**（最重要）：**拆头**（outcome / search / oracle）+ **search value 用 query 解出来的值** +
  **uncertainty penalty** + **p_td1 可调**。我们的 CVPN 现在只有一个头，这是与蓝本最大的结构性差距。

### B8. DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker（Science 2017）
- **问题**：不完全信息下的实时搜索需要 leaf value。
- **方法**：continual re-solving + 一个 counterfactual value network；value net 在**随机生成的 PBS**上用搜索值训练。
- **证据**：44,000 手对职业选手显著胜出。
- **局限**（ReBeL 论文 §2 的评价）：随机 PBS 绝大多数在真实对局中不会出现，样本效率极低；
  靠手工特征降维和领域抽象绕过。
- **链接**：<https://arxiv.org/abs/1701.01724>
- **可迁移**：**反面教材**——不要用"随机采样对手手牌/骰子"造 search value 训练集；
  要用实际对局中搜索真正查询过的 PBS（我们的 `CFRSearchDecision` 正好记录了这些）。

### B9. Libratus: Superhuman AI for heads-up no-limit poker（Science 2018）
- **问题**：固定规则、固定对手分布下的最强打法。
- **方法**：blueprint（离线 CFR）+ 在线 nested subgame solving + **self-improvement**：
  每手结束后，针对对手实际使用的动作/尺度算 best response，并把结果补进 blueprint。
- **证据**：对 4 名顶级人类 +147 ± 39 mbb/g（ReBeL 论文 Table 1 引用）。
- **局限**：需要大量在线算力；self-improvement 依赖对手动作可观测。
- **链接**：<https://doi.org/10.1126/science.aao1733>
- **可迁移**：**对固定脚本池，"剥削"比"Nash 自博弈"更对齐目标**。我们的 dgeo PG +16.8pp 正是这个现象；
  Libratus 的 self-improvement 对应"每轮针对最差 matchup 做定向 best response/PG"。
  这也是 E2 把池权重放到 60% 的理论依据。

### B10. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL（2024）
- **问题**：MSE 回归的价值头难以随网络/数据 scale。
- **方法**：把标量 target 投影到 m 个等距 bins 上的类别分布（HL-Gauss）：
  `p_i = F_N(z_{i+1/2}) − F_N(z_{i−1/2})`，`N(μ=target, σ)`，`σ/ς=0.75`（ς=bin 宽），
  loss = `−Σ p_i log p̂_i`，推理取 `ŷ=Σ p̂_i z_i`。
- **证据**：Atari MoE +30% IQM；multi-task Atari 1.8–2.1×；Chess 70%；机器人 67%；
  **但 §5.2.1 明确说：确定性动力学下 HL-Gauss 与 MSE 相当**，优势主要来自噪声 target/非平稳。
- **局限**：需要选 `[v_min,v_max]`、bins、σ；对 3 值 outcome 几乎没有分辨率增益；不改善兄弟动作排序。
- **链接**：<https://arxiv.org/abs/2403.03950>
- **可迁移**：**只用在连续值的 search 头上**（root value 在 [−1,1] 连续），不要用在 ±1/0 的 outcome 头上；
  我们环境是确定性的（固定种子 + 确定性脚本），论文预测增益很小——它是"顺手做"的第三臂，不是主菜。

### B11. A Distributional Perspective on Reinforcement Learning（C51, ICML 2017）
- **问题**：只学期望值丢失了回报分布。
- **方法**：51 个 atom 上的类别分布 + 分布式 Bellman 投影 + cross-entropy。
- **证据**：Atari SOTA（2017 年时点）。
- **局限**：投影/atom 调参；比 HL-Gauss 复杂；后续工作显示 HL-Gauss 在多数设定更好。
- **链接**：<https://arxiv.org/abs/1707.06887>
- **可迁移**：分布输出的**方差**可以给搜索的 `uncertainty_penalty` 用（SoG 参考实现用的是 particle 间 std，
  不是 value 分布；两者可以叠加）。

### B12. Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning（NeurIPS 2023）
- **问题**：离线 critic 的尺度不可比，在线微调时被过度保守/过度乐观的价值拖死。
- **方法**：把 Q 标定成"对学习策略真实值的下界 + 对参考（behavior）策略值的上界"；
  在 CQL 上加一行约束即可。
- **证据**：11 个 fine-tuning benchmark 里 9 个优于 SOTA。
- **局限**：离线→在线设定；需要 behavior policy 的价值估计。
- **链接**：<https://arxiv.org/abs/2303.05479>
- **可迁移**：**"价值在被用于改进之前，必须先和参考策略的尺度对齐"**——这正是我们 `V^expert` vs `V^σ` 的问题。
  具体做法：按 matchup 把 value head 的中心对齐到该 matchup 的实际胜率（见 §3 R3b），
  这样"根价值"才有绝对意义；对 argmax 无影响（同一 matchup 内是常数），但对诊断和门控阈值有影响。

### B13. PSRO / PFSP / OpenAI Five：对手采样与历史策略
- **PSRO**（<https://arxiv.org/abs/1711.00832>）：在策略集合上算 meta-game，按 meta-strategy 采样对手，
  对 mixture 求 best response；统一了 fictitious play / double oracle / iterated best response。
- **PFSP**（AlphaStar, Nature 2019, <https://doi.org/10.1038/s41586-019-1724-z>）：
  按"当前 agent 对该对手的胜率"给对手加权，优先采样**打不过的**对手（具体函数形式见 supplement，**未逐字核实**）。
- **OpenAI Five**（<https://arxiv.org/abs/1912.06680>）：**80% 对当前策略、20% 对历史策略**；
  历史策略的采样细节在 Appendix N（未核实）。
- **证据**：三者都把"历史/难度加权对手"作为核心组件；OpenAI Five 的 80/20 是工程选择，论文未给"纯 current"的同预算消融（未核实）。
- **局限**：PSRO 需要 payoff matrix 和大量策略；我们的池固定只有 20 个，做完整 PSRO 过重。
- **可迁移**：把"20 套脚本"当成 league 的固定成员，用 PFSP 式**难度加权**采样；
  历史 checkpoint 用 10% 的 recency 权重防止 PG 在固定池上循环/遗忘（我们 dvalin 连续两轮 −0.063 → −0.128 就是漂移信号）。

### B14. The Predictive–Optimization Gap in GuanDan Self-Play（arXiv 2026-10-21，preprint）
- **问题**：更好的"结果预测器"是否等于更好的优化信号？
- **方法**：在 DanZero 式 DMC 自博弈里，用冻结的 match-win predictor 的差分 ΔΦ 作为 round-level 监督，
  5 个预注册 replication seeds。
- **证据**：单个 lineage 可以显著更好，但 **5 个预注册种子没有一个是正的**；10 seed 合并区间含 0；
  跨迭代效应反复变号 → "predictive–optimization gap"。
- **局限**：preprint、未同行评审（未核实）；GuanDan 特定。
- **链接**：<https://arxiv.org/abs/2610.29411>
- **可迁移**：这是对我们最直接的警告——**"value/GRP 的预测精度提升"不能当成"策略变强"**。
  所有门禁必须是引擎胜率；AUC/ECE 只能当机制检查。

---

## 3. 最小可复现 recipe（按性价比排序）

### R1（≈0 成本，先做）：修 frontier 的两个语义 bug

**输入**：任一 CVPN ckpt（E1 里用该 deck 基座自带的 value 头；`data/checkpoints/kaggle/.../bc_e150.pt` 也可做冒烟）。
**输出**：`envs/cfr_probe._frontier_probability` 返回真正的 P(player 0 wins)，不再截断负值。

```python
# envs/cfr_probe.py（示意）
def _frontier_probability(built, *, value_spec, acting_player):
    if value_spec:
        v = resolver.value(built)                  # identity: [-1,1]; sigmoid: [0,1]
        sigmoid = getattr(resolver.model.config, "value_activation", "identity") == "sigmoid"
        p_acting = v if sigmoid else (v + 1.0) / 2.0
        p0 = p_acting if acting_player == 0 else 1.0 - p_acting
        return min(1.0, max(0.0, p0))              # clamp 只做在最后
    return heuristic_frontier_probability(built.context.player_view)   # 已是 p0

# _ProbeState.record: "frontier_p0": _frontier_probability(built, value_spec=..., acting_player=who)
```

**要改的文件**：`envs/cfr_probe.py`（`_frontier_probability` + `record`）、`train/resolver.py`（加
`value_p0(built, acting_player)` 或返回 `(value, perspective)`，避免下一个调用者再错）、
`tests/test_cfr_probe.py`（新增：fake resolver=0.8 @ acting=1 → `p0=0.2`，traverser 0 的 centred=`−0.6`）。
**成本**：1–2 h（含单测）；0 训练；0 采集。
**预期信号**：修复后对 200 个 val 状态统计原始值 <0 的比例（若 >20%，D1 确认）；
用 100 局 d1 搜索重算"赢局/输局根价值"与"同状态 action-value spread"。
**失败判据**：若修复后 spread 仍 <0.01、根价值 AUC 仍 ≈0.5，则 D1/D2 不是主因，进入 R3。

### R2（≈0 成本）：候选集与门控

**输入**：R1 修好的 frontier + 现有 base policy 的 prior。
**输出**：只在"显著更好"时才覆盖专家；候选由 prior 排序而非引擎序。

```python
# envs/cfr_search.py，solution 之后
base_key = keys[base_idx]
v_base = solution.action_values.get(base_key)
v_best_key, v_best = max(solution.action_values.items(), key=lambda kv: kv[1])
if v_base is None or (v_best - v_base) <= tau:      # tau ∈ {0.03, 0.05, 0.10}
    return base_code                                 # 不覆盖
chosen_key = v_best_key
```

- `candidate_limit` 必须同时传 `prior`（`CFRSearchPolicy(prior=PriorResolver(...))`），
  且 `prior` 用**同一基座**的 policy 头，保证候选不是任意序。
- 若预算允许（A4 B 档 0.44 s/搜索），用 **fork rollout 的配对均值**替代 `V(s')`：
  同一粒子的兄弟分支从同一 `branch_random` 出发（`reps/belief._sample_branch_random` 用公共字段派生种子），
  在**首个随机事件之前**是严格配对的；之后会因 RNG 消耗次数不同而分叉。它不是严格的 common random numbers，
  但比"两个独立学习的 V(s') 之差"更接近配对比较，方差更低。
- **要改的文件**：`envs/cfr_search.py`（`choose` 的覆盖逻辑）、`scripts/run_cfr_ladder.py`（加 `--gate-tau`）。
**成本**：0.5 h；0 训练。
**预期信号**：动作改变率从 86–89% 降到 20–40%；win→loss 翻转数下降。
**失败判据**：若改变率降到 <10% 且 Δ 不变，说明搜索本来就没在起作用（价值/候选都无信息），直接进 R3。

### R3（1 次训练，0 新采集）：拆头 + oracle + 可选 HL-Gauss

**输入**：现有 store（`sa_all.pt` / `mixed_four_decks.pt`），已含 `privileged_state`。
**输出**：CVPN 增加两个头，训练目标分离：

| 头 | 输入 | target | loss | 用途 |
|---|---|---|---|---|
| `value_head`（现有） | pooled public state | `outcome_target`（±1/0） | MSE 或 sigmoid+Brier | PG baseline；depth-1 frontier |
| `search_value_head`（新） | pooled state + belief summary（可选） | `search_value`（MCCFR/rollout） | **Huber(δ=1)**；权重=`f(visits, particles)` | depth≥2 / 网络基座的 frontier |
| `oracle_value_head`（新） | `privileged_state` | `outcome_target` | MSE；+ `(value − oracle.detach())²` 蒸馏 | 降低部分可观测下的 value 方差 |

```python
# train/model.py
self.search_value_head = nn.Sequential(nn.Linear(d + belief_dim, 128), nn.SiLU(), nn.Linear(128, 1))
self.oracle_value_head = nn.Sequential(nn.Linear(privileged_dim, 128), nn.SiLU(), nn.Linear(128, 1))

# train/distill.py
value_loss = F.smooth_l1_loss(value, outcome, beta=1.0)          # 替代 MSE
search_loss = F.smooth_l1_loss(search_value_pred, search_value, beta=1.0) * w_i
oracle_loss = F.mse_loss(oracle_pred, outcome)
distill_loss = F.mse_loss(value, oracle_pred.detach())
```

- 所有新头默认关闭（`search_value_dim=0` 等），旧 checkpoint 不变。
- HL-Gauss 作为**第三臂**（只加在 `search_value_head`）：101 bins over [−1,1]，`σ/ς=0.75`，CE。
- **要改的文件**：`train/model.py`（两个头 + config）、`train/distill.py`（Huber/权重/蒸馏）、
  `scripts/train_bc_scale.py`（`--value-loss {mse,huber,hlgauss}`、`--oracle-coef`、`--search-value-coef`）、
  `tests/test_value_heads.py`（头形状/旧 ckpt 兼容/HL-Gauss target 均值为原值）。
- **成本**：1 次 150ep（30k–200k 样本）≈ **0.1–0.9 GPU·h**（T4，9.2k 样本/s）；CPU（200 样本/s）≈6–40 h，
  所以本地先跑 15ep（30k：≈40 min；200k：≈4 h）看分层 AUC 方向，再上 GPU 长训。
- **预期信号**：分层 AUC（按 matchup）与 ECE；oracle 蒸馏后 `value_head` 的 AUC 提升 ≥0.02；
  **不是**引擎胜率。
- **失败判据**：若分层 AUC ≈ pooled AUC（说明价值头本来就不是 matchup prior），且 oracle 蒸馏无增益，
  则价值学习不是瓶颈，停止这条线，把资源投 E2。

### R3b（0 训练）：matchup 条件化的价值中心对齐（严格意义的 calibration）

- 在**训练种子**上按 `(deck, opponent)` 算实际胜率 `p̄_{d,o}`，把 value head 的输出 shift 到
  `V_cal(s) = V(s) + (p̄_{d,o} − mean_{s∈(d,o)} V(s))`。
- 用途：让"根价值"有绝对意义（门控阈值 τ 才可跨 matchup 复用）；**不改变同状态内的 argmax**。
- 成本：30 min（只读 store + ckpt）。失败判据：若 shift 后根价值 AUC 仍 ≈0.5，说明价值头连 matchup 先验都没学好。

### R4（低代码）：belief 的正确接法

按"性价比"分三步，**不要一次全上**：

1. **dice 分布**（先做，20 行）：`reps/belief._sample_hidden_dice` 改为按引擎掷骰分布采样多重集，
   权重乘 `P(multiset)`。这一步不需要 belief 头，直接提升所有 determinization 的质量。
2. **学到的 belief → 根粒子权重**（30 行）：`train/cfr_solver.belief_root_particles` 增加
   `belief_histogram` / `belief_group_hand_probs` / `belief_group_deck_probs` 参数，
   `envs/cfr_search._root_particles` 从 belief 头取输出传入。
   注意：`build_public_belief_state` 已支持这些参数（`reps/belief.py:337-349`），只是没接线。
3. **belief summary → 独立 search 头**（不碰主策略）：照 SoG 参考实现，
   给 `search_value_head` / `search_option_head` 加 `self_range_summary` + `opponent_range_summary`
   + `search_context_summary`；主 policy/value 仍然只看公共态。
4. **对手 deck token**（可选）：把 20 个 deck slug 加进 `opponent_tag` 词表，collector 传 `opponent=<slug>`。
   注意：这会让 `token_dim` +20，旧 ckpt 不兼容——要么重训，要么用独立 embedding（A2 的教训：先做消融）。

**成本**：1+2 ≈ 1 h + 单测；3 需要 search 数据（R5/E1 顺带）。
**预期信号**：dice 修复后，搜索根价值在"对手有/无关键反应牌"的状态上出现可分辨差异；
belief 接线后，根粒子的权重熵下降（不再近似均匀）。
**失败判据**：若 dice 修复后 action-value spread 不变，说明 dice 不是主导隐藏量，回到手牌 belief。

### R5（collector-only）：自博弈采样分布

不改训练代码，只改采集任务表（`deck × opponent × seeds × spec`）。推荐分布（按局数）：

| 组分 | 权重 | 采样 | 目的 |
|---|---:|---|---|
| 20 套脚本池 | **0.60** | `opponent_spec=expert:<opp>`；池内按 PFSP 权重 `w_j ∝ (1−p̂_j)² + 0.2` | 对齐评测分布 + 长尾 |
| 当前策略自博弈 | 0.20 | 两席同 ckpt，`#t=1.0`，**换座**（一半 teacher 在 p1） | 探索/覆盖 + 防对 20 个确定脚本过拟合 |
| 专家 anchor | 0.10 | `teacher_spec=expert:<deck>` vs 池/自博弈 | 防漂移（dvalin 教训） |
| 历史 checkpoint | 0.10 | 最近 K=3 个**已认证** ckpt，权重 0.5/0.3/0.2（不足 3 个就用已有基座/上一轮） | 防循环/遗忘（OpenAI Five 的 20%） |

- **条件**：若该 deck 的当前 Δ < −2pp（未达 parity），把自博弈/历史降到 0，先 100% 池 + anchor
  （依据：PG 只在 parity+ 的 deck 上有效，SA/dvalin 的负结果）。
- **换座**：`train/data.py` 的 `collect_samples` 固定 teacher=p0；自博弈要覆盖 p1，需要加
  `teacher_player` 参数并在 `value_target` 上乘 `+1/-1`。这是唯一需要的代码改动（~20 行）。
- **成本**：2,000 局/臂 ≈ 0.5 h（3,700–4,900 局/h）。
- **失败判据**：若自博弈占比 20% 的臂在同等局数下不优于 0%（E2），永久移除自博弈组分。

---

## 4. 预注册实验（2 个，引擎胜率为唯一主指标）

> 种子纪律：所有认证种子必须与选型/调参种子不重叠。建议 E1 用 **1200–1219**，E2 用 **1220–1239**；
> 实验开始前把区间写进脚本参数并提交，禁止中途换种子。
> 统计口径：20 对手 × 20 seeds = 400 配对局/臂；Wilson/配对 bootstrap CI；同时报 `rate<0.45` 的 matchup 数。

### E1（主实验）：修语义 + 门控搜索（4 臂，2 deck）

**假设**：在 depth-1 + expert 基座下，搜索失败主要来自 frontier 消费端（D1/D2）与任意候选/无条件覆盖（D7），
不是价值学习。修好之后，**门控搜索**至少不劣于不搜索，并可能小幅正增益。

| 臂 | frontier | 候选 | 覆盖规则 |
|---|---|---|---|
| A | — | — | 同一 ckpt 纯策略（`neural:<ckpt>` 贪心，无搜索；对照） |
| B | 现 heuristic（无 `value_spec`） | 引擎序前 4 | 总是覆盖（复现 A4 B 档） |
| C | 现 heuristic | **prior top-4** | `Δ > τ=0.05` 才覆盖 |
| D | **该 deck 基座 ckpt 自带的 value 头**，修 D1/D2 | prior top-4 | `Δ > τ=0.05` 才覆盖 |

- **基座**：`double_geo_navia` 的 `rl_dgeo3`（已验证 0.880，强基座，搜索若有害最容易看出来）
  + `unyielding_geo` 的 `rl_ungeo3`（0.909，第二 deck 做方向性复验）。
  两者都通过 `scripts/run_cfr_ladder.py --checkpoint <ckpt>` 同时作为 base+prior（不是 `expert:` 基座）。
- **D 臂的头**：用 A 臂同一 ckpt 的 `value_head`（同 deck 的训练数据，避免 `sa_all` 头的跨 deck 偏移）；
  若其 `value_activation=identity`，R1 的 `(v+1)/2` 转换必须生效。
- **要改的 CLI（前置，~10 行）**：`scripts/run_cfr_ladder.py:156` 现在 `value_spec = spec.value_spec or spec.checkpoint`，
  即 `--checkpoint` 会**同时**设 base/prior/value。E1 需要把 base 与 frontier 解耦：
  加 `--base-checkpoint`（只作 base+prior）或 `--no-value-from-checkpoint`，否则 B/C 无法用 heuristic frontier。
- **采集**：0 新训练局（只有 100 局诊断 + 3,200 局评测）。
- **训练**：0 GPU·h（A–D 都不训练；D 用基座 ckpt 自带的 value 头）。
- **评测**：2 deck × 4 臂 × 400 配对局 = **3,200 局**；d1 搜索约 3,100 局/h → **≈1.0–1.5 h**（12 核）。
  若加 depth-2 档（C′/D′），每臂 +400 局 × 2 deck ≈ 0.6 h。
- **机制门禁（跑引擎前必须过）**：
  1. 单测：acting=1、v=+0.8 → p0=0.2（D2）。
  2. 100 局 d1 搜索日志：同状态 `action_values` 的 `max−min` 中位数 **≥0.02**（D1/D7）。
  3. matchup 中心化后的根价值 AUC **≥0.60**（D1/D2 修好后的价值头）。
- **主判据**：对 A 的配对 Δ，**CI 下界 ≥ −0.02 且点估计 > 0**（非劣 + 方向），
  且 **D 比 C 至少 +0.03**（同一候选/门控下只换 frontier，隔离语义修复的贡献；B 只用于复现 A4）。
- **停止条件**：
  - 若 B 复现 ≤ −0.03 且 C/D 未达非劣 → **正式关闭搜索分支**，在 PLAN §0 记录（不再排产）。
  - 若机制门禁 2 未过（spread <0.02）→ 不要跑引擎，直接进 R3（拆头/search target）。
- **预期成本汇总**：0 采集局、0 GPU·h、≈3,200 评测局、1.5–2 h 墙钟。

### E2（次实验）：自博弈采样分布（3 臂，1 deck）

**假设**：在固定脚本池下，"池为主 + 少量自博弈 + anchor + 历史"比"100% 池"或"高比例自博弈"更能提升
**对池 pooled 胜率**，且不损伤 floor。

| 臂 | 池 | 自博弈 | anchor | 历史 | 局数/臂 |
|---|---:|---:|---:|---:|---:|
| S1 | 100% | 0 | 0 | 0 | 2,000 |
| S2 | 60% | 20% | 10% | 10% | 2,000 |
| S3 | 40% | 40% | 10% | 10% | 2,000 |

- **deck**：`shenhe_double_swirl`（最新一轮 400 局：pooled 0.745，配对 Δ **+0.0675** [+0.018,+0.117]，
  见 `reports/explore/rl_shenhe2_shenhe_double_swirl.json`，2026-10-27；strict floor 仍因 10-seed 功效不足标红）；
  若 S2/S3 有正信号，再到 `electro_manifestation_nuke`（pooled 0.7175，Δ +0.030 [−0.021,+0.081]）复验。
- **基座**：该 deck 的稠密底座（`gb_*`/Group C 专用 ckpt），三臂同基座、同 seed、同 PG 超参（2 epoch）。
- **采集**：3 × 2,000 = **6,000 局**（≈1.5 h，12 核；自博弈局同价）。
- **训练**：每臂 ~92k 样本 × 2 epoch；CPU 200 样本/s → ~15 min/臂；GPU 更快。合计 **GPU <0.1 h**
  （或 ≈45 min CPU）。
- **评测**：3 臂 × 400 配对局 = **1,200 局**（新种子 1220–1239）≈0.5 h；
  另加当前 specialist 作参照（+400 局）。
- **主判据**：`pooled(S2 或 S3) − pooled(S1) ≥ +0.02` 且配对 Δ 的 CI 下界 > 0；
  同时**没有任何 matchup 退化 >5pp**、`rate<0.45` 计数不增加。
- **停止条件**：若 S2/S3 ≤ S1 → 自博弈/历史组分不进入默认配方；
  若 S2 正、S3 负 → 说明自博弈比例有上限，取 20%。
- **预期成本汇总**：6,000 采集局 + 1,600 评测局 + GPU <0.1 h。

> 两个实验都不使用 val_acc / AUC 作为主指标；AUC/ECE 只出现在 E1 的机制门禁里。
> 这是 A2 的直接教训：`store val_acc 0.866` 的模型在引擎上 dgeo 只有 0.266（`reports/explore/a2_id_screening.md`）。

---

## 5. 与现有负结果的关系（为什么这次不会重演）

| 已有负结果 | 本报告是否重复 | 为什么不同 / 如何避免 |
|---|---|---|
| **DMC / Q / AWR 全负**（`plan_strength_2day_kaggle.md`） | **不重复**：不把 DMC 作为主线 | 只做 C1/C2/C5 三个**诊断**（每个 <1 GPU·h），目的是解释历史；不再用离线 Q 的 argmax 选招。E1/E2 都不用 DMC。 |
| **search-as-teacher / MCCFR 蒸馏全负**（A4：−0.066 / −0.114） | **不重复**：E1 不蒸馏、不训练策略 | E1 只改**搜索消费端**（符号/截断/候选/门控），策略本身不变；且先过机制门禁（spread、AUC）再跑引擎。若门禁不过，不跑引擎、不蒸馏。 |
| **"价值标定是重开搜索的硬前置"**（D23 / H3） | **修正而非重复** | H3b 已证"标定本身不够"；本报告进一步指出 D23 的"根价值≈0"可能是 D1/D2 的测量伪影。E1 里 B→C 隔离"候选/门控"，C→D 隔离"只换 frontier（修语义）"——能直接分辨"价值学习不行"与"消费者坏了"。 |
| **H5 matchup 重加权负** | **不重复**：E2 改**采集分布**，不改 loss 权重 | H5 是在 BC 损失上加权（偏离模仿目标 → 有害）；E2 是改变**对局从哪来**（同等局数对照），这正是 H4/Group C 已证有效的杠杆。 |
| **胜局筛选 BC 负** | **不重复** | E2 不做结果筛选；自博弈/历史对局全部入训，避免生存者偏差。 |
| **A2 identity embedding 假阳性** | **不重复**：不以 val_acc 为门禁 | E1/E2 主指标是引擎胜率；AUC 只做机制检查；所有 +Δ 必须在**新种子**上复验（`a2_id_screening.md` 的教训）。 |
| **纯 self-play 会优化错目标** | **正面处理** | E2 的自博弈上限是 40%，且必须有池 60%/40% 的对照；若不如 100% 池，永久移除。 |

---

## 6. 不要做什么（negative guidance）

1. **不要把 identity 激活的价值头直接当概率用**，也不要在消费端 `max(0, v)` 截断——这会把所有劣势状态压成同一个值。
2. **不要用 pooled 根价值 AUC 判断搜索质量**：跨 matchup 的均值会被对位先验主导；至少按 `(deck, opponent)` 中心化，并同时看同状态 action-value spread。
3. **不要用 val_acc / ECE / AUC 作为主门禁**。A2 已经证明它们可以和引擎胜率反号。
4. **不要在随机 determinization 上训练 PBS/search 价值**（DeepStack 的教训，ReBeL 的 Fig. 2 反证）；训练状态必须来自真实对局/搜索查询。
5. **不要把 depth-1、1 粒子、4 个任意候选的 MCCFR 根策略当 teacher 去蒸馏**；A4 已经证明它是噪声。
6. **不要把纯 self-play 当成固定脚本池的主数据源**；它优化的是"打败自己"，不是"打败 20 套脚本"。
7. **不要对 terminal-only ±1 奖励用 γ<1 的 Q 目标**：同一状态内 argmax 不变（所有动作共享同一个 γ^k），所以它不是"奖励输得慢"；但它把不同深度的样本缩放到不同尺度，使 Q 不再是回报、跨状态不可比，并压低早期决策的信噪比。DouZero 用 γ=1 是有原因的。
8. **不要只监督被走动作的 Q 然后对全部合法动作 argmax**；off-support 的 Q 没有定义。
9. **不要在未做消融前把 belief 塞进主策略**：belief 错了会直接污染动作；先当辅助头/只喂 search。
10. **不要把"预测更准"当成"策略更强"**（B14 的 predictive–optimization gap）；所有结论回到引擎胜率。
11. **不要在认证种子上调参**；E1/E2 的种子区间事前写死，报告同时给点估计和 CI。
12. **不要用单一 pooled 数字掩盖长尾**；逐 deck + `rate<0.45` 计数 + floor 一起报。

---

## 7. 参考（链接、版本/日期、可信度）

| # | 文献/实现 | 链接 | 版本/日期 | 可信度 |
|---|---|---|---|---|
| 1 | DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning | <https://arxiv.org/abs/2106.06135> | ICML 2021；代码 main 分支 2026-10 读取 | 高（论文 + 官方代码逐文件读） |
| 2 | DouZero 代码（dmc.py / models.py / utils.py） | <https://github.com/kwai/DouZero> | main，2026-10-27 读取 | 高 |
| 3 | DouZero+: Opponent Modeling and Coach-guided Learning | <https://arxiv.org/abs/2204.02558> | arXiv 2022-04 | 中（只读摘要） |
| 4 | DanZero: Mastering GuanDan with RL | <https://arxiv.org/abs/2210.17087> | arXiv 2022-10 | 中（只读摘要） |
| 5 | DanZero+ | <https://arxiv.org/abs/2312.02561> | arXiv 2023-12 | 中（只读摘要） |
| 6 | PerfectDou: Perfect Information Distillation | <https://arxiv.org/abs/2203.16406> | NeurIPS 2022 | 中（只读摘要；代码未核实） |
| 7 | Suphx: Mastering Mahjong with Deep RL | <https://arxiv.org/abs/2003.13590> | arXiv 2020-03 | 高（读 §3.1–3.4 与实验） |
| 8 | ReBeL: Combining Deep RL and Search for Imperfect-Information Games | <https://arxiv.org/abs/2007.13544> | NeurIPS 2020 | 高（读 §4–§5、附录算法 + 代码） |
| 9 | ReBeL 代码（cfvpy/selfplay.py） | <https://github.com/facebookresearch/ReBeL> | archived 2024-11；2026-10 读取 | 高 |
| 10 | Student of Games | <https://arxiv.org/abs/2112.03178>；<https://doi.org/10.1126/sciadv.adg3256> | Science Advances 9, eadg3256 (2023) | 高（读正文 + 附录 Table S2） |
| 11 | SoG 蓝本（本仓 refs） | `refs/Rebel_base_RL/research/world_model/src/gitcg_world_model/` | 本地只读 | 高（逐文件读） |
| 12 | DeepStack | <https://arxiv.org/abs/1701.01724> | Science 2017 | 中（读摘要 + ReBeL 对其评价） |
| 13 | Libratus | <https://doi.org/10.1126/science.aao1733> | Science 359(6374):418–424 (2018) | 中（读 ReBeL Table 1 引用 + DOI 元数据） |
| 14 | Stop Regressing（HL-Gauss） | <https://arxiv.org/abs/2403.03950> | arXiv 2024-03 | 高（读 §3.1/§5.2 与实验） |
| 15 | C51: A Distributional Perspective on RL | <https://arxiv.org/abs/1707.06887> | ICML 2017 | 中（读摘要） |
| 16 | Cal-QL | <https://arxiv.org/abs/2303.05479> | NeurIPS 2023 | 中（读摘要） |
| 17 | PSRO: A Unified Game-Theoretic Approach to MARL | <https://arxiv.org/abs/1711.00832> | NeurIPS 2017 | 中（读摘要） |
| 18 | AlphaStar（PFSP） | <https://doi.org/10.1038/s41586-019-1724-z> | Nature 575, 350–354 (2019) | 中（PFSP 具体函数形式未逐字核实） |
| 19 | OpenAI Five | <https://arxiv.org/abs/1912.06680> | arXiv 2019-12 | 高（"80% 当前 / 20% 历史"原文）；Appendix N 细节未核实 |
| 20 | The Predictive–Optimization Gap in GuanDan Self-Play | <https://arxiv.org/abs/2610.29411> | arXiv 2026-10-21，preprint | 中（preprint，未同行评审） |
| 21 | 本仓 A4 搜索预算阶梯 | `reports/search/a4_search_budget_ladder.md` | 2026-10-12 | 高（本仓实测） |
| 22 | 本仓 H3 价值标定 | `reports/train/value_calibration_kg150.json`、`plan_strength_2day_kaggle.md` §Day1b/Day2b | 2026-10-17 | 高（本仓实测） |
| 23 | 本仓 A2 identity 假阳性 | `reports/explore/a2_id_screening.md` | 2026-10-27 | 高（本仓实测） |

---

### 附：一页纸执行清单（如果只做三件事）

1. **今天**：修 `envs/cfr_probe.py` 的符号 + 截断（R1），加单测；跑 100 局 d1 搜索，量 `action_values` 的 spread。
2. **明天**：若 spread ≥0.02，跑 E1 的 A/B/C/D 四臂（3,200 局，2 h）；若 B 复现负而 D 转正/打平，搜索分支保留并进 R3。
3. **并行**：E2 的 6,000 局采集（1.5 h）→ 3 臂 PG（GPU <0.1 h / CPU ≈45 min）→ 1,200 局评测；若 S2/S3 不胜 S1，自博弈组分永久移除。
