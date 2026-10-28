# 文献调研汇总：从 A2 假阳性到可执行实验队列（2026-10-27）

> 口径：**纯策略**（LLM 不参与，PLAN §6.3）；强度只看引擎胜率，不看 store `val_acc`。
> 方法：7 个 subagent 各读一手论文 + 官方实现 + 本仓代码，分报告在 `reports/explore/lit/`。
> 本文只做交叉裁决、纠错与排产；未改代码、未跑训练/评测。子报告中的「未核实」标记保留。

---

## 0. 一页结论

1. **A2 的负结果不能采信。** compact 语料默认 `float16`，把 `definition_id / 400000` 这个可逆编码压成了**不可逆的假 id**：
   152 张卡在 store 里只剩 **28 个不同 id**，真实 bucket 142 个，**交集只有 4 个**。
   A2 的 id embedding 训练时读假 id、推理时读真 id，等于给卡牌类 token/option 注入随机向量。
   这不是「id embedding 有害」，是**数据落盘精度事故**。
2. **P0 不是「再加一个结构」，而是先过 parity 门禁**：逐元素 store↔engine 对比 + definition 列按整数比 +
   重采干净语料（fp32 或 int32 id 列）+ 修评测/初始化种子。约 **0.3 CPU·h 建设 + 20 min 重采**，
   比任何一次结构重训便宜两个数量级。
3. **DAgger 是「BC→parity」的分布修复算子，不是超专家算子**；先做 200 局零训练诊断，gap <3pp 就停。
   超专家仍只能靠已验证的 PG（dgeo +16.8pp / ungeo +5.5pp），且**绝不在 PG 之后再做 DAgger**。
4. **floor 违规大部分不是策略问题，是 deck 天花板**：19 个有认证臂的 deck 里，严格可修的 matchup 只有 **10 个**；
   natlan / beidou / chasca / skirk 系有 18–20 个对手**连脚本专家自己都输**。验收必须同时报「绝对 floor」和
   「相对同 deck 专家的配对 Δ」，否则会把 deck 强度算到策略头上。
5. **specialist 组合优先于 generalist**：先把 dvalin / shenhe / electro 三个 parity+ deck 扩成 PASS，
   再用 deck 级、按公开阵容的 router；不要 20-deck 单模型，也不要逐决策 contextual bandit。
6. **候选集 attention 不是当前瓶颈**：DouZero/Suphx 都用逐动作独立打分；先做 0 成本歧义诊断，
   没有信号就停 A4。即使做，也只做冻结 trunk + zero-init 的 1 层 probe，只认引擎胜率。
7. **评测口径必须升级**：800 配对局只能判 ~6pp；「CI 下界 ≥ −0.02」需要约 **1,400 配对局**；
   5pp 需要 20–28 seeds，3pp 需要 2,000–3,000 局。现在的 400/800 局筛选不能做认证级结论。

**排产一句话**：`P0 一致性/数据精度` → `P1 可修 matchup + DAgger 诊断 + specialist 扩张` →
`P2 league/value/belief` → `P3 候选集/记忆`。在 parity 未过之前，冻结所有新结构训练。

---

## 1. A2 复核：负结果为什么不能采信

### 1.1 直接证据（f05 + f07 独立复核，数学可证）

- `train/data.py:350` 的 `DEFAULT_COMPACT_DTYPE = "float16"`；`train/data.py:474-475` 把
  `token_features` / `option_features` **整体**压成 fp16；`scripts/datafirst_bc_store.py:47-51`
  只做 fp16→float32 提升，不恢复 id。
- definition 列（token 42 / option 30）是 `id / 400000`。fp16 在 `x≈0.5–0.8` 的绝对精度约 `1e-4–2e-4`，
  乘 400000 后 id 误差达 **±几十到上百**；`reps/id_features.py:20-23` 再 `round(x*400000)` 得到假 id。
- f07 实测：152 张卡经 fp16 往返只剩 **28 个不同 id（20 组歧义）**，只有 8 张能唯一反推；
  真实 bucket 142 个，store bucket 与真实 bucket **交集 4/142**。
  - dvalin：真实 17 bucket vs store 9，**交集 0**；
  - dgeo：真实 15 vs store 9，**交集 1**；
  - ungeo：真实 24 vs store 12，**交集 0**。
- 角色 id（4 位）大部分没坏（35/36 精确），所以「切人/选人」多的 deck 还能保留部分 id 信号；
  依赖 6 位卡 id 的 deck 基本被随机向量淹没——这解释了 A2 的 deck 依赖性。
- **读盘时无法修复**：fp16 是多对一，不是四舍五入到最近真 id。只能重采、加 int32 id 列，或整体 fp32。

### 1.2 A2 还叠加了四个口径问题

| 问题 | 证据 | 影响 |
|---|---|---|
| **只采 p0，评测双边** | `train/data.py:157-173` teacher 只包 p0；`scripts/eval_protocol.py` 默认 `sides=(0,1)`；store `acting_player` 1128/1128 为 `(1,0)` | side=1 的输入分布模型从未见过；f07 建议用 `seat_swap` 量化 |
| **val_acc 跨语料比较** | A2 报告表头写「store val_acc (dgeo)」，但 0.866 是 mixed_four_decks 混合 val、0.838 是 h4m 另一套 val、0.397 是 dgeo val | 三个数字不可横比；「val_acc 涨、胜率跌」的叙事被削弱 |
| **模型初始化不受 `--seed` 控制** | `scripts/datafirst_bc_train.py:144` 先建模型，`scripts/train_bc_scale.py:614` 才 `torch.manual_seed` | A0/A0i 对比混入 init 方差；同 seed 重跑 init 不同 |
| **只保存最后一个 epoch** | A2 ep70 val_acc 0.884 best，ep149 0.866 final；认证用了 final | 过拟合 checkpoint 被认证；需要 `--save-best` |
| **eval seed 双倍偏移** | `scripts/eval_protocol.py:519,534,94`：`seeds=range(seed_start, seed_start+n)` 且 `base_seed=seed_start`，`to_match` 再加一次 | `--seed-start 900 --seeds 10` 实际引擎 seed 是 **1800–1809**；配对 A/B 仍有效，但报告里的 seed 标签错误 |

### 1.3 结论

- **A2 判负无效**：A0i 的 0.266/0.358/0.678 是「id 分支读到假 id + init 方差 + final epoch」的混合结果，
  不能推出「id embedding 有害」。
- **A0 仍是当前 4-deck 通用 BC 基线**；A0i_noid 的 dgeo 0.741 只说明「同一 trunk 在某些 deck 上可用」，
  ungeo 0.670 / dvalin 0.303 仍不如 A0，不能当通用底座。
- **id embedding 路线暂停，不是关闭**：等 exact int id + parity 通过后，用 2 个 deck（dgeo/dvalin）重做一次
  预注册 A/B；如果仍无增益，再正式归档。
- 现有 specialist（dgeo 0.880 / ungeo 0.909）不受影响：它们是引擎认证过的强度，不依赖 A2 的结论。

---

## 2. 七条线的结论卡片

### 2.1 卡牌不完全信息自博弈与价值校准（f01）

**核心判断：不是价值头没标定，是三种价值被当成一种用。**

| 量 | 现在的载体 | 正确用途 |
|---|---|---|
| `V^π(s)`（outcome 回归） | `train/model.py` 的 `value_head`，target = 整局胜负 | PG baseline；depth-1 + expert 基座的 frontier |
| `V^σ(β)`（搜索根价值） | **没有独立头**；`train/distill.py` 在 `search_value` 存在时把同一头 target 硬切过去 | depth≥2 / 网络基座时的 frontier |
| `Q(s,a)`（候选动作价值） | 没有独立头；DMC 把 policy logits 当 Q | 候选打分/裁剪；DMC |

- **两个消费端 bug（0 训练可修）**：
  1. `envs/cfr_probe.py:91-92` 把 identity 激活的价值头输出当概率并 `min(1,max(0,v))` 截断；
     150ep 头第一分箱 `mean_p=-0.0616`，负值全被压成 0 → 兄弟叶子排序被抹平 → regret matching 全 0。
  2. `train/resolver.py:95-105` 的 value 是 **P(当前行动者赢)**，`envs/cfr_probe.py:146` /
     `envs/cfr_search.py:207-214` 却按 **P(player 0 赢)** 解释；在行动者=1 的叶子符号反了。
- **belief 的实锤 bug**：`reps/belief.py:97-100` 的 `_sample_hidden_dice` 是**确定性轮转**，不是采样；
  dice 是最影响对手合法动作的隐藏量。学到的 belief 也没有接进 search（`train/cfr_solver.belief_root_particles`
  和 `envs/cfr_search._root_particles` 都不传 belief 参数）。
- **对手身份通道是死的**：`reps/observation_encoder.py` 支持 `opponent_tag`，但 `train/data.py:183` 和
  `agents/neural.py` 都不传 → 恒为 `"unknown"`。
- **D23 的「根价值区分度≈0」可能是测量坏了**：value 头在 `sa_all` val 上 AUC 0.845 / ECE 0.045；
  H3b 换价值前沿结果逐值相同，更像「所有负值被截断成同一个值」。
- **DMC 负结果不构成「DMC 不行」**：DouZero 是 γ=1 + 在线自博弈 + 千万局；我们是离线、只监督被走动作、
  γ^k 缩放、规模差 3–4 个数量级。C1（off-support argmax）最可能，C2（γ^k）次之；但**不建议重开 DMC 主线**。
- **自博弈采样**（固定池）：池 60% + 当前策略自博弈 20% + 专家 anchor 10% + 历史 ckpt 10%；
  未达 parity 的 deck 把自博弈/历史降到 0，先做 100% 池 + anchor。

**对计划的影响**：先修 `cfr_probe` 语义（P0，0 训练），再判断搜索是否真的死了；value 拆头/HL-Gauss
放到 P2，且只在 depth≥2 或基座换网络时做。

### 2.2 DAgger / 离线 RL 与分布偏移（f02）

- **DAgger 的定位**：BC→parity 的分布修复；定理单边，收敛到专家，**不能超专家**。
- **我们的特殊性**：脚本专家是确定性纯函数、标签 0.1–1ms，**不要**移植 SafeDAgger/Ensemble/Thrifty 的省查询机制；
  真正的成本是局数 + 训练。
- **特有失败模式**：模型走偏后专家规则命中率下降，落到 `_fallback_choice` 启发式兜底 →
  必须丢弃 `fallback=True` 标签，逐轮监控，>10% 停。这是 HG-DAgger「人接管」的对应物。
- **聚合必须按 batch 混**：DAgger 占 30–50%，轮次衰减 γ=0.7；把 12k 新样本追加进 334k 基座只占 3.7%，等于没做。
- **AWR 负结果是定义级的**：`scripts/train_bc_scale.py:500` 的 `adv = values - value.detach()`，
  `w=exp(β(G−V(s)))` **与动作无关** ⇒ 只是状态重加权 BC，无法改变状态内动作排序。
  β=1 在 150ep 降到 0.352（−0.156），β=2 更差；15ep ≈ BC。**永久归档，不再扫 β。**
- **Q 的失败是 Fujimoto 外推误差**：只监督被走动作（`logits.gather(teacher)`），无 CQL/双 Q/target net/BC 锚；
  `K̄=25.5, max=256`。唯一值得重开的形态是 **TD3+BC 离散版**（CE + BC 锚），一次实验判定。
- **先做 200 局零训练诊断**：`agree_expert_states − agree_model_states`；gap <3pp 就别做 DAgger，
  转去查特征/指标（A2 的另一支）。gap ≥5pp 才做。
- **配方**：DAgger 2–3 轮 → 达标即停（CI 下界 ≥ −0.02）→ PG 阶梯；**绝不在 PG 之后再做 DAgger**。
- **成本/功效**：每轮 300 局 ≈12.4k 查询、≈2,700 步；3 轮 900 局、≈37k 查询。
  「非劣 Δ≥−0.02」需 **≈1,400 配对局**；800 局在 Δ=0 时下界约 −0.026，过不了 −0.02。

**预注册 E1**：2 deck × {base, +900 局 DAgger, +900 局等量专家语料}，同种子 800–1,600 配对；
接受规则 `Δ(A−B)≥+2pp 且 CI 下界>0`，同时机制指标 `agree_model_states(A)−(B)≥+3pp`。约 11h CPU / 3.5h GPU。

### 2.3 League / population + specialist portfolio / router（f03）

- **固定 20 套脚本对手 → 做「约束最优回应」，不是 Nash。** 主效用 pooled，floor 是硬约束；
  训练 surrogate 用 `pooled − λ·floor_deficit` 或 `CVaR_3`，不要 raw min。
- **采样**：`0.7*PFSP(1−w)^2 + 0.2*uniform + 0.1*worst3`；每对手上限 15%、下限 2%；历史 checkpoint 只占 20–30%。
- **防漂移顺序**：replay mix + incumbent 非劣门禁 → root-anchor KL → 历史池 + PFSP；EWC 最后。
- **Router 只做 deck 级、按对手公开阵容硬路由**，不做逐决策 contextual bandit；
  每 cell ≥100 局，经验贝叶斯收缩 + LCB + holdout，收益门槛 +3pp，最差格不降 >3pp。
- **扩张先扩 deck**：dvalin / shenhe / electro 是下一批；SA 与 Skirk 系结构性劣势，排除。
- **评测**：固定 20 脚本锚 + 历史 checkpoint 池；fixed-anchor BT 为主，Nash average 只作 secondary。
- **citation hygiene**：PFSP 不是 Lanctot 2017（1709.04326 是 LOLA）；PSRO 是 1711.00832；
  AlphaStar 主论文是 Nature DOI 10.1038/s41586-019-1724-z，1902.01724 是 GECCO 视角文。

**预注册 E1**：扩 3 deck（18k 采集、≤8.4k 评测、<0.5 GPU·h）；**E2** checkpoint router（12k 评测、0 GPU）。

### 2.4 候选集 / 组合动作建模（f04）

- **CVPN 现状**：token mean-pool + 每个 option 独立 MLP 打分 = DeepSets 形式；候选之间没有交互。
  `TransformerCVPN` 有 option→token cross-attention，但仍无候选间 attention。
- **DouZero/Suphx 都是逐动作独立打分**，没有候选集 attention；这是 A4 最强的反证——候选交互不是必要条件。
- **先做 0 成本诊断**：top-2 margin、同 kind/同 target 但不同费用的近重复选项比例、A0 错误是否集中。
  歧义率 <5% 或错误不集中 → **直接停 A4**。
- **三种最小结构**：候选 self-attention（相对费用/支配/近重复）、context-candidate cross-attention（目标绑定）、
  pairwise relation（显式支配/替代，更省样本）。
- **置换等变四条**：参数共享、无位置编码/engine index、mask 贯穿 attention/pairwise/pooling、归一化用 valid 数。
- **训练-评测一致**：target 用语义 key 重映射，不只存位置索引；argmax tie-break 语义化；
  单样本 vs batch 一致；全 mask 行报错而不是 NaN。
- **骰子费用/多目标**：引擎已保证合法，BC 只需学「值不值得」；先补 post-cost dice 等特征，
  原子 joint option 打分保持 100% 覆盖；pointer/分解只在 K 爆炸或子集动作时用。
- **A2 假阳性归因**（f04 视角）：val_acc 指标错位、id shortcut、p0-only/both-sides、final epoch 过拟合；
  不是同局泄漏或列错位。
- **Probe**：冻结 A0 trunk，1 层 SAB + zero-init delta，1–2 epoch；screen 400 局/deck 需 ≥2/3 deck +6pp
  且无 deck ≤−2pp；confirm 2,000 局/deck 需 ≥2/3 deck +3pp 且 CI 下界 >0。

**对计划的影响**：A4 排 P3；先做 0 成本诊断，不要先训。

### 2.5 Train/eval 一致性、校准与评测口径（f05）

- **A2 根因已实锤**：fp16 破坏 definition 列；1 seed × 1 matchup × 48 决策探针中，card 实体行
  **1267/1277 = 99.2%** id 不一致，char 行 0/348；失配处 id bucket 100% 不同。
- **parity harness 三层**：
  1. P0 确定性：同 `(deck,opp,seed)` 跑两遍，比动作序列 + `sha256(player_view)`；
  2. P1 同状态双路编码：`include_training_targets=True/False`，token/option 必须逐位相同（该 flag 只影响标签）；
  3. P2/P3 序列化 + 跨进程/服务路径：从 replay 读回样本，与新进程/`neural:` 真跑的特征比。
- **分列容差**：one-hot 精确；definition 列先 `round(x*400000)` 转 int 再精确比；其余 fp16 `atol=1e-3`；
  mask 不作为等价性判据（当前全 True，由长度重建）。
- **硬不变量**：`len(option)==len(codes)==len(specs)`；`low_to_high != -1`；`isfinite`；definition id <400000；
  **语料 dtype 下 id 可逆**（fp16 当前 150/152 不满足，fp32 152/152 满足）。
- **分布级监控**：T/K 分位数、P(K=1)、hand token/决策、opponent token 占比、definition id 词表外比例
  （**当前 fp16 语料实测 63.8%**）、request_type 份额、fallback/illegal、value AUC/Brier、每列 PSI。
- **checkpoint 选择**：val_acc 与引擎胜率脱钩；改用 `val_ce` early stop + on-policy teacher agreement +
  200 局配对引擎筛选 + 冻结 cert 块；`--save-best` 必须加。
- **评测统计**：800 局仅判 ~6pp；5pp 需 20–28 seeds；「非劣 CI 下界 ≥−0.02」需 ~1,400 局。
  预注册、固定种子块、不看中间结果、null-pair 校准、split-half 复现。
- **6 步诊断顺序**：确定性 → 双路编码 → definition 列 → 分布 → 选择散点 → 认证；合计 ≈3.5 CPU·h。

**预注册**：E-PARITY-1（门禁，不是强度声明）+ E-SELECT-1（ep70 vs ep150，≥6pp 判定）；
**parity 未过前冻结所有新结构训练**。

### 2.6 Worst-matchup 靶向采集与对手条件化（f06）

- **先分层，不要把所有 floor 违规都当可修**：
  - 19 个有认证臂的 deck 里，严格可修 matchup 只有 **10 个**（expert Wilson low ≥45% 且 student point <45%）；
  - natlan（expert 0.237，18/20 老师也输）、beidou、chasca、skirk_ayaka 等 **不应进 BC 靶向采集**；
  - 验收应同时报绝对 floor 与「相对同 deck 专家的配对 Δ」。
- **H5 matchup 重加权失败有机制解释**：`scripts/train_bc_scale.py:308-349` 用 `store.value`（teacher 胜负）
  算权重，**抬高了「老师也输」的 matchup**；方向错了，不是强度没调好。
- **正确做法**：只用 student-vs-expert 的可修 gap 选目标；**改采集层构成**，不改损失权重；
  target 25–35% + uniform/anchor 65–75%；单 matchup ≤25%；每轮用 fresh seeds 更新。
- **对手条件化**：当前 `opponent_tag` 只有 7 个粗标签，20 套脚本全落到 `baseline_scripted`；
  `opponent_deck_id` 65 桶对 20 deck 只有 18 个唯一 id（3 套撞 57）。不要用 65 桶；
  用**公开阵容（19 个唯一阵容）embedding + FiLM**，zero-init，放在 `global_vec` 上。
- **数据密度是主因，条件化是第二杠杆**：20-deck 崩塌约 3/4 来自每 matchup 密度，1/4 来自任务数；
  容量不是瓶颈。
- **最小 pilot**：nahida × 1 target，1,400 采集局 + 960 评测局，2–3h；target Δ≥+10pp 且 CI low>0 才扩到
  3 deck 的 full E1（8,400 采集 + 5,840 评测，8–13h）。
- **不要**：10 seeds（n=20/matchup）选 target（真实 0.50 有 25% 概率显示 <45%）；只采失败 matchup；
  focal loss 当第一杠杆；PCGrad/GradNorm 先上。

### 2.7 代码级审计：store→engine 不一致（f07）

**按概率排序的三个点**：

1. **确定不一致（已实证）**：compact fp16 破坏 definition id（见 §1.1）。
2. **确定不一致（结构）**：BC 样本只来自 p0，评测双边；store `acting_player` 恒为 `(1,0)`，side=1 变 `(0,1)`。
3. **可疑但当前未触发**：store 不记录 encoder config / definition 列号；`definition_feature_indices()` 加载时重算，
   布局块同宽重命名/重排会静默读错列。A2 的 8 个 manifest fingerprint 与今天一致，所以不是本次原因。

**其他立即要修的**：

- **eval seed 双倍偏移**：`--seed-start 900 --seeds 10` 实际 1800–1809；配对仍有效，但「fresh seeds」表述错误。
- **模型初始化不受 `--seed` 控制**：先建模型、后 `manual_seed`；A/B 混入 init 方差。
- **`#t>0` 采样不播种**：`torch.multinomial` 用进程 RNG；任何温度采样评测不可配对复现。
- **`A0i_noid` 本身是干净的**（`id_buckets=0`，14/14 strict load，14 个 tensor 与 A0i 非 id tensor 逐位相等）；
  但 `load_state_dict_compatible` 对缺失的 6 个 id 参数默认**随机初始化**，只有 `--id-zero-init` 才是 no-op。
- **A2 的 val_acc 跨语料比较**：0.866 mixed val / 0.838 h4m val / 0.397 dgeo val，不能横比。
- **其他**：`--contestant-deck` 默认 SA（忘传就用错 deck）；replay fingerprint 不含 deck 牌表；
  engine lock 只记录不校验；`Store` 丢掉 `privileged_state`/`request_type`/`opponent_token_mask`/`codes`；
  `b_mask` 由标签非零推断，合法全 0 标签会被 mask 掉。

**parity harness**（f07 §3）：输入 `--replay-dir/--deck/--opponent/--seed/--checkpoint`；
先比 `player_view` digest；再按 `step_index` 配对；分类为 `CONTEXT_MISMATCH / STATE_MISMATCH /
FORMAT_FP16_ID / LAYOUT_DRIFT / FEATURE_DRIFT / ACTION_INDEX_DRIFT / MODEL_DRIFT / NUMERICAL`；
用例至少 dgeo_compact、dvalin_compact、jsonl_control、a0_control、a0i_vs_a0i_noid、seat_swap。

---

## 3. 交叉裁决：子报告之间的冲突与取舍

| 冲突/问题 | 子报告说法 | 裁决 |
|---|---|---|
| 400/800/1,400 局 | f02 说 1,400 才过 −0.02；f05 说 800 只判 ~6pp、5pp 需 20–28 seeds；f06 说 100 局/target 是筛选级 | **分档**：400=大效应筛选；800=≥6pp 筛选；1,400–1,600=非劣/4–5pp 确认；3pp 需 2,000–3,000 或更多。禁止用 400 做认证。 |
| DAgger vs 靶向数据 | f02 说 DAgger 先做 200 局诊断；f06 说 floor 可修集合只有 10 个，靶向数据更便宜 | **先做两个 0 成本诊断**：DAgger 一致率 gap + f06 的 teacher/student 分层。若可修集合 <5，靶向数据收益低，转 DAgger/PG；若 DAgger gap <3pp，转靶向数据/特征。 |
| 候选集 attention | f04 说不是当前瓶颈，先 0 成本诊断；f01 说搜索的候选裁剪是任意序 | **区分两个问题**：BC scorer 的候选交互（P3，先诊断）；搜索的候选裁剪（P2，必须改成非任意 + 门控覆盖）。不要混为一谈。 |
| floor 口径 | 项目 M4 要求「无单套 <45%」；f06 说多数 deck 老师也输 | **M4 改为双口径**：绝对 floor + 相对老师的配对 Δ；对 teacher-limited matchup 不要求绝对 45%，但要求「不低于老师 −ε」。需在 PLAN §4.3 明确。 |
| A2 id embedding | 项目现记录「A2 判负」；f05/f07 说实验无效 | **A2 改判「无效，待重做」**；A0 保持基线；id 路线暂停不关闭。 |
| p0-only | f03/f04/f07 都指出；现有 specialist 是 p0-only 训练 | **先量化再改**：`seat_swap` 看 side=1 的 argmax 翻转率；>10% 则采集层加 side 覆盖，否则只记录为已知偏差。不要先重采全部。 |
| 价值头「不校准」 | D23 说根价值区分度≈0；f01 说 AUC 0.845/ECE 0.045，测量有 bug | **先修 cfr_probe 语义再用历史日志重算**；在修复前不重开搜索，也不把 D23 当最终结论。 |
| league 目标 | f03 说约束最优回应，不是 Nash；老师建议 league/self-play | **一致**：固定脚本池上不做 Nash；league 只用于 specialist 防漂移和历史池，不作为主目标。 |

---

## 4. 实验队列（P0 → P3）

> 所有实验预注册：假设 / 臂 / 主指标（引擎配对胜率）/ 样本量 / 预算 / 成功与停止判据。
> 采集种子必须避开 `val_every=10` 的 val 残差（取模 10 余 1–9）；认证种子用未使用区间并记录实际 `match_seed`。

### P0（1–2 天，0 新训练；先做）

#### P0-A. Parity 门禁（最高优先级）

- **假设**：store 与 engine 的特征在同一 `(deck,opponent,seed,step)` 上逐元素一致；definition id 精确可逆。
- **做法**：新增 `envs/parity.py`、`scripts/parity_store_live.py`、`tests/test_feature_parity.py`；
  用例 dgeo_compact / dvalin_compact / jsonl_control / a0_control / a0i_vs_a0i_noid / seat_swap。
- **主指标**：definition 列 id 失配数 = 0；其余列在容差内；argmax 一致（jsonl 路径）。
- **预算**：0.3 CPU·h 建设；CI <1 min；20 局审计 ~5 min。
- **判定**：任一 id 失配/列超容差/不变量失败 = FAIL；FAIL 则**冻结所有新结构训练**，先修数据。
- **预期**：dgeo/dvalin compact 在第一个 hand-card token 上 `FORMAT_FP16_ID`；A0i logits 大偏移，
  A0i_noid 回到 A0 附近。这已足以否定 A2，不需要等重训。

#### P0-B. 修数据精度 + 重采干净语料

- **方案**（按优先级）：
  1. payload 增加 `token_def_ids`/`option_def_ids`（int32，无损）；或
  2. `--compact-dtype float32` 作为新默认；或
  3. 只对 definition 列走 fp32（per-field dtype）。
- **重采**：4 deck × 20 对手 × 20 seeds = **1,600 局**（~20 min）；可复用现有 `train.collect`，
  加 `--format compact --compact-dtype float32`（或新 int 列）。
- **同时**：`scripts/datafirst_bc_store.py` 的 `meta` 加 encoder config / 解析后的 def 列号 / deck 牌表 hash；
  加载时校验，不再重算。
- **判定**：152/152 卡 id 可逆；`definition id 词表外比例 ≈0`；parity PASS。
- **预算**：重采 20 min + store 重建 ~10 min。

#### P0-C. 修可复现性与评测口径

- `torch.manual_seed(config.seed)` 放到模型构造**之前**（`datafirst_bc_train.py`、`train_loop.py`、`pipeline.py`）。
- `--save-best`（按 `val_ce`，同时保存 last）；认证 ep70/best 与 final 两个 checkpoint。
- `#t>0` 用 `random.Random(seed)` 或 `torch.Generator`；`NeuralPolicy` 接收 seed。
- `eval_protocol` 的 `--seed-start` 只进 `seeds` 或只进 `base_seed`，不要两次相加；report 记录实际 `match_seed`。
- `neural:` 强制 `--contestant-deck`；report 写清 deck。
- 新增 `tests/test_eval_seed.py`、`tests/test_train_init.py`、`tests/test_id_embeddings.py` 的 fp16 用例。
- **判定**：同 seed 重跑 init/动作序列逐位一致；`--seed-start 900 --seeds 10` 实际 900–909。

#### P0-D. A2 重做（仅在 P0-A/B/C 之后）

- **臂**：A0（无 id）vs A0i（int id / fp32），dgeo + dvalin，2 个 init seed，150ep 或 early stop。
- **主指标**：同种子引擎配对 Δ（A0i − A0），800 局 screen + 1,600 局 confirm；同时报 store `val_ce`（不报 val_acc 作门禁）。
- **判定**：screen ≥ +3pp 且无 deck ≤−2pp → confirm；confirm CI 下界 >0 → id 路线重开；
  否则归档。**不要再用 mixed val_acc 横比。**
- **预算**：2 deck × 2 臂 × 150ep ≈ 4–6 GPU·h（A800）；重采 1,600 局已含在 P0-B。

#### P0-E. Seat 覆盖量化（0 新训练，先测再改）

- **做法**：`seat_swap` 用例 + 200 局 A/B：p0-only 训练 vs 奇偶 seed 交换 deck/opponent 采集。
- **主指标**：side=1 的 argmax 翻转率、side=1 vs side=0 的引擎胜率差。
- **判定**：翻转率 >10% 或 side 差 >5pp → 采集层加 `side`/`swap`，所有新语料双边覆盖；
  否则记录为已知偏差，不阻塞 P1。
- **预算**：200 局采集 + 800 局评测 ≈1h。

#### P0-F. 修 `cfr_probe` 价值语义（0 训练）

- identity → `p=(v+1)/2`；sigmoid → `p=v`；**不要先截断**，保留负值用于兄弟比较；
- `frontier_p0 = p_acting if who==0 else 1-p_acting`；
- 用历史搜索日志重算 root value spread，判断 D23 的「区分度≈0」是否测量假象。
- **判定**：修复后 action-value spread 显著 >0，且与胜负相关；否则搜索仍搁置。
- **注意**：这只修测量，不等于搜索会变强；搜索重开仍受 P2 门禁约束。

### P1（3–5 天；数据与组合）

#### P1-A. DAgger 诊断 + E1

- **Stage 0（200 局，0 训练）**：模型自走 vs 专家状态的一致率 gap；fallback 率。
- **Go 条件**：gap ≥3pp（理想 ≥5pp）且 fallback <10%。
- **E1**：2 deck（nahida、dual_mualani，或按 Stage 0 选）× {base, +900 DAgger, +900 等量专家}；
  DAgger 2–3 轮 × 300 局；batch 混合 40%；fallback 标签丢弃；同种子 1,600 配对。
- **判定**：`Δ(A−B)≥+2pp` 且 CI 下界>0，且机制指标 +3pp；否则归档 DAgger，转 P1-B/P1-C。
- **预算**：3,600 采集局 + 训练 ~7h CPU（GPU 24min）+ 6,400 评测局 ≈11h CPU。

#### P1-B. 靶向 worst-matchup pilot → full E1

- **Stage 0（0 新对局）**：用 f06 的分层表 + 50 fresh seeds 复核可修 target；可修集合 <3 → 停。
- **Pilot**：nahida × `superconduct_mika_attack`；treatment 300 target + 1,100 uniform；control 1,400 uniform；
  focused eval 960 局；判定 target Δ≥+10pp 且 CI low>0。
- **Full E1**：3 deck（nahida/mika/SA）× 4 target；8,400 采集 + 5,840 评测；判定 ≥2/4 target +10pp、
  floor 违规合计 −≥2、pooled CI low ≥−0.02。
- **禁止**：target-limited deck（natlan/beidou/chasca/skirk）；loss 层 `--matchup-weight deficit`。
- **预算**：pilot 2–3h；full 8–13h CPU、<1 GPU·h。

#### P1-C. Specialist 扩张（dvalin / shenhe / electro）

- **判据**：expert pooled ≥0.60、expert floor 违规 ≤3、base pooled ≥ expert−0.03、clean。
- **配方**：每 deck 3 轮 PG（现有 `run_single_deck_rl.sh`），每轮 2,000 局自博弈；每轮换新种子；
  20 seeds 认证；失败即停，不三 deck 一起加轮。
- **预算**：18k 采集局（4.5h）+ 9 个 1-epoch PG（CPU 1.5h / GPU <0.5h）+ 评测 2.4k–6k 局（1.6–4h）。
- **判定**：逐 deck `CI_low ≥ −0.02` 且点估计 >0；floor 违规不增；2/3 通过则扩下一批。

#### P1-D. Portfolio / router（0 新训练）

- **先做 0 成本**：用 `scripts/matchup_router.py` 在现有 arm 上做 even/odd seed 路由；
  若 routed floor 违规 ≤2，优先 portfolio，不补采。
- **Router**：deck 级、按公开阵容；每 cell ≥100 局；经验贝叶斯收缩 κ=20–50；LCB 选择；
  top-2 差距 <3pp 回退 pooled 最佳；holdout seeds；收益门槛 +3pp，最差格不降 >3pp。
- **判定**：routed 比最佳单 arm 高 ≥3pp 且最差格不降；否则不做 router。

#### P1-E. 公开阵容 FiLM（0 新采集）

- **做法**：`EncodedObservation` 加 `opponent_lineup`（3 个角色 id，从 `player_view` 取，公共信息）；
  CVPN `global_vec` 上加 zero-init FiLM；19–20 类 embedding。
- **A/B**：nahida + mika，同 store/epoch/seed；base vs FiLM；1,600 评测局。
- **判定**：≥1/2 deck floor 违规 −≥2 且 pooled CI low ≥−0.02；否则停条件化。
- **预算**：3–5h CPU + 1.6h 评测；info-leak 测试必须 PASS；不要用 65 桶 `opponent_deck_id`。

### P2（策略改进与搜索；P0/P1 之后）

#### P2-A. League / PFSP（specialist 防漂移）

- 每 deck 20–30% 局对历史 checkpoint；PFSP `(1−w)^2`；replay mix 20–30%；
  incumbent 非劣门禁；root-anchor KL 只在漂移出现后加。
- 预算：每 deck 每轮 2,000 局；只对已 PASS 的 specialist 开。

#### P2-B. Value 拆头 + 搜索重开门禁

- 三件事按顺序：
  1. `V^π`（outcome）保留给 PG baseline；
  2. `V^σ`（search）独立头，target 用搜索根值或 N 次 rollout 均值，Huber；HL-Gauss 只用于连续 search 值；
  3. `Q(s,a)` 若要重开，必须加 BC 锚/CQL 式支撑约束，且先做 C1/C2 诊断。
- **搜索重开门禁**（全部满足）：parity PASS；value AUC ≥0.6 且 within-matchup AUC ≥0.55；
  候选裁剪非任意（prior/top-k + 覆盖）；门控式覆盖（只在 value spread 足够时覆盖专家）；
  修复 dice 采样与 acting-player 符号。
- 预算：value 头 1 次训练；搜索阶梯 3,200 局；不含重采。

#### P2-C. Belief 正确接法

- 修 `_sample_hidden_dice` 为 iid 采样（按引擎掷骰分布/至少 uniform over 8 types，count 多重集）；
- 把 `belief_histogram`/`group_hand_probs`/`group_deck_probs` 接进 `belief_root_particles` 和
  `envs/cfr_search._root_particles`；
- 消融「belief 作为策略输入 vs 辅助头」：这是本项目自己的问题，文献没有直接消融答案。
- 预算：0 新采集；搜索重开后再评。

### P3（结构；只在 P0 诊断有信号时）

#### P3-A. 候选 self-attention probe

- 前置：P0 的 0 成本歧义诊断显示 top-2 margin/近重复选项与 A0 错误集中；否则停。
- 结构：冻结 A0 trunk，1 层 SAB + zero-init delta；1–2 epoch；置换/batch 独立/全 mask 测试。
- Screen：400 局/deck，≥2/3 deck +6pp 且无 deck ≤−2pp；Confirm：2,000 局/deck，≥2/3 deck +3pp 且 CI low>0。
- 预算：0 新采集；GPU 2–5 min / CPU 1–2h；评测 1.2k–6k 局。
- **失败判据**：store 涨、引擎不涨（A2 模式）→ 立即停，不加层/不训久。

#### P3-B. 决策记忆 / option attention / 辅助头

- 只做冻结 encoder 的 probe（GRU/option self-attn），不端到端；P3-A 通过后再排。

---

## 5. 代码修复清单（file-level）

| 优先级 | 文件 | 改动 | 原因 | 测试 |
|---|---|---|---|---|
| P0 | `train/data.py` | payload 加 `token_def_ids`/`option_def_ids`（int32）；`--compact-dtype` 默认 fp32 或 per-field dtype | fp16 破坏 definition id | `test_id_embeddings` fp16 用例 |
| P0 | `scripts/datafirst_bc_store.py` | store `meta` 加 encoder config/def 列号/deck hash；读入 int id 列 | 静默列漂移 | `test_bc_scale_store` |
| P0 | `scripts/datafirst_bc_train.py` | `manual_seed` 移到模型构造前；`--save-best`（val_ce）+ last | init 不可复现；final epoch 过拟合 | `test_train_init` |
| P0 | `scripts/train_bc_scale.py` | `manual_seed` 前移；保存 best；`#t` seed | 同上 | 同上 |
| P0 | `scripts/eval_protocol.py` | 修 seed 双倍偏移；记录 `match_seed`；强制 contestant deck；加 McNemar/paired bootstrap | 评测口径不可预测 | `test_eval_seed` |
| P0 | `agents/neural.py` | `NeuralPolicy` 接收 seed；`torch.Generator` 采样；语义 tie-break | 温度评测不可复现 | `test_neural_seed` |
| P0 | `envs/cfr_probe.py` | identity→p；acting→p0；不提前截断 | 根价值测量假象 | `test_cfr_probe_value` |
| P1 | `train/data.py` / `train/collect.py` | 加 `side`/`swap`；奇偶 seed 交换 deck/opponent | p0-only vs both-sides | `test_seat_coverage` |
| P1 | `reps/observation_encoder.py` / `train/data.py` | 传入 `opponent_tag`/公开阵容；`EncodedObservation` 加 `opponent_lineup` | 对手身份死特征 | `test_info_leak` + parity |
| P1 | `train/model.py` | `lineup_emb` + zero-init FiLM | 公开阵容条件化 | `test_lineup_film` |
| P1 | `reps/belief.py` | `_sample_hidden_dice` 改 iid；加 dice 头/权重 | 搜索粒子错误 | `test_belief_dice` |
| P2 | `train/cfr_solver.py` / `envs/cfr_search.py` | 接 belief 参数；非任意候选裁剪；门控覆盖 | 搜索重开前置 | `test_cfr_search` |
| P2 | `train/model.py` / `train/distill.py` | 独立 search value 头；Huber/HL-Gauss | 三种价值混用 | `test_value_calibration` |
| P3 | `train/model.py` | `CandidateSetBlock` + zero-init delta；`forward_all` | 候选集交互 | `test_candidate_set` |
| P3 | `scripts/datafirst_bc_train.py` | `--candidate-layers/--freeze-trunk` | probe | 同上 |

---

## 6. 决策记录草案（PLAN §0，追加不改写）

> 建议新增 **D25（2026-10-27，文献调研 + A2 复核）**：

**D25：A2 判负无效；先过 parity 门禁，再谈结构。**

- A2 的 compact 语料默认 fp16，把 `definition_id/400000` 压成不可逆假 id：152 张卡只剩 28 个不同 id，
  真实 bucket 142 个、交集 4 个；A2 的 id 分支训练/推理读两套索引。**A2 不能作为 id embedding 的负证据。**
- A2 还叠加：只采 p0/评测双边、模型 init 不受 seed 控制、val_acc 跨语料横比、只保存 final epoch、
  eval seed 双倍偏移。**A2 的实验设计整体不成立。**
- **决定**：
  1. A2 改判「无效，待重做」；A0（h4m e150）继续作为 4-deck 通用 BC 基线；id embedding 路线暂停不关闭。
  2. 在 parity 未 PASS 前，**冻结所有新结构训练**（A3/A4/A5、id、FiLM、candidate attention）。
  3. P0 修复：parity harness + exact int id/fp32 语料 + `--save-best` + seed 修复 + eval seed 修复。
  4. 重采 1,600 局干净语料（4 deck × 20 对手 × 20 seeds，fp32/int id），重建 store。
  5. A2 重做：dgeo/dvalin，A0 vs A0i，800 screen + 1,600 confirm；只看引擎配对 Δ。
- **M4 floor 口径补充**：同时报「绝对 floor」与「相对同 deck 脚本专家的配对 Δ」；
  teacher-limited matchup（expert Wilson low <45%）不要求绝对 45%，但要求不低于老师 −ε。
- **评测功效**：400 局只做筛选；800 局只判 ≥6pp；非劣/4–5pp 用 1,400–1,600 局；3pp 需 2,000–3,000 局。
- **下一步排产**：P0 → P1（DAgger 诊断、可修 matchup 靶向、specialist 扩张、router/FiLM）→
  P2（league/value/belief）→ P3（候选集/记忆）。

> 同时建议在 `AGENTS.md` 的「现状」里把 A2 一行改成「A2 实验无效（fp16 definition id 事故），
> 待 exact id + parity 后重做；A0 仍是基线」，并在 `reports/explore/a2_id_screening.md` 顶部加「已复核，
> 结论无效」的说明。按项目规范，**不改写历史，只追加更正**。

---

## 7. 对老师建议的逐条回应

| 老师建议 | 我们的结论 | 调整 |
|---|---|---|
| **P0 修 train/eval 一致性**：store vs engine 逐元素、`--save-best`、固定 harness | **完全同意，而且找到了具体根因**：fp16 破坏 definition id；另有 p0-only、init seed、eval seed、final epoch | P0-A/B/C 先行；parity 未过冻结新结构 |
| **P1 DAgger** | 同意，但它是 parity 修复算子，不是超专家；先做 200 局诊断，gap <3pp 就停 | P1-A；带等量专家语料对照；之后接 PG |
| **P1 specialist 组合 + router** | 同意；固定池做约束最优回应，不做 Nash；先扩 dvalin/shenhe/electro | P1-C/D；router 只做 deck 级、公开阵容、100 局/cell |
| **P1 靶向 worst-matchup + 对手条件化** | 同意，但先分层：只有 10 个严格可修 matchup；natlan 等老师也输，不要补 | P1-B/E；采集层混合，不做 loss 重加权；FiLM 用公开阵容 |
| **P2 league/self-play** | 只用于 specialist 防漂移/历史池，不作为主目标 | P2-A；PFSP + anchor + incumbent 门禁 |
| **P2 离线 RL/价值/搜索** | AWR 定义级无效；Q 是外推误差；搜索的「根价值≈0」可能是测量 bug | 先修 cfr_probe；value 拆头；搜索重开需 5 条门禁 |
| **P2 belief 作为策略输入** | 先修 dice 采样、把 belief 接进 search；输入 vs 辅助头做本仓消融 | P2-C；文献没有直接消融答案 |
| **P3 A3/A4/A5** | 候选集 attention 不是当前瓶颈；先 0 成本诊断 | P3-A；冻结 trunk probe；失败即停 |
| **文献调研方法** | 已按「读代码/复现细节」执行；纠正了 3 处引用错误 | 7 份子报告 + 本文 |

---

## 8. 子报告与关键参考

**子报告（按本文引用顺序）**

- `reports/explore/lit/f01_selfplay_value.md` — DouZero / Suphx / ReBeL / SoG，价值语义与 belief。
- `reports/explore/lit/f02_dagger_offline.md` — DAgger / AWR / IQL / CQL / TD3+BC / 分布偏移。
- `reports/explore/lit/f03_league_portfolio.md` — PFSP / PSRO / AlphaStar / specialist router。
- `reports/explore/lit/f04_set_actions.md` — Set Transformer / pointer / option-set attention。
- `reports/explore/lit/f05_train_eval.md` — parity harness / calibration / 评测统计。
- `reports/explore/lit/f06_targeted_data.md` — worst-matchup / 对手条件化 / 数据选择。
- `reports/explore/lit/f07_code_audit.md` — store→engine 代码审计 + parity 设计。

**关键一手文献（完整引用见子报告）**

- DouZero: arXiv:2106.06135；Suphx: arXiv:2003.13590；ReBeL: arXiv:2007.13544；
  Student of Games: arXiv:2112.03178；DeepStack: arXiv:1701.01724。
- DAgger: arXiv:1011.0686；AWR: arXiv:1910.00177；IQL: arXiv:2110.06169；CQL: arXiv:2006.04779；
  BCQ/Fujimoto: arXiv:1812.02900；TD3+BC: arXiv:2106.06860；Decision Transformer: arXiv:2106.01345。
- PSRO: arXiv:1711.00832；AlphaStar: Nature DOI 10.1038/s41586-019-1724-z；OpenAI Five: arXiv:1912.06680。
- Set Transformer: arXiv:1810.00825；DeepSets: arXiv:1703.06114；Pointer Networks: arXiv:1506.03134；
  Relation Networks: arXiv:1706.01427；AGILE: OpenReview `MljXVdp4A3N`。
- Calibration: Guo arXiv:1706.04599；Kuleshov arXiv:1807.00263；HL-Gauss arXiv:1806.04613；
  Farebrother arXiv:2403.03950。
- Evaluation: Henderson arXiv:1709.06560；Colas arXiv:1904.06979；Agarwal arXiv:2108.13264；
  Breck ML Test Score；Sculley 2015。
- Data selection: OHEM arXiv:1604.03540；PER arXiv:1511.05952；RHO-LOSS arXiv:2206.07137；
  PLR arXiv:2010.03934；DoReMi arXiv:2305.10429；FiLM arXiv:1709.07871；PCGrad arXiv:2001.06782。

**引用勘误（已在子报告核实）**

- PFSP 不是 Lanctot 2017 的 1709.04326（那是 LOLA）；PSRO 是 1711.00832。
- AlphaStar 主论文是 Nature DOI 10.1038/s41586-019-1724-z；1902.01724 是 GECCO 视角文。
- Mindermann 2022 是 arXiv:2206.07137（不是 2202.01365）。
- PCGrad 是 arXiv:2001.06782（不是 2005.10447）。
