# DAgger / on-policy expert correction / 离线 RL 与模仿学习分布偏移

> 2026-10-27。调研方式：读一手论文（arXiv abs/HTML 全文，优先读算法伪代码、定理与实验数字）+ 读本仓代码
> （`train/data.py`、`train/collect.py`、`envs/policy.py`、`envs/match.py`、`agents/neural.py`、
> `agents/scripted/agent.py`、`scripts/train_bc_scale.py`、`scripts/datafirst_bc_train.py`）+ 读 `reports/SPECIALISTS.md`
> 与 `reports/explore/plan_strength_2day_kaggle.md` 的既有结果。**未改代码、未跑训练/评测、未写 `data/`。**
> 标注「未核实」的地方是没有拿到一手材料直接确认的。

---

## 0. 一句话结论（可执行）

**在本项目里，DAgger 是「把 BC 从 −2~−8pp 拉到 parity」的分布修复算子，不是提升上限的算子**：
脚本专家是确定性纯函数、标签成本≈0，所以 SafeDAgger/ThriftyDAgger 那套「省查询」机制对我们没有价值，
真正的成本是**局数与训练算力**；而它的收益上界被「当前 Δ vs 同 deck 专家」硬性封住（目标 deck 约 5–8pp）。
因此正确做法是：**先花 200 局做零训练判别测量**（模型自走状态上的 expert 一致率 vs 专家状态上的一致率），
gap ≥ 5pp 才做 DAgger；做的时候**必须带「同等局数 expert-vs-expert」对照臂**（否则分不清「分布修复」与「更多数据」，
而后者在我们这儿已经把 6 个 deck 拉到 parity 了）；DAgger 达标即停，**之后接已验证的 PG 阶梯**去超专家，
**绝不能在 PG 之后再做 DAgger**（那等于把 PG 的改进按 expert 标签抹回去）。

---

## 1. 与我们问题的对应关系

### 1.1 Q1：我们这套环境里 DAgger 的正确接法

#### 1.1.1 我们和论文设定有 3 个结构性差异（决定了接法）

| 维度 | 论文（DAgger 系列） | 我们 |
|---|---|---|
| 专家查询成本 | 人类遥操作，每次查询 $1–10s，是**主要瓶颈** | `expert:<deck>` 是确定性脚本纯函数（`ExpertRuleAgent.choose`，`self.random` 只在 `__init__` 赋值、规则里未使用 → 同状态同标签），**≈0.1–1ms/次**；一局 42 个决策的标签成本 <50ms |
| 专家质量 | 假设近最优 $\pi^\*$ | 脚本规则，对池胜率 0.19–0.85；且**它自己也在 20 个对手里**（完美克隆 ⇒ 对该对手 ~50%，对池 = 专家自己的 pooled） |
| 标签合法性 | 观测即状态 | 标签必须与 actor 同源：`build_decision_context` 的枚举走 `visible_state=player_view`（`reps/action_adapter.py:132`），专家与网络看到的都是掩码视图 → **DAgger 标签不引入信息泄漏**（C10 不破） |
| 目标 | 追上专家 | 先追平、再靠 PG/自博弈超过（见 §1.3） |

**结论 1**：不要移植查询预算机制（SafeDAgger 的 safety policy、EnsembleDAgger 的 ensemble 方差、ThriftyDAgger 的
novelty+risk 门控）。它们解决的是「查询贵」，我们查询免费。把工程量花在**局数、聚合权重、标签门控**上。
（如果之后要省**引擎时间**而不是省标签，可以再考虑门控——但引擎时间本来就是采集预算的主要项，门控只会减少数据。）

**结论 2**：因为专家是确定性函数，`β_i`（执行专家动作的概率）在我们这里**只影响状态分布，不影响标签覆盖**。
所以 β 的作用从「让专家接管以避免走得太偏」变成「控制采样状态的分布」，可以设得很小甚至 0。

#### 1.1.2 采集协议（推荐默认值）

```
第 r 轮：
  对手  = 20 套脚本池（与 eval 同分布）
  局数  = 300（20 对手 × 15 seeds，seeds 与 eval/val 都不重叠）
  温度  = r=1: 一半 T=1.0 / 一半 T=0（贪心）；r≥2: 全 T=0
  β     = r=1: 0.3；r≥2: 0
  标签  = expert:<同 deck>，对**每个** p0 决策都取（不抽样、不选点）
  门控  = 丢弃 (a) `fallback=True` 的标签；(b) 只有 1 个合法选项的决策；(c) 标签不在编码选项内
  记录  = observation / 执行动作 index / 标签 index / fallback / 选项数 K / 网络熵 / K 内 top1
```

为什么 `r≥2` 用贪心而不是采样：**部署是贪心**（`NeuralPolicy.temperature=0.0`，`agents/neural.py:56-57`），
DAgger 要修的是「部署分布」上的损失 $E_{s\sim d_{\pi_\theta}}[\ell(s,\pi_\theta)]$，所以采样分布应当**就是贪心分布**。
贪心 + 固定种子 ⇒ 每局确定，300 个种子就是 300 条确定的部署轨迹，没有采样噪声浪费。
`r=1` 保留一半 $T=1$ 是为了覆盖「网络不确定时的走法」（否则贪心会在同一个错误上反复循环，见 DAgger 论文里
Mario 的 stuck-at-obstacle 现象，以及 β=0.5 优于 β=I(i=1) 的实测）。

#### 1.1.3 聚合的 β 混合：**按 batch 混合，不按语料混合**

这是最容易做错的一点。我们已有 334k 样本的专家语料（`dfbc_sa_store.json`），
一轮 300 局 ≈ 12.4k 决策（p0 侧实测 41.5 决策/局 = 334,734/8,061；含双方约 46）。如果把 DAgger 数据**追加**进语料，
新数据占比 3.7%，**几乎不可能对损失产生影响**。

推荐：把「聚合」实现为**每个 epoch 的采样比例**，而不是数据集拼接：

$$
\mathcal{L}=\sum_{i=0}^{R} \gamma^{\,R-i}\cdot \frac{1}{|\mathcal{D}_i|}\sum_{(s,a)\in\mathcal{D}_i}\mathrm{CE}\big(\pi_\theta(s),a^\*_E(s)\big),
\qquad \gamma\in\{0.5,0.7\}
$$

工程上等价于：每个 epoch 从 (base, r1, r2, r3) 四个 store 按权重抽固定配额拼 `order`（bs=128），
其中 **DAgger 总量占 batch 的 30–50%**，且 r3 > r2 > r1。

- 理论依据：DAgger 要求 $\bar\beta_N=\frac1N\sum\beta_i\to0$（论文 §3），原版用 $\beta_i=p^{i-1}$ 控制**状态**分布；
  我们用 $\gamma^{R-i}$ 控制**损失**分布，是对偶的同一件事。均匀聚合（不做衰减）在「基座语料远大于每轮新数据」时
  会退化成「基座语料 + 噪声」，这正好解释为什么很多团队照抄 DAgger 无效。
- 不要用 `--train-limit` 随机截断来「平衡」，那会破坏 game-level 去重与确定性（`train_bc_scale._plan_budgets` 的排序保证）。

#### 1.1.4 查询哪些决策（全量 + 3 个门控 + 1 个可选加权）

我们的默认是**全量查询**（免费）。但下列四类要区别对待，否则会**主动引入噪声**：

1. **`fallback=True` 的标签要丢**。`ExpertRuleAgent._fallback_choice()`（`agents/scripted/agent.py:869`）
   在规则全部不命中时按 `_fallback_action_score` 打分，这是「启发式兜底」，不是脚本策略本身。
   **在模型自走的状态上，规则命中率会下降**——这是 DAgger 在本项目的**特有失败模式**：
   越采越偏 ⇒ 兜底越多 ⇒ 标签退化成 heuristic ⇒ DAgger 变成「蒸馏一个笨启发式」。
   **所以 fallback 率必须逐轮记录，>10% 立刻停**（§1.1.6 的 `fallback_rate`）。
   这等价于 HG-DAgger 的「人只在必要时接管」在这里的自然对应物：**不接管，但丢弃不可信标签**。
2. **单一合法选项**的决策没有梯度信号（CE 恒 0），丢掉省内存。
3. **`reroll_dice` / `switch_hands` / `select_card`** 与 `action` 分开统计一致率。若某类一致率 >0.99，
   它对策略几乎无贡献，可以降权（0.3×）以把容量让给 `action`。
4. **可选加权**：按网络熵/间隔加权（把梯度集中在「网络不确定」处），这是 EnsembleDAgger/ThriftyDAgger 的
   「不确定性查询」在免费标签下的正确用法——不是省查询，而是**重加权**。要先验证「熵能预测不一致」
   （见 §3 R1 的 AUROC），不能预测就别加。

#### 1.1.5 「防止专家标签把模型拉回专家上限」——能防的和不能防的

先把话说死：**DAgger 的不动点就是专家在访问分布上的策略**。论文 Theorem 3.2 是单边界
$J(\hat\pi)\le J(\pi^\*)+uT\epsilon_N+O(1)$，它保证不比专家差，**从不保证更好**。所以：

- **不能防**「DAgger 训练到收敛 ⇒ 行为≈专家」。这是定义决定的，不是调参问题。
- **能防**的是「把已经超过专家的行为抹掉」。具体三条：
  1. **顺序锁死：DAgger 只在 PG 之前做**。PG 阶梯（REINFORCE+baseline，`--loss pg`）的收益正是「偏离专家」；
     在 `rl_dgeo3`（0.880）上再跑一轮 DAgger，会把 16.8pp 的偏离按专家标签拉回去。写进流程，不做「定期回归 DAgger」。
  2. **只做到 parity 就停**（预注册门槛：配对 Δ 的 CI 下界 ≥ −0.02），不要「练到一致率最高」。
     `reports/explore/plan_strength_2day_kaggle.md` 已经证明**模仿保真度与强度脱钩**（val_acc 0.764→0.895，pooled 0.476→0.429），
     所以「一致率高」不是目标函数。
  3. **保留项（retention）只能用在「已认证的偏离」上**：模型动作 ≠ 专家、且该 deck 的这一臂已在**独立种子**上认证过更好
     （例如 dgeo 的 PG 臂），才对这批状态加自我蒸馏项 $\mathrm{CE}(\pi_\theta,\mathrm{sg}(\pi_\theta^{old}))$；
     **不要**用「这局赢了」当保留条件——胜局筛选已实测 −0.14（`plan_strength_2day_kaggle.md` §Day2b）。
- **不要做 soft label**：`ExpertRuleAgent` 是规则级联，只输出单个 `RuleChoice`，没有对全部合法选项的分数分布
  （只有 `_fallback_choice` 会给每个选项打分）。要做 soft label 得改 20 套 deck 规则，性价比极低。
  退而求其次：对**一致**的决策用 label smoothing 0.05，对不一致的用 one-hot。

#### 1.1.6 落到本仓的具体接口改动

**最小改动集（按文件）**

| 文件 | 改动 | 说明 |
|---|---|---|
| `train/data.py` | 新增 `collect_dagger_samples(deck, opponent, seeds, *, actor_spec, label_spec, temperature, beta, gate, encoder)`，与 `collect_search_samples` 并列；`Sample` 增加 4 个**可选**字段 `behaviour_index: int\|None`、`label_fallback: bool=False`、`actor_entropy: float\|None`、`source: int=0`；`sample_to_payload/from_payload` 仅在存在时写出（旧 replay 完全兼容，做法照抄 `policy_target`） | 核心采集器 |
| `envs/policy.py` | `ScriptedPolicy` 增加 `last_choice: RuleChoice\|None`（`choose` 里存一下）；`_neural_policy` 加 `@lru_cache`（key=`(path,temp,device)`） | 拿 `fallback` 标志；避免每局重载 ckpt（500 局 × 0.3–1s = 3–8 分钟纯浪费） |
| `train/collect.py` | 新增 `DaggerSpec` dataclass（`actor_spec/temperature/beta/label_spec/gate/value_from`），`CollectSpec.dagger` 参与 `as_dict()` → 自动进 manifest 指纹；`run_collect_task` 增加 `if task.dagger: return run_dagger_collect_task(task)`；CLI `--dagger-from/--dagger-temp/--dagger-beta/--dagger-gate/--label-spec` | 复用现有可续跑/分片/指纹护栏 |
| `envs/rollout.py` | **不改**（`CollectTask` 已够用） | — |
| `scripts/train_bc_scale.py` | `Store` 增加 `source: list[int]`；`_replay_record` 读 `payload.get("source",0)`；`--source-weights "1.0,0.7,0.5"` 用于构造每 epoch 的 `order`（替换 `rng.shuffle(order0)`）；`--init-from` 复用 `train.model.load_state_dict_compatible` | 轮次加权 + 微调 |
| `scripts/datafirst_bc_train.py` | 不动（已有 `--init-from`、`--loss ce/awr/q/pg`、`--matchup-weight`） | 单 store 快速实验 |
| 新增 `scripts/dagger_diag.py` | 读一轮 DAgger replay，输出：per-deck 一致率、per-request-type 一致率、fallback 率、熵/间隔 → 不一致的 AUROC、K 分布 | 判别测量，`§3 R1` |

**两个必须注意的坑**

1. **val 种子残差**：`train_bc_scale.load_store` 用 `(seed - seed_start) % val_every == 0` 划 val。
   现有 store 是 `seed_start=0, val_every=10` ⇒ **所有 10 的倍数都是 val**。
   DAgger 采集若用 `--seed-start 12000 --seed-count 20`，其中 12000/12010 会落进 val ⇒ **训练集污染**。
   ⇒ **DAgger 种子一律取模 10 余 1–9**（如 12001–12019），或显式给 DAgger 目录单独 `--val-every 0` 再在合并时按来源划。
2. **PG 需要「执行动作」而不是「标签」**：`run_batch` 的 `loss=pg` 注释明确写「teacher is the action the *sampling* policy took」
   （`scripts/train_bc_scale.py:511`）。DAgger 一轮同时产生「执行动作」和「专家标签」，
   ⇒ `behaviour_index` 让 **一份采集两用**：CE 用 `teacher_index`（专家），PG 用 `behaviour_index`（旧策略动作）。
   这是把 DAgger 与 PG 串起来的关键 3 行改动，能省掉一轮 PG 采集（~800 局）。

**没必要的改动**：`fork_search`/`cfr_search`/belief 头不用动；`reps/` 不用动；信息隐藏测试不受影响
（actor 与 labeler 都只吃 `player_view`）。

---

### 1.2 Q2：为什么 AWR / Q(DMC) 在本项目全是负结果（可检验归因）

先把最重要的一个**从代码里直接读出来的结论**摆在前面：

> **我们的 AWR 权重与「动作」无关：$w_t=e^{\beta(G_g-V(s_t))}$ 只依赖整局胜负 $G_g$ 与该状态的 $V(s_t)$，
> 在同一状态下对所有候选动作完全相同。所以它在数学上只是一个「按状态重加权 + 梯度缩放」的 BC，
> 无法改变状态内的动作排序。**

`scripts/train_bc_scale.py:500`：`adv = values - value.detach()`，其中 `values` 来自 `store.value`
= `payload["value_target"]` = **整局胜负**（`train/data.py:176` `value = outcome_score(record.winner, 0)`；
一局约 42 个决策共享同一个 $G_g$），`value` 是模型自己的 $V(s_t)$。梯度是

$$\nabla_\theta \mathcal{L}=\sum_{t} w_t\,\nabla_\theta \mathrm{CE}(s_t,a_t),\qquad w_t=e^{\beta(G_g-V(s_t))}$$

$w_t$ **不依赖 $a_t$** ⇒ 它改变「哪些状态被强调 / 步长多大」，不改变「某状态下该选哪个动作」。
（$G_g$ 在局内是常数，$V(s_t)$ 在局内变化有限，所以 $w_t$ 近似局级常数——但精确的说法是「动作无关」，
这个性质足以推出下面的结论。）
所以 AWR 最多等于「BC + 状态重加权」，而这正是 H5 `--matchup-weight` 做的事，**已实测显著负**（SA −0.091）。
更糟的是 $\beta$ 大时 $w$ 把有效数据压到少数胜局上（β=2 权重上限 6.04），退化成「胜局筛选 BC」（−0.14）。
这完美对应实测：15ep 时 AWR≈BC（0.475 vs 0.476），150ep 时随 β 单调恶化（β=1 → 0.352，−0.156）。

| # | 归因 | 我们的证据 | 判别实验 | 预测 |
|---|---|---|---|---|
| A1 | **AWR 的权与动作无关 ⇒ 只是状态重加权**（上面） | 代码 + β 单调恶化 + H5/胜局筛选同族已负 | **权重置换对照**：把 $w_t$ 在局间随机打乱（保持分布），其余逐值不变 | AWR ≈ AWR-shuffled ⇒ $(G,V)$ 的信息量为零，AWR 永久归档（不是「没调好」，是定义上做不到） |
| A2 | **Q 的支撑约束缺失 = 外推误差**（Fujimoto 2019 的核心结论） | `q_played = logits.gather(teacher)` 只监督被走动作（`train_bc_scale.py:491`），无 CQL 惩罚/无双 Q/无 target net/无 BC 项；q_mse 最好（0.0609）但 pooled 0.275、与老师一致率 0.187 | 训练后统计 `mean Q(未走合法动作) − Q(被走动作)`；再做离散版 TD3+BC：$\mathcal{L}=\mathrm{CE}(\pi_\theta,a_{beh})+\lambda\cdot(-\log\sum_a e^{Q(s,a)})$ 或直接 $\mathrm{CE}+\alpha\,\mathrm{MSE}(Q_\theta,\text{target})$ + BC 锚 | Q 在未走动作上系统性高估（差值 >0 且随 K 增大）；加支撑约束后 pooled 从 0.27 → ≥0.45 |
| A3 | **动作候选太宽**：$\bar K=25.5$、$\max K=256$（`reports/train/a3_bc_v1.json`），每状态只观测 1 个动作 ⇒ 单次访问只覆盖 1/K ≈ 4% 的 (s,a) 格 | 同上 | 按 K 分桶看 argmax 命中率/Δ；把评测限制在 K≤10 的决策上看 Q 臂是否恢复 | 大 K 桶贡献了绝大部分错误；K≤10 上 Q 明显没那么差 |
| A4 | **奖励极稀疏 + 信用分配**：整局 ±1，42 决策共享；要分辨「两个动作差 2pp」需 ~2,400 局/状态-动作对（$ (1.96/0.02)^2\times0.25$），不可能 | 价值头 AUC 0.84、ECE 0.027（排序可以，尺度也够），但**决策级优势**没有可估的信号 | 把优势换成 `search_value`（已有字段）或 paired-seed 优势，看 AWR 是否翻正 | 仍然 ≤BC：因为 `search_value` 的根价值区分度 ≈0（D23 已测） |
| A5 | **训练/评测口径不一致**（不一定是特征 bug） | A2：store val_acc 0.866 / 引擎 0.266；更普遍地 val_acc 0.764→0.895 而 pooled 0.476→0.429（**负相关**） | 见 §4 E1 的 Gate-0：在**模型自走状态**上重算同一个一致率指标 | 若一致率从 ~0.85 掉到 ~0.6，A5 = 协变量偏移（DAgger 可修）；若仍是 ~0.85 而胜率低，**A5 不是偏移，是指标/特征问题**，DAgger 无用 |
| A6 | **DMC 的前提我们都不满足**（DouZero 是**在线自博弈 + 千万局 + 全回合 MC 回报**） | 我们 8k–15k 局离线固定数据，差 2–3 个数量级；且 DMC 有 3 个 bidding/playing 网络共享结构 | 不需要实验：属于「单位不一致」的比较 | — |

**一句话**：AWR 的负结果是**定义级**的（权重与动作无关 ⇒ 只是状态重加权），Q 的负结果是**文献已预言**的（无支撑约束的 argmax），
两者都**不构成「离线 RL 在我们这儿不行」的证据**。真正没被否定的离线变体只有「带支撑约束的 Q」
（TD3+BC/CQL 式），见 §4 E2。

---

### 1.3 Q3：DAgger 之后接什么才能超过专家（组合配方）

一句话：**DAgger 只负责把底座送到 parity，超专家只能靠「用结果奖励优化」的那一族**；本项目的实测已经指出是哪一族。

配方（顺序不可换）：

```
① 稠密专家语料 BC（已有）                      → Δ ≈ −8…0pp
② DAgger 2–3 轮（本文档）                       → Δ ≈ −2…+1pp（parity）
③ on-policy PG（已有 --loss pg，REINFORCE+value baseline）
     仅对 parity+ 的 deck 开；每轮换新种子；离线锚混 30% batch 防漂移  → Δ ≈ +3…+17pp
④（可选）自博弈扩展对手池 / 剥削式定向   → 只在 ③ 饱和后再开
```

依据（全部来自本项目已有数据，不是论文推断）：

- **③ 是唯一被复现的「超专家」机制**：dgeo +16.8pp / unyielding +5.5pp，两套独立种子、800 配对局、完整门禁 PASS
  （`reports/SPECIALISTS.md`；`plan_strength_2day_kaggle.md` §2026-10-26）。
- **③ 的前提是 ②**：PG 只在「BC 已 ≥ 老师」的 deck 上有效；落后 deck 上 PG 显著退化（dvalin −0.063→−0.128，SA −0.058→−0.078）。
  这正是 AWAC/SQIL 那条「离线初始化质量决定在线微调能否奏效」的经验规律
  （[AWAC](https://arxiv.org/abs/2006.09359)、[SQIL](https://arxiv.org/abs/1905.11108)）。
- **③ 之后的「再 DAgger」是负向操作**（见 §1.1.5）。
- **不要指望 filtered BC / %BC 帮我们超专家**：%BC = 按**整局回报**取 top X% 再 BC（[DT 论文 §5 定义](https://arxiv.org/abs/2106.01345)）。
  我们的回报是「同一策略对不同对手的运气」，不是「策略质量」——胜局筛选 −0.14 已经把这个机制否掉了。
  要过滤就用**配对种子**的优势（同种子、同对手、两臂对打），不是原始 return。

---

### 1.4 Q4：成本与功效估算

**测量口径的现有统计量（实测锚点，来自 `reports/SPECIALISTS.md`）**

| 配对数 | 观测 95% CI 半宽 | 推出 SE(Δ) | 80% power 可检出 Δ | Δ=0 时的 95% 单边非劣下界（margin −0.02 需 ≥−0.02） |
|---|---|---|---|---|
| 400（10 seeds） | ±0.052 | 0.0265 | **7.4pp** | −0.044 ⇒ **不可能过** |
| 800（20 seeds） | ±0.031 | 0.0158 | 4.4pp | −0.026 ⇒ 也过不了 |
| 1,400 | ±0.023 | 0.0120 | 3.4pp | **−0.0197 ⇒ 刚好过** |
| 1,600 | ±0.022 | 0.0112 | 3.1pp | −0.018 |
| 3,200 | ±0.016 | 0.0079 | 2.2pp | −0.013 |

> 结论：**「非劣」门禁（CI 下界 ≥ −0.02）需要 ≈1,400 配对局**。我们现在用 400 局做筛选，
> 在真实 Δ=0 时下界就是 −0.044，必然 FAIL——这是统计功效问题，不是策略崩。
> 若要比「两臂之差」（同种子配对、共同随机数），方差更小，约 0.7×（经验值，未核实），
> 1,200–1,600 局可检出 2–3pp 的臂间差。

**一致率指标（机制终点）的功效**：42 决策/局，局内相关 ICC=ρ，设计效应 $1+41\rho$：

| 局数 | 决策数 | $n_{eff}$(ρ=0.1) | $n_{eff}$(ρ=0.3) | SE @ A=0.7 (ρ=0.1 / ρ=0.3) |
|---|---|---|---|---|
| 100 | 4,200 | 824 | 316 | 1.6pp / 2.6pp |
| 200 | 8,400 | 1,647 | 632 | **1.1pp / 1.8pp** |
| 400 | 16,800 | 3,294 | 1,263 | 0.8pp / 1.3pp |

⇒ **每轮 200–400 局就够把「一致率」测到 ±1–2pp**，足以看到「三轮把 gap 修掉一半」这种量级的效应。

**每轮的真实成本（按本机实测吞吐）**

| 项 | 数量 | 折算 |
|---|---|---|
| 采集 | 300–400 局（20 对手 × 15–20 seeds） | 3,000–4,900 局/h ⇒ **6–10 分钟**（含网络推理开销） |
| 专家标签 | ≈13k–17k 次 `choose` | **<20 秒 CPU**，可忽略（这是与机器人 DAgger 的本质区别） |
| 训练 | 3 epoch × (100k base + 14k DAgger) ≈ 114k 样本/epoch ≈ **2,700 步**（batch 内 DAgger 占 40%） | CPU 150–300 样本/s ⇒ **30–40 分钟**；T4（9.2k/s）⇒ **1–2 分钟** |
| 评测 | 800 配对局 | **20–30 分钟** |

⇒ **3 轮 DAgger + 2 次评测 ≈ 3 小时（纯 CPU）/ 1.3 小时（GPU 训练）per deck**。
⇒ 要把 covariate shift「显著降低」（定义：模型状态一致率 +5pp，且引擎 Δ 改善 ≥2pp）：
**3 轮 × 300 局 = 900 局、≈37k 专家查询、≈8,400 训练步**。

---

## 2. annotated bibliography

### 2.1 DAgger 家族

**1. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning（DAgger）**
Ross, Gordon, Bagnell, AISTATS 2011 — https://arxiv.org/abs/1011.0686 ；PDF https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
- 问题：BC 在自身诱导分布上误差累积，$J(\pi)\le J(\pi^\*)+T^2\epsilon$（Theorem 2.1，引自 Ross & Bagnell 2010）。
- 方法：Algorithm 3.1 —— 每轮用 $\pi_i=\beta_i\pi^\*+(1-\beta_i)\hat\pi_i$ 采样 $m$ 条轨迹，**在访问到的状态上问专家**，
  聚合 $\mathcal{D}\leftarrow\mathcal{D}\cup\mathcal{D}_i$，再在整个 $\mathcal{D}$ 上重训。
  理论：Lemma 4.1 $\|d_{\pi_i}-d_{\hat\pi_i}\|_1\le 2T\beta_i$；Theorem 3.1/3.2：$N=\tilde O(T)$ 时
  $J(\hat\pi)\le J(\pi^\*)+uT\epsilon_N+O(1)$；Theorem 3.3/4.2 有限样本 $mN$ 需 $O(T^2\log(1/\delta))$。
  $\beta_i$ 只需 $\bar\beta_N\to0$；**实际最好的是 $\beta_i=\mathbb{I}(i=1)$**，Mario 上 $0.5^{i-1}$ 略好（3030 vs 2980 距离）。
- 证据：Super Tux Kart（1 圈/轮 × 20 轮，~1000 点/轮；15 轮后 0 次掉赛道，BC 完全不改善）、
  Super Mario Bros（5000 点/轮 × 20 轮）、手写 OCR（83.6%→85.5%）。
- 局限：专家必须「处处近最优」；$\beta$ 与轮数的选择无原则；连续控制为主。
- 可迁移点：**$N=\tilde O(T)$，我们 $T\approx42$ ⇒ 理论上几十轮，实践 3–6 轮足够**；
  「BC 加数据不改善、加状态才改善」这个现象在我们这儿会同样出现（我们已有 val_acc↑/胜率↓ 的证据）。
- **关键警告**：定理是**单边**的，上限是专家。任何「DAgger 超专家」的说法都是误读。

**2. Efficient Reductions for Imitation Learning**
Ross & Bagnell, AISTATS 2010 — https://proceedings.mlr.press/v9/ross10a.html
- 问题：序列决策下 i.i.d. 假设失效。方法：Forward Training（每步一个策略）与 SMILe（策略混合，$\alpha$ 递减）。
- 证据/结论：BC 的 $T^2\epsilon$ 是**紧的**（构造性反例）；forward/SMILe 可到 $T\epsilon$。
- 可迁移点：解释了为什么「更多专家数据」不改善——数据仍在 $d_{\pi^\*}$ 上。我们的 val_acc 0.895 而 pooled 0.453 就是这句话的实证。

**3. DART: Noise Injection for Robust Imitation Learning**
Laskey, Lee, Fox, Dragan, Goldberg, CoRL 2017 — https://arxiv.org/abs/1703.09327 ；代码 https://github.com/BerkeleyAutomation/DART
- 方法：**往专家动作里注噪**（不是让机器人自己走），专家被迫示范恢复动作；用 KL 上界
  $|E_{p(\pi^\*,\psi)}J-E_{p(\pi_\theta)}J|\le T\sqrt{\tfrac12 D_{KL}}$ 优化噪声协方差
  $\Sigma^\alpha_{k+1}=\frac{\alpha}{T\,\mathrm{tr}\hat\Sigma}\hat\Sigma$，$\alpha$ 是「预期最终误差」的标定旋钮。
- 证据：MuJoCo 四任务与 DAgger 打平，但**省 3× 计算**；Humanoid 上专家累计回报只掉 5%（DAgger 掉 80%）；
  真机抓取 BC 49% → DART 79%（$\alpha=3$），$\alpha=6$ 反而 72%（噪声过大）。
- 局限：需要标定 $\alpha$；高维图像上噪声方向难定；专家必须是「能对扰动做出反应」的（人类可以，脚本不行）。
- 可迁移点：**我们的「噪声」= 网络的 temperature/采样**。$\alpha$ 的对应物是「模型与专家的动作差异度」，
  可以逐轮测量并**把 temperature 调到差异度匹配上轮的实测值**（而不是拍 1.0）。这是我们的 `T` 调度唯一有原则的定法。

**4. Query-Efficient Imitation Learning for End-to-End Autonomous Driving（SafeDAgger）**
Zhang & Cho, AAAI 2017 — https://arxiv.org/abs/1605.06450
- 方法：额外训一个 **safety policy** 预测「primary 会不会犯错」，只对「不安全」的状态问专家（$\beta_i\pi^\*+(1-\beta_i)\pi_{i-1}$ 的改进版）。
- 证据：赛车模拟器上查询更少、无碰撞圈数更多、损伤更小；作者归因于「按 safety 选子集 ⇒ 自动课程」。
- 局限：safety policy 本身要标注；阈值手调；只在单一驾驶域验证。
- 可迁移点：**safety policy ≡ 我们的「网络熵/间隔 → 是否会与专家不一致」预测器**。
  在免费标签下不要用它省查询，用它**重加权**（§1.1.4 第 4 条）。要先验证 AUROC，否则别加（见 §3 R1）。

**5. HG-DAgger: Interactive Imitation Learning with Human Experts**
Kelly, Sidrane, Driggs-Campbell, Kochenderfer, ICRA 2019 — https://arxiv.org/abs/1810.02890
- 方法：把 $\beta$ 换成**人控门控** $g(x_t)$：$\pi_i=g(x_t)\pi_{exp}+(1-g)\pi_{novice}$；
  另外学一个**不确定性风险指标的安全阈值**，用来预测训练好的 novice 在哪些状态会出问题。
- 证据：仿真 + 实车，HG-DAgger 的碰撞/压线最少；转向分布比 DAgger 更接近人类驾驶数据（Bhattacharyya 距离近 **21.1%**）。
- 局限：需要人在环；作者指出 on-policy 采样在人手上会因执行器延迟/「人不在控制环里」而降低标签质量，
  且实车数据量有限、统计显著性不足（论文自述）。
- 可迁移点：**「专家只在必要时接管」在脚本专家上的对应物 = 丢弃 fallback 标签**（§1.1.4 第 1 条），
  因为 fallback 恰是「脚本不确定/规则没覆盖」的显式信号——比学一个 safety net 便宜得多。

**6. EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning**
Menda, Driggs-Campbell, Kochenderfer, IROS 2019 — https://arxiv.org/abs/1807.08364
- 方法：用**网络集成**近似 GP，以方差为置信度；决策规则：只有方差低于阈值才让 novice 执行，且
  $\beta_i=\lambda^i\beta_0$ 与阈值一起调，目标是「最大化 novice 执行比例，同时约束失败概率」。
- 证据：倒立摆 + MuJoCo HalfCheetah，安全性与学习性能优于 VanillaDAgger/SafeDAgger。
- 局限：训练 N 个模型（成本 ×N）；只验证低维。
- 可迁移点：**不要**为 DAgger 训集成（我们训练预算紧），但「集成方差」的廉价替代是 **MC-dropout 或 top-2 logit 间隔**，
  在 R1 里测它的 AUROC 即可。

**7. ThriftyDAgger: Budget-Aware Novelty and Risk Gating**
Hoque, Balakrishna, Novoseller, Wilcox, Brown, Goldberg, CoRL 2021 Oral — https://arxiv.org/abs/2109.08273
- 方法：切换准则 = **novelty（集成方差）× risk（学一个 Q 估计任务失败概率）**，按用户给的切换预算 $\alpha_h$
  取历史分数的分位数当阈值；执行期也能用（人可随时接管）。
- 证据：peg insertion / 三机器人编队用户研究（N=10）/ 真机布线；**在 $\alpha_h=0.01$ 下实际切换率 1.15%（novelty）+0.79%（risk）**；
  执行期 100% 成功率；用户研究里人与机器人表现分别 +58%/+80%。
- 局限：两个额外模型；阈值依赖历史分数分布；仍未处理「专家本身不最优」。
- 可迁移点：**这是「预算感知」最完整的实现**，但对我们是**反例**——我们不该限制查询率，而该限制**局数**。
  它唯一有用的部件是 risk = 「该状态导致输的概率」，我们已有价值头（AUC 0.84）可直接当 risk，若要做门控用它。

**8. MEGA-DAgger: Imitation Learning with Multiple Imperfect Experts**
arXiv 2303.00638 — https://arxiv.org/abs/2303.00638
- 方法：多个**不完美**专家 + 数据过滤器（筛掉低质量片段）+ 冲突消解（同一状态多个专家标签冲突时择优）。
- 证据（作者声称）：学到的策略**能超过所有单个专家**，靠的是专家之间的互补性。
- 局限：需要「专家互补」这一前提，且过滤器/消解规则是启发式。
- 可迁移点：**这条在我们按 deck 的设定下不成立**——一个 deck 只有一个脚本专家，其余 19 个专家的卡池不同，
  标签不可互换。⇒ **「多专家互补」这条路我们走不了**，只能靠 PG（§1.3）。这个否定结论本身很重要，避免再试。

**DAgger 家族横评（查询预算 / 聚合方式 / 稳定性）**—— brief 要求的三条对比轴：

| 方法 | 何时问专家（查询预算） | 聚合方式 | 稳定性 / 已知失效模式 | 对我们的取舍 |
|---|---|---|---|---|
| DAgger | 每轮在**全部**访问状态问（$m$ 条轨迹 × $T$ 步；$mN=\tilde O(T^2\log(1/\delta))$） | $\mathcal{D}\leftarrow\mathcal{D}\cup\mathcal{D}_i$，**均匀**（旧的也在，占比随时间稀释到 1/N） | $\beta$=1 只在第 1 轮最稳；$0.5^{i-1}$ 略好但收敛慢；纯 novice 采样会「卡死」（Mario 顶墙） | **主方案**；但把「均匀聚合」改成 **batch 级加权**（§1.1.3），否则被 334k 基座语料淹没 |
| DART | **不问**（离线）；改为往专家动作注噪 | 无迭代聚合，所有示范混一起 | 各向同性噪声**会训出不安全策略**（论文实测）；$\alpha$ 过大会伤专家（79%→72%） | 借「按实测动作差异度标定噪声/温度」这一条；不借「纯离线」 |
| SafeDAgger | safety policy 预测「会不会错」⇒ 只问子集（查询 ↓） | 与 DAgger 同（累积） | safety policy 自身的漏检 = 系统性盲区；阈值手调；作者归因「子集选择产生自动课程」而非查询本身 | 免费标签下**不省查询**；把 safety 分数当**样本权重** |
| HG-DAgger | **人**决定何时接管（$g(x_t)$），只在接管段打标签 | 与 DAgger 同 | 依赖人的反应速度；作者指出实车数据少、显著性不足；只验证驾驶 | 「人接管」⇒ 我们**丢弃 fallback 标签**（脚本的显式不确定信号） |
| EnsembleDAgger | 集成方差 < 阈值才让 novice 走（最大化执行率、约束失败概率） | 与 DAgger 同 | ×N 训练成本；阈值与 $\beta_i=\lambda^i\beta_0$ 联合调，敏感 | 用 top-2 margin/MC-dropout 当廉价代理；**不训集成** |
| ThriftyDAgger | novelty（集成方差）+ risk（学一个失败概率 Q），按预算 $\alpha_h$ 取分位数阈值 | 与 DAgger 同 + 执行期可继续接管 | 两个额外模型；阈值依赖历史分布；「专家不最优」仍未处理 | 若要做门控，risk 用**现成价值头**（AUC 0.84）而不是新训 Q |
| MEGA-DAgger | 多个不完美专家都要问（成本 × 专家数） | 过滤 + 冲突消解后聚合 | 依赖「专家互补」前提；规则是启发式 | **不适用**：一个 deck 只有一个脚本专家（§1.3） |

### 2.2 离线 RL 与 return-conditioned

**9. Off-Policy Deep RL without Exploration（BCQ；extrapolation error）**
Fujimoto, Meger, Precup, ICML 2019 — https://arxiv.org/abs/1812.02900
- 核心定义：**extrapolation error = 数据集分布与当前策略真实访问分布不匹配导致的值估计误差**；
  在「数据与策略不相关」时 DDPG 的值估计**发散**（Hopper 实验）。
- 方法：batch-constrained —— 用扰动模型把候选动作**限制在数据支撑内**，再用 Q-filter 选。
- 可迁移点：**这正是我们 `--loss q` 失败的原因**（无支撑约束的 argmax over 25.5 个选项）。
  离散动作空间的对应做法：CQL 惩罚 / 对未走动作加 BC 锚（§4 E2）。

**10. Conservative Q-Learning for Offline RL（CQL）**
Kumar, Zhou, Tucker, Levine, NeurIPS 2020 — https://arxiv.org/abs/2006.04779
- 方法：在 Bellman 误差上加「压低数据动作以外 Q、抬高数据动作 Q」的正则（$\log\sum_a e^{Q}$ 项），
  理论给出策略价值的**下界**。
- 证据：D4RL 上常达 2–5× 最终回报，尤其在多峰数据上。
- 可迁移点：我们的动作是离散且 K 可达 256，$\log\sum e^Q$ 可精确算（有 mask），**实现成本极低**；
  但它压的是「未走动作」，而我们的问题是「未走动作本来就该低」——只有在 Q 值和策略抽取都修好后才有意义。
  优先级低于 TD3+BC 式 BC 锚（后者只需一个 CE 项）。

**11. Advantage-Weighted Regression（AWR）**
Peng, Kumar, Zhang, Levine, arXiv 1910.00177 — https://arxiv.org/abs/1910.00177
- 方法：两步监督学习 —— 回归 $V$，再 $\max_\pi E[\exp(\beta(A(s,a)-V(s)))\log\pi(a|s)]$。
- 证据：Gym 连续控制竞争力强；纯静态数据上也能超过多数 off-policy 方法。
- 局限（对我们致命）：AWR 的权重是**逐 (s,a)** 的，前提是同一状态有多个动作的 A；我们每状态只观测 1 个动作，
  且我们实现里 A 是局级常数 ⇒ **退化为按局重加权**（§1.2 A1）。
- 可迁移点：**归档**。若要保留，唯一形态是「用 paired-seed 的决策级优势」，成本高于收益。

**12. Implicit Q-Learning（IQL）**
Kostrikov, Nair, Levine, ICLR 2022 — https://arxiv.org/abs/2110.06169 ；代码 https://github.com/ikostrikov/implicit_q_learning
- 方法：**从不查询数据外动作**：用 expectile 回归 $L_V=E[L_2^\tau(Q_{\hat\theta}(s,a)-V_\psi(s))]$（$\tau=0.7$）拟合
  「状态内上 expectile」，再 $L_Q=E[(r+\gamma V_\psi(s')-Q_\theta)^2]$，策略抽取用
  $L_\pi=E[\exp(\beta(Q-V))\log\pi]$（论文对 Franka/Adroit 用 $\beta=0.5$）。
- 证据：D4RL SOTA；在线微调也强。
- 可迁移点：**这是唯一在结构上适配「每状态一个动作」的离线 RL**（它不查未见动作）。
  但我们的动作是 25.5 维离散、回报是局级 —— IQL 的 expectile 会把「胜局里的所有动作」都抬高，与 A1 同病。
  **⇒ IQL 只值得在「同一状态有多个动作标签」之后才试**，而那需要搜索/多策略数据。优先级：低。

**13. A Minimalist Approach to Offline RL（TD3+BC）**
Fujimoto & Gu, NeurIPS 2021 Spotlight — https://arxiv.org/abs/2106.06860 ；代码 https://github.com/sfujim/TD3_BC
- 方法：$\pi=\arg\max E[\lambda Q(s,\pi(s))-(\pi(s)-a)^2]$，$\lambda=\alpha/\frac1N\sum|Q|$（默认 $\alpha=2.5$），
  只加一项 BC 正则 + 状态归一化。
- 证据：与 SOTA（Fisher-BRC）同级，实现极简（作者强调「无额外模型、单超参」）。
- 可迁移点：**离散版落地只需几行**：在 `run_batch` 的 `q` 分支上加
  $\alpha\cdot\mathrm{CE}(\text{logits},a_{beh})$，或用 $\mathrm{softmax}(Q)$ 的 CE 做策略抽取。
  这是「救 Q」的最低成本实验（§4 E2 arm 4）。

**14. Decision Transformer（含 %BC 的定义）**
Chen et al., NeurIPS 2021 — https://arxiv.org/abs/2106.01345
- 方法：把 RL 变成「以 return-to-go 为条件的自回归序列建模」，测试时给目标回报，逐步递减。
- 关键（对我们）：论文 §5 明确定义 **%BC = 只对数据集中按整局回报排名前 X% 的 timestep 做 BC**，
  $X=100\%$ 退化为 BC；他们发现 DT 大致等价于「在某个百分位上做 %BC」（Atari 上验证）。
- 可迁移点：**%BC 在我们这儿无效**（回报由对手/种子决定，不由策略决定；胜局筛选 −0.14 实测）。
  另外我们的 return 只有 {0,1} 两个值 ⇒ **return conditioning 没有可条件的信号**，DT 的结构性优势消失。
  ⇒ 不要为了「return-conditioned」换架构；要换就换容量（transformer 已显示每 epoch 更省样本）。

**15. Offline RL as One Big Sequence Modeling Problem（Trajectory Transformer）**
Janner, Li, Levine, NeurIPS 2021 Spotlight — https://arxiv.org/abs/2106.02039
- 方法：整条轨迹离散化成 token，自回归建模 + beam search 当规划器；可做动力学预测、goal-conditioned、离线 RL。
- 证据：稀疏奖励长程任务上作为 planner 很强。
- 局限：beam search 随层数/beam 宽度指数增长；需要长上下文的完整轨迹。
- 可迁移点：**只在「有可信动力学模型」时才有价值**；我们没有（引擎在环，但不可微）。**不建议**。

### 2.3 游戏/卡牌里的 expert correction 与「超专家」

**16. Expert Iteration（ExIt）**
Anthony, Tian, Barber, NeurIPS 2017 — https://arxiv.org/abs/1705.08439
- 方法：搜索当「专家」产生改进行动 → 网络蒸馏 → 网络反过来引导搜索（policy 改进循环）。
- 证据：Hex 上超过 REINFORCE，最终 tabula rasa 击败 MoHex 1.0。
- 可迁移点：**这正是我们的 `--search` 蒸馏分支**，已实测全池 Δ 显著为负（−0.066/−0.114）。
  失败原因与 ExIt 的差异：ExIt 的搜索比网络**强**（有真实动力学 + 足够深度）；我们的 MCCFR 根价值区分度≈0，
  搜索并不比脚本专家强。**⇒ 「teacher 更强」是这类方法的前提，不是结果**。

**17. Suphx: Mastering Mahjong with Deep RL**
Li et al., arXiv 2003.13590 — https://arxiv.org/abs/2003.13590
- 方法链（消融逐级提升）：SL → +policy gradient（RL-basic）→ +Global Reward Prediction（RL-1）→
  **+oracle guiding（RL-2）**：训练时用「能看全部暗牌」的 oracle agent，**逐步 dropout 特权特征**过渡到正常 agent；
  线上再加 pMCPA（按本局手牌做参数化 MC 策略适配）。
- 证据：天凤特上 5,760 局，最高 10 段、稳定段位 8.74（表 4）；作者称是首个记录段位达 10 段的 AI。
- 可迁移点：**不完全信息游戏里「教师有特权」是合法的加速手段，但它是课程，不是 DAgger**；
  而且增益来自**在线 RL 阶段**（SL 单独不够）。这与我们「BC 到 parity、PG 才超专家」的经验一致。
  我们的特权教师版本（`search` + belief root）已证无增益 ⇒ 差别在「教师是否真的更强」。

**18. DouZero（DMC 基线，解释 Q 的负结果）**
Zha et al., ICML 2021 — https://arxiv.org/abs/2106.06135
- 方法：Deep Monte-Carlo —— 用整局回报做**在线自博弈**的 Q 回归，无需 TD。
- 可迁移点：**它的成功条件我们一个都不满足**：在线自博弈、千万级局数、动作空间小（合法牌型组合数十）、
  全回合 MC 回报、三个网络（地主/农民）共享 bidding。我们的 `--loss q` 是「8k 局离线 + 单步 Q + 无支撑约束」，
  属于把 DMC 的所有前提都抽掉后的形态。**⇒ 不能用它论证「DMC 不适合七圣」**。

**19. Accelerating Self-Play Learning in Go（KataGo）**
Wu, arXiv 1902.10565 — https://arxiv.org/abs/1902.10565
- 方法：自博弈 + 一系列「加速」技巧（playout cap randomization、**ownership/score 辅助目标**、forced playouts 等）。
- 证据：同算力下远超当时 AlphaZero 复现，且**辅助目标显著改善价值标定**。
- 可迁移点：**「价值头的辅助目标（ownership/score）」是标定的正解**，我们只做了 episode outcome +
  HL-Gauss（H3 已验证「标定靠预算不靠损失」）。若重开搜索，应先加**可解释的辅助目标**（如「本局剩余回合数 / 血量差」），
  而不是换损失函数。

**20. Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents**
Haluška & Schmid, ALA 2024 — https://arxiv.org/abs/2404.16689
- 结论：ByteRL 是 LOCM/Hearthstone 的 SOTA（曾击败中国 top-10 炉石选手），但作者证明它在 LOCM 上**高度可被剥削**。
- 可迁移点：**「对固定池胜率」≠「强度」**。我们的 20 个脚本对手是固定池，dgeo 的 0.88 可能同样可被针对性剥削；
  `eval/scoreboard.py` 的 exploitability 下界要一直报，不能只看 pooled。

**21. Dota 2 with Large Scale Deep RL（OpenAI Five）**
OpenAI, arXiv 1912.06680 — https://arxiv.org/abs/1912.06680
- 方法：**先用人类回放做 BC 预训练，再用大规模 self-play RL 超过人类**；作者报告 BC 初始化加速但远不足以达到最终强度。
- 可迁移点：这是「BC → RL 才能超数据」在复杂游戏里最干净的证据，和 §1.3 的配方同构。

---

## 3. 最小可复现 recipe（按性价比排序）

### R1（≈0 成本，先做）：**协变量偏移判别测量**（`scripts/dagger_diag.py`）
- 输入：一个已认证 ckpt（如 `data/checkpoints/gc_nahida_bc_d256_e150.pt`）+ 同 deck 的 base replay + 200 局自走。
- 输出 5 个数：
  1. `agree_expert_states`：在 store val（专家状态）上的 argmax 一致率；
  2. `agree_model_states`：在 200 局模型自走状态上的一致率（**用同一个 ckpt、同一个编码器**）；
  3. `fallback_rate`：专家标签 `fallback=True` 的比例（按轮/按 request_type）；
  4. `auroc(entropy → disagree)`、`auroc(margin → disagree)`：熵/间隔能不能预测不一致；
  5. `K` 分布与 per-request-type 一致率。
- 需要改：`train/data.py`（`collect_dagger_samples`）、`envs/policy.py`（`last_choice` + ckpt 缓存）、新脚本。
- 成本：**400 局（2 deck）≈ 8 分钟** + 打分 2 分钟，无训练。
- 预期信号：`agree_expert_states − agree_model_states` = 5–20pp。
- **失败判据（go/no-go）**：gap < 3pp ⇒ DAgger 无头寸，转 §5 的其他解释（特征/指标）；`fallback_rate > 10%` ⇒ 标签不可信，先修门控。

### R2（主菜）：**DAgger 2–3 轮 + 轮次加权微调**
- 输入：base store（每 deck 2,000 局稠密语料）+ 每轮 300 局 DAgger replay。
- 关键超参：`T=1.0/0` 各半（r1）、`T=0`（r≥2）；`β=0.3→0`；`γ=0.7`；DAgger 占 batch 40%；
  `--init-from`(上一轮 ckpt)；`lr=3e-4`；`epochs=3`；`fallback` 标签丢弃。
- 伪代码：
  ```
  ckpt = base
  for r in 1..3:
      collect(actor=neural:ckpt#t=T_r, beta=β_r, label=expert:d, seeds=S_r)  → dir_r
      diag(dir_r)                       # 记录 agree/fallback，决定是否继续
      ckpt = train_bc_scale(replays=[base, dir_1..dir_r], weights=[1, γ^{r-1}..1],
                            init_from=ckpt, epochs=3)
      certify(ckpt, deck, seeds=新种子, pairs=800)
  ```
- 需要改：`train/data.py`、`train/collect.py`、`scripts/train_bc_scale.py`（`source` + `--source-weights` + `--init-from`）。
- 成本：见 §1.4，**per deck ≈3h CPU / 1.3h GPU**。
- 预期信号：一致率 gap 每轮 −2~−4pp；引擎 Δ 每轮 +1~+3pp，2–3 轮进 parity。
- 失败判据：两轮后 Δ 无改善且一致率不再收敛 ⇒ 停（不是「再调超参」）。

### R3（零代码）：**同等局数 expert-vs-expert 对照**
- 同一 deck、同一对手集、**同一批种子**，采 900 局专家对局，与 R2 的 900 局 DAgger 对撞。
- 纯 CLI：`python -m train.collect --deck d --all-opponents --seed-start 12001 --seed-count 45 ...`
- 作用：把「状态分布修复」与「更多数据」分离。**没有这一臂，R2 的正结果不可信**（我们已知更多数据能到 parity）。
- 成本：900 局 ≈ 20 分钟。

### R4（超专家）：**DAgger ckpt → PG 阶梯**
- 复用 `--loss pg`（REINFORCE + value baseline + entropy）。新增两件事：
  (a) `behaviour_index` 让 DAgger 的 rollout 直接当 PG 数据（省一轮采集）；
  (b) 每个 batch 混 30% 离线 BC 数据当**锚**（防漂移），实现为 `order` 里固定配额。
- 成本：每轮 800 局采集 + 2 epoch 训练 + 800 配对认证 ≈ 1h。
- 预期：仅在 parity+ deck 上有效（已有证据）；失败判据同历史（Δ 连续两轮为负 ⇒ 停）。

### R5（可选消融）：**标签门控 + 保留项**
- arm A：丢 fallback（默认）；arm B：全收；arm C：丢 fallback + 对「已认证偏离」加自我蒸馏。
- 成本：3 次训练 + 3×800 局认证；只在 R2 有正信号后才做。

---

## 4. 预注册实验（2 个）

### E1（主实验）：**DAgger vs 等局数专家语料（2 deck，双臂对照）**

- **假设**：在 expert-vs-expert 语料上追加 DAgger 数据，比追加**同等局数的专家对局**更能提升引擎胜率
  （机制：模型自走状态上的 expert 一致率提升更大）。
- **deck（预注册，不改）**：`nahida_kokomi_raiden`（Δ −0.080，专家 0.6125，base 2,000 局/78.6k 样本）
  与 `dual_mualani_stacks`（Δ −0.070，专家 0.555，base 2,000 局/102k 样本）。
  选它们的理由：**都已有稠密 base 语料**（所以唯一变量是新增数据的**来源**）、Δ 都在 7–8pp（有头寸、又没到无望）。
- **臂**：
  - `C0` base（冻结基线，只用于测量，不参与比较）；
  - `A` base + 900 局 DAgger（3 轮 × 300）；
  - `B` base + 900 局 expert-vs-expert（同对手、同种子数、**种子不重叠**）。
  - A、B 训练预算**逐值相同**（同 epochs/lr/batch/初始化）。
- **种子卫生**：采集种子 `12001–12019`（模 10 余 1–9，避开 `val_every=10` 的 val 残差）；
  认证种子取**未用过的区间**（建议 `900–919`；运行前先核对 `reports/explore/*.json` 里出现过的 seeds——
  已知已用 0–229、400–569、600–639、810–869），20 seeds × 20 对手 = 800 配对。
- **指标**：
  - 主指标：`A − B` 的**配对**引擎 Δ（同种子、同对手、双向），800 配对局；
  - 机制指标：`agree_model_states`（200 局/臂，同种子）；
  - 报告但不作门禁：pooled、floor 违规数（rate<0.45 计数）、per-matchup Δ 全表。
- **判定**（预注册）：
  - 成功：`Δ(A−B) ≥ +2pp` 且 95% CI 下界 > 0，**且** `agree_model_states(A) − agree_model_states(B) ≥ +3pp`；
  - 部分成功：`Δ(A−B) > 0` 但 CI 含 0 ⇒ 只把 DAgger 保留为「floor 修补」候选，不进主线；
  - 失败：`Δ(A−B) ≤ 0` ⇒ DAgger 在本项目归档（与 AWR/Q 同级），把「BC→engine gap」归因转到特征/指标（§1.2 A5 的另一支）。
- **停止条件**：任一 deck 在 r1 后 `fallback_rate > 10%` ⇒ 该 deck 停；
  `agree_model_states` 在 r2 后仍 < `agree_expert_states − 3pp` 且 Δ 不涨 ⇒ 停。
- **成本**：采集 2 deck × (900 + 900) = 3,600 局 ≈ 1.2h；训练 2 deck × 2 臂 × 3 轮 × 35min ≈ 7h CPU（或 2 deck × 2 臂 × 3 轮 × 2min ≈ 24min GPU）；
  认证 2 deck × 2 臂 × 800 配对 = 6,400 局 ≈ 2.2h。**合计 ≈ 11h 墙钟（CPU）/ 3.5h（GPU 训练）**。

### E2（负结果归因，复用现有 store，1 deck）：**AWR 的权重信息量检验 + Q 的支撑约束修复**

- **假设 1（A1）**：`--loss awr` 的权重与动作无关（只依赖 $G_g$ 与 $V(s_t)$），
  因此它等价于「按状态加权 BC」；其全部信息量都在 $(G_g,V)$ 里——若打乱权重后结果不变，则信息量为零。
- **假设 2（A2）**：`--loss q` 的失败由「未走动作无约束」导致；加 BC 锚（TD3+BC 离散版）可救回 ≥0.45 pooled。
- **臂（同 deck，`superconduct_aggro`，用现成 `sa_all` store，同初始化、同种子、同预算）**：
  1. `BC`（对照，`--loss ce`）；
  2. `AWR-β1`（`--loss awr --awr-beta 1`）；
  3. `AWR-shuffled`（把权重 $w_t$ 在**局间**随机置换后喂给同一实现——即 `--loss awr` + 一个
     `--awr-permute-seed`，或离线算好 $w_t$ 存成 per-sample weight 走 `--matchup-weight` 的通用权重通路）；
  4. `Q+BC`（`--loss q` + $\alpha\cdot\mathrm{CE}(\text{logits},a_{beh})$，$\alpha=1$；另跑 $\alpha=0$ 作 `Q` 的原样复现）。
- **指标**：引擎配对 Δ vs **同 deck 专家**（`scripts/eval_protocol.py`，同一种子集 900–919，800 配对/臂），
  逐臂全部报出——`BC` 臂就是「同 deck 专家 BC 基线」；机制指标 = `q_mse` 与 `mean Q(未走) − Q(已走)`。
- **判定**：
  - 若 `|AWR-β1 − AWR-shuffled| ≤ 2pp`（3 seeds × 800 配对；单臂 SE≈1.6pp、臂间差 SE≈2.2pp）
    ⇒ **A1 成立**（真实权重与随机权重等价 = 信息量为零），AWR 永久归档；
  - 若 `Q+BC` 的 pooled ≥ 0.45 且显著高于 `Q`（α=0）⇒ **A2 成立**，离线 Q 路线重开（并直接给出改造方案）；
    否则 ⇒ 离线 Q 在本项目归档，写进 PLAN §0。
- **成本**：训练按 50 ep 档（T4 实测 334k 样本 ≈36 s/epoch ⇒ 50ep ≈30 min/run），5 runs ≈ **2.5 GPU·h**；
  若必须复现 150ep 结论则 5 ×（83–98 min）≈ **7.5 GPU·h**（Kaggle 周额度 30h，够）。
  认证 5 臂 × 400 配对（先筛）≈ 1.5h，赢家再 800 配对 ≈ 40min。**合计 ≈ 3–9.5h 墙钟（GPU 2.5–7.5h + CPU ≈2h）**。
- 为什么值得：这两条判定会**永久关掉两条已烧了 ~10 GPU·h 的路线**（或给出唯一可行的改造），比再跑一轮超参搜索便宜一个数量级。

---

## 5. 与现有负结果的关系（为什么这次不会重演）

| 已有负结果 | 本文档的解释 | 为什么 E1/E2 不是重演 |
|---|---|---|
| AWR 全负 | **定义级**：权重与动作无关（代码可证），AWR ≡ 状态重加权 BC；H5 matchup 重加权已独立证伪同一机制 | E2 用**权重置换对照**测「信息量」而不是再调 β；若等价就关掉，不再花 GPU |
| Q/DMC 全负 | 无支撑约束的 argmax（Fujimoto 2019 已预言）+ K̄=25.5 的候选宽度；q_mse 最好但策略最差是指标错位 | E2 用 TD3+BC 离散版做**唯一一次**带支撑约束的尝试，而不是换超参 |
| 搜索 teacher 全负 | ExIt 的前提是「搜索比网络强」；我们的根价值区分度≈0，搜索不比脚本强 | 不重开搜索（除非先做价值辅助目标，KataGo 式） |
| 胜局筛选 −0.14 | 回报由对手/种子决定，不由策略决定 | 不再用 return 过滤；E1 用**配对种子**做臂间比较 |
| matchup 重加权 −0.09 | 与 AWR 同一机制族（按状态/局重加权，不改变状态内动作排序） | E1 不碰权重重加权，只改**状态来源** |
| 「更多数据」能到 parity（H4/Group C） | 数据量与状态分布是两个变量 | E1 的 B 臂就是「等量更多数据」，把两者拆开 |
| PG 只在 parity+ 有效 | 离线初始化质量决定在线微调能否奏效（AWAC/SQIL 的经验） | E1 的目标就是**造出更多 parity deck**，再交给已认证的 PG |
| A2 假阳性（val_acc↑ 胜率↓） | 指标测在专家状态上（off-distribution），与实际部署分布不符 | R1 的核心就是**换到模型状态上测同一个指标** |

---

## 6. 不要做什么（negative guidance）

1. **不要移植 SafeDAgger/EnsembleDAgger/ThriftyDAgger 的查询预算机制**。我们标签免费，省查询 = 白扔数据。
   唯一可借的是「不确定性重加权」和「fallback 门控」。
2. **不要在 PG 之后再做 DAgger**（会把已认证的 +5~17pp 拉回专家）。
3. **不要用整局回报做过滤/条件化**（%BC、胜局筛选、return-conditioned 架构）。我们的 return 只有 {0,1}，
   且由对手决定；return conditioning 没有可条件的信号。
4. **不要用「一致率高」当选型指标**。val_acc 0.764→0.895 时 pooled 从 0.476→0.429（本项目实测）。
   选型只看引擎配对 Δ + 独立种子复验。
5. **不要在没有 B 臂（等量专家语料）的情况下宣称 DAgger 有效**。
6. **不要用 AWR 再试 β 网格**。β 只是「重加权强度」，机制上不能改变状态内动作偏好。
7. **不要为 DAgger 训集成**（EnsembleDAgger 的 ×N 成本），用 top-2 margin/MC-dropout 做不确定性代理。
8. **不要在 fallback 率高的情况下继续采**——那是在蒸馏启发式兜底，不是蒸馏专家。
9. **不要用 400 配对局做「非劣」判定**（Δ=0 时下界 −0.044 必然 FAIL）；要 1,400+。
10. **不要把 DAgger 当超专家手段**。定理是单边的；超专家只能靠 PG/自博弈（本项目已复现三次）。

---

## 7. 参考（链接、版本/日期、可信度）

**一手论文（已读全文/算法细节）**

| 文献 | 链接 | 版本/可信度 |
|---|---|---|
| Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* | https://arxiv.org/abs/1011.0686 | v3 (2011-03-16)，AISTATS 2011；Algorithm 3.1 + Thm 2.1/3.1/3.2/3.3/4.1/4.2 已核对 |
| Ross & Bagnell. *Efficient Reductions for Imitation Learning* | https://proceedings.mlr.press/v9/ross10a.html | AISTATS 2010，PMLR v9；$T^2\epsilon$ 紧性由 DAgger 论文 Thm 2.1 转述 |
| Laskey et al. *DART: Noise Injection for Robust Imitation Learning* | https://arxiv.org/abs/1703.09327 | v2 (2017-10-18)，CoRL 2017；Lemma 4.1/Prop 4.1/α 标定/79%-vs-49% 已核对 |
| Zhang & Cho. *Query-Efficient Imitation Learning...*（SafeDAgger） | https://arxiv.org/abs/1605.06450 | AAAI 2017；机制已核对，具体查询减少倍数**未核实** |
| Kelly et al. *HG-DAgger* | https://arxiv.org/abs/1810.02890 | v2 (2019-03-11)，ICRA 2019；门控公式与 21.1% Bhattacharyya 已核对 |
| Menda et al. *EnsembleDAgger* | https://arxiv.org/abs/1807.08364 | v3 (2019-07-19)，IROS 2019；决策规则 $\beta_i=\lambda^i\beta_0$ 已核对 |
| Hoque et al. *ThriftyDAgger* | https://arxiv.org/abs/2109.08273 | CoRL 2021 Oral；1.15%/0.79% 切换率、$\alpha_h=0.01$ 已核对 |
| MEGA-DAgger | https://arxiv.org/abs/2303.00638 | v3；「超过所有单个专家」为作者声称，未独立复核 |
| Fujimoto et al. *Off-Policy Deep RL without Exploration*（BCQ） | https://arxiv.org/abs/1812.02900 | v3 (2019-08-10)，ICML 2019；extrapolation error 定义与「DDPG 发散」已核对 |
| Kumar et al. *CQL* | https://arxiv.org/abs/2006.04779 | v3；2–5× 为论文声称 |
| Peng et al. *AWR* | https://arxiv.org/abs/1910.00177 | v3；权重形式已核对 |
| Kostrikov et al. *IQL* | https://arxiv.org/abs/2110.06169 | v1；$\tau=0.7$、$\beta=0.5$（Franka/Adroit）已核对；locomotion 的 β **未核实** |
| Fujimoto & Gu. *TD3+BC* | https://arxiv.org/abs/2106.06860 | v2 (2021-12-03)，NeurIPS 2021 Spotlight；$\lambda=\alpha/\text{mean}|Q|$ 已核对 |
| Chen et al. *Decision Transformer* | https://arxiv.org/abs/2106.01345 | v2；**%BC 定义在本文 §5**，已核对 |
| Janner et al. *Trajectory Transformer* | https://arxiv.org/abs/2106.02039 | v4，NeurIPS 2021 Spotlight |
| Anthony et al. *ExIt* | https://arxiv.org/abs/1705.08439 | v4；Hex 击败 MoHex 1.0 已核对 |
| Li et al. *Suphx* | https://arxiv.org/abs/2003.13590 | v2；oracle guiding 与「首个记录 10 段」已核对正文；稳定段位 8.74 系表 4 转述，**未直接读表** |
| Zha et al. *DouZero* | https://arxiv.org/abs/2106.06135 | ICML 2021 |
| Wu. *KataGo / Accelerating Self-Play Learning in Go* | https://arxiv.org/abs/1902.10565 | v5；辅助目标的价值标定作用为论文声称 |
| Haluška & Schmid. *Learning to Beat ByteRL* | https://arxiv.org/abs/2404.16689 | ALA 2024；「SOTA 但高度可剥削」已核对摘要 |
| OpenAI. *Dota 2 with Large Scale Deep RL* | https://arxiv.org/abs/1912.06680 | v1 |

**本仓证据（用于所有「我们」的陈述）**

- `reports/SPECIALISTS.md` —— 19 deck 的认证最优臂、400/800 配对局 CI、floor 计数。
- `reports/explore/plan_strength_2day_kaggle.md` —— AWR/Q/搜索/胜局筛选/matchup 重加权/预算轴/混合语料/PG 阶梯的全部数字。
- `scripts/train_bc_scale.py:477-538` —— AWR 权重与动作无关（`adv = values - value.detach()`）、Q 只监督被走动作（本文 §1.2 A1/A2 的代码依据）。
- `agents/scripted/agent.py:869-888` —— `_fallback_choice` 打分兜底（fallback 门控的依据）。
- `train/data.py:137-199` —— `collect_samples` 的 teacher/opponent/`value_target` 语义（DAgger 采集器的挂载点）。
- `reps/action_adapter.py:102-134` —— 合法动作枚举走 `player_view`（DAgger 标签不破信息隐藏的依据）。
- `train_bc_scale.load_store`（`scripts/train_bc_scale.py:186+`）—— `val_every` 按 seed 取模划 val（§1.1.6 的种子残差坑）。

**未核实项（明确标注）**

- SafeDAgger 的具体查询减少倍数与圈数数字（只核对了「更少查询 + 更少碰撞」的定性结论）。
- ThriftyDAgger 在各任务上的绝对成功率（只核对了 peg insertion 的 1.15%/0.79% 与执行期 100%）。
- IQL 在 locomotion 上的 $\beta$ 取值。
- 「两臂配对比较方差 ≈0.7×」为经验估计，需在 E1 的第一批数据上实测校准。
- MEGA-DAgger「超过所有专家」的具体实验设置与效应量。
