# FirstLight CR 架构深挖：名人堂级皇室战争 Agent 的配方与取证（2026-11-19）

> 来源：<https://github.com/Jaasssoooonnnnn/FirstLight_CR>，本地只读副本 `/tmp/opencode/FirstLight_CR`。
> `main` pinned `4c82496`（2026-10-31）；另读 `yes-delay` 分支 `d29716d`（Everlight，最新模型）。
> 方法：4 个并行 lane 逐文件读 `training/v4/*`、`battle_env.py`、`cache*`、`docs/*`；主会话读
> `README`、`docs/training_history.md`、`checkpoints/README.md`，并直接加载 checkpoint 统计参数、
> 运行上游测试子集。**本报告未改本仓代码、未跑训练/评测**；验证记录见 §10。
>
> 与上一份 `daybreak_cr_20261119.md` 的关系：daybreak-cr 的 specialist 配方明确写着"FirstLight's Hog 2.6 recipe"、
> "FirstLight's expert BC"——**本仓是那个配方的源头**；daybreak 是它的复现 + 门禁化 + 搜索实验。
> 本报告只搬协议、诊断与建模思想，不搬 Android/模拟器基础设施。
>
> 路径约定：`training/v4/*`、`battle_env.py`、`cache*.py`、`docs/*.md`、`checkpoints/*` 均指**上游仓库**；
> 本仓路径写成 `scripts/...`、`train/...`、`envs/...`、`reports/...`。上游缩写：
> `train_ppo_self_play_cluster.py`（PPO 入口）、`async_cluster_self_play.py`（采集）、
> `training_history.md`（训练史）。

---

## 0. 一页结论

1. **"名人堂"是作者自报的账号成绩**，仓库自己把它与模型对战数字严格分开
   （`docs/training_history.md:88`、`checkpoints/README.md:5,19`）。可核验的强度证据是：
   Hog 2.6 specialist 2 对 IL **89.87%**、对 Active IL **86.97%**、对 General **88.03%**（每对手 760 局、双向、argmax）。
   → 引用时应写"作者用该模型把账号打上名人堂"，不要写成"模型评测达到名人堂"。
2. **这个仓库最值钱的不是网络，是"胜率 ≠ 强度"的取证方法。** 他们有一个 80.10% 胜率的 PPO 实验，
   回看对手行为发现 **296/306 场胜利来自 IL 只行动 ≤3 次的对局**——赢得是"对手摆烂"，不是自己强
   （`training_history.md:41-45`）。从此他们的评测固定输出 `actions_by_owner`、`rejected_actions`、`truncated`
   （`checkpoints/README.md:55`）。**这一条直接适用于我们的 20 套脚本池**：脚本专家不会摆烂，但我们的策略
   可能"靠对手失误赢"，指标必须能看见。
3. **他们的统计纪律明显弱于我们**：IL 无留出集（`--validate` 只是张量校验，`train_imitation_cache.py:84-95,362`）、
   评测无置信区间、无同种子配对（只做左右换边 + `seed+index`，`evaluate.py:60-73`）。
   **不要向他们降级**；我们的 D25 配对种子 + Wilson/opponent-cluster CI 是优势。
4. **动作被分解成 5 个头**：gate（ACT/WAIT）× 候选（≤6）× 目标格（576）× 延迟档（5）× continue（是否接第二段），
   logp 相加；**每个头有独立温度**：gate 0.20、action 1.0、continue 2.0（发布 checkpoint 里 Hog 模型是 5.0）。
   这给我们的"分层 option + 温度"一个可直接照抄的骨架：不是"全局一个温度"，而是"结构决策与内容决策分开调"。
5. **观测是"诚实隐藏信息"的**：对手只看到已揭示的牌（≤8 张 + 一个 unknown 比例），对手 elixir 用
   **区间 + 不确定度**而不是精确值（`tensorizer.py:2069-2079`），critic **不特权**——和 daybreak 的"给 critic
   喂对手手牌"相反，而 daybreak 实测特权 critic 反而更差。这两个独立证据合起来支持我们的 info-leak 硬约束。
6. **他们的 PPO 有三个可直接搬的工程纪律**：每个 rollout 带 `policy_state_id`，与 learner 不一致直接报错（零陈旧）；
   任何 rejected/truncated 的整批拒训（all-or-nothing）；resume 要求 reward/PPO/温度常数逐字节一致（contract hash）。
7. **奖励比想象中朴素**：发布配方里所有 Hog/溢出/Fireball 惩罚系数都是 **0**，只剩终局 ±1 + 势函数 shaping（β=0.05）。
   而 shaping 的实现有两个细节值得抄：**终局势函数置 0**（否则等价于给快速三星额外奖励）+
   **显式零和反对称化**（`battle_env.py:4198-4226`）。
8. **专家 BC 是独立于 PPO buffer 的后置步骤**：系数 0.2、每 rank 2 条、128 步窗口、按全局有效计数归一化；
   六个头都算，但 gate 的 ACT 类权重 8.0、delay 邻域 0.2，还可选 `gradient_scope=gate-head-only` 只回传 gate 头。
   我们 PG 阶梯的"防遗忘锚"可以照这个做，但更好的版本在
   `yes-delay` 分支：**per-head KL anchor + 自适应 β**（初值 0.05、目标 KL 0.02、k3 估计）。
9. **专家 shaping 是拐杖**：Hog 专用奖励让 Hog1 更主动，但**移除全部 Hog 奖励后继续 PPO 得到的 Hog2 更强**
   （89.87% vs 84.74%）。验收标准应是"去掉 shaping 仍然更强"，不是"加了 shaping 分数涨了"。
10. **他们仍在开放的问题**：放置/时序的细粒度决策、argmax 下的被动开局、模拟器与真实执行时序差
    （`yes-delay` 分支就是在补这个：卡牌 20 tick 后落地）。

---

## 1. 系统全貌

### 1.1 分层

| 层 | 入口 | 职责 |
|---|---|---|
| 原生探针 | `native_runner/probe/*.h/.inc/.S` | 内存布局、hook、遥测、动作执行（ARM64） |
| 构建/部署 | `offline_build.py`、`offline_install.py` | 打补丁、签名、隔离、安装（离线 APK） |
| 引擎传输 | `cr_native_env.py`、`resident_batch_channel.py` | 单引擎控制 + 常驻多局批通道（CRBH/CRBQ/CRBS） |
| 对战环境 | `battle_env.py`、`contracts.py` | `reset/step`、FAIR 观测、动作、终局 |
| 策略 | `training/v4/model.py`、`tensorizer.py`、`decoding.py` | 语义张量、循环 actor-critic、动作解码 |
| 模仿 | `cache_builder.py`、`train_imitation_cache.py` | 回放重建 → 序列缓存 → 监督更新 |
| 强化 | `train_ppo_self_play_cluster.py`、`matchmaking.py`、`league.py` | 自博弈采集、PPO、对手选择 |
| 评测/服务 | `evaluate.py`、`serve_policy.py` | 完整对局、JSON-lines 策略服务 |

关键规模（全部来自仓库自述/实测）：122 张卡、252,238 局人类回放 / 17,836,160 个动作、5 个 checkpoint
（各约 48.4MB）、PPO 使用 1,528 个并发对局、40 游戏秒的 rollout 段、8 GPU（`README.md:50`、
`training_history.md:37`）。

### 1.2 部署形态

- 引擎是用户自备的 Null's Royale APK（15.535.13 / content 15.535.86）打补丁后跑在 MuMu/AVD 里；
  设备网络被 `iptables` 切断、每个原生变更都有 attestation（`attestation.py:45-68`）。
- 训练/评测都在本机或 Linux AVD 集群；常驻多局：一个引擎进程 0–15 个 slot，共享一条原生执行通道
  （`cr_native_env.py:3224-3290`）。
- 这部分**不可迁移**（我们没有 Android/模拟器），但它解释了为什么他们的"每局成本"能压到很低。

---

## 2. 模型、观测与动作

### 2.1 规模（实测，非自述）

我直接加载了 5 个 checkpoint：

```
General  step=460    stage=ppo-league-snapshot  12.68M 标量 / 481 tensors / 48.38MB
IL       step=29396  stage=imitation            12.68M …
ActiveIL step=30     stage=ppo-league-snapshot  12.68M …
Hog1     step=192    stage=ppo-league-snapshot  12.68M …
Hog2     step=616    stage=ppo-league-snapshot  12.68M …
```

按模块拆（General）：

| 模块 | 参数 | 占比 |
|---|---:|---:|
| `relation_transformer`（4 层 × 8 头，token 256） | 3.162M | 24.9% |
| `lstm_core`（512→512） | 2.101M | 16.6% |
| `candidate_encoder` | 0.899M | 7.1% |
| `cross_attention`（解码器） | 0.788M | 6.2% |
| `mechanic_profile_encoder` | 0.467M | 3.7% |
| `spatial_encoder`（64ch × 3 残差块） | **0.288M** | 2.3% |
| 其余（event/child/group/tower/heads/…） | ~5.0M | 39% |

这解释了 daybreak 文档里那句"FirstLight 12.7M 参数，但 board network 只有 0.29M"——**容量压在输入语义
（卡牌机制 + 关系）而不是空间卷积**。对我们 CPU 场景是个正面证据：我们也不该在 token encoder 上堆宽度。

### 2.2 观测契约

`UniversalSemanticBatchV4` 的字段与上限（`config.py:233-314`、`tensors.py:770-888`）：

- `match_scalars[18]`、`own_cards[8]`（自己的 8 张牌 + `in_hand`/`is_next_card`）、`opponent_cards[9]`、
  `towers[6]`、`groups[48]`（含 `children[160]`）、`active_effects[M]`、`events[32]`、
  `relation_edges[E]`、`spatial_planes[14,32,18]`、`previous_action[2]`、`candidates[≤6]`。
- 对手可见性：只有**已揭示**的卡牌身份（`revealed_cards` ∪ 公开 tracker），外加一个
  `unknown_opponent_card_fraction` 聚合行；没有对手手牌/下一张（`tensorizer.py:747-770`）。
  对手循环/进化只作为公开 tracker 特征（`public_availability_over_3` 等）。
- 对手 elixir 是**区间**：`enemy_elixir_upper_over_10` + `enemy_elixir_uncertainty_over_10`，
  未观测时取 (1.0, 1.0)。这比"假装知道"和"完全不知道"都更诚实，值得我们在 belief 特征里借用。
- 归一化全是手写尺度（`/10000`、`/5000`、`/10`、`/30000ms`），没有 running stats——**训练/服务数值一致**，
  不需要同步归一化统计。
- **critic 不特权**：value 读同一套公开张量，甚至刻意丢掉 candidate/prev-action 特征
  （`model.py:424-445`）。与 daybreak 的"critic 看对手手牌"相反；daybreak 实测那并没有更好。
- 关系注意力：14 种关系类型 × 8 头的稀疏 additive bias + 8 维几何 MLP（dx、dy、距离、重叠），
  O(E) 而不是 O(N²)（`components.py:585-646`）。

### 2.3 动作分解（最值得抄的部分）

一个决策 = **gate**（WAIT / ACT）× 最多 2 个 **micro-action**，每个 micro-action =（候选指针，可选目标格，延迟档），
外加一个 **continue** Bernoulli（决定要不要接第二段）（`model.py:602-709`）：

```
logp = logp(gate) + Σ logp(candidate) + Σ logp(target_cell) + Σ logp(delay) + logp(continue)
```

- 每头独立温度：gate 0.20（把 ACT/WAIT 变尖）、action 1.0（保留内容多样性）、continue 2.0（压平 →
  更愿意接第二段）。发布 checkpoint 把这三个温度写进元数据，PPO 重算 logp 时**强制一致**，不一致直接报错
  （`distributed_ppo.py:155-162`）。General 的 continue 是 2.5，Hog 两个专精是 5.0（实测 checkpoint 元数据）。
- 掩码是**开环 shadow state**：每选一个候选就扣 elixir、候选不可复用、Mirror 不能当第二张、每回合一个技能、
  同一手牌行只能出一次、建筑不能与第一张重叠、延迟档必须非降（`decoding.py:67-157`）。
  sampler 与 replayer 用同一份合法性推导，消灭"采样出非法动作"这一类 bug。
- 回放对候选按 **`blake2b` UID** 匹配，不按行号（`decoding.py:38-64`）——防止候选顺序变化导致标签错位。
- 数值：fp16 打包时**候选的 elixir/cost 保持 fp32**，避免 `4.999 → 5.0` 把买不起的牌放进合法集
  （`tensors.py:797-816`）；location head 也强制 fp32（`train_ppo_self_play_cluster.py:1185-1187`）。
  这与 daybreak 的"tile 头 fp32"、我们的"id 不能进 fp16"是同一类教训：**预算/索引类数值不能降精度**。

---

## 3. 训练配方的演进（本报告的核心）

`training_history.md` 是我读过的同类项目里最诚实的失败记录之一。五个模型是一条有因果的链：

```
人类回放 → IL → Active IL（PPO 调整，更主动）
                ↘ General（从 IL 初始化，混合对手）→ Hog1（示范 + shaping）→ Hog2（去掉 shaping，继续 PPO）
```

### 3.1 IL：像人，但弱

跨 deck 可辨认的打法，整体强度一般；既是权重初始化，也是后续 PPO 的基线对手（`training_history.md:13,35`）。

### 3.2 "100% 打 IL" 的假阳性（最重要的方法论段落）

- 设置：从 IL 初始化，**100% 对冻结的 argmax IL**，无自博弈、无历史池。选中 checkpoint 对 IL **80.10%**。
- 回看对手行为：**306 场胜利里 296 场，IL 的非 WAIT 动作 ≤3 次**——IL 几乎不出牌。
- 另一组自博弈为主的实验里复现：5 场 Hog vs IL 全胜，IL 一张牌没出；learner 首次行动在 196–240 秒，
  比赛 17–26 秒后就结束了（`training_history.md:41-45`）。
- 结论：**高胜率在选"剥削被动对手"，不是强度**。他们从此要求同时看回放、对手行动数与手感。

对我们的直接映射：我们的对手是确定性脚本，不会"不出牌"；但反过来，**我们的策略可能因为对手脚本的
固定漏洞而赢**，pooled 胜率会掩盖这一点。可执行的检查：按对手统计"我方获胜局中对手的有效动作数/回合数"，
与专家基线对比；异常低的 matchup 单列（这类胜利不可迁移到人类/新脚本）。

### 3.3 Active IL：为了造一个"会还手"的对手

- 从 IL 出发，**50% 自博弈 + 50% 冻结 IL**，关掉 forced-opening fallback，加 elixir 溢出惩罚；
  先强惩罚、后弱惩罚，选后者（`training_history.md:49`）。
- 结果：大费 deck 变好，小费 deck 退化（"急于花费用"破坏节奏）。它的价值是当**主动对手**，
  不是当最终模型。

### 3.4 General：从 IL 初始化，混合对手

- 同样从**原始 IL**初始化（不是 Active IL），一半自博弈；固定对手的一半里，**按对手牌组平均费用选策略**：
  cost ≤3 用 IL，>3 用 Active IL（`training_history.md:55`）。这是一个便宜且有效的"对手多样性"机制：
  用哪套对手由局面属性决定，而不是固定比例。
- 结果：小费 deck 更强，大费一般；argmax 下仍会在空场等待。

### 3.5 Hog 2.6 专精：配方与"去 shaping"

- 固定牌组：Evolved Cannon、Evolved Skeletons、Hero Musketeer、Hog Rider、Ice Golem、Ice Spirit、
  Fireball、The Log（`training_history.md:61`）。
- 固定对手 **40% IL / 60% General**；对手牌组 **30% focus / 40% weighted / 30% hard**（`training_history.md:69`）。
- 先加 Hog 部署奖励 + 开局窗口奖励 + 首个 Hog 时机奖励 + 错过截止惩罚 + Fireball 打睡着的国王惩罚
  （`async_cluster_self_play.py:60-171`、`king_activation_reward.py:100-139`）。
- 效果：Hog 出得更多但常常太晚 → 加入**真实 Hog 回放的示范监督**（expert BC）补动作与开局。
- 最后一轮：**移除全部 Hog 专用奖励/惩罚，换新优化器，只做 PPO** → Hog2。
- 结果：Hog2 测试成绩更好（89.87% vs Hog1 84.74% 对 IL），但打法**更被动**（`training_history.md:71`）。
  → **shaping 教会行为，移除后让胜负主导**；验收要问"去掉 shaping 还强不强"。

### 3.6 评测数字（他们的口径）

| 候选 | vs IL | vs Active IL | vs General |
|---|---:|---:|---:|
| General（混合牌组） | 73.30% | 60.21% | — |
| Hog 2.6 specialist 1 | 84.74% | 84.08% | 79.74% |
| Hog 2.6 specialist 2 | **89.87%** | **86.97%** | **88.03%** |
| Hog 2.6 specialist 2（混合牌组） | 66.75% | 60.73% | 59.16% |

- 一般模型：每对手 382 局、覆盖 172 个候选牌组；专精：每对手 760 局、候选固定 Hog 牌组；双方 argmax。
- 一个必须写进报告的口径：**原始 IL 在评测中享有"15 秒未行动则强制第一步"的兜底**，其它模型没有
  （`training_history.md:84`）。这说明他们知道 IL 会摆烂，并用兜底避免"0 动作对局"污染统计。
- 专精不是全面更强：剔除所有含 Hog 的牌组后，Hog2 对 General 仍有 59.45%，但它在一般牌组上对 IL 比
  General 低 **6.54pp**（`training_history.md:86`）。**专精改变能力结构，不是单调提升。**

---

## 4. PPO 与分布式采集

### 4.1 拓扑与一致性

- 每 GPU 一个 rank（NCCL），每个 rank 分配到不相交的原生引擎；每引擎 8 个常驻 lane
  （`train_ppo_self_play_cluster.py:1114-1145`、`build_ppo_multinode_topology.py:88-89`）。
- 数据面是 TCP + 长度前缀 pickle header + 原始 tensor buffer（不是队列）；worker 独立进程、
  单 CPU、`torch.set_num_threads(1)`（`remote_worker_transport.py:137-145`、`async:856-869`）。
- **零陈旧**：每个 rollout 带 `policy_state_id`，与 learner 期望不一致直接 `ValueError`
  （`distributed_ppo.py:216`）。每 40 游戏秒在段边界停引擎 → 回传 bootstrap → learner 回 `resume` 才继续
  （`async:1511-1537`）。批内异步、批间同步。
- **all-or-nothing**：任何一场 truncated 或任何一次原生动作被拒，整批采集审计失败、不进训练
  （`train_ppo_self_play_cluster.py:1634-1652`）。这与 `training_history.md:37` 的"rejected actions must be zero"一致。

### 4.2 超参（代码默认；checkpoint 里另有记录）

| 项 | 值 | 位置 |
|---|---|---|
| γ / λ | 0.9997 / 0.98（发布 ckpt 记录 λ=0.99） | `train_ppo_self_play_cluster.py:936-937`；ckpt 元数据 |
| clip（策略/价值） | 0.1 / 0.1 | `train_ppo_self_play_cluster.py:955` |
| value coef | 0.5 | 同上 |
| entropy coef | **0.0** | `train_ppo_self_play_cluster.py:1457,1475` |
| lr / 计划 | AdamW 1e-5、eps 1e-5、**无 warmup/衰减** | `train_ppo_self_play_cluster.py:1188` |
| target KL 早停 | 0.01（超过则放弃剩余 epoch） | `distributed_ppo.py:491-495` |
| grad clip | 0.5 | `train_ppo_self_play_cluster.py:936-955` |
| lane minibatch / micro / chunk | 256 / 96 / 48 | `train` 默认 |
| 温度 | gate 0.20 / action 1.0 / continue 2.0 | `train_ppo_self_play_cluster.py:50-54` |
| snapshot 间隔 | 8 updates | `train_ppo_self_play_cluster.py:1795-1807` |

注意 **entropy 系数是 0**：探索完全靠三个温度 + 分类采样，不靠熵奖励。这与我们 PG 的 `pg_entropy=0.01`
形成对照——我们的熵奖励在"把 argmax 变尖"这件事上是反向力（见 daybreak 报告 §3.1）。

### 4.3 奖励

- 基础：`r = 终局 ±1 + β(γΦ' − Φ)`，β=0.05；`Φ = 冠差 + Σ 塔权重×失血比例`（国王 3、公主 1）
  （`battle_env.py:4198-4226`）。
- 两个实现细节：**终局时 Φ 置 0**（否则等于给快速三星额外奖励，改变最优策略）；**结果显式反对称化**
  `(r0 − r1)/2`，保证严格零和。
- 采集端再叠加每步惩罚：溢出、Hog 部署、开局窗口、首个 Hog 时机、错过截止、Fireball 激活国王……
  **发布配方里这些系数全是 0**（`train_ppo_self_play_cluster.py:940-970`）。也就是说他们最终靠"终局 + PBRS"赢，而不是靠手工 shaping。
- PBRS 的时间常数写成 `shaping_tau_ticks = 5/−ln(γ)`，保证 shaping 与决策级 γ 完全一致（`ppo_runtime.py:28-29`）。

---

## 5. 对手池、牌组池与专家 BC

### 5.1 对手与牌组采样

- 默认牌组桶（每批固定配比，和 = 100）：`focus-focus 30 / focus-tail-coverage 30 / focus-tail-weighted 20 /
  tail-tail 10 / focus-tail-hard 10`（`matchmaking.py:42-46`）；对手策略默认 `0 自博弈 / 100 IL / 0 历史`。
- 专精模式换成 `FixedLearnerOpponentDeckSamplerV4`：`30 focus / 50 weighted / 20 hard`
  （`matchmaking.py:92`），并在**每个策略 matchup 内部**做最大余数分层，保证配额不是泊松噪声。
- 尾牌组覆盖用**不放回队列**轮转；weighted 用 `sqrt(uses)`；hard 是显式的 focus→tail 对位表。
- 冻结对手时，当前策略 **75%** 拿 focus 牌组（`focus_tail_current_focus_percent=75`）。
- 等级：11 与 16 按批 50/50 精确切分；卡槽顺序按种子重排（`matchmaking.py:523-533,386-399`）。

**在线 hard-deck 挖掘**（与 daybreak 的公式同源，FirstLight 是源头）：

```python
smoothed = (score + 2.0) / (games + 4.0)          # score: 1 / 0.5 / 0
reliability = games / (games + 4.0)
priority = max(0.0, 0.55 - smoothed) * reliability
# 取 top-128，抽样权重 = priority**2；每批 games/score 乘 0.8 衰减
```

（`matchmaking.py:212-237,99-101`）

### 5.2 League

- `LeagueEntry` = checkpoint + W/D/L。PFSP 权重：
  `0.05 + 0.7·w(1−w) + 0.3·(1−w)²`（`league.py:41-46`）——**同时奖励"势均力敌"和"learner 打不过"**，
  与 daybreak 的 `(1−w)²+0.05` 不同。
- 每批先**不放回抽 8 个候选**，再按权重选一个（`league.py:102-123`）。
- 快照每 8 个 update 追加，但**只有在没有固定对手时才追加**（`train_ppo_self_play_cluster.py:1803-1818`）。
- **没有晋升门禁、没有回滚、没有 Elo**：选哪个 checkpoint 是人手决定的（对比 daybreak 的 `loop.py`）。

### 5.3 牌组池

- 从回放 Parquet 统计双方 8 张牌 + 塔兵的**精确形态**，按 `replay_tag` 去重，≥5 次使用才入池
  （`deck_pool.py:289-329`、`build_ppo_deck_pool.py:57`）。
- deck id = 排序后卡牌 key 的内容 hash（含 `-evN` / `-hero` 后缀）+ 塔（`deck_pool.py:183-207`）。
- focus = 排名 1–20，必须已在池中；`focus_decks_20.json` 是 20 套人用最多的卡组（排名 1 是
  Royal Ghost Evo + Battle Ram Evo + Hero Wizard + Giant Skeleton + Mother Witch + Vines + Zappies + Barrel）。
- 等级不属于身份，来自 50/50 的等级切分。→ **同一 deck id 可跨等级/种子复现**，这对我们逐 deck 报表很有用。

### 5.4 专家 BC（Hog 示范）

- 作为**独立于 PPO buffer 的后置步骤**：每 update 后取 `2 records/rank`、128 步窗口，`coef=0.2`；
  六个头都算，但 gate 头权重 8.0、delay 邻域 0.2、value loss 0；**按全局有效计数归一化**
  （`ppo_expert_bc.py:156-201,321-335`）。
- 可选 `gate-only`：`gradient_scope=gate-head-only`，只改 ACT/WAIT 头——用于修开局被动而不扰动内容策略。
- 示范准入：精确 Hog 牌组 + 公主塔；胜局权重 1.0，负局只在 `|HP 差| ≤5%` 时权重 0.25；
  至少一张 Hog；丢弃尾部 200 tick（`build_hog_expert_manifest.py:64-86`）。
  → **他们用"接近的负局"当示范**，不是只学胜局（对比我们 failures/003 的"胜局筛选"负结果，方向一致）。

### 5.5 评测协议

- 严格镜像：第 i 局 A 在 `i%2` 侧，牌组/形态/塔兵随模型交换，`seed = match.seed + i`
  （`evaluate.py:60-73`）。默认 argmax，`--sample` 可采样。
- 输出 `identity.json` / `matches.jsonl` / `summary.json`：A/B 胜、平、`truncated`、
  每局 `actions_by_owner`、`rejected_actions`、终局状态（`evaluate.py:121-146`）。
- **没有 CI、没有 bootstrap、没有同种子配对**。这是他们的短板，也解释了为什么他们要靠
  "296/306" 这种取证而不是统计检验来发现假阳性。

---

## 6. 数据与环境工程

### 6.1 回放 → 缓存

- 输入只有 Parquet：`replays/part-*.parquet`，列 `replay_tag` + `payload_json`
  （schema `royaleapi-battle-replay.v1`，20Hz 动作 tick、双方牌组/形态/塔、坐标、结束时间）。
- **预检先拒**：零动作、`data_i` 缺失/混合、动作早于 tick 90、5 tick 窗口内 >2 个动作——
  在碰引擎之前就分类拒绝并计入 index（`dataset.py:22-41`）。我们目前是先花引擎时间再发现坏样本。
- 重建：prepare → 发牌不变性校准 → 专家动作提取 → 张量化；每局记录 `simulated_end_tick` vs `source_end_tick`、
  `winner_matches`、`crowns_match`、`executed_expert_action_count` 作为 attestation
  （`cache.py:315-329,500-503`）。
- 缓存分片 = magic `CRILC001` + JSON manifest + safetensors + zstd；存 fp16
  （`cache.py:451-474`、`cache_builder.py:530`）。
- **契约哈希**：每个分片带全部 `*.py/*.cpp/*.h/*.inc/*.S` 与 `data/competitive/*` 的 SHA-256、
  模型配置 hash、`decision_ticks`、γ、存储 dtype、headless 标志（`cache_builder.py:51-64,522-544`）。
  旧 `battle_env.py` 的已知 hash 会触发内存里的掩码升级，而不是静默复用。
- `--verify-roundtrip`：重载后的张量与内存逐位比较（`cache.py:867-898`）。
- **没有留出集**：`--train-tags` 是平铺的 tag 文件，`--validate` 只是张量校验
  （`train_imitation_cache.py:84-95,340-362`）。checkpoint 选型靠 step 和下游表现——这是我们要明确拒绝的做法。

### 6.2 环境

- 只有一个动力学源（`native`），也只产一个观测层：`BattleEnvV1.observe()` 只输出 `ObservationTier.FAIR`
  （`battle_env.py:4608-4683`）。训练/缓存/评测/服务共用 `ObservationV1 → tensorizer`。
- `reset(config|match_config, seed)` → `{owner: ObservationV1}`；`step(actions)` → PettingZoo 并行五元组。
- 常驻多局：一个进程 0–15 slot，每局命令带 `env <id>`，共享一条原生执行通道
  （`cr_native_env.py:3224-3290`）；批数据面是自定义二进制协议（`resident_batch_channel.py:1-45`）。
- 安全：设备网络 `iptables REJECT`、root ADB、每个原生变更 fail-closed attestation。

### 6.3 服务

- `serve_policy`：stdin/stdout JSON-lines，`start | observe | act | end`；**一个进程一个 episode**，
  `act` 推进 LSTM 与 previous-action；契约错误返回 `ok:false` 并丢弃 episode，意外错误直接退出
  （`policy_service.md:6-13`、`serve_policy.py:27-64`）。
- 延迟：batch-1 CUDA graph，effect/relation 轴固定填充 512/2048，`inference_ms` 逐次返回
  （`policy_session.py:33-176`）。CPU 推理也支持。

---

## 7. `yes-delay` 分支（Everlight）：时序与锚

这是仓库的最新工作，两个改动都值得单独记：

1. **执行延迟进模型**：卡牌/技能在打出后 **20 tick（1 秒）** 才生效；卡牌 token 从 16 维加到 21 维，
   新增 5 维描述"自己正在路上的牌"（是否在途、剩余 tick、目标 x/y、是否技能）；对手的牌落地前仍不可见。
   checkpoint 记录 `delay_policy: earliest` 与 ≤200ms 的延迟档，加载时直接应用。
   网络同时放大：空间 CNN 64ch×3 → **128ch×5**，LSTM 512 → **768**，文件 67.3MB（`checkpoints/README.md` on branch）。
2. **RLHF 式 IL 锚**（`ppo_anchor.py`）：对四个头分别算 `KL(π_θ ‖ π_ref)`，参考策略是冻结的 IL checkpoint，
   在 learner 自己的轨迹上、从零 LSTM 状态评估；gate/continue 是二分类，用选中动作的 logp 解析求 KL；
   candidate/target 用 k3 估计 `(r−1)−log r`（前向 KL 的无偏估计，梯度方向是 mode-covering，正好防遗忘）；
   delay 头不参与。系数用 Ziegler 比例控制器自适应：初值 0.05、目标 KL 0.02、每 update
   `β ← β·(1 + clip(KL/target − 1, ±0.2)/horizon)`，`[0.005, 1.0]` 夹住。

对我们的意义：**这是"PG + BC 锚"的正确写法**。我们之前设想的"每 epoch 加一步 CE、权重 0.2"是它的粗糙版；
per-head KL + 自适应 β 不会把策略拉回专家（KL 到目标就松手），而且 continue/gate 这类二分类头有解析解。

---

## 8. 与 daybreak-cr 的对照

| 维度 | FirstLight（本仓） | daybreak-cr |
|---|---|---|
| 定位 | 原始配方 + 名人堂账号 + 评测取证 | 复现 + 门禁化 + 搜索/蒸馏实验 |
| 数据 | 252K 人类回放，1 局 2 视角，无留出集 | 176K 同源回放，convert 带 exact 状态码，2% 哈希留出 |
| 模型 | 12.1–12.7M，五头分解，gate 0.2/action 1.0/continue 2–5 | 8.06M，what/which/tile，卡与格 0.05–0.1、时序 1.0 |
| Critic | 不特权，甚至丢 candidate 特征 | 特权（对手手牌），实测**不如**不特权的 IL value 头 |
| PPO | 1,528 并发局、8 GPU、零陈旧、all-or-nothing、entropy 0 | 3–4 A100 + 87 collectors、entropy 0.002、KL anchor |
| 晋升 | **无门禁**，人手选 checkpoint | `loop.py` 状态机：promote/worse/rollback/stall |
| 统计 | **无 CI、无配对种子**，只看计数 | cluster bootstrap、配对门禁、400/800 局功效表 |
| 对手采样 | 五分桶 30/30/20/10/10、专精 30/50/20、hard 挖掘 | 固定份额 + PFSP 余量，hard 挖掘（抄自 FirstLight） |
| 专家 BC | 独立后置步骤、全局归一化、可只训 gate 头 | 抄成 per-update 1 步、coef 0.2（同源） |
| 搜索 | 未做 | 做过，玩家强 20pp，蒸馏只带走 ~10%，放弃 |
| Shaping | 最终发布全关，只剩终局 + PBRS | 塔血 shaping 0.1 + margin 0.2，长期保留 |

**两个仓合起来的判断**：FirstLight 提供了"什么配方能把模型打上高分段"，daybreak 提供了"怎么用门禁和统计
证明它真的变强了"。我们缺的正是前者（配方层面的对照实验），而统计纪律我们已经比两者都严。

**他们对温度的分歧值得注意**：daybreak 说"卡/格温度 0.05–0.1 远强于 1.0，完全贪心最差"；
FirstLight 的发布配置是 gate 0.2（尖）、action 1.0（不尖）、continue 2–5（压平）。两者并不矛盾——
**被调的是不同的头**：daybreak 的 IL 在"选哪张/放哪格"上多峰，所以要收紧；FirstLight 的问题是
"该不该现在出牌"，所以收紧 gate、放松 continue。结论不是"抄某个温度值"，而是
**温度必须按"决策类型"分别调，并且要和训练时打分用的温度一致**。

---

## 9. 可迁移清单（按性价比排序）

### 9.1 0 训练、本周可做

| # | 做法 | 我方落点 | 成本 | 判定 |
|---|---|---|---|---|
| F1 | **评测取证**：每场记录双方有效动作数、`rejected/fallback`、`truncated`；把"对手动作异常少"的胜局单列 | `scripts/eval_protocol.py` 输出 JSON 已有逐局记录；加聚合列 | ~0.5 天 | 出现"对手有效动作 < 基线 50%"的 matchup → 该 matchup 的胜率不进认证 |
| F2 | **温度按决策类型**：先在现臂上扫 t=0/0.05/0.2/1.0；再按 option 类别（出牌/技能/切人/结束回合）分段加温 | `envs/policy.py` 的 `#t=` 扩展为 `#t_<class>=`；`agents/neural.py` 采样处按 class mask 分段缩放 | 0 训练 + ~2,000 局（1 臂 × 4 个全局温度 + 1 个分类型变体，各 400 配对局） | 任一类别温度相对全局 t=0 的配对 Δ CI 下界 >0 且 ≥2pp |
| F3 | **零陈旧批次**：collect 产物带 checkpoint id + 配置 hash，混合批次拒训 | `train/collect.py` manifest 已有指纹，扩成 per-batch 校验 | ~30 行 | 人为混入旧 ckpt 的批次必须被拒 |
| F4 | **all-or-nothing**：批次含 fallback/error/rejected 就整批不进训练 | 现有 `fallback=True` 标签丢弃逻辑升级为批次门 | ~20 行 | 与 DAgger 诊断共用同一个"有效批次"定义 |
| F5 | **缓存位级 roundtrip**：store 重载后与内存逐位比较 | `scripts/datafirst_bc_store.py` 加 `--verify-roundtrip` | ~50 行 | 位级一致；不一致的批次不进训练 |

### 9.2 小改动 + 一轮训练

| # | 做法 | 落点 | 成本 | 判定 |
|---|---|---|---|---|
| F6 | **PBRS**：终局 Φ=0 + 显式零和；Φ = 血量差 + 场面价值 | PG 的 reward 组装处（`envs/match.py` / rollout 记录） | 1 天 + 1 轮 400 局 | dvalin（+3pp 平台）上 Δ ≥ +2pp 且 CI 下界 >0；否则归档 |
| F7 | **专家 BC 后置步骤**：PG 每轮后 1 步 CE，全局计数归一化，可只训 END_TURN/pass 类 | `scripts/train_bc_scale.py` PG 分支 | 0 采集 + 1h | 与无锚对照同种子比较，`CI_low ≥ −0.02` 且点估计 >0 |
| F8 | **对手/牌组五分桶 + hard 挖掘**：30/30/20/10/10（专精 30/50/20），priority² 抽样，0.8 衰减 | `train/collect.py` 的 `tasks()`（并入 daybreak 报告 T7） | ~40 行 | 目标 matchup 的 Δ ≥ +10pp，否则停（沿用 failures/003 的重开条件） |
| F9 | **deck id = 内容 hash**（排序 key + 形态 + 塔）写进 checkpoint/报告 | 现有 deck slug 旁加 hash | ~30 行 | 同一 deck 的多次认证可机器对齐 |

### 9.3 P2

| # | 做法 | 说明 |
|---|---|---|
| F10 | **per-head KL anchor**（yes-delay 版） | 比 F7 更强的防遗忘：gate/END_TURN 用解析 KL，option/target 用 k3，β 自适应；目标 KL 0.02 |
| F11 | **五头动作分解** | 若要做结构改动（parity PASS 后）：把"是否结束回合/是否继续"从 flat option 里拆成独立二分类头，与内容打分分开调温 |
| F12 | **对手策略按局面属性路由**（他们按牌组费用选 IL/Active IL） | 我们有 20 套脚本专家，可按"对手卡组类型/回合数"路由不同难度的脚本，制造对手多样性 |

### 9.4 明确不采纳

- **无留出集 / 无 CI / 无配对种子**：我们的 D25 协议更严，保留。
- **Android/APK/模拟器/常驻多局/attestation**：与本项目目标无关。
- **12M 参数 + CUDA graph 服务**：CPU 场景不划算；他们的"容量在输入语义"结论比"容量在网络宽度"更值得学。
- **100% 打单个冻结对手**：已被他们自己的 296/306 证伪。
- **长期依赖专用 shaping**：他们的最终模型是"去掉 shaping 后更强"。

---

## 10. 验证记录（本报告自身可复现）

**checkpoint 实测**（解释器 `~/miniconda3/envs/gitcg/bin/python`，torch 2.14.1+cpu）：

```
General  step=460   stage=ppo-league-snapshot  params=12.68M tensors=481 48.38MB
IL       step=29396 stage=imitation            12.68M
ActiveIL step=30    stage=ppo-league-snapshot  12.68M
Hog1     step=192   stage=ppo-league-snapshot  12.68M
Hog2     step=616   stage=ppo-league-snapshot  12.68M
```
- 元数据：`gamma_per_decision=0.9997, gae_lambda=0.99, ppo_gate_temperature=0.2,
  ppo_action_temperature=1.0, ppo_continue_temperature=2.5`（General）/`5.0`（Hog1/Hog2）；
  IL 为 `0.99 / None / 1.0 / 1.0 / 1.0`。
- 模块拆分：`relation_transformer 3.162M`、`lstm_core 2.101M`、`spatial_encoder 0.288M`
  ——独立复现了 daybreak 文档"board network 只有 0.29M"的说法。

**上游测试子集**（全量测试需要 `sc-compression` 与用户自备 APK，本机未跑）：

```
$ PYTHONPATH=/tmp/opencode/fl_deps pytest -q \
    native_runner/tests/test_training_v4_model.py \
    native_runner/tests/test_training_v4_ppo_matchmaking.py \
    native_runner/tests/test_training_v4_decoding.py \
    native_runner/tests/test_training_v4_ppo_expert_bc.py
84 passed in 30.24s
```

**代码级核对**（抽查，非全量）：
- `distributed_ppo.py:216` = `policy_state_id` 不一致即报错；`:391-421` = clipped surrogate + clipped value。
- `learning.py:209-217` = GAE（terminated/truncated 分别处理）。
- `battle_env.py:4198-4226` = 终局 Φ 置 0 + 显式零和。
- `train_ppo_self_play_cluster.py:936-955` = γ/λ/clip/β 默认；`:1457` = entropy 0.0；`:1188` = AdamW 1e-5。
- `matchmaking.py:42-50` = 五分桶与 75% focus；`:212-237` = hard 挖掘公式。
- `league.py:41-46` = PFSP `0.05 + 0.7w(1−w) + 0.3(1−w)²`。
- `evaluate.py:60-73` = 换边 + `seed+index`（非配对种子）。
- `train_imitation_cache.py:84-95,362` = `--validate` 是张量校验，不是留出集。
- `async_cluster_self_play.py:60-171` = Hog 部署/开局/时机 shaping；`king_activation_reward.py:100-139` = Fireball 惩罚。
- `yes-delay` 分支 `ppo_anchor.py` = per-head KL + k3 + 自适应 β（初值 0.05、目标 0.02）。

---

## 11. 参考

- 上游：`README.md`、`docs/training_history.md`（本报告主要证据源）、`docs/architecture.md`、
  `docs/training.md`、`docs/policy_service.md`、`docs/validation.md`、`checkpoints/README.md`、
  `checkpoints/manifest.json`；分支 `yes-delay` 的 README 与 `ppo_anchor.py`。
- 本仓：`reports/explore/daybreak_cr_20261119.md`（姊妹报告，含温度/门禁/搜索的对照）、
  `docs/PLAYBOOK.md`、`docs/EVAL_PROTOCOL.md`、`docs/failures/003-matchup-reweight-winonly.md`。
