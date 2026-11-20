# daybreak-cr 架构深挖：可迁移经验与不可迁移部分（2026-11-19）

> 来源：<https://github.com/Jason-XII/daybreak-cr>，本地只读副本 `/tmp/opencode/daybreak-cr`，
> pinned commit `693057d`（2026-11-12，"Fix live diagnostic ID normalization and miss accounting"）。
> 方法：4 个并行 lane 逐文件读 `ppo.py` / `cr/*` / `train.py` / `convert.py` / `loop.py` / `experiments/*`，
> 主会话读 `README.md` + `docs/TRAINING.md`（1,071 行，作者按时间序记录"改了什么、测到什么、为什么"）。
> **本报告未改本仓代码、未跑训练/评测**；第 6 节是本报告自身可复现的验证记录。
>
> 对方项目：皇室战争（实时、5 tick 决策、连续放置坐标、176,732 局人类回放）。
> 本项目：七圣召唤（回合制、分层离散候选、20 套脚本专家池、约 4k 局/h CPU）。
> 因此下面只搬**协议与诊断方法**，不搬规模与实时系统细节。
>
> 路径约定：未标"上游"的 `README.md`、`docs/TRAINING.md`、`docs/CONVERSION.md`、`docs/GAME_INTEGRATION.md`、
> `cr/*`、`convert.py`、`ppo.py`、`loop.py`、`train.py`、`arena.py` 均指**上游仓库**；
> 本仓路径一律写成 `train/...`、`envs/...`、`scripts/...`、`reports/...` 或 `docs/PLAYBOOK.md` 等本仓文件名。

---

## 0. 一页结论

1. **最大的单点可迁移经验是"训练温度 = 部署温度"。** 他们对同一模型只改卡/放置温度：
   v120 在 T=1 时线上 2–3，T=0.25 时 14–4–1；自我对弈 T=0.1 对 T=1 赢 68.5%
   （`docs/TRAINING.md:349-371`）。他们后来把 learner 也切到部署温度，但从 v220 到 v380 连续九次门禁
   对 v200 仍只有 47–53%；kl_ref 涨到 0.035，而尖锐策略实际打出的 top 选择没变（`:382-384`）。
   **我们的 PG 阶梯正是"T=1 采集 + argmax 部署 + 熵奖励 0.01"**，
   dvalin/shenhe/electro 卡在 +3pp 平台，与他们的平台签名同构。这是一个 0 训练、半天内可判定的实验。
2. **他们独立复现了我们的 D23 结论**：搜索作为"玩家"很强（v200 上 65.5±4.6%，v60 上 70.3%），
   但蒸馏只捕获约 10% 的搜索增益，arena 无增益；早期把搜索损失混进共享网络，30 个 update 就崩溃
   （`TRAINING.md:195-211, 666-685`）。**不要再投搜索蒸馏预算。**
3. **评估门禁的自动化图纸可以直接照抄**：`loop.py` 是一个文件状态机（champion/challenger、
   gate、promote、rollback、stall、report），常数 `promote=0.55, worse=0.45, bench_drop=0.03,
   rollback_after=2, stall_after=10, min_entropy=0.05, keep_ckpts=3`（`loop.py:29-31`，本报告第 6 节实跑导出）。
   他们还量化了门禁的统计代价：400 局下"没变强的模型"能以约 12% 概率过 53% 线，800 局约 4.5%
   （`TRAINING.md:800-804`）——与我们的 D25 功效表一致，可作外部佐证。
4. **他们的 critic 给了我们一个反直觉的负面结果**：让 critic 看对手手牌（特权信息）**并没有更好**。
   解释方差：PPO critic（见对手手牌）0.186 人类/0.278 bot，而只看自己视野的 IL value 头是 0.274/0.321；
   离线重训 critic 只追平 IL value 头，单独换 critic 重启的首个门禁 49.6% 后被停（`TRAINING.md:470-490`）。
   → 我们 D23/D26 之后重开价值路线，应先做**真值校准测试**，而不是先加特权输入。
5. **他们有一件我们没有的、便宜且决定性的价值诊断**：把位置 fork 16 份打到终局，用平均胜负当真值，
   再和 critic 的估值比（`TRAINING.md:914-920`）。我们已有 `train/replay_branch.py`（重放到决策点 + 注入/直接 rollout 求均值），
   200 个位置 × 9 局 ≈ 1,800 局 ≈ 0.5 CPU·h，就能回答"D23 的根价值区分度≈0 是测量坏了还是头坏了"。
6. **数据面有两个结构性改进**：(a) 他们每局产出**双方各一份 ego-rotated 视角**（176,732 局 → 约 6M 决策，
   `TRAINING.md:150`），我们只采 teacher=p0（`train/collect.py:116`），这同时是我们未验证的 P0-E seat 覆盖问题；
   (b) 他们把"回放能精确重放"做成**逐局状态码**（`exact` / `exact by seed` / `dice` / `not reproduced` / `refused` / `crashed`，
   95.6% exact，`convert.py:581-583`、`docs/CONVERSION.md`），我们只有 spot-check parity。
7. **防遗忘的正则比 KL-to-anchor 更适合我们的逐 deck 阶梯**：他们的 specialist 在每个 PPO update 后
   追加 1 步 BC（8×128 chunk，权重 0.2）在同 deck 人类胜局上（`ppo.py:643-646`、`TRAINING.md:960-962`）。
   我们的 PG 阶梯只有 entropy 0.01，没有 anchor；这是 dvalin +3pp 平台的一个候选解释，且 **0 新采集**即可 A/B。
8. **他们试过且无效的一长串**（UPGO、短信用 λ_place、镜像平均、响应式搜索、强制多按技能、13 tick 提前视野、
   每 tick 决策）都写进了文档。这些恰好都是"看起来聪明"的改动，我们不要花预算（见 §5.5）。

---

## 1. 系统全貌

### 1.1 五个程序 + 一个共享包

| 程序 | 职责 | 关键事实 |
|---|---|---|
| `convert.py` | 人类回放 → 原生引擎重放 → 每局一个 `.npz`（双方视角 + 标签） | 1,051 行；版本 oracle + 逐局 exact 校验；95.6% exact |
| `train.py` | 模仿学习（IL） | DDP、bf16、`torch.compile`、chunk=128、LR warmup+线性衰减 |
| `play.py` | 模拟器/真机读取内存、模型决策、点击 | 实时系统，与本项目无关 |
| `arena.py` | 两模型配对座位的原生对局 | 400 局、双向、同种子 |
| `ppo.py` | PPO：`learn`（DDP learner）+ `act`（collector + decision server） | 3 A100 learner + 13–87 collectors；36–48k 局/h |
| `loop.py` | champion/challenger 晋升循环 | 只在登录节点跑，只提交作业和读小文件 |

`cr/` 包：`observation.py`（单决策视野 + 合法掩码 + 词表）、`policy.py`（网络/采样/解码）、
`imitation.py`（loss/chunk/collate）、`engine.py`（原生模拟器 + reward 部件）、
`matchmaking.py`（PFSP/固定份额/对位牌组）、`actors.py`（collector 与 decision server 的线格式）。

值得注意的工程取向：**没有常驻服务**（loop 只读写文件；actor 通过 endpoint.json 自发现），
**所有长任务可续跑**（`latest.pt` 每 1000 步、`settings.json` 覆盖 CLI、manifest 全齐才开训）。
这与我们 D14/C8 的约束同构，说明这套形态在 CPU/机会算力下也能落地。

### 1.2 训练主线

IL（人类回放，1.26M → 3.4M → 8.06M 参数）→ PPO（从 IL 初始化，独立 critic）→ 门禁制晋升。
他们明确记录：IL 只是冷启动，真正的强度来自 PPO；PPO 的增益**主要来自"何时出牌"**
（head transplant：timing 头单独 74.1%，tile 头 51.1%，card 头 56.0%，整体 83.3%，`TRAINING.md:418-431`）。

### 1.3 观测与网络（决定上限的模块）

**观测**：`PlayerView` 是唯一入口，回放/arena/线上三条路径共用同一个 `step()`
（`cr/observation.py:412-420`）——这是他们防 train/serve 漂移的手段，等价于我们的"只取 `notification.state`"纪律。
- 最多 128 个对象：卡 id、数据 id、阵营、类型（单位/建筑/区域/弹道/定时）、精确位置与移动、
  hp、部署时刻、当前目标、**31 个 unit-state 浮点**（眩晕/减速/攻击周期/蓄力/冲刺/护盾/高度…）、最多 6 个 buff。
- 位置按 owner 旋转（双方视角对称）、按地图尺寸归一；时间用秒（`_ms_s(cap)`），elixir 用 1/100000、时钟用 6 维。
- 隐藏信息处理：**不读**对手手牌/循环牌（线上读到的也是空），只记"已出的牌 + 由出牌顺序推出的回手时间"；
  对手英雄从场上技能控制器推断（`:506-510`）。刻意保留的两处泄漏是**对手精确 elixir** 与冠数（`:514`）。
  值得一提：他们后来实测"提前 13 tick 看到对手出牌"价值为 0（`TRAINING.md:996-1002`）——
  即便有特权视野也没有转化成强度，这与我们"严格禁止特权信息进观测"的取向不冲突，反而支持它。
- 词表 append-only：id → 索引 = 追加序 + 2，0 是 padding、1 是 unknown；加载时未知 id clamp 到 unknown
  （`cr/policy.py:68-78`、`cr/observation.py:140-148`）。

**网络（arch 1，默认到 8M）**：对象 MLP → 1–6 层 Transformer（4 头，可选距离 bias）→ `scatter_add` 到 18×32 网格
→ 卷积主干（4–12 层普通卷积，或 `--res_blocks` 预激活残差 + GroupNorm）→ core：网格 mean/max + 对象池化 +
手牌/循环/技能/在途/已见牌/标量 → MLP → LSTM → 头：`what`（wait/play/技能1/2）、`which`（卡槽打分）、
`tile`（FiLM(core, card) 条件化的整盘卷积 → 576 logits）、value、辅助 `next_when/next_which`。
配方：1.26M（card/object 32、1 层 attention、48ch×4、core 256）／3.4M（64/96、3 层、96ch×6、core 384）／
8.06M（64/160、6 层、160ch×12、core 448、8×4 粗地图）。

**网络（arch 2，`TokenPolicy`）**：24 个固定 token（summary/手牌4/循环4/技能2/在途3/已见8/英雄2）+ 128 个对象 token，
一个 384 宽 × 10 层 transformer；选牌 = 与手牌 token 做点积，选格 = 576 个学习到的 cell query 与
(core, 选中卡) 做 2 块 cross-attn（`cr/policy.py:334-501`）。这是"把分层动作做成集合同一注意力"的另一种写法，
与我们 `action_hierarchy` 的 option 打分是同一问题的两个解。

**掩码与解码**（与我们最接近，值得逐条对照）：
- 合法性来自预计算的 18×32 bitmap（`assets/placement.json`，4 个 owner×pocket 变体），镜像座位复用同一张表；
- tile logits 在非法处 `-inf`；空卡槽在 `which` 上 `-inf`；有牌在途时 `what` 只允许 WAIT；
- **买不起的牌直接折进 WAIT**（`playable`），不是让模型学"别选它"——这条对我们是可借鉴的掩码纪律；
- 联合 log 概率是**分层相加**：`logp = logp(what) + [play]·(logp(card) + logp(tile))`（`policy.py:674`），
  温度只作用于卡与格，时间维保持 1（`:666-671`）。

**消融（30k 局 / 1 epoch / 1,000 留出局，`TRAINING.md:527-595`）**：残差主干 B（6.019→5.886）与
放置头 2 个残差块 D（5.850）有效；在途绘制 C、更细粗地图 E 无效；算力消融里"更少的通道 + 更多深度"
（128ch×4res，0.84 G MACs，5.832）优于 v200 的 160ch×12 普通卷积（1.64 G，5.923）；
去掉 GroupNorm 更快但明显更差（6.044）；距离 bias 有小增益但吃掉全部加速。全量数据上 card stats 无增益（r4 ≈ r3）。

**对我们的直接含义**：如果 parity 后要动结构（P3 候选 attention），优先加在**打分头**（残差/一层 cross-attn），
不要加宽 token 编码器——他们的 8 组消融和我们的 f04 结论一致。分层掩码与"买不起折进 WAIT"是可以立刻对照检查的纪律。

---

## 2. 与本项目结构对照

| 维度 | daybreak-cr | 本仓 | 可迁移性 |
|---|---|---|---|
| 引擎 | 原生 C++ 模拟器，Python 进程内 | gitcg pybinding + 子进程 JSON-lines | 协议层可搬，实时细节不可搬 |
| 观测 | 一个 `PlayerView.step()` 供回放/arena/线上三条路径共用 | `reps/observation_encoder.py`，只取 `notification.state` | **同构**，可互相印证 |
| 动作 | 分层：what（wait/play/ability）× which（卡槽）× tile（576 格），logp 相加 | `action_hierarchy` 分层候选，option 独立打分 | 分层掩码/分层温度可搬 |
| 数据 | 人类回放，每局双方视角 | 脚本专家对局，只采 p0 | **双边采集可搬** |
| 冷启动 | IL 4–4.5 epoch，warmup 500–2000，线性衰减到 1/10 | 逐 deck BC 15 epoch，Adam 固定 lr | LR 计划可搬 |
| RL | PPO + 独立 critic + KL anchor + PFSP | REINFORCE + value baseline + entropy，无 anchor | **温度/BC 正则/PFSP 可搬** |
| 晋升 | `loop.py` 文件状态机，自动 rollback/stall | 人肉跑 `eval_protocol` 读 JSON | **状态机可搬** |
| 候选统计 | 4000 次 cluster bootstrap | Wilson + opponent-cluster CI（更严） | 我们更严，保留 |

---

## 3. 可迁移经验（按收益/成本排序）

每条格式：**他们的做法与证据 → 我方现状（代码位置）→ 适配动作 → 成本与判定门槛 → 风险**。

### 3.1 训练温度 = 部署温度（最高优先，0 训练）

**他们**：卡与放置位置的 softmax 温度在部署时用 0.05–0.25，"何时出牌"仍按 1 采样。
- 线上：v120 T=1 → 2–3；T=0.25 → 14–4–1，上 6 级；v200 T=0.25 在 Trophy Road 20–0（`TRAINING.md:353-360`）。
- 模拟器配对：v120@T0.5 / 0.25 / 0.1 对自身 T=1，分别 66.3% / 55.7% / 68.5%（`:365`）。
- 部署扫描：T=0.05 对 T=0.1 是 55.4±4.8%；**完全贪心 T=0 反而最差（48.0±4.9%）**（`:496-501`）。
- 训练端修正：`frozen_temperature=0.1`（对手/门禁/benchmark 按部署温度打）→ `learner_temperature=0.1` +
  `learner_mix=0.1`（采 90% 尖锐 + 10% 未加温，logp 按同一混合分布精确计算），因为"PPO 的步长只改分布宽度、
  不改 argmax"（`:373-384`；实现 `policy.py:642-657`、`ppo.py:175-177,517-519`）。

**我方现状**：
- 采集 `--teacher-spec neural:...#t=1.0`（`kaggle/grids/run_single_deck_rl.sh`）。
- PG loss 用**未加温** `log_softmax`（`scripts/train_bc_scale.py:527`），外加 `pg_entropy=0.01`
  （`scripts/train_bc_scale.py:616`）——熵奖励直接对抗"把 argmax 变尖"。
- 评测默认 `temperature=0.0`（`envs/policy.py:127`，纯 argmax），但 `neural:<ckpt>#t=<temp>` 已支持任意温度
  （`envs/policy.py:120-155`、`scripts/eval_protocol.py:29-32`）。

**适配动作**（三步，按顺序）：
1. **诊断**（不改任何训练）：取一个已认证臂（dgeo `final20_dgeo` 与 dvalin `rl_dvalin1_*`），
   在 t ∈ {0, 0.05, 0.1, 0.25, 1.0} 上各跑 400 配对局，对手固定 20 套池、配对种子。
   命令形状：
   ```bash
   for T in 0 0.05 0.1 0.25 1.0; do
     OMP_NUM_THREADS=1 python -m scripts.eval_protocol \
       --contestant "neural:<ckpt>#t=$T" --contestant-deck <deck> \
       --seeds 10 --seed-start 900 --workers 10 --strict-ci \
       --baseline "expert:<deck>" --tag temp_<deck>_t$T
   done
   ```
2. **判定**：若 t∈[0.05,0.25] 相对 t=0 的配对 Δ 的 95% CI 下界 > 0 且点估计 ≥ +2pp，则"认证温度"改为
   该值并写进 `docs/EVAL_PROTOCOL.md`（预注册，换全新种子复验一次）；若 CI 含 0，归档为"无增益"，不动口径。
3. **训练端**（仅在 2 通过后）：PG 采集温度改为 T*，logp 打分用同一温度；把 `learner_mix`（少量未加温质量）
   作为可选 A/B。`agents/neural.py:59-62` 已有温度采样，加混合分布约 15 行。

**成本**：诊断 5×400 = 2,000 局 ≈ 0.5 CPU·h；训练端改动 ≤ 0.5 天 + 1 轮 400 局。
**风险**：(a) 他们的低温柔性收益来自"IL 模仿了很多人类、分布多峰"；我们的专家是确定性函数，BC 分布可能本来就尖，
收益会小——这正是要先测的原因。(b) T=0 最差这一条可能是他们 576 格平局的 tie-break 伪影，我方 option 数少，
未必复现；诊断里顺手记录 **argmax tie rate**（见 §3.10）。

### 3.2 独立 critic + 特权信息：他们有，但实测不更好（重要负面结果）

**他们**：critic 是策略网络的副本 + 对手手牌/循环牌/技能，两个头（P(win) + 短期 shaping），
前 20（或 60/10/3）个 update 只训 critic；GAE γ=1 用于胜负流、γ=0.999 用于 shaping 流
（`ppo.py:211-252, 483-489`）。收益证据是**负面的**：

| 预测器 | 人类对局 EV | bot 对局 EV |
|---|---:|---:|
| PPO critic（见对手手牌） | 0.186 | 0.278 |
| IL value 头（只看自己） | **0.274** | **0.321** |
| v200 自己的 value 头 | 0.244 | 0.294 |
| 离线重训 critic（见对手手牌） | 0.260 | 0.327 |

来源 `TRAINING.md:470-483`；单独把 critic 换成离线重训版重启（gen-critic），首个门禁 49.6% 后被停（`:485-490`）。

**我方现状**：value 头语义/符号 bug 已在 D26 修复（`PriorResolver.value_probability()`、acting→p0、dice i.i.d.），
`scripts/eval_value_calibration.py` 已能算 Brier/ECE/AUC，但那是**对局终局标签**的离线校准，没有"同一状态的真值"。

**适配动作**：不要把"给 critic 喂特权信息"当作提升路线（我们的 info-leak 门禁也不允许它进策略路径）；
若要重开价值/搜索（D23 三前置之一），先走 §3.3 的真值测试。

### 3.3 Critic truth test：用 fork 到终局当价值真值（便宜、决定性）

**他们**：v240 对 v240，每局取 2 个随机决策，fork 16 份打到终局，均值即该位置真值；再对 1 次出牌的
8 个候选格各 fork 8 份（同随机流），得到"动作后 0.25–10 秒"的真实价值（`TRAINING.md:914-920`）。

**我方现状**：`train/replay_branch.py` 已实现同一形状——从初始态重放到决策点、注入候选、用 rollout 策略打完、
按 rollout 种子求均值（模块 docstring 明确写"O(depth) 重放 vs O(1) fork，离线 teacher 用途"）；
`scripts/run_replay_branch.py` 已有 CLI。也就是说**不需要新引擎能力**。

**适配动作**：写一个只读诊断（约 80 行，复用 `run_match` + `build_policy`）：
在固定 (deck, opponent, seed) 上取 200 个决策点，每点 `rollouts=8`（同策略、不同 rollout 种子），
输出 (a) 位置真值 vs 模型 value 的 Brier/ECE；(b) 专家动作与该状态"最佳候选"的真值差。
判定：若 value 的 Brier 明显差于常数基线（均方误差 ≥ 0.25 即不如猜 0.5），则价值头不可用于前沿，
D23 的三前置继续冻结；若可用，则搜索重开有了测量基础。

**成本**：200 位置 × 9 局 ≈ 1,800 局 ≈ 0.5 CPU·h + 0.5 天实现。
**风险**：rollout 策略弱会让真值有偏（他们同样承认"rollouts assume the opponent plays like our policy"）；
缓解：同一位置的所有候选共享 rollout 种子（配对），只比较**相对**价值。

### 3.4 "收益在哪个头"的归因诊断（0 局成本）

**他们**：把 v200 的单个头移植回 IL 网络，分别对 IL 打约 450 局：tile 51.1%、card 56.0%、timing 74.1%、
整体 83.3% → 结论"PPO 的增益主要是何时出牌"（`TRAINING.md:418-431`）。后来又用
"与搜索候选的 pair accuracy / gain captured"证明 PPO 一直在移动放置位置但没变好（`:838-841`）。

**我方现状**：没有时间头，但 option 有类别（出牌/技能/切人/结束回合，见 `reps/` 的候选构造）。
**适配动作**：离线（store 上，0 局）比较 PG 前后模型 argmax 与专家动作的**分类别一致率**，
并记录每类的熵；若 PG 只改动了低影响类别（如无意义的切人），说明 400 局筛选的分辨率不足以看到真实改进，
应换更大的 n 或换指标，而不是继续加轮次。实现 ≈ 60 行，分钟级运行。

### 3.5 逐 deck specialist 的 BC 正则（0 新采集）

**他们**：specialist 固定单 deck（`self_play:0`），对手 40% IL + 40% general + 20% PFSP（自己的快照与 league），
对位牌组 30% 最常见 / 30% 最差 / 40% 池内原配；**每个 PPO update 后追加 1 步 BC**
（`bc_batch=8` 个 128 决策 chunk，权重 `bc_coef=0.2`）取自同 deck 人类胜局（`ppo.py:643-646`、`TRAINING.md:949-966`），
后来放宽到 0.05（`:1028`）。

**我方现状**：PG 阶梯（`docs/PLAYBOOK.md` §1）从当前策略 `--init-from`，只有 entropy 0.01，无任何 anchor；
dvalin 三轮 +3.8/+3.0/+3.3pp 后平台（`reports/SPECIALISTS.md`）。
**适配动作**：在 `scripts/train_bc_scale.py` 的 PG 分支里，每个 epoch 追加 1 步 CE（该 deck 的专家 store，权重 0.2），
即 "PG + 轻量 BC 锚"。这是 REINFORCE 版的 `bc_coef`，实现 ≈ 20 行。
**成本**：0 采集，1 次训练（~1h）+ 400 局评测。
**判定**：与不加重整的同一轮 PG 在同一批新种子上比；`CI_low ≥ −0.02` 且点估计 > 对照，才进下一轮。
**风险**：BC 锚会把策略拉回专家（天花板）；所以只作"防遗忘"用，权重 ≤0.2，且只在 PG 轮启用。

### 3.6 晋升自动化：`loop.py` 的状态机与判据

**他们**：文件契约 `champion.json` / `eval.json` / `evals.jsonl` / `rollback.json` / `report.md`；
每 tick：作业恢复 → 指标 → 健康 → 门禁 → 超时 → 报告。规则（本报告第 6 节实跑导出）：
```python
LOOP_DEFAULTS = dict(promote=0.55, worse=0.45, bench_drop=0.03,
                     champion_games=400, bench_games=200, eval_timeout_min=90,
                     rollback_after=2, stall_after=10, hang_min=30,
                     min_entropy=0.05, max_hours=72.0, keep_ckpts=3)
```
- 晋升 = 对 champion ≥55% **且** benchmark 均值不低于 champion 的 3pp；benchmark 游戏不足 → `inconclusive`，
  **绝不盲晋升**（`loop.py:190-212`）。
- 回滚 = 连续 2 次门禁 <45%，或连续 3 个 update 熵 <0.05（策略崩溃）；回滚到**完整 checkpoint**（含优化器）。
- 停 = 连续 `stall_after` 次无晋升，或 72h；inconclusive 不计数。
- 升级版判据（curriculum）：按**同一门禁内 paired gain vs stage-start** 接受，而不是原始分，因为原始分在门禁间漂移 44.4–51.6%
  （`experiments/deck_curriculum.py:234-244`）。
- 他们对门禁统计的自评："400 局下无改进模型约 12% 概率过 53%，800 局约 4.5%"；且"55% 的门禁挡住了真实的 ~5% 改进"
  （`TRAINING.md:713, 800-804`）。

**我方现状**：`scripts/eval_protocol.py` 输出 JSON，`--strict-ci` + opponent-cluster CI 比他们的点估计更严；
但晋升/回滚/停止是人肉判断，specialist 扩张（P1，18k 采集）会放大这个成本。
**适配动作**：新建 `scripts/promotion_loop.py`（≈200 行）：读 eval JSON，按**预注册**判据
（`CI_low ≥ −0.02` 且 Δ>0，外加"其它 deck 退化 ≤3pp"的 benchmark guard）输出
`promote / kept / rollback / inconclusive`，并重写 `reports/<tag>/report.md`。
判据不用他们的 53% 点估计（我们有更好的统计），但**照抄三条纪律**：不完整=inconclusive、回滚要含完整状态、停要有计数。
**成本**：0.5–1 天；之后每轮 specialist 省 30–60 分钟人工。
**风险**：自动化会放大"同一批种子选型又报分"的错误；脚本必须强制 `--seed-start` 不重叠并记录在报告里。

### 3.7 PFSP 与 league（P2 的设计蓝图，现在不做）

**他们**：权重 `(1 - win_rate)^2 + 0.05`（win_rate 来自最近 ±1 结果），**先扣固定份额，余下按 PFSP 归一**
（`cr/matchmaking.py:59-70`）；池 = IL + 最近 6 个快照 + **所有**晋升过的 champion（永久保留）；
exploiter 模式只打另一 run 的当前 champion。他们的实战教训：
- 从 champion 自己的谱系出发的 exploiter，400 个 update 没找到洞；建议从 IL/anchor 出发（`TRAINING.md:404-405`）。
- 自适应对手采样会让训练内分数震荡（"sine wave"：赢得越多 → 打的 IL 越少 → 忘了它的风格 → 又输回去），
  **必须用固定池 arena 判强度**（`:111-115`）。

实跑验证（第 6 节）：
```
hard  win=0.00  pfsp=(1-win)^2+0.05=1.0500  share=1.0500
easy  win=1.00  pfsp=(1-win)^2+0.05=0.0500  share=0.0500
mid   win=0.50  pfsp=(1-win)^2+0.05=0.3000  share=0.3000
```
**适配**：我们 P2 的 league 设计直接照抄公式与"固定份额优先"；anchor 不要只挂一个 champion（他们的 55% 门禁+
陈旧 champion 作为 KL 参照相互锁死）。认证永远在固定 20 套池上，不用 PFSP 权重。

### 3.8 数据面：每局双边视角（解决 P0-E + 省一半采集）

**他们**：每局在 `convert.py` 里按两个 owner 各 ego-rotate 一份输入与标签，
`len(entries)//2` 即局数（`convert.py:691, 793-797`）；176,732 局 ≈ 6M 决策 ≈ 2 视角/局（`TRAINING.md:150`）。
两份视角共享同一个 train/val 标记（`convert.py:755-756, 801`），所以不会出现兄弟视角泄漏。

**我方现状**：`train/collect.py:116` 的注释即事实——"teacher (player 0) vs opponent (player 1)"，
只记录 p0 的决策；`docs/STATUS.md` 的 P0-E 明确说 seat 覆盖未验证、side=1 分布偏移可能仍在。
**适配动作**：在 collector 里同时包住两方（记录方 + 教师方），一行一局保存两段 samples，存储按 owner 标记。
收益：(a) 同一批对局得到两倍样本；(b) 天然覆盖 side=1；(c) 采集是 CPU 瓶颈（~4k 局/h），这等于**把采集成本砍半**。
**成本**：`train/collect.py` + `train/data.py` 约 50–80 行；先用 2 局 smoke 验证双段 payload 与 parity。
**风险**：compact fp16 的 id 事故刚修（D25），新 payload 必须沿用 exact int sidecar，并对 side=1 跑一次
`scripts/parity_store_live.py`。

### 3.9 数据门禁：逐局 exact 状态码 + 版本 oracle

**他们**：`exact(result)` 三个条件——塔血误差 0、胜负一致、冠数一致（`convert.py:581-583`）；
`assignment.jsonl` 由日期表给出"哪局用哪个引擎构建"，不可复现的局被**标记原因**而不是静默丢弃；
taxonomy：`exact` / `exact with a seed` / `dice` / `not reproduced` / `refused` / `crashed`；
逐日 exact 率当漂移检测器；95.6%（1,076/1,126）exact（`docs/CONVERSION.md`）。
**我方现状**：`envs/parity.py` + `scripts/parity_store_live.py` 是**抽点**校验（5 decisions/(deck,opp,seed)），
已经抓到了 fp16 id 事故。
**适配动作**：不做全量重放（那是 2× CPU），而是 (a) 每个采集批次抽 1/20 局做全量重放，写状态码进 manifest；
(b) 批次级不变量：definition id 的集合/hash、argmax tie rate、side=1 覆盖率；
(c) 任何非 `exact` 的局不进训练集，只进 `reports/`。这是把我们的 P0 从"抽检"升级成"统计门禁"的最小改动。

### 3.10 数值与表示：id 只走整数、score 头 fp32、容量放在打分头

**他们**：
- id 词表 append-only（879→898 只追加，旧索引不变），加载时把未知 id clamp 到 `unknown`（`policy.py:68-78`）；
  新增行默认等于 unknown 行——用 base+offset 初始化实测 48.3%（CI 44.6–52.0），不如直接 unknown（`tools/extend_vocab.py`）。
- **tile 头强制 fp32**：bf16 下 tile 分数漂到 ~2000，相邻格被舍入成同一个值，85% 的出牌有平局，tile loss 2.8→5.0；
  fp32 下同一 checkpoint 是最好的（`TRAINING.md:238-247`、`policy.py:315-322`）。小模型也有 49% 平局。
- 消融结论：**残差放置头（D/N）几乎免费且稳定 -0.05/-0.07 loss**；加深 board trunk（E、flight-grid）无增益；
  card stats 输入全量数据后无增益（r4 ≈ r3，`TRAINING.md:748-757, 895-897`）；距离 bias 有小增益但吃掉全部加速。
**我方现状**：compact 用 fp16 存全体特征（D25 已用 int sidecar 修补 id 列）；模型本身在 CPU 上是 fp32。
**适配动作**（低成本、高保护性）：
1. 在 store 读盘与模型前向处加**断言**：definition 列必须是整数类型（而不是"看起来像整数"）。
2. 加 `argmax tie rate` 指标（离线，分钟级），纳入采集批次报告；平局率高说明特征/精度有问题，不是策略变强。
3. 若 parity PASS 后要做结构（P3 候选 attention）：把新增容量放在**option 打分头**（残差/1 层 cross-attn），
   不要加宽 token encoder——这是他们用 8 组消融换来的结论，和我们 f04 的"song 候选集 attention 不是当前瓶颈"相互印证。

### 3.11 LR warmup + 线性衰减（小改动，需 A/B）

**他们**：IL AdamW 3e-4，warmup 500（大模型 2000）步，线性衰减到 `lr·final_lr`（默认 1.0，常用 0.1），
grad clip 1.0，loss 非有限则中止并保留 `latest.pt`（`train.py:174-181`）。
**我方现状**：`scripts/train_bc_scale.py:623` 用固定 lr 的 Adam，无 warmup/衰减。
**适配动作**：加 `--warmup` 与 `--final-lr` 两个参数（≈15 行），在一个 deck 的现成 store 上做 2 组 A/B（各 ~1h）。
判定：同种子、同 epoch 数下，以 `val_ce` 与 400 局引擎 screen 双指标；引擎不涨就不采纳（记住 D25：不看 val_acc）。

### 3.12 辅助头：next-play 预测（可选，未验证）

**他们**：IL loss = `CE(what) + CE(which) + CE(tile) + 0.5·MSE(value) + 0.5·(CE(next_when) + CE(next_which))`，
即"预测下一次出牌在几个 tick 后、出哪张"（`cr/imitation.py:43-51`；`NEXT_EDGES=(1,2,3,5,9,17)`）。
**我方**：已有 `belief_head` 做隐藏量预测，形状相同。
**适配**：如果 belief 头已有正向证据，可把"predict 下一次自己/对手的动作类别"作为额外自监督头试一次；
若 belief 头本身没有引擎增益，就不要加第二个辅助头。**优先度低于 3.1/3.5/3.8。**

---

### 3.13 他们的开放问题 = 我们的风险清单

读这份仓库最有价值的部分不是成功项，而是他们**在 8 个月、上千 GPU·h 之后仍然没解决**的东西：

| 他们的开放问题 | 证据 | 对我们的含义 |
|---|---|---|
| 精细放置学不会：搜索能多 20pp，蒸馏只带走 ~10%，且 200 局后学习曲线就平 | `TRAINING.md:838-841, 888-893` | 我们的"候选 self-attention probe"（P3）也要按小预算做，别指望结构一换就解决 |
| critic 解释方差 ~0.3，无法支撑长程信用分配 | `:470-483` | value 拆头/搜索重开前必须先过 §3.3 的真值测试 |
| 模拟器增益不等于线上增益（对手/benchmark 全是自家模型） | `:708-712` | 我们的所有强度都只对 20 套脚本池成立；报告必须带池版本 |
| 门禁既能挡假阳性也能挡真实 5% 改进 | `:713` | 用 CI 而不是点估计晋升，并预注册"回滚成本" |
| 专精与通用冲突（专精最快，但违背 any-deck 目标） | `:220-221` | 我们的逐 deck specialist 路线是刻意的权衡，不是缺陷 |

一句话：**他们的强项是工程纪律（可复现、门禁、诊断），弱项是同样卡在表示/信用分配上**。
我们能直接拿走的是前者；后者要用我们自己的真值测试去判，而不是照抄他们的结论。

---

## 4. 明确不可迁移 / 不要照搬

| 他们的东西 | 为什么不能搬 |
|---|---|
| 内存读取、模拟器控制、点击时序（上游 `play.py`、`docs/GAME_INTEGRATION.md`） | 实时客户端工程，与本项目目标无关 |
| CUDA graph decision server、actor 自发现、staleness-by-version | 我们是短命子进程 + JSON-lines，GPU 只在 A800 窗口；瓶颈在引擎 4k 局/h，不在推理 |
| 5 tick 决策、放置格几何、13 tick 提前可见、`when-to-play` 头 | 回合制离散候选游戏没有这些自由度（他们对"提前可见"的实测价值也是 0，`TRAINING.md:996-1002`） |
| 176k 人类回放、人类 BC 正则 | 我们没有人类语料；我们的老师是确定性脚本，BC 的上限就是老师 |
| 1.26M→8M 参数、4×A100 17k 局/h | 我们 CPU 约 4k 局/h、d256；只搬协议不搬规模 |

还有一条**方法论上的反向教训**：他们最大的自评是"模拟器增益没有清晰传导到天梯，因为训练对手与 benchmark
全是自家模型，人类风格对手只占 ~1%，人类会惩罚的错误在训练里从不被惩罚"（`TRAINING.md:708-712`）。
我们的评测池是固定 20 套脚本专家（D11），没有这个"自博弈闭环"问题，但也有对应风险：
**所有强度声明都是对这一个池的**。他们用"多谱系 benchmark 模型"（不同来源的 IL/PPO/specialist）当留出集，
我们可以等价地用"不参与采集的对手子集 + 全新种子"做留出，且报告里必须写清池版本。

---

## 5. 建议排产（与 `docs/STATUS.md` 对齐）

### 5.1 P1 内、0 训练即可做（本周）

| # | 实验 | 预算 | 预注册判定 | 失败后 |
|---|---|---|---|---|
| T1 | 温度扫描（§3.1）dgeo + dvalin 各 5 个温度 × 400 配对局 | 4,000 局 ≈ 1 CPU·h | 任一 T 相对 T=0 的配对 Δ CI 下界 >0 且 ≥+2pp | 归档"温度无增益"，认证口径不变 |
| T2 | critic truth test（§3.3）200 位置 × 9 局 | 1,800 局 ≈ 0.5h + 0.5 天实现 | value 的 Brier 优于 0.25 且 AUC>0.6 | D23 价值前置继续冻结 |
| T3 | 分类别归因（§3.4）PG 前后各一个 arm | 0 局，分钟级 | 若 PG 只改了低影响类别 → 加轮次无效 | 把 dvalin 平台写成"测量分辨率不足"而非"策略饱和" |

### 5.2 P1 内、小改动 + 一轮训练

| # | 实验 | 预算 | 判定 |
|---|---|---|---|
| T4 | PG + BC 锚（§3.5），选 dvalin（+3pp 平台） | 0 采集 + 1h 训练 + 400 局 | 新种子上 `CI_low ≥ −0.02` 且优于无锚对照 |
| T5 | LR warmup/衰减 A/B（§3.11），选 ungeo（已 PASS，稳定） | 2×1h 训练 + 2×400 局 | 引擎 screen 不劣且 val_ce 更低 |
| T6 | 双边采集 smoke（§3.8）：2 局验证 payload/parity | 0.5 天实现 + 分钟级 | side=1 parity PASS；否则不推广 |

### 5.3 P1 内、采集策略（与既有 "matchup 靶向采集 pilot" 合并）

T7：把 `docs/failures/003-matchup-reweight-winonly.md` 已批准的重开条件（target 25–35% + uniform 65–75%）与他们的
**hard-deck 选择公式**合并：平滑胜率 `(w+2)/(n+4) < 0.55`，priority = `(0.55 − rate)·n/(n+4)`，
取 priority 前 N，**抽样权重 = priority²**（`cr/matchmaking.py:10-13, 174-184`；方次已内含，不要再平方）。
我方已有数据源：`reports/SPECIALISTS.md` 的 `matchups <45%` 列 + `eval_protocol` 每对手 JSON。
实现点：`train/collect.py:288-310` 的 `tasks()` 加配额（重复对手元组或权重表），≈30 行。

### 5.4 P2

- T8：PFSP + 固定份额 + league（§3.7），照抄公式；exploiter 从 anchor 出发。
- T9：`scripts/promotion_loop.py`（新建，§3.6），在 specialist 扩张前落地，否则 18k 采集的评审成本不可控。

### 5.5 明确不做（有他们的实测背书）

搜索蒸馏（第三次）、UPGO、短信用折扣、镜像平均、响应式搜索、强制多按技能、每 tick 决策、
自适应对手采样下的训练内分数作强度声明、把结构容量加在 token encoder 上。

---

## 6. 验证记录（本报告自身可复现）

**上游代码可运行性**（无需 GPU/引擎/模拟器，作者 README 如此声明）：

```
$ cd /tmp/opencode/daybreak-cr && OMP_NUM_THREADS=2 python -m pytest -q tests
116 passed, 2 skipped, 3 warnings in 8.46s
```
（解释器：`~/miniconda3/envs/gitcg/bin/python`，torch 2.14.1+cpu）

**引用的公式与常数是实跑导出的，不是从文档抄的**：

```
$ python -c "
import numpy as np
from cr.matchmaking import weights
res={'hard':[-1]*10,'easy':[1]*10,'mid':[-1,1]*5}
for k,(p,s) in weights({'hard':'h.pt','easy':'e.pt','mid':'m.pt'},res).items():
    w=(np.mean(res[k])+1)/2; print(f'{k:5s} win={w:.2f} pfsp=(1-win)^2+0.05={(1-w)**2+0.05:.4f} share={s:.4f}')
import loop; print('LOOP_DEFAULTS:', loop.LOOP_DEFAULTS)"
hard  win=0.00 pfsp=(1-win)^2+0.05=1.0500 share=1.0500
easy  win=1.00 pfsp=(1-win)^2+0.05=0.0500 share=0.0500
mid   win=0.50 pfsp=(1-win)^2+0.05=0.3000 share=0.3000
LOOP_DEFAULTS: {'promote': 0.55, 'worse': 0.45, 'bench_drop': 0.03, 'champion_games': 400,
 'bench_games': 200, 'eval_timeout_min': 90, 'rollback_after': 2, 'stall_after': 10,
 'hang_min': 30, 'min_entropy': 0.05, 'max_hours': 72.0, 'keep_ckpts': 3}
```

**逐条核对的原文位置**（抽查，非全量）：
- `convert.py:581-583` = `exact()` 三条件；`convert.py:755-756` = 2% 哈希留出；`convert.py:793-797` = 双方视角。
- `cr/imitation.py:43-51` = IL loss 组合；`train.py:174-181` = LR warmup/衰减/非有限中止。
- `cr/policy.py:642-657` = 温度与 mix；`policy.py:315-322` = tile 头 fp32。
- `ppo.py:643-646` = specialist 的 BC 穿插；`ppo.py:211-230` = 带对手私有信息的 critic。
- `loop.py:29-31` = 门禁常数；`loop.py:190-212` = inconclusive/promote 判定。

**我方现状的核对位置**：
- `envs/policy.py:127` = 评测默认温度 0；`envs/policy.py:120-155` = `#t=` 解析。
- `scripts/train_bc_scale.py:527` = PG 用未加温 `log_softmax`；`:616` = `pg_entropy=0.01`。
- `kaggle/grids/run_single_deck_rl.sh` = 采集 `#t=1.0`。
- `train/collect.py:116` = "teacher (player 0) vs opponent (player 1)"；`:288-310` = 任务笛卡尔积。
- `train/replay_branch.py:1-40` = 重放分叉 MC 的既有能力。
- `scripts/eval_value_calibration.py` = 现有 Brier/ECE/AUC 工具。
- `reports/SPECIALISTS.md` = 每 deck 的 `matchups <45%` 计数（T7 的数据源）。

---

## 7. 参考

- 上游：`README.md`（程序与模块地图）、`docs/TRAINING.md`（16 节时间序实验记录，本报告的主要证据源）、
  `docs/CONVERSION.md`（数据门禁）、`docs/LIVE_PENDING_BUDGET.md`（实时资源预留，不适用）。
- 本仓：`reports/explore/literature_review.md`（7 条文献线；本报告是其"同类系统工程实证"补充）、
  `docs/PLAYBOOK.md`（有效方法）、`docs/EVAL_PROTOCOL.md`（认证口径）、
  `docs/failures/001-search-as-teacher.md` 与 `docs/failures/003-matchup-reweight-winonly.md`（同一路线的负结果）。
