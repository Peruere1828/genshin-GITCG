# 当前目标 · 模型架构 · 输入输出（2026-11-20 快照）

> 口径：纯策略（无 LLM、无测试期搜索）、固定 20 套脚本池、配对种子、双方各半。
> 问题刻画见 `reframe_20261120.md`；失败与重开条件见 `docs/failures/007-search-actor-n800.md`。
> 本文件描述**代码里现在真实存在的东西**，不描述计划。

---

## 1. 目标（D29 + D30）

### 1.1 一句话

在**一副固定牌组 `Ax`** 上产出一个**纯策略** `π(a | o_t)`，逐决策对局 20 个脚本专家
（专家各自打自己的牌组 `A1..A20`），并且**显著强于"同一副 `Ax` 的脚本专家"**。

### 1.2 判据

| 判据 | 数值 | 现状 |
|---|---|---|
| 主指标：配对 Δ（我方 − `expert:Ax`，同 seed、双方各打一次、只算干净局） | 95% CI 下界 > 0 | 2/20 deck 达标 |
| ≥3 个 deck 显著超专家 | n≥800 对 | 2/20 |
| ≥1 个 deck 绝对强度 | pooled ≥0.75 且非 teacher-limited 对手 ≥0.45 | 2 个（`double_geo_navia` 0.880、`unyielding_geo` 0.909） |
| 已 PASS 项非劣复验 | CI_low ≥ −0.02 | 待跑 |

### 1.3 明确不是目标

- 求解 Nash / 可利用度（对手是固定脚本 → 这是 **best response**，不是 equilibrium，D30）；
- 测试期搜索（CFR / fork-MC / PUCT）当决策来源（n=800 干净臂 −6.63pp，007）；
- 搜索蒸馏（D23）；
- belief 作为**策略**输入；
- 卡组构筑 / 选牌 / 先手选择（调用方指定，D28）；
- 20-deck 通才单模型（failures/004）。

---

## 2. 模型架构（当前已认证的实装）

**类**：`train/model.py::CVPN` —— mean-pool 的 "toy CVPN"，不是 Transformer。
已认证最强臂 `final20_dgeo` 用的是 `rl_dgeo3_double_geo_navia.pt`：

```json
{"token_dim": 65, "option_dim": 68, "d_model": 256, "option_hidden": 256,
 "value_hidden": 128, "dropout": 0.0,
 "belief_group_dim": 0, "belief_card_dim": 0, "value_activation": "identity"}
```

### 2.1 数据流

```
tokens (B,T,65) ──► Linear(65→256)+LN+SiLU ──► masked mean-pool ──► g (B,256)
                                                                     │
                                        options (B,K,68) ──► Linear(68→256)+SiLU ──► e (B,K,256)
                                                                     │
                              logits = MLP([g ⊕ e]) : 512→256→1  ────┘   （masked_fill −inf）
                              value  = MLP(g)       : 256→128→1
```

- **没有 self-attention、没有 RNN**：T 个 token 只被**平均池化**成一个向量 `g`；
  token 之间不交互。
- **选项打分是逐项独立的**（pairwise `[g ⊕ e_k]`），选项之间也不交互。
- 参数量 ≈ 20 万（fp32 checkpoint 806 KB）。
- 存在一个 Transformer 版本（`TransformerCVPN`：2 层、4 头、d=128、选项 cross-attention），
  接口与 `CVPN` 相同，但**当前已认证的臂都不是它**。

### 2.2 可选子模块（已认证臂全部关闭）

| 子模块 | 开关 | 作用 | 认证臂状态 |
|---|---|---|---|
| belief head | `belief_group_dim / belief_card_dim > 0` | 从 `g` 预测对手手牌直方图（V 类）、牌库直方图、手牌张数、burst 就绪数 | **off** |
| identity embeddings (A2) | `id_buckets > 1` | 把 definition id 当离散类查表，加到 token/option 投影上 | **off**（A2 实验因 fp16 事故判无效，D25） |
| value activation | `value_activation` | `identity` / `sigmoid`（概率） | `identity` |

### 2.3 当前架构相对目标的缺口（这三条是 D30 的直接工作项）

1. **特权信息完全没接**：`EncodedObservation.privileged_state` 在采集时**已落盘**，
   它是 god's-eye `encode_state(full_state)`——**包含双方手牌/牌库的卡牌直方图与真实骰子面值**，
   但 `CVPN.forward` 从不读它。维度随 deck 卡表变化（`superconduct_aggro` 28 类 → 286；
   `double_geo_navia` 152 类 → 1022）。→ PTIE 要做的事就是把它接进 value 头。
2. **对手隐藏信息只有"张数"**：模型看不到对手手牌身份/牌库身份/骰子面值；
   belief head 是**唯一**估计通道，而认证臂把它关了。
3. **历史通道是空的**：编码器支持 32 个 history token，但**所有调用点都传 `history=()`**
   （`agents/neural.py`、`train/data.py`、`train/distill.py`、`train/batched_inference.py`、
   `train/resolver.py`）。实测一个真实 store 样本：
   `global=1, player_summary=2, character=6, hand_card=5, history=0`。
   也就是说 **"公开动作历史"这个特征目前在实现上是缺的**，不是"已承载"。

---

## 3. 输入结构

### 3.1 token 张量：`(T, 65)`，`token_mask: (T,)`

`T` 逐决策变化：实测均值 29、最大 46（`max_tokens` 理论上限 105）。每行是按 token 类型拼接的定长向量，**65 = 10 + 3 + 6 + 6 + 9 + 7 + 24**：

| 段 | 维度 | 内容 |
|---|---:|---|
| token type | 10 | `global / player_summary / character / character_entity / status / summon / support / hand_card / dice / history` |
| owner | 3 | `self / opponent / neutral` |
| phase | 6 | 游戏阶段 |
| request type | 6 | `action / choose_active / reroll_dice / select_card / switch_hands / terminal` |
| option kind | 9 | 动作大类 |
| opponent tag | 7 | 对手标签（脚本族） |
| scalars | 24 | 该 token 的连续量：血量/充能/元素/数量/骰子面值/回合等 |

token 配比（`max_tokens = 105`，是**上限**不是定长）：`global 1` + `player_summary 2` + `character 6`
+ `character_entity 3×6` + `status 6×2` + `summon 4×2` + `support 4×2`
+ `hand_card 10` + `dice 8` + `history 32`。

**可见性**：tokens 由 `reps.public_state.mask_state_for_player` 之后的 `player_view` 构造。
- 己方手牌**身份可见**（最多 10 张）；
- 对手手牌只有**张数**（`player_summary` 的 scalar），无 `hand_card` token；
- 对手骰子在 `notification.state` 里就已被抹成 0（实测 1283 条通知中 943 条双方非空，对手侧恒 `[0,0,…]`）；
- 己方牌库按 `definition_id` 排序 → 只保留**多重集**，不含抽牌顺序。

### 3.2 选项张量：`(K, 68)`，`option_mask: (K,)`

`K` = 该决策的合法**低层** option 数（实测 `superconduct_aggro` store：均值 28、最大 256，无硬上限）；
`68 = kind 9 + metadata 38 + cost 5 + target 16`。

- 选项是**低层**动作（选骰子、选目标、选卡的具体组合），不是 51 类高层动作；
  高层分类（`reps/action_hierarchy.py`，`high_action_vocab_size()=51`）用于统计与语义键，不用于打分。
- 选项特征**相对于当前局面**编码（含目标血量/骰子代价），所以模型对进程级 action code 不变
  （AGENTS.md 的"重放确定性坑"）。

### 3.3 辅助输入（当前未进网络）

| 字段 | 维度 | 现状 |
|---|---:|---|
| `privileged_state` | 286（28 类卡表）/ 1022（152 类） | 采集落盘，**模型不读**（PTIE 的入口）；god's-eye `encode_state`，含对手手牌/牌库直方图与真实骰子 |
| `belief_target` | 手牌直方图 V + 牌库直方图 V + 10 组边际 + 2 标量 | **只做损失标签**，不是输入 |
| `opponent_tag_id / opponent_entity_id / opponent_deck_id` | 3 | 编码进 token，未单独接 head |

### 3.4 决策上下文（不在张量里）

`DecisionContext` 还带 `legal_low_level_codes`（进程级整数码）、`request_type`、`acting_player`。
`NeuralPolicy.choose()` 用 `argmax(logits)` 取索引，再映射回 `legal_low_level_codes[index]`。

---

## 4. 输出结构

```python
logits : (B, K)      # 每个合法低层 option 一个分数，pad 位置 = −inf
value  : (B,)        # 标量；value_activation="identity" 时无界，"sigmoid" 时是 P(win)
belief : BeliefOutput|None
         # group_hand_logits (B,10), group_deck_logits (B,10),
         # hand_card_logits (B,V), hand_size (B,), burst_ready (B,)
```

- **决策**：`temperature = 0` → `argmax`（纯策略、确定性）；`temperature > 0` → 用 per-policy
  `torch.Generator` 采样（配对可复现）。
- **value 的语义**：`0..1` 概率时表示"从当前**公共**状态出发、当前行动方的胜率"
  （`PriorResolver.value_probability()`）；认证臂是 `identity` 线性头，未强制概率。
- **训练目标**：policy = CE vs 专家动作；value = 局面胜负回归；belief = BCE/CE（开启时）。

### 4.1 牌组绑定

`card_vocabulary` 是**每个 deck 各自的卡表**（`dgeo` 152 类、`superconduct_aggro` 28 类），
写进 checkpoint 的 `encoder_config`。所以 **checkpoint 与牌组一一绑定**，
这正好匹配"per-deck specialist"的设计；跨 deck 换模型必须换 encoder。

---

## 5. 目标架构（PTIE，尚未实现）

```
公共观测 tokens ─┬─► actor encoder ─► g_pub ─► 选项打分（不变）
                 │                        └─► value 头（不变，作 baseline）
                 └─► critic encoder ─┐
特权输入 x（对手手牌直方图/牌库直方图/骰子面值）
                    └────────────────┴─► 特权 value 分支 ─► V(s, x)   ← 只训练期存在
```

三条硬约束：

1. **actor 输入不变**：只吃 `o_t`（公开 + 己方私有 + 历史）。信息泄漏门禁
   （`tests/test_info_hiding.py`、`tests/test_info_leak.py`）必须仍然 PASS。
2. **特权只进 value**：`V(s, x)` 用于 PG 的 advantage 基线 / 搜索叶值（若重开）；
   actor 的反向传播路径不得经过 `x`。
3. **先过真值测试再谈结构**：M-A（`replay_branch` 在固定位置 fork 求经验胜率）
   比较「常数 0.5 / 现有 value 头 / 特权 critic」的 Brier/AUC。

---

## 6. 一句话总结

> 现在跑的是一个 **mean-pool 的 20 万参数 CVPN**，输入变长 `(T×65)` 的公开 token 序列
> 与 `K`（均值 28、最大 256）个低层候选，输出 logits + 标量 value；它已经比同 deck 专家强 2 个 deck 达标。
> 缺的三块是：**历史 token 空着、特权状态落盘但没进网络、belief 头在认证臂里关着**。
> D30 之后的路线是把第三块接成**只服务 critic 的特权分支**，不动 actor。
