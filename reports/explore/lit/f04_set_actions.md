# 组合动作/候选集建模（Set Transformer / Pointer Network / option-set attention）

> 调研日期：2026-10-27。只读代码与 `refs/`，未跑训练/评测、未写 `data/`。对应老师问题 4、项目 A4。
> 外部引用均给 URL；内部事实给文件与行号。不能从一手材料确认的地方标「未核实」。

## 0. 一句话结论（可执行）

先做 **0 成本诊断**：在现有 store 上跑 A0，统计 top-2 margin、同 kind/同 target 但不同费用的近重复选项比例、以及 A0 错误是否集中在这些决策上；如果歧义率 <5% 或错误不集中，**直接停止 A4**。有信号才在**冻结 A0 trunk** 上加 1 层候选 self-attention（零初始化残差，初始 logits 与 A0 完全相同），训练 1–2 epoch，只认**同种子引擎胜率**：screen 400 局/deck 需 ≥2/3 deck 达 +6pp 且无 deck ≤−2pp；confirm 2,000 局/deck 需 ≥2/3 deck 达 +3pp 且 95% CI 下界 >0。`store val_acc` 只作诊断，probe 失败即停止，不进入全量训练。

## 1. 与我们问题的对应关系（不泛泛复述论文）

### 1.1 现状：CVPN 是「全局池化 + 逐 option 独立 MLP」的 DeepSets

`train/model.py` 的 `CVPN.forward_all`：

```python
projected = token_proj(tokens)                       # (B,T,d)
global_vec = masked_mean(projected, token_mask)      # (B,d)
option_vec = option_proj(options)                    # (B,K,d)
repeated = global_vec.unsqueeze(1).expand(-1, K, -1)
logits = option_scorer(torch.cat([repeated, option_vec], -1)).squeeze(-1)
logits = logits.masked_fill(~option_mask, -inf)
```

`option_scorer` 是 `Linear(2d→256) → SiLU → Dropout → Linear(256→1)`（`train/model.py:79-84`）。每个 option 的分数只由「全局池化向量 + 自己」决定，**候选之间没有交互**。这正是一个 DeepSets 形式的 set-to-vector 打分器：对 token 置换不变、对 option 置换等变，但逐元素独立（[DeepSets](https://arxiv.org/abs/1703.06114) 的 Theorem 2 / Lemma 3）。

`TransformerCVPN`（`train/model.py:294-391`）多了一步 token self-attention 和 **option→token cross-attention**：

```python
attended, _ = cross_attn(query=option_vec, key=encoded, value=encoded,
                         key_padding_mask=~token_mask)
logits = option_scorer(torch.cat([attended, option_vec], -1))
```

但它仍然**没有候选之间的 attention**；cross-attention 解决的是「这个 option 该看哪些 token」，不是「这个 option 相对其他 option 值多少」。当前 A0 基线是 `CVPN d256`（`data/checkpoints/kaggle/h4m_bc_d256_e150.pt`，见 `scripts/certify_a2.sh`），不是 `TransformerCVPN`，所以两条结构轴都还没在 A0 线上验证过。

**option 68 维的构成**（`reps/features.py:181-194`，9 种 `OptionKind`）：

| 块 | 维度 | 内容 |
|---|---:|---|
| kind one-hot | 9 | use_skill / play_card / switch_active / elemental_tuning / declare_end / choose_active / select_card / reroll_dice / switch_hands |
| used dice histogram | 8 | 实际支付的骰子类型 |
| auto-selected dice histogram | 8 | 引擎自动选择的骰子 |
| scalars | 10 | 支付数、自动数、is_fast、target 数、target_dice、subject/select/discard definition id、choose_active_slot、switch_hand_slot_mask 计数 |
| required cost buckets | 5 | exact / aligned / void / energy / legend |
| target slots | 2×6 | owner(self/opponent)、is_character、is_active、is_defeated、definition id |
| target aggregate | 4 | self / opponent / character / active 计数 |
| removed hand | 1+2 | 数量 + 两个被换手牌的 definition id |
| reroll mask | 1+8 | 数量 + 8 类骰子命中 |

这些特征已经相当丰富：费用、目标、换手、重掷都在里面。**缺的不是「费用特征」，而是费用/目标之间的相对比较**。

**K 的规模**：内部报告 `reports/explore/model_compare_20261011T132909.json` 的 pool stats 是 `mean_options=26.94, max_options=192`；`reports/train/a3_bc_v1.json` 是 `mean_options=25.48, max_options=256`。也就是说，绝大多数决策 K≈25–30，长尾到 192–256。K 不大，**全量候选打分是可行且精确的**；不需要 Wolpertinger 式的 proto-action + kNN。

**其他相关工程事实**：

- `train/data.py:137-199` 的 `collect_samples` 只把 `_CaptureTeacher` 包在 **p0** 上，p1 是普通 opponent；A2 的采集管线因而是 **p0-only** 的专家语料。`scripts/eval_protocol.py:17,107,120-122` 明确 **both sides** 评测。这是已核实的训练/评测不对称。
- store 的默认 train/val 切分按 `(seed - seed_start) % val_every == 0` 在**局级**进行（`scripts/datafirst_bc_store.py:58-61,113`），不是同局随机切样本；A2 报告没有显示同局泄漏，所以不能把假阳性归因于「val 里混了训练局」。
- `teacher_index` 是 `codes.index(code)` 得到的**位置索引**（`train/data.py:180-186`），不是语义 key；引擎 order 变了标签就会静默错位。
- `agents/neural.py:53-63` 用 `argmax`，并列时取最小 index；没有语义 tie-break。

### 1.2 问题 1：候选集交互到底应建模什么？三种最小结构

候选集交互要建模的不是「所有 option 的联合分布」（那是 $2^K$），而是**一个 option 相对于当前备选集合的 contextual value**：

$$
f(s,a_i,\mathcal{A}) \quad\text{而不是}\quad f(s,a_i).
$$

对我们最有价值的四类信号：

1. **相对费用/支配关系**：同一个技能/卡牌常有多个支付方式或目标变体；3 个匹配骰 vs 3 个万能骰，价值取决于当前骰池和后续技能。独立打分只能学一个全局费用惩罚，不能表达「现在省 1 个骰子给下回合」。
2. **目标绑定**：option 的 target slot 只编码了 definition id / owner / active / defeated；目标的 HP、aura、能量、状态在 token 里。全局 mean-pool 把 3 个角色平均掉，逐 option MLP 无法把「这个技能打的是不是那个残血角色」绑定起来。
3. **集合可用性/机会成本**：如果「切人」或「打出某张牌」可用，另一个攻击 option 的价值会变；这是 set-level 的 explaining-away。
4. **近重复选项的校准**：同一效果、不同费用/目标的 option 很多；独立 scorer 可能给多个近重复项都高分，softmax 质量被摊薄，top-1 在边界上不稳定。

三种最小结构：

> 说明：本文把「option-set attention」理解为**在 option 集合上做 attention**（Set Transformer 的 SAB / AGILE 的 action graph），不是某篇论文的专名；老师问题里的 Set Transformer、pointer network 都归到这一框架下比较。

| 结构 | 公式（单层） | 建模什么 | 解决的失败模式 | 成本 | 备注 |
|---|---|---|---|---:|---|
| **候选 self-attention**（SAB-lite） | $H=\mathrm{LN}(X+\mathrm{MHA}(X,X,X;\text{mask}))$，$H=\mathrm{LN}(H+\mathrm{FFN}(H))$，$z_i=\mathrm{MLP}([H_i,X_i,g])$ | 候选之间的软比较、相对费用、近重复、集合可用性 | 独立分数无法比较「A 比 B 便宜/强」；近重复质量摊薄；集合变化不敏感 | $O(K^2d)$ | K≈27 时很便宜；K=256 时需 ISAB/分桶 |
| **context-candidate cross-attention** | $C_i=\mathrm{MHA}(Q=X_i,K=V=\text{tokens};\text{token mask})$，$z_i=\mathrm{MLP}([X_i,C_i,g])$ | option 该看哪些 token：目标 HP/aura/状态、手牌/骰子 | 全局池化把目标信息平均掉；target slot 只有 definition id | $O(KTd)$ | `TransformerCVPN` 已有；CVPN 没有 |
| **pairwise relation** | $r_{ij}=g([X_i,X_j,X_i-X_j,X_i\odot X_j])$，$h_i=\frac{1}{\lvert\mathcal{V}\rvert}\sum_{j\in\mathcal{V}}r_{ij}$，$z_i=\mathrm{MLP}([X_i,h_i,g])$ | 显式的支配/替代/同目标比较 | 严格支配、同效果更便宜、同目标互斥；比 self-attention 更省样本 | $O(K^2d)$ | 可只算同 kind/同 target 的对，降成本 |

**在七圣召唤里的具体例子**：

- 「use_skill」有 3 匹配 / 3 万能 / 2 匹配+1 万能等多种支付；独立 scorer 看到的是三个 68 维向量，费用块不同，但不知道「现在骰池只剩 1 个万能」。候选 self-attention 让每个 option 看到其他 option 的费用分布；pairwise relation 直接学「同技能、同目标，费用更低者优先」。
- 「play_card」带 `select_card_definition_id` / `discarded_card_definition_id`；这张卡的价值取决于手牌里有没有可弃的牌、弃了会不会破坏 combo。cross-attention 让 option 查询手牌 token；self-attention 让 option 看到其他可打的牌。
- 「switch_active」 vs 「use_skill」：切人本身不造成伤害，但能让下回合用更强的技能；这个价值取决于当前 available skills 集合。self-attention 的集合上下文能表达「因为切人可用，所以现在打这个技能的机会成本高」。
- 「declare_end」：结束回合的价值取决于剩余骰子、对手威胁、下回合资源；它和「use_skill」是直接替代关系。独立 scorer 可以学一个 end bias，但不能比较「现在打光骰子 vs 留骰子」。
- `choose_active` / `select_card` / `reroll_dice` / `switch_hands` 的 K 通常很小（3–8，reroll 可到 256），语义也更局部；**先按 request_type 分组观察，不要一上来对所有类型共享一个候选交互块**。

**不该建模的**：

- 不要用 PMA（Pooling by Multihead Attention）直接产出一个 set summary 当 policy logits；PMA 是 set-to-vector，适合 value head，不适合 per-option 输出。policy 要用 SAB（equivariant），不是 PMA（invariant）。
- 不要给 option 加位置编码或 engine index；那会把「引擎枚举顺序」当成语义，破坏置换等变，也会在引擎版本变化时失效。
- 不要建 full joint distribution over subsets；引擎一次只选一个 action，per-option softmax 已经足够。

### 1.3 问题 2：动作数可变、顺序由引擎决定时，如何保证置换等变/不变与训练-评测一致？

**定义**：设 $X\in\mathbb{R}^{K\times d}$ 是 option 特征，$m\in\{0,1\}^K$ 是合法 mask。逐 option 打分器 $f$ 应满足

$$
f(PX,Pm)=P\,f(X,m)\quad\text{对所有置换矩阵 }P.
$$

策略分布是 $ \mathrm{softmax}(f) $，因此分布也等变；损失在目标索引同步置换时不变。若模型是 set-to-scalar（value head），则要求 $f(PX,Pm)=f(X,m)$（不变）。

**保证等变的四条**：

1. **参数共享**：所有 option 过同一个 `option_proj` / scorer；不要 per-slot 或 per-index 参数。
2. **无位置编码**：不加 option position embedding、不加 engine index 特征、不把 option 按固定槽位拼接。
3. **mask 贯穿所有交互**：attention 的 `key_padding_mask`、pairwise 的 pair mask、pooling 的 valid count；pad 不能参与 softmax/归一化。
4. **归一化用 valid 数**：$\frac{1}{\sum_j m_j}\sum_j$，不要用固定的 K；否则 K 变化时 logit 尺度漂移。

**训练-评测一致的六个检查**：

- **目标用语义 key，不用位置索引**。当前 `teacher_index` 是位置索引；建议在 `Sample` 里同时存 `teacher_semantic_key`（`reps.action_hierarchy.semantic_action_key_for_spec`），训练/评测时按当前 `legal_low_level_specs` 重新映射。若保留位置索引，至少加一个「同一 context 打乱 option 后标签重映射」的测试。
- **规范顺序（canonical order）**：如果确实需要顺序（例如 tie-break、缓存、batch 分桶），按语义 key 排序（kind、subject id、target ids、cost），并保存 `order`/`inverse_order` 映射回 engine code；不要按 engine 返回顺序做位置特征。注意：排序本身是 set 的函数，仍可保持等变，但 tie 需要稳定的二级 key。
- **tie-break 语义化**：`argmax` 并列时用最小语义 key，不要用最小 index；否则打乱 option 后同分选项会选到不同语义动作。
- **mask/padding**：
  - `collate` 已按 batch 内最大 K padding（`train/model.py:432-452`；`scripts/train_bc_scale.py:401-421`）。pad 的 option 特征是 0，`option_mask=False`。
  - self-attention：`key_padding_mask = ~option_mask`；pad query 的输出在 block 后 `* option_mask`；**保证每个样本至少 1 个合法 option**，否则 PyTorch MHA 全 mask 行会产生 NaN。
  - cross-attention：`key_padding_mask = ~token_mask`；option query 的 pad 行同样清零。
  - pairwise：`pair_mask = option_mask[:, :, None] & option_mask[:, None, :]`；无效对置 0，分母 `pair_mask.sum(-1).clamp(min=1)`；自配对可排除或单独加一个 self term。
  - pooling：`(H * mask).sum(1) / mask.sum(1).clamp(min=1)`。
  - loss：`logits.masked_fill(~option_mask, -inf)` 后 CE；`teacher` 必须 `< option_mask.sum(-1)`；全 mask 行跳过或报错。PG/`log_softmax` 用 `torch.where(mask, logp, 0)`，避免 `0 * -inf = NaN`（`scripts/train_bc_scale.py:525-529` 已有这个保护，CE 路径也应加）。
- **batch 独立性**：不要用 BatchNorm 或跨样本的候选归一化；LayerNorm 是逐元素的，安全。测试「同一条样本单独 forward vs 与 K 更大的样本同 batch forward，合法 logits 必须一致」。
- **训练/评测同分布**：同一 `encoder_config`（checkpoint 已存）、同一 `engine.lock`、同一卡池、同一 option 顺序、同一 dtype、`model.eval()`。A2 已核对 token_dim 65 / option_dim 68 / definition 列 42/30 一致（`reports/explore/a2_id_screening.md:61-64`），但仍建议加运行时断言。
- **p0-only vs both sides**：A2 的采集函数只采 p0，而 eval 跑双方；新结构实验要么训练也采 p1，要么只在 p0 侧认证并明确声明。否则 p1 的增益/损失会被误读。

**最小 pseudocode**：

```python
def masked_sab(x, mask, mha, ln1, ff, ln2):
    pad = ~mask
    a = mha(x, x, x, key_padding_mask=pad, need_weights=False)
    x = ln1(x + a)
    x = ln2(x + ff(x))
    return x * mask.unsqueeze(-1).to(x.dtype)

def pairwise(x, mask, g):
    pair = mask[:, :, None] & mask[:, None, :]          # (B,K,K)
    r = g(x[:, :, None, :], x[:, None, :, :])           # (B,K,K,d)
    r = r * pair.unsqueeze(-1).to(r.dtype)
    h = r.sum(dim=2) / pair.sum(dim=2, keepdim=True).clamp(min=1)
    return h

def remap_target(specs, teacher_semantic_key):
    for i, spec in enumerate(specs):
        if semantic_action_key_for_spec(spec) == teacher_semantic_key:
            return i
    raise ValueError("teacher action not in legal set")
```

**测试**（建议新建 `tests/test_candidate_set.py`）：

```python
perm = torch.randperm(K)
logits_perm, _ = model(tokens, token_mask, options[:, perm], option_mask[:, perm])
assert torch.allclose(logits_perm, logits[:, perm], atol=1e-5)
assert argmax_semantic(logits_perm) == argmax_semantic(logits)
```

再加：zero-init 时 adapter logits 与 base logits 差 <1e-6；单样本 vs batch 一致；全 mask 行报错而不是 NaN。

### 1.4 问题 3：骰子费用/多目标动作是否需要特殊结构（费用约束、组合动作分解、pointer 选择目标）？还是特征工程足够？

**结论：先特征工程，后 pairwise；pointer/分解只在 K 爆炸或需要子集动作时考虑。**

- **费用约束**：引擎给出的 `legal_low_level_specs` 已经保证合法；BC 不需要学「能不能付」，只需要学「值不值得」。所以不要加 feasibility classifier；加了反而可能把引擎的合法性判断学错。当前 5 个 required-cost bucket + 8 维 used/auto dice histogram + target_dice 已经覆盖了大部分一阶费用信号。最值得补的是**支付后剩余骰池**：`pool[type] - used[type] - auto[type]`（语义需对照 `reps/schema.py` 的 `LowLevelActionSpec` 核实），以及「同 kind 中最便宜 option 的 cost」作为相对特征。这是纯特征工程，成本最低，优先级高于新结构。
- **多目标**：`_TARGET_SLOTS=2`，每个 slot 6 维；引擎把 joint target assignment 枚举成独立 option。所以**原子打分**天然保持 100% 覆盖，不需要 pointer。只有当目标组合爆炸（例如 2 个 slot 从 5 个目标里选，K 变成 20+）或需要泛化到未见组合时，才考虑分解。当前 3 个角色 × 2 slot 最多 9 种 joint target，K 不大，原子打分更稳。
- **pointer 选择目标**：Pointer Networks 的输出字典 = 输入位置，适合「从输入 token 里选一个/一串」；如果要做，正确用法是 **每个 target slot 一个 pointer head，mask 已选目标，训练 NLL，推断时枚举/beam 并检查引擎合法性**。不要用它替换整个 option list；它会把「选择哪个目标」和「选择哪个 kind/费用」拆开，必须保证组合后仍是引擎合法动作。Pointer Networks 的原始实验也说明顺序敏感：Delaunay 输出是 set，作者必须按 incenter 规范排序，否则效果变差（[Pointer Networks](https://arxiv.org/abs/1506.03134)，§3.2）。
- **动作分解（action factorization）**：把 $p(a)$ 分解为 $p(\text{kind})p(\text{cost}\mid\text{kind})p(\text{target}\mid\text{kind},\text{cost})$ 可以降低输出维度、共享统计强度；但推断时必须枚举/beam 所有组合并 mask 非法组合，且要精确归一化才能和原子 softmax 比较。BDQ 的 action branching 在连续控制上有效（[BDQ](https://arxiv.org/abs/1711.08946)），但它允许各分支独立选择，可能产生非法组合；我们的引擎已经枚举 joint action，原子打分更符合 100% 覆盖约束。独立选择 vs 自回归选择动作分量的系统比较可参考 [Factored Action Spaces in Deep RL](https://openreview.net/forum?id=naSAkn2Xo46)（具体结论未核实）；离线 RL 的线性 Q 分解见 [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dda7f9378a210c25e470e19304cce85d-Paper-Conference.pdf)。**不要在当前阶段做分解**；如果以后做，先按 `request_type`/`OptionKind` 分头（Suphx 的五模型就是这种），并在全局 legal set 上归一化，而不是每个 kind 独立 sigmoid。
- **什么时候 pointer/分解才值得**：`reroll_dice` 的 mask 空间是 $2^8=256$，`switch_hands` 类似；如果这些决策占比高、且 K=256 的 self-attention 显存吃紧，可以用 8 个 binary pointer/sigmoid 头 + 合法性 mask 来因子化。但这类决策只占一部分，先做 request-type 分桶/K 分桶即可。

### 1.5 问题 4：为什么 A2 的 id embedding 会产生 val_acc 假阳性？新结构实验如何避免重演？

**A2 的已核实事实**（`reports/explore/a2_id_screening.md`）：

| 模型 | store val_acc | 引擎 dgeo | 引擎 ungeo | 引擎 dvalin |
|---|---:|---:|---:|---:|
| A0（h4m e150） | 0.838 | 0.720 | 0.828 | 0.760 |
| A0i（CVPN d256 + id1024） | **0.866** | **0.266** | **0.358** | 0.678 |
| A0i_noid（同 trunk，id 关） | 0.397（dgeo） | 0.741 | 0.670 | 0.303 |

- A0i 的 val_acc **涨了 2.8pp**，dgeo/ungeo 引擎胜率 **掉 45–47pp**；dvalin 只掉 8pp。三个 deck 全部显著变差，但幅度强 deck 依赖。
- A2 最终 checkpoint 是**过拟合的最后一个 epoch**：best val 0.8839 @ ep70，final 0.8658 @ ep149；val_ce 从 0.3747 升到 0.5316（`a2_id_screening.md:29-33`）。`scripts/datafirst_bc_train.py:176` 只 `model.save(...)` 最后一个 epoch，没有 `--save-best`。
- A0i_noid 的 val_acc 掉到 0.397，但引擎 dgeo 反而 0.741：**trunk 已经学会依赖 id 分支**，把 shortcut 拿掉后 store 指标崩、引擎不一定崩。这说明 id 分支不是「可插拔的增益」，而是改变了 trunk 的表示。
- A2 的 val 切分规则是局级 seed 切分（`datafirst_bc_store.py:58-61`）；A2 报告没有显示同局泄漏，所以不能把假阳性归因于「val 里混了训练局」。A2 报告自己也说 definition 列 42/30、token 65/option 68 都核对过（`a2_id_screening.md:61-64`），不是简单列错位。

**最可能的四个机制**（按证据强度排序）：

1. **指标错位（已核实）**：val_acc 衡量「在专家状态上模仿 scripted expert 的 top-1」，不是「赢」。BC 的 val_acc 与引擎胜率没有单调关系；A2 是最干净的展示。
2. **shortcut/容量（强怀疑，未逐元素核实）**：id embedding 让模型能识别卡/角色/牌组。4-deck 语料里，自己的卡牌 id 几乎是 deck identity 的代理；模型可以学「这套 deck 的专家偏好」而不是「这个状态下的最优动作」。val 来自同 4 个 deck，shortcut 被奖励；eval 时模型走自己的状态分布，deck-specific prior 变脆。`reps/id_features.py:26-36` 的 1024 桶还有碰撞风险（次要）。
3. **分布偏移（已核实一半）**：A2 的采集函数是 expert-vs-expert 且只 capture p0（`train/data.py:137-199`），eval 是 self-generated 且 both sides（`scripts/eval_protocol.py:17,107`）。这是标准的 BC covariate shift（[DAgger](https://arxiv.org/abs/1011.0686)）；p1 侧完全是训练时没见过的角色。A2 报告怀疑的「store→engine 特征不一致」仍未逐元素核实，但 p0-only/both-sides 是确定的不对称。
4. **聚合掩盖长尾（已核实）**：pooled val_acc 把所有 request type 混在一起；`reroll_dice`、`choose_active` 这类简单决策占比不小，2.8pp 的池化增益可能来自简单决策，而 `action` 子集在退步。A2 的 deck 表已经显示 dvalin 只掉 8pp、dgeo/ungeo 掉 45pp；pooled 数字会掩盖这种长尾。

**新结构实验的设计检查表**：

| 风险 | A2 的表现 | Probe 的防护 |
|---|---|---|
| 指标错位 | val_acc↑、胜率↓ | 主指标 = 引擎胜率；val_acc 只作诊断 |
| shortcut/容量 | id 分支记 deck/card | 不加 id；共享参数；deck-held-out |
| trunk 耦合 | A0i_noid val 崩 | **冻结 trunk**；零初始化 adapter；初始 logits = A0 |
| 分布偏移 | p0-only 训练、both-sides eval | 训练也采 p1，或只在 p0 认证并声明 |
| 过拟合 | 最后一个 epoch 更差 | 只训 1–2 epoch；`--save-best`；early stop |
| 聚合 | pooled 掩盖 deck 长尾 | 逐 deck、逐 request type、逐 side 报 |
| 顺序泄漏 | 位置索引 target | 语义 key；打乱 option 测试 |
| padding 泄漏 | pad 参与 attention | mask 测试；batch 独立性测试 |
| 噪声 | 小 val 集 | 配对种子；CI；随机 adapter 负对照 |

**一句话**：A2 不是「id 特征列错了」，而是 **proxy metric + shortcut + 分布偏移 + 过拟合** 的叠加；新实验必须把「结构增益」从「trunk 变化/捷径/评测噪声」里隔离出来，所以冻结 trunk + 零初始化 + 引擎胜率 + 逐 deck 是硬要求。

### 1.6 问题 5：最小成本 probe——冻结 trunk，只加候选 self-attention，1–2 epoch，看什么信号才值得放大？

**Stage 0（0 成本，先做）**：在现有 store 上抽 20k 决策，跑 A0 forward，统计：

- K 分布（按 request_type）。
- top-2 margin：`logit_top1 - logit_top2`；margin <0.1 / <0.5 的比例。
- 近重复率：同 `OptionKind` + 同 target slots、但 cost bucket 或 used dice 不同的 option 对比例。
- A0 的 teacher rank：teacher 是 top-1 / top-2 / top-3 的比例。
- 按 request_type 的 top-1 error；`action` 子集单独看。
- 费用敏感子集：同 kind 有更便宜 option 时，A0 是否选了更贵的；teacher 是否选了更便宜的。

**判据（建议阈值，按实际分布调整）**：如果 margin<0.1 的比例 <5%，或近重复率 <5%，或 A0 的错误不集中在这些子集（例如 `action` 错误率与整体差不多），**停止**，不要训练。若 `action` 子集错误率 ≥2× 整体，且其中 ≥30% 是近重复/费用敏感/目标绑定，才有做结构的意义。

**Stage 1（1–2 epoch adapter）**：

- 载入 A0，冻结全部参数；在 `option_proj` 后加 1 层 4-head SAB（d256，FF512），再接一个 zero-init 的 `delta_head`：`logits = base_logits + delta`。`delta_head` 最后一层权重/偏置置 0，初始 logits 与 A0 完全一致。
- 只训练 SAB + delta head（约 0.6M 参数）；batch 512，lr 1e-3，weight decay 0，dropout 0，seed 0；1–2 epoch；`mixed_four_decks.pt` 542k/28.8k。zero-init 的 delta head 首步只更新最后一层，SAB 从第 2 步开始拿到梯度；若担心太慢，可把最后一层初始化为 1e-3 并在报告里注明「非严格 A0 起点」。
- 记录：train/val CE、top-1、**按 deck 分组**的 top-1、按 request_type 的 top-1、平均 `|delta|`、相对 A0 的 decision flip rate、attention 是否 attends to pads（应为 0）。严格 deck-held-out 需要另训 leave-one-deck-out adapter（成本×4），probe 阶段可先按 deck 分组报；确认阶段再做严格 held-out。

**Stage 2（引擎 screen）**：3 deck × 20 opponents × 10 seeds × 2 sides = 400 局/deck；A0 与 adapter 同种子；paired Δ。**Stage 3（confirm）**：50 seeds × 20 opponents × 2 sides = 2,000 局/deck。

**什么信号才值得放大**：

1. **不是惰性**：平均 `|delta logit| > 0.05`（或 > A0 logit 标准差的 5%），decision flip rate 在 2%–30%。<1% 说明没学到；>50% 说明在乱翻，先查 mask/lr。
2. **store 诊断正向**：按 deck 分组的 top-1 至少 +1.5pp（≥2/3 deck），且没有 deck 掉 >0.5pp；`action` 子集有提升，而不只是 `reroll_dice`/`choose_active`。严格 leave-one-deck-out 的检查放到确认阶段。
3. **引擎 screen 正向**：≥2/3 deck Δ ≥ +6pp，无 deck ≤ −2pp，pooled Δ >0。400 局/deck 只能抓大效应；+2pp 在这个样本量下基本测不出，别把「不显著」当「无效」，但也别为 +2pp 放大。
4. **confirm 正向**：≥2/3 deck Δ ≥ +3pp 且 95% CI 下界 >0，无 deck ≤ −2pp；pooled Δ ≥ +2pp 且 CI 下界 >0。
5. **机械检查通过**：permutation test、batch independence、zero-init identity、无 NaN/全 mask、value Brier 不劣化 >0.01（probe 冻结 value head，理论上不变）。
6. **负对照不显著**：随机 adapter（同参数量、随机特征）不应在 store val 或引擎上「提升」；如果它也提升，说明评测噪声/流程有问题。

**停止判据**（任一即停）：

- Stage 0 无信号（歧义/近重复 <5%，或错误不集中）。
- zero-init 不成立、permutation/batch 测试失败、NaN。
- 2 epoch 后 `|delta| < 0.05` 或 flip rate <1%，或 deck-held-out top-1 增益 <1pp。
- screen <2/3 deck +6pp，或任一 deck ≤−2pp，或 pooled ≤0。
- confirm <2/3 deck +3pp 且 CI 下界 >0，或任一 deck ≤−2pp。
- store val 涨但引擎不涨（A2 模式）→ 停，不要用「再加一层/再训久一点」救。

## 2. annotated bibliography

> 可信度：**A** = 读了一手正文/关键代码；**B** = 读摘要/README/部分代码；**C** = 只有搜索片段/二手，标「未核实」。访问日期均为 2026-10-27。

### 2.1 DeepSets: A Framework for Set-Structured Data（Zaheer et al., NeurIPS 2017）

- **问题**：如何构造对集合置换不变/等变的网络。
- **方法**：Theorem 2 证明置换不变函数可写成 $\rho(\sum_{x\in X}\phi(x))$；Lemma 3 给出线性层等变的充要条件 $\Theta=\lambda I+\gamma\mathbf{1}\mathbf{1}^T$；等变层可写成 $\sigma(\lambda x+\gamma\,\mathrm{pool}(X))$。
- **证据**：集合异常检测中，等变层 75% vs 非等变基线 6.3%（随机）；点云 ModelNet40 90.3% @5000 点；数字求和/人口统计等任务。
- **局限**：sum/mean pooling 会丢弃元素间交互；set-to-scalar 的万能性不等于 set-to-set 的万能性。
- **链接**：[arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- **可迁移**：我们的 CVPN 就是 $\rho([\mathrm{mean}(\phi(token)),\psi(option)])$；它是合法的 DeepSets，但逐 option 独立。加 SAB 是在保持等变的前提下补交互；不要 flatten option 或加位置。

### 2.2 Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks（Lee et al., ICML 2019）

- **问题**：pooling 架构无法建模集合元素间交互。
- **方法**：MAB / SAB / ISAB / PMA。`SAB(X)=MAB(X,X)`，`MAB(X,Y)=LN(H+rFF(H))`，`H=LN(X+MHA(X,Y,Y))`；ISAB 用 $m$ 个 inducing points 把 $O(n^2)$ 降到 $O(nm)$；PMA 用 seed vectors 做 set-to-vector 聚合。
- **证据**：unique character counting：SAB+PMA 0.6037 vs mean pooling 0.4382；max regression MAE 0.2085 vs mean pooling 2.133；meta clustering ARI 0.9015 vs 0.5593。说明「需要元素间比较/计数」时，self-attention 明显优于 pooling。
- **局限**：SAB 是 $O(K^2)$；官方代码（`modules.py`）**没有 mask**，直接搬来处理变长 option 会把 pad 也算进 attention；PMA 输出 set summary，不适合 per-option policy。
- **链接**：[arXiv:1810.00825](https://arxiv.org/abs/1810.00825)；[官方代码](https://github.com/juho-lee/set_transformer)
- **可迁移**：用 SAB 做候选 self-attention；PMA 只用于 value head；必须自己加 `key_padding_mask`（官方实现没有）。

### 2.3 Pointer Networks（Vinyals et al., NeurIPS 2015）

- **问题**：输出字典大小随输入长度变化，固定 softmax 无法处理。
- **方法**：$u_j^i=v^T\tanh(W_1e_j+W_2d_i)$，$p(C_i\mid C_{<i},\mathcal{P})=\mathrm{softmax}(u^i)$；输出是输入位置的指针；训练用 NLL，推断用 beam search + 合法性约束。
- **证据**：convex hull n=50：Ptr-Net 72.6% vs attention-LSTM 38.9%；可泛化到 n=500（1.3% exact，99.2% area）；TSP n=5–20 接近最优。Delaunay 输出是 set，作者必须规范排序，否则效果变差（§3.2）。
- **局限**：顺序敏感；输出必须是输入位置的子集/序列；不是通用的 candidate scorer；beam search 需要合法性 mask。
- **链接**：[arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- **可迁移**：只用于「从输入 token 里选目标/骰子/手牌」的子集动作（reroll mask、switch hands、target slot）；不要替换整个 option list；用规范顺序 + mask。

### 2.4 A Simple Neural Network Module for Relational Reasoning（Santoro et al., NeurIPS 2017）

- **问题**：标准 MLP/CNN 难以做实体间关系推理。
- **方法**：$RN(O)=f_\phi(\sum_{i,j}g_\theta(o_i,o_j))$；所有对象对过同一个 $g_\theta$，sum pooling 保证置换不变；可条件化 $g_\theta(o_i,o_j,q)$。
- **证据**：CLEVR 95.5% vs 之前 SOTA 68.5%；Sort-of-CLEVR 关系题 RN 94% vs CNN+MLP 63%；bAbI 18/20。说明「全局向量 + MLP」在关系题上会卡在 63%，加 pairwise relation 能到 94%。
- **局限**：$O(n^2)$；只建模 pairwise，不建模高阶；需要定义 object。
- **链接**：[arXiv:1706.01427](https://arxiv.org/abs/1706.01427)
- **可迁移**：我们的 CVPN 就是「全局 pooled 向量 + per-option MLP」，和 CNN+MLP 的结构同构；target binding / dominance 是关系题。pairwise relation 可作为 self-attention 的低成本替代/补充；只算同 kind/同 target 的对。

### 2.5 Deep Reinforcement Learning in Large Discrete Action Spaces（Dulac-Arnold et al., 2015；即 Wolpertinger）

- **问题**：动作数巨大（最高 1M）时，逐动作 Q 评估不可行。
- **方法**：actor 输出连续 proto-action $\hat a=f_\pi(s)$；$g_k(\hat a)$ 取 k 个最近邻离散动作；再用 critic 在候选内 refine：$\pi(s)=\arg\max_{a\in g_k\circ f_\pi(s)}Q(s,a)$。用 DDPG 训练；kNN 可近似到对数时间。
- **证据**：cart-pole 1M actions、puddle world 2^20 plans、13k recommender；k=1 在 13k recommender 上失败；5–10% 动作即可接近全量；k 的收益递减（Lemma 1，$O(p^k)+O(1/(k+1))$）。
- **局限**：需要动作嵌入 + 近似最近邻；kNN 引入近似误差；不是为「每状态只有几十个合法动作」设计的。
- **链接**：[arXiv:1512.07679](https://arxiv.org/abs/1512.07679)
- **可迁移**：**不要**把 Wolpertinger 直接搬来。我们 K≈27，全量打分精确且便宜；proto-action/kNN 只会增加复杂度和 off-policy 问题。可借鉴的只是「共享 action encoder + Q head」——DouZero 已经这么做。另：该文正文/附录未出现 log-density off-policy correction；若老师指的是后续工作，**未核实**。

### 2.6 Learning to Rank: From Pairwise Approach to Listwise Approach（Cao et al., ICML 2007；ListNet）

- **问题**：pairwise ranking 忽略「列表是一个整体」。
- **方法**：permutation probability 与 top-one probability；top-one 为 $P_s(j)=\phi(s_j)/\sum_k\phi(s_k)$，用 cross-entropy 训练。ListNet 的 listwise loss 对每个 query 的整个列表定义。
- **证据**：IR 数据集上 listwise 优于 pairwise（论文报告；具体数字未逐表核对，标 B）。
- **局限**：需要每个 item 的 relevance/ranking 标签；我们只有「专家选了哪个」一个正例，top-1 退化就是 softmax CE。
- **链接**：[MSR-TR-2007-40 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)；[DOI](https://doi.org/10.1145/1273496.1273513)
- **可迁移**：**我们的 CE 已经是 top-1 listwise / Plackett-Luce**；问题不在 loss，在 scorer。候选 self-attention 改的是 $s_j$，loss 不用换。若以后有 search 的软策略目标，`train/distill.py` 已支持 soft CE，可无缝接。

### 2.7 ListMLE / Plackett-Luce（Xia et al., ICML 2008；Guiver & Snelson, ICML 2009）

- **问题**：如何直接优化 permutation likelihood；PL 模型的参数推断。
- **方法**：Plackett-Luce 似然 $P(y\mid x)=\prod_{i=1}^{n}\frac{\phi(s_{y_i})}{\sum_{k=i}^{n}\phi(s_{y_k})}$；ListMLE 最大化该似然；top-1 特例就是 softmax CE。Guiver & Snelson 给 PL 的 Bayesian 推断。
- **证据**：ICML 2008 论文报告 listwise 方法在 ranking 任务上的优势；PL Bayesian 推断是标准方法；具体数字未逐表核对（B）。
- **局限**：需要完整 ranking 标签；我们只有 chosen action，不能构造完整 permutation；PL 的 top-1 特例不提供额外信息。
- **链接**：[ListMLE ICML 2008 PDF](https://icml.cc/Conferences/2008/papers/167.pdf)；[PL DOI](https://doi.org/10.1145/1553374.1553423)
- **可迁移**：确认「top-1 CE 是 listwise / PL 的一个特例」；不要为了「listwise」而引入不可得的 full ranking 标签；候选交互只改 logits，不改变 PL 的归一化。

### 2.8 DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning（Zha et al., ICML 2021）

- **问题**：斗地主合法动作随手牌剧烈变化，全局动作空间 27,472。
- **方法**：状态特征 + **每个合法动作的 54 维 one-hot 编码**（4×13 牌 + 2 joker）拼接，state 特征对每个 legal action 重复；MLP/LSTM 给每个动作打分；动作序列用 15 步 × 54 维编码。**没有候选间 attention**。
- **证据**：ICML 2021，DouDizhu 超人类；单机 4 GPU 数天；Botzone 第一（论文声称）。代码：`douzero/env/env.py` 的 `_get_obs_*` 把 `my_action_batch` 与 state 特征 `hstack`，`legal_actions` 动态枚举。
- **局限**：动作编码是固定的 54 维牌集合，适合「动作=牌组合」；我们的 option 是 kind/cost/target 组合，68 维更丰富但也更异构。其 `InfoSet.other_hand_cards` 是其他两家手牌并集，信息口径与我们的掩码不同；引用其强度前需先核实论文对训练/评测信息口径的正式说明（**未核实**）。
- **链接**：[arXiv:2106.06135](https://arxiv.org/abs/2106.06135)；[代码](https://github.com/kwai/DouZero)；[env.py](https://github.com/kwai/DouZero/blob/main/douzero/env/env.py)
- **可迁移**：**这是 A4 最强的反证**——一个大型卡牌游戏用逐动作独立打分就能超人类；候选交互不是必要条件。可借鉴的是「每个动作有规范、可共享的编码」和「state 特征按动作重复」，而不是 set attention。若我们的 option 编码已经够规范，A4 的增益可能很小。

### 2.9 Suphx: Mastering Mahjong with Deep Reinforcement Learning（Li et al., 2020）

- **问题**：日本麻将，复杂动作类型（打牌/立直/吃/碰/杠），隐藏信息巨大。
- **方法**：五个模型分别处理 discard / Riichi / Chow / Pong / Kong；discard 是 34 类输出，Riichi/Chow/Pong/Kong 是二分类；输入是 34×838/958 的 CNN 特征 + look-ahead features；Chow/Pong/Kong 输入还包含「用哪张牌吃/碰」的信息。没有候选集 attention。
- **证据**：Tenhou 10 dan、stable rank 8.74、高于 99.99% 人类；RL + global reward prediction + oracle guiding。论文报告 discard 测试准确率 76.7%，Chow 95.0%，Pong 91.9%，Kong 94.0%。
- **局限**：动作类型少且固定；五模型是人工分头，不是通用 candidate encoder；没有公开代码。
- **链接**：[arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
- **可迁移**：**按动作类型分头**可能比「一个 scorer 处理所有 option」更有效；我们的 5 个 request_type / 9 个 OptionKind 可以先用 shared trunk + per-kind scorer 做消融。这也是候选集建模之外更便宜的杠杆。

### 2.10 DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker（Moravčík et al., Science 2017）

- **问题**：HUNL 的不完全信息 + 巨大动作空间。
- **方法**：continual re-solving；深度受限 lookahead；sparse lookahead tree 只保留 fold/call/2–3 个 bet/all-in（≤6 个动作）；叶子用 learned counterfactual value network（1,000 个 hand clusters）估计；不需要显式 action abstraction 的 translation step。
- **证据**：44,852 手对 33 位职业玩家，492 mbb/g，>4σ；LBR 无法找到可利用性下界。
- **局限**：动作抽象/稀疏化会 void 理论保证；value network 依赖 hand clustering；不是 BC 的候选 scorer。
- **链接**：[arXiv:1701.01724](https://arxiv.org/abs/1701.01724)；[Science DOI](https://doi.org/10.1126/science.aam6960)
- **可迁移**：动作抽象是**搜索时**控制 branching 的技术，不是 BC 的候选编码；如果以后做 fork/search，可以只展开少量候选，但必须保证合法动作覆盖或不删 action；BC 阶段保留全部 legal options。

### 2.11 Libratus: The Superhuman AI for No-Limit Poker（Brown & Sandholm, Science 2018 / IJCAI 2017）

- **问题**：HUNL 的 action abstraction 与 subgame solving。
- **方法**：blueprint 用 action abstraction（多数 bet size 是 pot 的「漂亮分数/倍数」；前几轮用参数优化选 bet sizes，后两轮更粗）；nested safe subgame solving；self-improvement 根据对手实际 bet size 在后台加分支并重算。
- **证据**：击败四位顶级职业玩家；Science 2018。具体每轮 bet size 数量/抽象规模**未核实**（未读全文表格）。
- **局限**：action abstraction + translation 会带来可利用性；需要大量离线计算；不是 learned candidate encoder。
- **链接**：[Science DOI](https://doi.org/10.1126/science.aao1733)；[IJCAI 2017 PDF](https://ijcai.org/proceedings/2017/0772.pdf)
- **可迁移**：同 DeepStack——搜索时的候选裁剪；不要拿来做 BC 的 action space 裁剪，否则违反 100% 覆盖。

### 2.12 ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games（Brown et al., NeurIPS 2020）

- **问题**：把 RL+search 扩展到不完全信息博弈。
- **方法**：public belief state（PBS）作为状态；在 **public tree** 上做 depth-limited subgame search；value network 估 infostate values；policy network 给 CFR warm start；搜索在离散表示上做 CFR-D；HUNL 把动作空间缩到 **≤9 个动作**（domain knowledge 的典型 bet sizes），测试时遇到 off-tree action 就把它加进 subgame。
- **证据**：HUNL 超人类；Liar's Dice 收敛到近似 Nash；对 Dong Kim 165 ± 69 mbb/g；LBR 881 ± 94。
- **局限**：PBS 维度随 infostate 数线性增长；理论保证限 2p0s；policy network 输出是按 infostate 的动作分布，不是 candidate-set attention。
- **链接**：[arXiv:2007.13544](https://arxiv.org/abs/2007.13544)；[代码（Liar's Dice）](https://github.com/facebookresearch/rebel)
- **可迁移**：「≤9 个动作 + off-tree 扩展」是搜索时的 branching 控制；核心原则是**不丢合法动作**。我们的 `option_mask` 已经是 off-tree 安全的（引擎给全 legal set）；不要为了效率静默丢 action。

### 2.13 Know Your Action Set: Learning Action Relations for Reinforcement Learning（Jain et al., ICLR 2022；AGILE）

- **问题**：动作集合随状态变化，动作之间存在依赖/关系。
- **方法**：把 state encoding 与每个 action representation 拼成 action graph 节点；GAT 在所有动作节点上做 attention，学「哪些其他动作与当前动作相关」；utility network 用 state + mean-pooled relational action features 给每个动作打分。消融包括 utility policy、mask input/output、Summary-GAT/LSTM/DeepSet、AGILE-only-action。
- **证据**：作者报告在 CREATE / MiniGrid / RecSim 上优于 utility/mask/DeepSet 基线；**具体效应量未核实**（OpenReview PDF 被浏览器验证挡住，只读了 README/代码结构）。
- **局限**：动作是 recommender item / grid action，不是卡牌组合动作；K 通常 20 左右；GAT 是全连接图，$O(K^2)$。
- **链接**：[OpenReview](https://openreview.net/forum?id=MljXVdp4A3N)；[代码](https://github.com/clvrai/agile)
- **可迁移**：这是与「option-set attention」最接近的公开实现：动作图 + GAT + utility head + mean-pool summary。我们的 1 层 SAB 是它的简化版；可以借鉴「state 也作为一个节点」和「mean-pool 的 relational summary 给 utility」。但证据不在卡牌游戏，不能当强度承诺。

### 2.14 Action Branching Architectures for Deep Reinforcement Learning（Tavakoli et al., AAAI 2018；BDQ）

- **问题**：多维度动作空间的组合爆炸。
- **方法**：shared decision module + 每个 action dimension 一个 branch；输出数随维度线性增长；BDQ 是 Dueling DDQN 的 branching 变体；共享模块负责协调各 branch。
- **证据**：连续控制任务上扩展到高维动作，和 DDPG 竞争；共享模块对协调重要。无卡牌游戏证据。
- **局限**：各 branch 独立选择，可能产生非法/冲突的联合动作；需要额外的约束/协调。
- **链接**：[arXiv:1711.08946](https://arxiv.org/abs/1711.08946)；[代码](https://github.com/atavakol/action-branching-agents)
- **可迁移**：如果按 `OptionKind`/`request_type` 分头，BDQ 是参考；但我们的引擎枚举 joint action，原子打分天然保证联合合法。分头时要在全局 legal set 上归一化，不要每个 branch 独立 sigmoid。

### 2.15 DAgger + Shortcut Learning（Ross et al., AISTATS 2011；Geirhos et al., Nature Machine Intelligence 2020）

- **问题**：BC 在序列决策中违反 i.i.d. 假设，测试时状态分布由策略自己诱导；模型还会学「在 benchmark 上有效但不迁移」的捷径。
- **方法**：DAgger 迭代采集「当前策略访问的状态 + 专家动作」，用 no-regret 在线学习聚合；Shortcut Learning 是综述/立场文章，给出一组诊断与基准建议（OOD、反事实、解释）。
- **证据**：DAgger 在多个模仿学习/结构化预测任务上优于朴素 BC；Shortcut Learning 跨视觉/NLP 的案例集合。具体数字未逐表核对（B）。
- **局限**：DAgger 需要专家在环，我们目前的 collector 是 expert-vs-expert；Shortcut 不是卡牌游戏实证。
- **链接**：[DAgger arXiv:1011.0686](https://arxiv.org/abs/1011.0686)；[Shortcut arXiv:2004.07780](https://arxiv.org/abs/2004.07780)
- **可迁移**：解释 A2 的 store→engine 偏移和 id shortcut；新结构 probe 至少要在 engine 上评，不能只看 store；用 deck-held-out、per-deck、engine 指标防 shortcut。后续如果 A4 有效，再接 DAgger 式采集。

## 3. 最小可复现 recipe（3–5 个，按性价比排序）

### R0. 先修「语义目标 + 置换测试」（0 训练，最高优先级）

- **输入/输出**：现有 store（`data/store/mixed_four_decks.pt`）与 `Sample.teacher_index`；输出是带 `teacher_semantic_key` 的新样本字段 + 一组测试。
- **关键改动**：
  - `train/data.py`：`Sample` 增加 `teacher_semantic_key`；`collect_samples` 里用 `semantic_action_key_for_spec(context.legal_low_level_specs[index])` 填充；保留 `teacher_index` 兼容旧 store。
  - `train/train_loop.py::_batch_loss`：`teacher = teacher.clamp(max=k_lens - 1)` 或 assert；全 mask 行报错。
  - `agents/neural.py`：argmax tie-break 用语义 key；或至少在文档里声明「engine order 固定」。
  - `tests/test_candidate_set.py`：打乱 option 后 logits 等变、teacher 重映射后 loss 不变、单样本 vs batch 一致。
- **成本**：0 新采集；CPU 几分钟跑测试。
- **信号**：测试通过；旧 store 仍能加载。
- **失败判据**：打乱后 loss 变 / argmax 语义变；说明还有位置依赖，先修再谈模型。

### R1. 冻结 trunk 的候选 self-attention adapter（首选 probe）

- **输入/输出**：A0 checkpoint + `mixed_four_decks.pt`；输出 `data/checkpoints/a4/A4_sab_probe.pt` + `reports/train/a4_sab_probe.json`。
- **结构**：
  ```python
  H0 = frozen_option_proj(options)                 # (B,K,d)
  H  = masked_sab(H0, option_mask, mha, ln1, ff, ln2)
  delta = delta_head(torch.cat([H, H0], -1)).squeeze(-1)
  delta = delta * option_mask
  logits = frozen_base_logits + delta
  logits = logits.masked_fill(~option_mask, -inf)
  ```
  `delta_head` 最后一层 zero-init；初始 logits 与 A0 差 <1e-6。zero-init 首步只更新最后一层，SAB 从第 2 步开始拿到梯度；1–2 epoch 足够。
- **超参**：1 层、4 head、d=256、FF=512、dropout 0、lr 1e-3、batch 512、1–2 epoch、seed 0、weight decay 0。
- **需要改的文件**：`train/model.py`（`CandidateSetBlock` + `CVPNConfig.candidate_layers/candidate_heads/candidate_ff`；`CVPN.forward_all`；`load_state_dict_compatible` 允许 `candidate_` 前缀）；`scripts/datafirst_bc_train.py`（`--candidate-layers/--candidate-heads/--freeze-trunk/--save-best`）；`train/train_loop.py`（mask 断言）；`tests/test_candidate_set.py`。
- **成本**：2 epoch，GPU ~2–5 min；CPU ~1–2 h；无新采集。
- **信号**：Stage 0 有歧义；adapter `|delta|>0.05`、flip 2–30%；按 deck 分组 top-1 +≥1.5pp；引擎 screen 见 §4。
- **失败判据**：惰性/NaN/permutation 失败/引擎不涨；任一即停。

### R2. 费用感知的 pairwise relation（R1 有信号后再做）

- **输入/输出**：R1 的 adapter + 新特征；输出 `A4_pair_probe.pt`。
- **结构**：只对**同 kind 或同 target** 的 option 对算 $r_{ij}=\mathrm{MLP}([x_i,x_j,x_i-x_j,x_i\odot x_j])$；`h_i` 用 masked mean；`delta` 头 zero-init。新增特征：`post_cost_dice`（池子减支付）、`min_cost_same_kind`、`is_cheapest_same_kind`。
- **超参**：pair MLP hidden 128；只算同 kind 对（K 小时可全对）；lr 1e-3；1–2 epoch。
- **需要改的文件**：`train/model.py`（`PairwiseRelationBlock`）；`reps/observation_encoder.py` / `reps/features.py`（新增费用特征，需同步 `option_dim` 与 checkpoint config）；`tests/test_candidate_set.py`（pair mask 测试）。
- **成本**：2 epoch GPU ~3–6 min；CPU ~1–2 h；注意改 `option_dim` 后旧 checkpoint 不兼容，需新 config/重训。
- **信号**：相对 R1 在费用敏感子集/`action` 子集上有额外增益；引擎不劣化。
- **失败判据**：不如 R1，或任一 deck 掉 >2pp；停，不要叠加。

### R3. context-candidate cross-attention（目标绑定错误明显时）

- **输入/输出**：A0 + cross-attention adapter；输出 `A4_cross_probe.pt`。
- **结构**：`C_i = MHA(query=H_i, key=tokens, value=tokens, key_padding_mask=~token_mask)`；`delta` 头 zero-init；token encoder 冻结。
- **超参**：1 层、4 head、d=256、dropout 0、lr 1e-3、1–2 epoch。
- **需要改的文件**：`train/model.py`（`CandidateCrossBlock`）；`scripts/datafirst_bc_train.py`（flag）；`tests/test_candidate_set.py`（token mask 测试）。
- **成本**：同 R1；KV 长度 T~50–100，比 SAB 略便宜。
- **信号**：Stage 0 显示错误集中在「target 错」；cross-attention 的 attention 权重应落在被选目标 token 上。
- **失败判据**：目标错误率不降；或引擎不涨；停。

### R4. ISAB / 全 Set Transformer（只有 K 长尾或 R1 明显有效才做）

- **输入/输出**：R1 的 adapter 换成 ISAB（m=16/32）或 2 层 SAB；输出 `A4_isab_probe.pt`。
- **结构**：`ISAB_m(X)=MAB(X,MAB(I,X))`；inducing points 是参数；mask 仍然要自己加。
- **超参**：m=16/32、2 层、4 head、d=256；lr 1e-3；1–2 epoch。
- **需要改的文件**：`train/model.py`；`tests/test_candidate_set.py`。
- **成本**：2 epoch GPU ~3–6 min；K=256 时显存明显低于全 SAB。
- **信号**：K 长尾（reroll/switch_hands）子集提升，且全量不劣化。
- **失败判据**：不如 1 层 SAB；停，不要为了「更大模型」而加层。

## 4. 预注册实验（只给 2 个）

### E1：A4-probe——冻结 A0 trunk 的候选 self-attention adapter

**假设 H1**：在 A0 trunk 固定时，1 层候选 self-attention 能在 ≥2/3 deck 上把引擎胜率提高 ≥3pp（confirm），且不损坏第三个 deck。零假设：Δ=0 或任一 deck 显著变差。

**对照**：
- Control：A0 checkpoint，同 seeds/opponents/sides。
- Treatment：A0 + 1 层 SAB + zero-init delta head；只训 adapter；2 epoch。
- 负对照：随机 adapter（同参数量、随机**非零** delta，不训练）不应在引擎上「提升」。

**指标**：
- 主：每 deck 引擎胜率、paired Δ vs A0、95% CI（cluster by seed/opponent）。
- 次：按 deck 分组的 store top-1（严格 leave-one-deck-out 在 confirm 阶段做）、按 request_type 的 top-1、decision flip rate、平均 `|delta|`、value Brier。
- 机械：permutation test、batch independence、zero-init identity、无 NaN/全 mask。

**Seeds 与功效**：
- Screen：3 deck × 20 opponents × 10 seeds（910–919）× 2 sides = 400 局/deck。配对设计、p≈0.7、ρ≈0.5 时，paired SE≈2.3pp；80% power 只能抓 ~6pp。**这是 screen，不是 certification。**
- Confirm：3 deck × 20 opponents × 50 seeds（920–969）× 2 sides = 2,000 局/deck；paired SE≈1.0pp，能抓 ~3pp。与项目 certification tier 一致（`reports/explore/eval_protocol.md:248`）。
- 如果真实效应只有 +2pp，本实验很可能漏掉；这是有意的：只放大稳健的大信号。

**预算**：
- 采集：0 新局。
- 训练：2 epoch，GPU ~2–5 min（CPU ~1–2 h）。
- Screen：1,200 局，~15 min @4,900 局/h。
- Confirm：6,000 局，~1.2 h。
- 合计：GPU ~1.5 h（含评测）；CPU ~2–3 h。

**复现命令（示意；`--candidate-*`/`--freeze-trunk`/`--save-best` 需按 R1 新增）**：

```bash
python -m scripts.datafirst_bc_train \
  --store-cache data/store/mixed_four_decks.pt --arch cvpn \
  --init-from data/checkpoints/kaggle/h4m_bc_d256_e150.pt \
  --candidate-layers 1 --candidate-heads 4 --freeze-trunk --save-best \
  --epochs 2 --lr 1e-3 --batch-size 512 --device cuda --seed 0 \
  --out data/checkpoints/a4/A4_sab_probe.pt --report reports/train/a4_sab_probe.json

# screen（400/deck）
SEEDS=10 SEED_START=910 CANDIDATES="a4:data/checkpoints/a4/A4_sab_probe.pt" \
  bash scripts/certify_a2.sh
# confirm（2,000/deck）
SEEDS=50 SEED_START=920 CANDIDATES="a4:data/checkpoints/a4/A4_sab_probe.pt" \
  bash scripts/certify_a2.sh
```

**判定阈值**：
- Screen PASS：≥2/3 deck Δ ≥ +6pp；无 deck ≤ −2pp；pooled Δ >0；机械检查通过。
- Confirm PASS：≥2/3 deck Δ ≥ +3pp 且 95% CI 下界 >0；无 deck ≤ −2pp；pooled Δ ≥ +2pp 且 CI 下界 >0。
- 任一 deck ≤ −5pp：立即停。

**停止条件**：
- Stage 0 无信号（§1.6）。
- zero-init/permutation/batch 测试失败，或 NaN。
- 2 epoch 后 `|delta|<0.05`、flip<1%、或按 deck 分组的 store top-1 增益 <1pp。
- Screen 或 confirm 未达 PASS。
- store val 涨但引擎不涨（A2 模式）。

**如果 PASS**：进入 E2；不要直接跳到 150 epoch，也不要在同一 run 里加 id/费用特征。

### E2：A4-scale——全 trunk BC 的候选 self-attention（条件于 E1 PASS）

**假设 H2**：E1 的增益在 unfreeze trunk 后仍存在；如果 joint training 把增益吃掉，说明 adapter-only 是伪信号，应停。

**对照**：
- Control：A0 继续 fine-tune 20 epoch（同数据、同 lr、同 seed）。
- Treatment：A0 + E1 的 SAB 结构，unfreeze trunk，同预算 fine-tune 20 epoch。
- 可选第二 arm：E1 adapter-only 作为「下限」；不放 id。

**指标**：同 E1；主指标仍是引擎胜率；比较 Treatment vs Control、Treatment vs E1 adapter。

**Seeds/预算**：同 confirm（3×2,000 局）；训练 20 epoch，GPU ~10–20 min（150 epoch 75 min 的线性估计，实际以 `datafirst_bc_train` 报告为准）；总 ~1.5 h GPU + 1.2 h 评测。无新采集。

**判定阈值**：同 E1 confirm；另外要求 Treatment 不劣于 E1 adapter（否则 joint training 把结构增益吃掉了）。

**停止条件**：任一 deck ≤ −2pp；pooled <0；Treatment 不如 Control；或不如 E1 adapter。停，不要再加 pairwise/pointer。

**如果 PASS**：才考虑 R2（费用感知 pairwise）或 R3（cross-attention）作为独立消融；每次只加一个结构，仍用同一门禁。

## 5. 与现有负结果的关系（为什么这次不会重演）

| 现有负结果 | A4 probe 的差异 | 仍需警惕 |
|---|---|---|
| **A2 id embeddings**：val_acc↑、引擎崩 | 不加 id；共享参数；冻结 trunk；零初始化；主指标引擎胜率；逐 deck | 结构本身也可能过拟合；确认阶段做严格 deck-held-out |
| **20-deck 单模型负迁移**：7/8 deck 变差 | 不增加 deck-specific 参数；候选交互是共享的；逐 deck 报 | 4-deck 语料仍可能 deck 依赖；不要 pooled |
| **数据量 8k→48k 不显著** | 只训 1–2 epoch；考的是归纳偏置，不是数据量 | 如果 1–2 epoch 没信号，不要用「更多数据」救 |
| **AWR/Q/DMC 负结果** | 仍是 BC；不做 value bootstrapping/Q | 不要偷换成离线 RL |
| **search-as-teacher 显著负** | 不改 target；不蒸馏搜索；只改 scorer | 搜索 target 的价值问题未解决；不要混入 |
| **逐 deck PG 是唯一稳定正结果** | 不替代 specialist；先在共享 BC trunk 上验证结构 | 如果结构只在某个 deck 有效，可能该 deck 的 specialist 更划算 |
| **A2 val_acc 假阳性** | 引擎胜率 + 配对 seeds + 逐 deck + save-best | 400 局 screen 仍可能噪声；confirm 必须 2,000 |

**核心论证**：A4 不是「再加一个更大的网络」，而是一个**受控的结构消融**：trunk 冻结、adapter 零初始化、初始策略与 A0 完全相同；唯一变化是候选间交互。如果引擎胜率不涨，就没有理由把参数量/训练预算投进去。A2 的教训是「val_acc 不是强度」；A4 的设计把这一点写进了门禁。

## 6. 不要做什么（negative guidance）

1. **不要用 PMA 直接出 policy logits**。PMA 是 set-to-vector；policy 需要 per-option logits。用 SAB（equivariant），PMA 只给 value head。
2. **不要给 option 加位置编码/engine index**。会把引擎枚举顺序当语义，破坏等变，换引擎版本就失效。
3. **不要直接复制 Set Transformer 官方 `modules.py`**。它没有 mask；变长 option 会把 pad 算进 attention，必须自己加 `key_padding_mask`。
4. **不要用 `-inf` mask 而不 guard 全 mask 行**。CE/PG 会 NaN；至少 assert 每行有 ≥1 合法 option，`log_softmax` 用 `torch.where`。
5. **不要把 id embedding 和候选 attention 一起上**。A2 已经证明 id 分支会污染 trunk；两个变量同时动，无法归因。
6. **不要用 val_acc 做 go/no-go**。A2 的 0.866 vs 0.838 是最直接的反例；主指标必须是引擎胜率。
7. **不要 pooled 报分**。A2 的 dvalin 只掉 8pp、dgeo/ungeo 掉 45pp；pooled 会掩盖长尾。
8. **不要只训 p0 却在 both sides 上宣称增益**。要么训练也采 p1，要么只报 p0。
9. **不要用 pointer 替换整个 option list**。Pointer 适合「从输入 token 选子集/序列」；引擎已经枚举 joint option，pointer 会破坏 100% 覆盖。
10. **不要独立 factorize 多目标动作**。各 branch 独立 sigmoid 可能产生非法组合；要分解就用全局 legal set 归一化的 hierarchical softmax，并枚举/beam 检查合法性。
11. **不要用 Wolpertinger 的 proto-action/kNN**。K≈27，全量打分精确且便宜；kNN 只增加近似误差和 off-policy 复杂度。
12. **不要用 action abstraction 裁 BC 的合法动作**。DeepStack/Libratus/ReBeL 的抽象是搜索时 branching 控制；BC 必须保留全部 legal options（100% 覆盖是硬约束）。
13. **不要在 K=256 上直接全 pairwise/SAB**。一个 K=256 样本会把 batch 的 attention 矩阵撑到 65k×heads；用 ISAB、按 request_type 分桶、或只算同 kind 对。
14. **不要只保存最后一个 epoch**。A2 的 best val 在 ep70，最终 ep149 已过拟合；`datafirst_bc_train.py` 目前只 save 最后一个，必须加 `--save-best` 或固定小预算。
15. **不要在 probe 失败后加层/加 epoch 救**。失败判据是预注册的；加层只会把 A2 的 val_acc 假阳性重演一遍。

## 7. 参考（链接、版本/日期、可信度）

| # | 材料 | 链接 | 版本/日期 | 可信度 | 备注 |
|---:|---|---|---|---|---|
| 1 | DeepSets | https://arxiv.org/abs/1703.06114 | v3, 2018-04-14 | A | 读了 Theorem 2 / Lemma 3 / 实验 |
| 2 | Set Transformer | https://arxiv.org/abs/1810.00825 | v3, 2019-05-26 | A | 读了 SAB/ISAB/PMA 与实验 |
| 3 | Set Transformer 官方代码 | https://github.com/juho-lee/set_transformer | 访问 2026-10-27 | A | `modules.py` 无 mask |
| 4 | Pointer Networks | https://arxiv.org/abs/1506.03134 | v2, 2017-01-02 | A | 读了公式/实验/排序说明 |
| 5 | Relation Networks | https://arxiv.org/abs/1706.01427 | v1, 2017-06-05 | A | 读了 RN 公式与 CLEVR/Sort-of-CLEVR |
| 6 | Wolpertinger / Dulac-Arnold 2015 | https://arxiv.org/abs/1512.07679 | 2015-12 | A | 读了算法/实验；未发现 log-density correction |
| 7 | ListNet | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf | MSR-TR-2007-40 | B | 读了摘要/定义；未逐表核对 |
| 8 | ListMLE | https://icml.cc/Conferences/2008/papers/167.pdf | ICML 2008 | B | 读了摘要/公式；未逐表核对 |
| 9 | Plackett-Luce (Guiver & Snelson) | https://doi.org/10.1145/1553374.1553423 | ICML 2009 | B | 标准 PL 定义；未读全文 |
| 10 | DouZero | https://arxiv.org/abs/2106.06135 | ICML 2021 | A- | 读了代码 `env.py`；论文表格未逐页 |
| 11 | DouZero 代码 | https://github.com/kwai/DouZero/blob/main/douzero/env/env.py | 访问 2026-10-27 | A | `_get_obs_*` / `_cards2array` |
| 12 | Suphx | https://arxiv.org/abs/2003.13590 | 2020-03 | A | 读了五模型/动作维度/实验 |
| 13 | DeepStack | https://arxiv.org/abs/1701.01724 | Science 2017 | A | 读了 action abstraction / 实验 |
| 14 | Libratus | https://www.science.org/doi/10.1126/science.aao1733 | Science 2018 | B | 读了摘要/搜索片段；bet size 数量未核实 |
| 15 | Libratus IJCAI | https://ijcai.org/proceedings/2017/0772.pdf | IJCAI 2017 | B | 动作抽象描述；未逐页 |
| 16 | ReBeL | https://arxiv.org/abs/2007.13544 | NeurIPS 2020 | A | 读了 PBS/≤9 actions/实验 |
| 17 | ReBeL 代码 | https://github.com/facebookresearch/rebel | 访问 2026-10-27 | B | Liar's Dice 实现；未细读 |
| 18 | AGILE | https://openreview.net/forum?id=MljXVdp4A3N | ICLR 2022 | B | 读了 README/代码结构；PDF 被验证挡住，效应量未核实 |
| 19 | AGILE 代码 | https://github.com/clvrai/agile | 访问 2026-10-27 | B | action graph / GAT / ablations |
| 20 | BDQ | https://arxiv.org/abs/1711.08946 | v2, 2019-01-25 | B | 读了摘要；实验细节未逐表 |
| 21 | DAgger | https://arxiv.org/abs/1011.0686 | v3, 2011-03-16 | B | 读了摘要；用于解释 covariate shift |
| 22 | Shortcut Learning | https://arxiv.org/abs/2004.07780 | v5, 2023-11-21 | B | 读了摘要；用于解释 id shortcut |
| 23 | A2 内部报告 | `reports/explore/a2_id_screening.md` | 2026-10-27 | A | 内部事实；val/引擎数字 |
| 24 | CVPN 代码 | `train/model.py` | 当前 HEAD | A | `option_scorer` / `forward_all` / `TransformerCVPN` |
| 25 | 观测编码 | `reps/observation_encoder.py`, `reps/features.py` | 当前 HEAD | A | option 68 维 / definition 列 42/30 |
| 26 | 数据采集 | `train/data.py` | 当前 HEAD | A | p0-only / 位置 teacher_index |
| 27 | 评测协议 | `scripts/eval_protocol.py`, `reports/explore/eval_protocol.md` | 当前 HEAD | A | both sides / 2,000 games certification |
| 28 | store 切分 | `scripts/datafirst_bc_store.py` | 当前 HEAD | A | 按 seed 局级切分 |
| 29 | 训练保存 | `scripts/datafirst_bc_train.py` | 当前 HEAD | A | 只 save 最后一个 epoch |
| 30 | 内部 K 统计 | `reports/explore/model_compare_20261011T132909.json`, `reports/train/a3_bc_v1.json` | 2026-10 | A | mean 25–29, max 192–256 |

---

### 附：如果只记三件事

1. **候选集交互是合理的归纳偏置，但不是当前首要瓶颈**：DouZero 用逐动作独立打分就能在 27,472 个动作的斗地主上超人类；Suphx 用五个小头；我们的 K≈27，先把特征/分头/数据做对，再谈 set attention。
2. **A2 的教训是「val_acc 不是强度」**：id embedding 的 val_acc 0.866 对应引擎 dgeo 0.266；新实验必须冻结 trunk、零初始化、引擎胜率、逐 deck。
3. **最小 probe 的 go/no-go 是引擎胜率**：2 epoch adapter + 400 局/deck screen（+6pp）+ 2,000 局/deck confirm（+3pp, CI 下界 >0）；不过就停，不要加层。
