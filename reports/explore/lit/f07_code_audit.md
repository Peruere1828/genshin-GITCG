# f07 代码级审计：store 训练分布 ↔ live engine 推理分布

> 范围：只读代码 + 只读 replay/store/checkpoint 元数据；未训练、未评测、未起引擎。
> 目标：找出所有可能让「训练时的 store 特征」和「引擎在线推理特征」不一致的代码路径，并给出可执行的 parity harness 设计。
> 代码基线：当前工作区（2026-10-27），`reps/`、`train/`、`envs/`、`scripts/`、`agents/`。

## 0. 一句话结论

**最可疑的 3 个点（按概率排序）**：

1. **确定不一致（已实证，最可能的 A2 崩盘主因）**：A2 语料是用 `--format compact --compact-dtype float16` 采的，`definition_id / 400000` 这个「可逆编码」被 fp16 量化后**不可逆**。`reps/id_features.py` 的 `round(x * 400000)` 在 store 里读回的是被 fp16 抬/压几十到上百的**假 id**；152 张卡在 store 里只剩 **28 个不同的 id（20 组歧义）**，而引擎在线读的是 142 个真实 bucket，**交集只有 4 个**。A0i 的 id 分支在推理时，**卡牌类** token/option（手牌、打卡、卡牌实体）几乎全部命中「训练时从未更新过」的 embedding 行，等同于给这些 token/option 注入随机向量；这正好解释「store 混合 val_acc 0.866 / 引擎 dgeo 0.266」以及「去掉 id 分支的 A0i_noid 反而回到 0.741」。这一条不是「列错位」，是**数据落盘精度**问题，且**无法在读盘时修复**（见 §2.2、§5）。
2. **确定不一致（结构）**：BC 样本（`collect_samples` / `collect_search_samples`）**只来自 player 0**（`train/data.py:157-173`，`collect_search_samples` 也只遍历 `captured[0]`，`train/data.py:267`），而 `scripts/eval_protocol.py` 默认双边各打一半（`sides=(0,1)`，`scripts/eval_protocol.py:432-439`）。`acting_player` 的 one-hot（`reps/observation_encoder.py:540-541`）在 store 里恒为 `(1,0)`（1128/1128 抽样验证），side=1 时整体翻成 `(0,1)`——模型从未见过这个输入。影响可能小于 #1，但会构成系统性的 side=1 分布偏移（需 §3.6 的 `seat_swap` 量化）。
3. **可疑（当前未触发，但无护栏）**：store 的 `meta` **不记录 encoder config / 解析后的 definition 列号**；`scripts/datafirst_bc_train.py:121` 和 `scripts/train_bc_scale.py:740` 都用**训练时**的 `default_encoder()` 重建模型并写进 checkpoint。`definition_feature_indices()` 在**加载时**重算（`train/model.py:100-106`，checkpoint 里只存 `-1`），一旦 token/option 的**布局块**发生**同宽重命名/重排**（`_TOKEN_TYPES`/`OptionKind`/`_OPPONENT_TAGS`），旧 checkpoint 会**静默读错列**；`include_opponent_tag` 切换会改宽度（通常会在 matmul 处报错，除非模型宽度也恰好是 58）；`max_*` 改的是 token 数/截断，不是列号，但会改变 T 分布。（vocabulary 本身只影响 `privileged_state`/belief 标签，不影响 policy 输入。）当前 A2 的 8 个 replay dir 的 manifest fingerprint 与今天的 `default_encoder()` 完全一致（已逐个重算），所以**不是本次崩盘原因**，但属于同一类「看起来一致」的雷。

---

## 1. 数据流图（文字版，标 file:line）

### 1.1 live 采集

```
train/collect.py:585  main
  → :607 CollectSpec → :288 tasks
  → envs/rollout.py:411 stream_task_shard
  → train/collect_worker.py
  → train/collect.py:242 run_collect_task
  → train/data.py:137 collect_samples
      :157 build_policy(expert:<deck>, seed=derive_seed(seed,"p0",deck), role="p0")
      :160 _CaptureTeacher(p0)                      # 只包 teacher
      :166 run_match(deck_spec(deck), deck_spec(opp), capture, opp,
                     seed=seed, full_state_for_agents=True)   # :172
          envs/match.py:603 reset_default_hierarchical_action_codebook()
          envs/match.py:619 full_state_provider = lambda: game.state().json()
          envs/match.py:625-633 PolicyPlayer(who, policy, full_state_provider=...)
              envs/match.py:431 _build()
                  view = snapshot_notification(notification)      # masked player_view
                  full_state = snapshot_state(None, game.state().json()) if provider
                  build_decision_context(acting_player=who, full_state=full_state,
                                         player_view=view, full_state_json=...)  # reps/action_adapter.py:102
          envs/match.py:464 policy.choose(built)
              train/data.py:131 _CaptureTeacher.choose → 记录 (built, code)
      :177 for built, code in capture.items():
      :179 codes = list(context.legal_low_level_codes)
      :182 index = codes.index(code)                # teacher_index（位置）
      :183 encoder.encode_context(context, history=(), include_training_targets=capture_targets)
          reps/observation_encoder.py:202
            :219        global token
            :221-316    player / character / entity / status / summon / support tokens
            :318-339    self hand / self dice tokens
            :340-361    history tokens（history=() → 0 个）
            :364-374    option_features = [encode_public_low_level_spec(spec) for spec in legal_low_level_specs]
            :375        option_mask
            :380-392    privileged_state / belief_target（仅标签，来自 full_state）
      :188 Sample(observation, teacher_index=index, value_target=outcome_score(winner,0), ...)
  → train/data.py:430 observation_to_payload
      :474-475 _encode_2d(token_features/option_features, compact_dtype="float16")
      :369-376 _pack_dense → np.float16 + base64
  → data/replays/<tag>/samples.jsonl（一行一局）
```

### 1.2 store 构建/读取

```
scripts/datafirst_bc_store.py:73 load_store_any
  :96  _iter_replay_records
  :118 _tensor(obs["token_features"])      # :47 _unpack_dense → fp16 值 → float32
  :120 teacher = payload["teacher_index"]
  :141 keys = (deck, opponent, seed, step_index)
  :150-152 sorted by keys → _reorder
  :198 torch.save({"train":..., "val":..., "meta":...})
```

legacy 路径：`scripts/train_bc_scale.py:186 load_store`，读 jsonl dense list 时 `:246-252 torch.tensor(obs["token_features"])`（无损）；读 compact 时同样会走 `json.loads` 后的 dense dict，但 `train_bc_scale.py` 的注释说它只支持 legacy list（`scripts/datafirst_bc_store.py:1-12`），实际 A2 用的是 `datafirst_bc_store`。

### 1.3 训练

```
scripts/datafirst_bc_train.py:117 main
  :121 encoder = default_encoder()
  :123 payload = torch.load(store_cache)
  :125-126 _subsample(train/val)
  :141-144 cfg = config_for_encoder(encoder, ...) ; model = CVPN(cfg)
  :146-159 可选 zero_init_identity_branch / load_state_dict_compatible(--init-from)
  :173 train(train_store, val_store, model, config, has_belief)
      scripts/train_bc_scale.py:613 train
        :614 torch.manual_seed(config.seed)        # 注意：模型已在 :144 构造完
        :637 rng = random.Random(config.seed)
        :645-650 for batch: run_batch
            :402-421 _collate (pad_sequence)
            :477-538 run_batch
                :481 model.forward_all(tokens, token_mask, options, option_mask)
                    train/model.py:124-143
                      :132 _with_ids(self.token_proj(tokens))     # id 分支
                      :137 _with_ids(self.option_proj(options), option=True)
                      :139 option_scorer(cat[global_vec, option_vec])
                      :140 masked_fill(~option_mask, -inf)
                :495 CE(logits, teacher) + :532 value loss
  :176 model.save(out, encoder_config=encoder.to_dict())
```

### 1.4 在线推理 / 评测

```
scripts/eval_protocol.py:508 main
  :519 seeds = range(seed_start, seed_start + n)
  :526-538 ProtocolSpec(..., base_seed=seed_start)
  :115-125 tasks: for opponent × seed × side
  :82-98 ProtocolTask.to_match:
        side=0: deck0=contestant_deck, policy0=contestant
        side=1: deck0=opponent,        policy1=contestant
        seed=spec.base_seed + task.seed          # ← 见 §2.6
  → envs/rollout.py:78 run_task
      :83 run_match(deck0, deck1, build_policy(policy0, seed=task.seed, role="p0"),
                                  build_policy(policy1, seed=task.seed, role="p1"), seed=task.seed)
      envs/policy.py:154 build_policy
        :171-172 neural: → _neural_policy
          :142 checkpoint_encoder_config(path)
          :146 TokenObservationEncoder.from_dict(encoder_config)   # 或 :150 default_encoder()
          → agents/neural.py:62 load_neural_policy
              → train/model.py:196 load_cvpn_model（strict load + .eval()）
      envs/match.py:464 policy.choose
          agents/neural.py:34 NeuralPolicy.choose
            :38 encode_context(built.context, history=(), include_training_targets=False)
            :43 collate([observation])
            :50 model(tokens, token_mask, options, option_mask)
            :57 argmax(logits) 或 :54 softmax/temperature
            :59 return codes[index]                # 位置 → code
```

---

## 2. 逐项核对表

| # | 项 | 代码位置 | 结论 | 证据（关键点） |
|---|---|---|---|---|
| 1 | live 采集：history / include_training_targets / full_state_for_agents | `train/data.py:157-188`；`envs/match.py:431-455,619`；`reps/observation_encoder.py:202-212,340-392`；`reps/event_features.py:102-115` | **一致（输入）**；只有标签不同 | 采集 `history=()`，推理 `history=()`；`include_training_targets` 只控制 `privileged_state`/`belief_target`（`:380-392`）；token/option 全部走 `player_view` |
| 2 | store 构建/读取：compact fp16、padding、截断、排序 | `scripts/datafirst_bc_store.py:45-55,73-168`；`train/data.py:369-376,400-406,474-475`；`train/collect.py:571-576` | **确定不一致（definition 列）**；其余一致 | fp16 把 `id/400000` 量化；`_tensor` 只做 fp16→float32，不恢复 id；无全局 padding；history=() 时 T≤73，`max_tokens=105` 不截断；排序按 `(deck,opp,seed,step)` |
| 3 | 训练路径：mask / dropout / value / id / belief | `scripts/datafirst_bc_train.py:121-176`；`scripts/train_bc_scale.py:402-538,613-617`；`train/model.py:96-143,432-452` | mask/dropout/value/belief **一致**；id 分支 **确定不一致**；init **可疑** | mask 用长度、`-inf`；dropout 默认 0；value/belief 不影响 logits；id 读 42/30 列；模型在 `torch.manual_seed` 之前构造 |
| 4 | 推理路径：encoder 来源、load、温度、device | `envs/policy.py:120-151`；`agents/neural.py:27-59`；`train/model.py:156-214` | **一致（当前 checkpoint）**；温度/旧 config 有可疑点 | 85/85 checkpoint 都有 `encoder_config` 且 `include_opponent_tag=True`；`load_cvpn_model` strict + `.eval()`；`#t>0` 用未播种的 `torch.multinomial` |
| 5 | 动作索引：teacher_index、argmax、codebook、候选顺序 | `train/data.py:177-198`；`reps/action_adapter.py:61-93,128-184`；`reps/action_hierarchy.py:140-184,319-327`；`agents/neural.py:35-59` | **一致（位置一一对应）** | `codes.index(code)` 与 `codes[argmax]` 用同一 `legal_low_level_codes`；`option_features` 来自同一 `legal_low_level_specs`；per-option scorer 对 option 顺序不敏感 |
| 6 | 评测路径：seed、座位、deck、p0 视角 | `scripts/eval_protocol.py:82-98,415-439,508-538,585-609`；`envs/rollout.py:78-97`；`envs/match.py:575-636` | 座位 **确定不一致**；seed **有 bug**；deck 一致 | 训练只 p0；eval 双边；`seed = base_seed + seed` 且 `base_seed=seed_start` → 实际 2×seed_start+i；deck 走同一个 `envs.decks.deck_spec` |
| 7 | A2 的 A0i_noid：实例化/加载、id_buckets、随机 embedding | `data/checkpoints/a2/A0i_noid.pt`；`train/model.py:196-260`；`scripts/datafirst_bc_train.py:146-159` | **无随机 embedding（该 artifact）**；`load_state_dict_compatible` 有静默随机初始化风险 | A0i_noid config `id_buckets=0`，14 个 tensor；14/14 与 A0i 的非 id tensor 逐位相等；缺失的 6 个 id 参数默认随机 init，只有 `--id-zero-init` 才是 no-op |
| 8 | 「看起来一致但实际可能不一致」 | 见 §2.8 | 混合 | format 精度、encoder config 不落 store、def 列号重算、from_dict 默认值、测试容差、init 种子、CUDA/CPU、bf16、metadata 常量等 |

### 2.1 live 采集：三个开关对输入特征的影响

逐条核对（不是标签）：

- **`history=()`**：`train/data.py:184` 显式传空；`agents/neural.py:39` 也传空。`reps/observation_encoder.py:340-361` 的 history token 块永远不会产生 token。**一致**。如果未来任何 collector 传了 history，而在线推理仍传空，会立刻出现 `history` token 分布差异；目前没有。
- **`include_training_targets`**：采集是 `capture_targets`（`train/data.py:172,184`，默认 True），推理是 `False`（`agents/neural.py:39`）。它只控制 `reps/observation_encoder.py:380-392` 的 `privileged_state` 和 `belief_target`，这两个字段**不进入 token/option 特征**。**一致（输入）**。但注意：如果以后把 belief 头作为策略输入（而不是辅助头），推理端拿到的是全 0，这会变成真不一致。
- **`full_state_for_agents`**：采集为 True（`train/data.py:172`），评测/推理为 False（`envs/match.py:589` 默认）。True 时 `PolicyPlayer._build` 用 `game.state().json()` 造 `full_state`（`envs/match.py:437-446`），但：
  - `encode_context` 的 token/option 一律用 `context.player_view`（`reps/observation_encoder.py:218,317-339`）；
  - option 特征由 `encode_public_low_level_spec` 计算，显式用 `context.player_view`（`reps/event_features.py:102-115`）；
  - `full_state_json` 只被 `enable_result_based_action_relabel` 使用（`reps/action_adapter.py:135-153`），而 `build_decision_context` 默认 `False`（`reps/action_adapter.py:112`）。
  **一致（输入）**；full_state 只影响标签。
- **`opponent_tag` / `opponent_entity_key` / `opponent_deck_name`**：`PolicyPlayer` 构造函数接受 `metadata`，但 `run_match` 创建时**没有传**（`envs/match.py:625-633`），所以 `context.metadata` 恒为空；`encode_context` 的 `opponent_tag` 默认 `"unknown"`（`reps/observation_encoder.py:207`），`opponent_entity_key` 回退到 `metadata`/`"unknown"`（`:393`）。训练与推理**都是常量**，**一致**，但这是三个「死特征」：`opponent_tag_id` 恒定、`opponent_entity_id/deck_id` 恒定。将来若只在一侧填 metadata，会立刻漂移。
- **`opponent_public_dice_count`**：`encode_context` 支持（`:210,224-225`），但训练/推理都没有传。**一致**。
- **真正的不一致**：teacher 只包 player 0（`train/data.py:160`），`collect_search_samples` 也只遍历 `captured[0]`（`:267`）。store 中 `acting_player` 恒为 0（抽样 1128 条：1029 条 `(act0=1,act1=0,turn0=1,turn1=0)`，99 条 `(1,0,0,1)`），而 eval side=1 时 `acting_player=1`。见 §2.6。

### 2.2 store 构建/读取：compact fp16 到底改了什么

**结论：compact fp16 改变了 token/option 的每一个 dense 浮点值；对 `definition_id` 这一列是「类别级」破坏，不是 1e-3 的数值噪声。**

代码路径：

- `train/collect.py:571-576`：`--compact-dtype` 默认 `float16`；A2 的 8 个 dir 在 manifest 里都是 `format=compact, compact_dtype=float16, include_privileged_state=false`（逐目录核对）。
- `train/data.py:474-475` → `_encode_2d`（`:400-406`）→ `_pack_dense`（`:369-376`）：`np.asarray(values, dtype=float16).tobytes()`，base64。**所有 token/option 行都按 fp16 落盘**。
- `scripts/datafirst_bc_store.py:47-51`：`_unpack_dense` 解出 fp16 → `np.ascontiguousarray(arr, dtype=np.float32)`。**只是把 fp16 的值提升到 float32，没有恢复原始 id**。
- `reps/id_features.py:20-23`：`round(column.double() * 400000).long()`；`bucket_ids`（`:26-36`）再 `id % (buckets-1) + 1`。这个往返**只在 float32/float64 精确时成立**（`tests/test_id_embeddings.py:71-74` 只测了 float32）。

实测（只读 replay，未训练）：

| 量 | 结果 |
|---|---|
| `token_dim` / `option_dim` | 65 / 68（与 brief 一致） |
| `definition_feature_indices()` | `(42, 30)`（token 第 42 列、option 第 30 列） |
| 152 张卡经 fp16 往返后的**不同 id 数** | **28**（20 组歧义；只有 8 张卡能唯一反推） |
| 152 张卡的**真实 bucket 数** | 142（`id % 1023 + 1`） |
| store 的 corrupted bucket 与真实 bucket 的**交集** | **4 / 142** |
| 角色 id（36 个） | 35/36 精确（1603 → 1604） |
| 6 位 id 的 fp16 往返精确率（10 万–40 万抽样） | **0.7%** |
| 5 位 id（1 万–4 万抽样） | 6.8% |
| 4 位 id（1000–3000 抽样） | 75.2%（实际角色词表 35/36） |

按 deck 的具体证据（每个 deck 的 store 只看到 9–12 个卡 bucket，真实是 15–24 个）：

| deck | deck 卡数 | 真实 bucket | store bucket | 交集 |
|---|---:|---:|---:|---:|
| dvalin_bonk | 18 | 17 | 9 | **0** |
| double_geo_navia | 16 | 15 | 9 | **1** |
| unyielding_geo | 24 | 24 | 12 | **0** |

token 级证据（只读 replay，前 4000 个 sample）：

| deck | hand-card token 数 | 解出的 id 在 deck 牌表中的比例 |
|---|---:|---:|
| dvalin_bonk | 16,873 | **0 / 16,873 = 0.0%** |
| double_geo_navia | 30,846 | 4,942 / 30,846 = 16.0%（其余是与其他 deck 卡的偶然碰撞） |

option 级证据（dvalin 前 500 sample 的 `action_play_card` 选项）：

- 1,684 个 play-card option，**0 个**解出 deck 真实卡 id；
- 只有 9 个不同的 corrupted id（对应 18 张 deck 卡）。

**这意味着什么**：

- 训练时 `option_id_emb` 对「打出某张卡」这个 option 的输入，只有 9 个不同的 bucket；`token_id_emb` 对手牌 token 也只有 9 个左右。id 分支在 store 里**不可能**区分具体卡牌，只能学到一个粗粒度的卡 id 区间；
- 推理时引擎给的是真实 6 位 id → 15–24 个真实 bucket，其中 14–24 个**训练时从未出现**（交集 0–1）。`nn.Embedding` 对未出现的行保留随机初始化（`train/model.py:107-108`），再经过训练过的 `option_id_proj`（`train/model.py:110,120`）变成一个**每张卡固定的随机向量**加到 `option_vec` 上；`option_scorer` 的排序因此被随机扰动。token 分支同理。
- 角色 id（4 位）大部分没坏，所以 `ACTION_SWITCH_ACTIVE`/`CHOOSE_ACTIVE` 的 option id 仍有真实信号；一种可能的解释是：切人/选人多的 deck 还能保留一部分 id 信息，依赖 6 位卡 id 的动作多的 deck 则基本被随机向量淹没——这与 dvalin 的 A0i 0.678 > A0i_noid 0.303、dgeo 的 A0i 0.266 < A0i_noid 0.741 的 deck 依赖性一致，但还需要 §3.6 的逐动作 parity 来确认。
- **不能靠读盘时 `round(x*400000)` 修复**：fp16 已经把 152 张卡压成 28 个值，信息不可逆（不是「四舍五入到最近的真 id」——fp16 值到真实 id 的映射是多对一）。要修只能：重采（jsonl 或 fp32）；或在 payload 里额外存一列 int 的 definition id；或把 `token_features/option_features` 整体改成 fp32。

**padding / 截断 / 排序**：

- `Store` 不做全局 padding（`scripts/train_bc_scale.py:64-79`），训练时 `_collate` 按 batch 内 max 用 `pad_sequence`（`:402-421`）；`train/model.py:432-452` 的 `collate` 同样按 batch 内 max，`token_mask/option_mask` 只按长度置 True。推理是单样本，无 padding。CVPN 的 mean-pool 和 Transformer 的 padding mask 都保证 padding 不影响结果（见 §2.3）。
- 截断：`reps/observation_encoder.py:362-363` 在 `max_tokens=105` 处截断；但 `history=()` 时最坏 T = 1 + 2×(1+3+9+6+4+4) + 10 + 8 = 73，**不会触发**。训练/推理同一配置，一致。
- 排序：`datafirst_bc_store.py:150-152` 与 `train_bc_scale.py:281-284` 都按 `(deck, opponent, seed, step_index)` 排序，和文件完成顺序无关；`datafirst_bc_train.py:56-61` 再用 `random.Random(seed)` 做样本级子采样。确定、可复现。注意子采样是**样本级**而不是对局级：它不会造成 train/val 泄漏（val 按 seed 分），但 `--train-limit` 会随机截断一些对局的决策序列，改变 step/局末样本的比例；这是训练分布的一部分，不是 store↔engine 的特征错位。
- **但 `meta` 里没有 encoder config / def 列号 / codebook 版本**（`datafirst_bc_store.py:153-167`）。见 §2.8。

### 2.3 训练路径：mask、dropout、value、id、belief

- **mask**：`scripts/train_bc_scale.py:413-418` 由长度生成 `token_mask/option_mask`；`train/model.py:140` 用 `masked_fill(~option_mask, -inf)`。CE 的 teacher 是有效位置。**一致**。`train/train_loop.py:76-94` 的 stock 路径用 `train/model.py:432 collate` + 同一个 `CVPN.forward`，语义相同；A2 不走它（A2 走 `train_bc_scale`）。
- **dropout**：`CVPNConfig.dropout` 默认 0（`train/model.py:32`），`datafirst_bc_train` 没有 `--dropout` 参数，A2 的 checkpoint `dropout=0.0`。`model.eval()`/`train()` 无差异。**一致**。
- **value head**：`train/model.py:85-87`，loss 在 `train_bc_scale.py:532`；A2 的 `value_activation=identity`。动作选择只取 logits（`agents/neural.py:50-58`），value 不参与。**不相关**。
- **id 分支**：`train/model.py:96-110`（`has_ids = id_buckets > 1`）、`:112-122`（`_with_ids` 把 `id_proj(token_id_emb(bucket))` 加到 token 投影、`option_id_proj(option_id_emb(bucket))` 加到 option 投影）。A0i 的 `id_buckets=1024`、`id_emb_dim=32`。**问题不在模型代码，在 §2.2 的输入**。
- **belief head**：`train/model.py:88-95`；A2 的 `a2_a0i.json` 里 `belief_head=false`。它只从 `global_vec` 分叉，不回写 `global_vec`，**不影响 logits**。**不相关**。
- **模型初始化种子（可疑）**：`scripts/datafirst_bc_train.py:144` 先 `model = CVPN(cfg)`，`:173` 才进 `train()`，而 `train()` 在 `scripts/train_bc_scale.py:614` 才 `torch.manual_seed(config.seed)`。**模型权重的随机初始化不受 `--seed` 控制**（受进程启动时的 torch 默认 RNG 控制）。同一 `--seed` 重跑不会得到同一初始化；A0/A0i 的对比也因此混入了 init 方差。`train/train_loop.py:239` 和 `train/pipeline.py:113` 是同样的「先建模型、后 seed」模式。
- `train_bc_scale._loss_on_index` 不对 teacher 做上界 clamp（`train_loop.py:86` 只 clamp min 0）；teacher 越界会直接 `cross_entropy` 报错，不会静默。**一致**。

### 2.4 推理路径：encoder 来源、加载、温度、device

- **encoder 来源**：`envs/policy.py:142` 先读 `checkpoint_encoder_config(path)`；有则 `TokenObservationEncoder.from_dict`（`:146`），无则 `default_encoder()`（`:150`）。当前 85 个 checkpoint **全部**有 `encoder_config`，且 `include_opponent_tag=True`、`token_dim=65`；A2 的 8 个 replay manifest fingerprint 也与当前 `default_encoder()` 一致。**一致**。
- **checkpoint 加载**：`train/model.py:196-214`：按 `arch` 建 `CVPN`/`TransformerCVPN`，`load_state_dict(payload["state_dict"])` 是 **strict**（默认），然后 `.eval()`。**不会丢参数/随机初始化**。缺 `arch` 的旧 checkpoint 按 `CVPN` 处理。
- **`#t=` 温度**：`envs/policy.py:129-138` 解析 `t`/`d`；`agents/neural.py:53-55` 在 `t>0` 时 `softmax(logits/t)` + `torch.multinomial`。**问题**：`_neural_policy` 没有把 `seed` 传给 `NeuralPolicy`（`load_neural_policy` 也不收 seed），`torch.multinomial` 用的是**进程级 torch RNG**，worker 进程之间/重跑之间不可复现。A2 用 `t=0`（greedy），所以不影响本次；但任何 `#t>0` 的评测都不是配对可复现的。
- **device**：`_neural_policy` 默认 `cpu`；`#d=cuda` 时 `NeuralPolicy.__post_init__` `.to(device)` + `.eval()`（`agents/neural.py:31-32`）。A2 训练用 `--device cuda`，评测用 CPU（`certify_a2.sh` 的 `neural:<ckpt>` 没有 `#d`）；float32 matmul 的 CUDA/CPU 差异是 ~1e-5 级，通常不翻 argmax，但属于「训练/推理精度环境不同」，见 §2.8。
- **`include_training_targets=False`**：推理端 `privileged_state/belief_target` 全 0（`reps/observation_encoder.py:384-392`）；不进入 logits。**一致**。
- **`from_dict` 的默认值**：`reps/observation_encoder.py:198` 把 `include_opponent_tag` 的缺省设为 **False**，而 dataclass 默认是 **True**（`:93`）。当前 checkpoint 都带这个 key，所以没事；但如果加载一个没有该 key 的旧 config，会得到 58 维 token 而模型期望 65，**会在 matmul 处报错**（不是静默）；只有「模型也是 58 维」时才会静默走错列（见 §2.8）。
- **`definition_feature_indices` 的重算**：`train/model.py:100-106` 在 `id_buckets>0` 时调用 `definition_feature_indices()`；checkpoint 里存的是 `token_def_index=-1/option_def_index=-1`，**解析后的 42/30 没有落盘**。见 §2.8。

### 2.5 动作索引：teacher_index、argmax、codebook、候选顺序

**结论：当前 live 路径没有 off-by-one，也没有重排风险；但 search-teacher 路径有一个「根动作集 ≠ live 动作集」的窄风险。**

- 采集：`train/data.py:179-182` 用 `codes.index(code)`；`codes` 是 `context.legal_low_level_codes`，`option_features` 是 `[encode_public_low_level_spec(spec) for spec in legal_low_level_specs(context)]`（`reps/observation_encoder.py:364-374`）。`legal_low_level_specs(context)` 优先返回 `context.legal_low_level_specs`（`reps/action_hierarchy.py:319-327`），而 `build_decision_context` 在 `reps/action_adapter.py:147-184` 用**同一个** `materialized_specs` 同时生成 `legal_low_level_codes` 和 `legal_low_level_specs`，顺序严格一致。
- 去重：`_dedupe_spec_payloads`（`reps/action_adapter.py:61-93`）的 key 与 `_low_level_semantic_key`（`reps/action_hierarchy.py:482-497`）逐字段相同，所以 `legal_low_level_codes` 不会出现重复 code，`.index()` 不会歧义。
- 推理：`agents/neural.py:35` 取 `codes`，`:57` argmax，`:59` `return codes[index]`。位置一一对应。`index = max(0, min(index, len(codes)-1))`（`:58`）只是防御。
- 非法动作：采集端 `train/data.py:180-181` 若 teacher 返回的 code 不在 `codes` 里就丢掉该样本；推理端 `envs/match.py:471-476` 若 code 不在 legal 里就 `_fallback_code`（优先 `ACTION_DECLARE_END`，否则第一个）。两边都不会把非法 code 喂给引擎。**一致**。
- codebook：`run_match` 每局开始 `reset_default_hierarchical_action_codebook()`（`envs/match.py:603-605`）；code 是进程级递增分配的（`reps/action_hierarchy.py:95-113,140-184`），但 **code 本身不进入模型输入**：token/option 特征只用 definition id / kind / dice / target 等语义字段，`low_level_action_codes` 只用于最后的 `code → payload` 映射。所以 codebook 的全局漂移**不会**改变特征。**一致（结构性安全）**。
- 候选顺序：CVPN 的 `option_scorer` 对每个 option 独立打分（`train/model.py:137-139`：`global_vec` 广播 + `option_vec` 逐行 MLP），Transformer 的 cross-attention 对每个 option query 独立（`:383-387`）。因此即使引擎的候选顺序变了，只要 `option_features` 和 `codes` 一起搬，argmax 的**语义**不变。**一致**。
- **search-teacher 的窄风险**：`envs/cfr_search.py:469-479` 的 `chosen_option` 是 `root_keys.index(key)`，而 `root_keys` 来自**信念根**的 `probe.legal_keys`（`envs/cfr_search.py:150-184`）；如果 determinization 后的根动作集和 live 动作集不同，`collect_search_samples` 把它当成 live `codes` 的位置（`train/data.py:289`）就可能指错动作。`strategy`/`action_values` 是按语义 key 对齐的（`train/data.py:89-120`），不受影响；只有硬 `teacher_index` 有风险。A2 是纯 BC，不走这条。

### 2.6 评测路径：seed、座位、deck、p0 视角

- **座位（确定不一致）**：训练 teacher 是 p0（`train/data.py:157`），store 的 `acting_player` 恒 0（1128 条抽样全部 `act0=1, act1=0`）。`eval_protocol` 默认 `sides=(0,1)`（`scripts/eval_protocol.py:432-439`），side=1 时 contestant 是 p1，`acting_player=1`，`acting_player/current_turn` 的组合分布也随之改变（store 里 `current_turn` 本身有 0/1 两种，但只以 p0 acting 的组合出现）。`reps/observation_encoder.py:540-541` 的 `acting_player` one-hot 只在 global token（`:457-474`）里写；store 里这个 token 永远是 `(1,0)`，side=1 时是 `(0,1)`。模型从未见过后者。**这是确定的特征分布不一致**，不是「视角归一化」能自动解决的（`_owner_label` 已经把 self/opponent 相对化了，但 `acting_player` 自身没有）。
- **seed（bug）**：`scripts/eval_protocol.py:519` `seeds=range(seed_start, seed_start+n)`，`:534` `base_seed=seed_start`，`:94` `seed=base_seed+seed`。所以 `--seed-start 900 --seeds 10` 实际引擎 seed 是 **1800–1809**，不是报告里写的 900–909。配对 A/B 仍然有效（baseline 走同一套），但「全新 seed 900–909」的表述是错的；`--seed-start 0` 时无影响（0+i）。`_run_llm_task` 同样（`:251`）。
- **deck 装配**：`ProtocolTask.to_match`（`:82-98`）把 contestant 的 deck 放在 deck0 或 deck1；`envs/rollout.py:64-75` 用 `envs.decks.deck_spec`，与训练 `train/data.py:167` 的 `deck_spec` 是同一个 registry。牌表/角色表一致。**一致**。
- **p0 视角的价值标签**：`train/data.py:176` `outcome_score(record.winner, 0)` 永远是 p0 视角；A2 的 value head 用这个标签训练。动作选择不看 value，所以对 A2 无影响；但如果以后用 value 做搜索/基线，side=1 需要取反，否则标签意义翻转。**当前不相关，潜在**。
- **`--contestant-deck` 默认值**：`scripts/eval_protocol.py:415-418` 对 `neural:` 默认 `superconduct_aggro`。如果忘了传 `--contestant-deck`，会用错 deck 评测；A2 脚本传了。**footgun**。
- **policy seed**：`envs/rollout.py:86-87` 用 `task.seed` 给两边 `build_policy`；expert 的 RNG 再用 `derive_seed(seed, role, "expert", slug)`（`envs/policy.py:170`）。训练采集的 expert seed 是 `derive_seed(seed,"p0",deck)` 再被 `build_policy` 二次派生（`train/data.py:158`），所以**具体 RNG 流不同**，但对手分布仍是同一个 expert。**一致（分布）**，不是逐局可复现的同一 RNG 流。
- **`full_state_for_agents`**：评测走 `run_match` 默认 False（`envs/match.py:589`）；推理 `include_training_targets=False`。标签全 0，不影响 logits。**一致**。

### 2.7 A2 的 A0i_noid：到底怎么实例化/加载的

从 artifact 直接核对（只读）：

- `data/checkpoints/a2/A0i.pt`：`arch=cvpn`，`token_dim=65, option_dim=68`，`id_buckets=1024, id_emb_dim=32`，20 个 state tensor（含 6 个 id 参数）。
- `data/checkpoints/a2/A0i_noid.pt`：`arch=cvpn`，`id_buckets=0`，**14 个 state tensor**，没有 `token_id_emb/option_id_emb/id_proj/option_id_proj`；14/14 与 A0i 的非 id tensor **逐位相等**（`torch.equal` 全 True）。
- 加载：`train/model.py:196-214` 的 `load_cvpn_model` 会按 `id_buckets=0` 建 `CVPN`（14 个参数），`load_state_dict` strict 14/14 成功，`.eval()`。**没有随机 embedding，没有静默初始化**。
- 如果换成「保留 `id_buckets=1024`、只删 6 个 key」的做法：`load_cvpn_model` 的 strict load 会报 missing keys（不会静默）。只有走 `load_state_dict_compatible`（`train/model.py:243-260`）才会允许缺失；它按 `ID_PARAM_PREFIXES = ("token_id_emb.", "option_id_emb.", "id_proj.", "option_id_proj.")`（`:217`）放行，**不校验缺失数量恰好是 6，也不校验 shape/value**；缺失的 id 参数默认是 `nn.Embedding/Linear` 的随机初始化，**只有调用 `zero_init_identity_branch`（`:220-240`）才保证初始输出为 0**。`scripts/datafirst_bc_train.py:146-159` 的 `--id-zero-init` 是**可选**的；`--init-from` 一个 pre-A2 checkpoint 且忘加 `--id-zero-init`，id 分支会以随机噪声起步。
- 结论：**A0i_noid 这个文件本身是干净的**（id_buckets=0，strict 14/14）；但「从旧 checkpoint 升级到 id 模型」的通用路径存在**静默随机初始化**的可能，靠调用方自觉加 `--id-zero-init`。
- 可复现性：仓库里没有生成 `A0i_noid.pt` 的脚本（`grep A0i_noid` 只命中报告），只能从 artifact 反推；如果这个 ablation 要进正式结论，应该补一个 `scripts/make_id_ablation.py`（读源 ckpt、只保留非 id key、写 `id_buckets=0` 的 config）。
- 另外：A0i_noid 是**同一个训练好的 trunk 在推理时去掉 id 分支**，不是「从零训练的无 id 模型」；它的 val_acc 0.397 不能和 A0 的 0.838 直接比（不同语料/不同 val 集）。A2 报告里把 0.397 与 0.838/0.866 并列，容易误读。
- 小问题：`reps/id_features.py:33-36` 用 `id % (buckets-1) + 1`，所以 `id_buckets=1024` 实际只有 1023 个可用槽（第 1024 行永远不用）；这不是本次崩盘的原因，但会加重桶碰撞。

### 2.8 「看起来一致但实际可能不一致」清单

| # | 点 | 位置 | 为什么危险 | 当前是否已触发 |
|---|---|---|---|---|
| 1 | compact fp16 的 definition 列 | `train/data.py:474-475`；`scripts/datafirst_bc_store.py:47-51`；`reps/id_features.py:20-36` | 6 位卡 id 被压成 28 个值；训练/推理 bucket 不同 | **是（A2 全部 8 个 dir）** |
| 2 | store 不记录 encoder config / def 列号 | `scripts/datafirst_bc_store.py:153-167` | 训练端用 `default_encoder()` 重建；config 漂移无法检测 | 否（fingerprint 当前一致） |
| 3 | 解析后的 def 列号不落 checkpoint | `train/model.py:100-106`；`reps/observation_encoder.py:57-76` | 布局块**同宽重命名/重排** → 旧 ckpt 静默读错列；`include_opponent_tag` 切换通常因宽度变化报错（除非模型也是 58 维） | 否（当前 42/30） |
| 4 | `from_dict` 的 `include_opponent_tag` 默认 False | `reps/observation_encoder.py:198` vs `:93` | 旧 config 缺 key → 58 维；若模型也是 58 维则静默走错列 | 否（85/85 有 key=True） |
| 5 | fp16 测试容差 | `tests/test_payload_format.py:75,121-128` | `atol=1e-3` 正好放过 ±200 的 id 误差；合成特征不含 ~0.8 的 def 标量 | **是（测试盲区）** |
| 6 | id 往返测试只测 float32 | `tests/test_id_embeddings.py:71-74` | 没有 fp16 用例 | **是（测试盲区）** |
| 7 | 模型初始化不受 `--seed` 控制 | `scripts/datafirst_bc_train.py:144,173`；`train_bc_scale.py:614` | 同 seed 重跑 init 不同；A/B 混入 init 方差 | **是（可复现性）** |
| 8 | `#t>0` 采样不播种 | `envs/policy.py:120-151`；`agents/neural.py:53-55` | `torch.multinomial` 用进程 RNG；评测不可配对复现 | 否（A2 t=0） |
| 9 | 训练 CUDA / 推理 CPU | A2 训练 `--device cuda`；`certify_a2.sh` 无 `#d` | float32 内核差异 ~1e-5；极端 argmax 翻转 | 低概率 |
| 10 | `BatchScorer(use_bf16=True)` | `train/batched_inference.py:47-58` | bf16 只有 8 位尾数，对 id 的破坏比 fp16 更大；`serve` 还用 `CVPN.load`（不认 Transformer，`:177-181`） | 否（eval_protocol 不走它） |
| 11 | belief 标签的静默回退 | `envs/match.py:437-446` | `game.state().json()` 抛异常时 `full_state=view`，belief/privileged 标签退化成 masked 值；不报错 | 仅影响标签 |
| 12 | `opponent_tag/entity/deck` 常量 | `envs/match.py:625-633`；`reps/observation_encoder.py:207,393-416` | 训练/推理都恒为 unknown；将来只在一侧填 metadata 就漂移 | 否（但死特征） |
| 13 | `collate` 忽略 `EncodedObservation.token_mask/option_mask` | `train/model.py:444-451` | 只按长度置 True；如果未来 mask 中间有 False，会被吞掉 | 否（当前 mask 全 True） |
| 14 | `Store` 丢掉 `privileged_state` | `scripts/datafirst_bc_store.py:116-140` | `--no-privileged-state` 的语料无法再训练 oracle/belief 辅助头（静默空向量） | A2 不用 |
| 15 | `datafirst_bc_store` 的 `b_mask` 由标签非零推断 | `scripts/datafirst_bc_store.py:133-140` | 全 0 的合法标签（例如真的空手牌）会被当成「无标签」，loss mask 掉 | 可能少量 |
| 16 | val_acc 跨语料比较 | `reports/explore/a2_id_screening.md:26-30,50-52` | A0i 的 0.866 是 `mixed_four_decks` 的**混合 val**（4 deck × 20 对手），h4m 的 0.838 是另一套 4-deck val，A0i_noid 的 0.397 是 dgeo val；表头写「store val_acc (dgeo)」但三个数字不是同一 val 集，不能横比 | **是（报告口径）** |
| 17 | eval seed 双倍偏移 | `scripts/eval_protocol.py:519,534,94` | `--seed-start 900` 实际 1800–1809；配对仍有效，但「新 seed」描述错误 | **是（协议 bug）** |
| 18 | `--contestant-deck` 默认 `superconduct_aggro` | `scripts/eval_protocol.py:415-418` | 忘传就用错 deck | 否（A2 传了） |
| 19 | replay fingerprint 不含 deck 牌表 | `train/collect.py:321-351`；`envs/decks.py:17-36` | fingerprint 只 hash deck **名字**，不 hash 角色/卡表；改 deck profile 后 resume 不会拒绝，旧语料会混入新牌表的局；store 也没有 deck 列表 | 否（A2 未改 deck） |
| 20 | engine lock 只记录不校验 | `train/collect.py:440`；`scripts/eval_protocol.py:581` | manifest/report 记录 `engine_lock`，但读取 replay 时不强制比对；引擎升级后旧 replay 仍可被读入 | 否（当前 pin 住） |
| 21 | `Store` 丢掉 `opponent_token_mask`/`high_level_mask`/`request_type` | `scripts/datafirst_bc_store.py:116-141`；`scripts/train_bc_scale.py:64-79` | 当前 CVPN 不用这些字段；但高层头/对手条件化/请求类型条件化模型无法从 store 训练（或缺 mask） | 否（当前不用） |
| 22 | `Store` 不保存 `low_level_action_codes` | `scripts/datafirst_bc_store.py:116-141` | teacher 只存位置；如果未来 option 顺序在 store 与 live 之间变化且无法用 features 对齐，就没有 codes 可供诊断 | 否（当前 per-option scorer 顺序不敏感） |

---

## 3. parity harness 设计（不要求现在实现）

### 3.1 目标

对同一个 `(deck, opponent, seed)` 的同一局、同一个决策序号，比较：

1. **store 里落盘的特征**（从 `samples.jsonl` 解码，模拟训练时的输入）；
2. **live 引擎重新编码的特征**（同一个 teacher、同一个 seed、同一个 `run_match` 路径，模拟推理时的输入）；
3. 同一个 checkpoint 在两组特征上的 `logits/argmax/code`。

要求能在**第一处分歧**处停下并分类，而不是只报一个 loss。

### 3.2 输入

```
scripts/parity_store_live.py
  --replay-dir data/replays/dfbc_dvalin_bonk_c80   # 或 jsonl dir
  --deck dvalin_bonk
  --opponent natlan_battleship
  --seed 0
  --checkpoint data/checkpoints/a2/A0i.pt          # 可选；不给则只比特征
  --limit 3                                        # 只比前 3 个决策，先快后全
  --teacher-spec expert:dvalin_bonk                # 默认 expert:<deck>
  --opponent-spec expert:natlan_battleship         # 默认 expert:<opponent>
  --mode corpus|game|model                         # corpus 不需要引擎
  --json-out reports/explore/parity/<case>.json
```

`corpus` 模式（秒级，无引擎）：只做 §5 的静态检查（manifest fingerprint、fp16 id 往返、hand-card id 成员、store meta）。
`game` 模式：重跑一局，比较特征/teacher_index。
`model` 模式：在 `game` 的基础上再比较 logits/argmax。

### 3.3 步骤（game 模式）

1. **对齐身份**：从 replay 读 `deck/opponent/seed`；先比较 `manifest["engine_lock"]` 与当前 `load_engine_lock()`，并把 `deck_spec(deck).characters/cards` 的 hash 与 manifest 里记录的 deck hash 比对（如果 manifest 没有 deck hash，先补记录；见 §2.8 #19）。然后用与 `collect_samples` 完全相同的构造：
   ```python
   teacher = build_policy(teacher_spec or f"expert:{deck}",
                          seed=derive_seed(seed, "p0", deck), role="p0")
   capture = _CaptureTeacher(teacher)
   opp = build_policy(opponent_spec or f"expert:{opponent}",
                      seed=derive_seed(seed, "p1", opponent), role="p1")
   record = run_match(deck_spec(deck), deck_spec(opponent), capture, opp,
                      seed=seed, full_state_for_agents=True)
   ```
   注意：`_CaptureTeacher` 是 `train/data.py:123-134` 的私有类；harness 可以直接复用（或复制 10 行），但**不要**改 `collect_samples` 的语义。
2. **按 `step_index` 配对**：`PolicyPlayer._decide` 在 `envs/match.py:458` 先 `_step_index += 1` 再 `_build`，所以第一个决策的 `context.step_index == 1`；store 的 `Sample.step_index` 也是同一个值（`train/data.py:194`）。按 `step_index` 配对，不要按 0-based ordinal 猜。
3. **解码 store**：
   - 用 `train/data.py:493 observation_from_payload`（自动识别 jsonl/compact）；
   - 保留原始 payload 的 `low_level_action_codes`/`teacher_index`（`Store` 不存 codes，必须从 raw payload 读）。
4. **重新编码 live**：
   ```python
   enc = TokenObservationEncoder.from_dict(checkpoint_encoder_config(ckpt))  # 或 default_encoder()
   obs_live = enc.encode_context(built.context, history=(),
                                 include_training_targets=False)
   ```
   如果你要验证「采集时的配置」而不是「checkpoint 的配置」，再加一个 `--encoder store|checkpoint` 开关；两者不一致本身就是 finding。
5. **逐字段比较**（顺序见 §3.4）。
6. **模型比较**（可选）：
   ```python
   t, tm, o, om = collate([obs_store]);  logits_s, val_s = model(t, tm, o, om)
   t, tm, o, om = collate([obs_live]);   logits_l, val_l = model(t, tm, o, om)
   arg_s, arg_l = int(logits_s.argmax()), int(logits_l.argmax())
   code_s, code_l = codes[arg_s], codes[arg_l]
   ```
   同时记录 `last_value`，但**不要**把 value 当作策略一致性的判据（A2 的 value 不参与动作）。

### 3.4 对比字段与容差

| 字段 | 类型/形状 | 容差 | 说明 |
|---|---|---|---|
| `len(token_features)` / `len(option_features)` | int | exact | 长度不同 → `CONTEXT_MISMATCH`（先查 seed/opponent/step） |
| `token_mask` / `option_mask` | bool[T]/[K] | exact | 当前全 True；若不同，先查 encoder 配置 |
| token 类型块（前 41 列：type/owner/phase/request/kind/tag） | float[T,41] | exact（0/1） | 不同 → `STATE_MISMATCH`（不是同一状态） |
| **token 第 42 列** | float[T] | **recovered id 必须 exact**；原始 float 对 compact 允许 `1e-3` | `id = round(x*400000)`；这是 A2 的核心 |
| token 其余列（41 与 43–64，共 23 列） | float[T,23] | jsonl: exact；compact: `atol=1e-3, rtol=0` | fp16 的数值噪声 |
| option kind 块（前 9 列） | float[K,9] | exact（0/1） | 不同 → `OPTION_SET_MISMATCH` |
| **option 第 30 列** | float[K] | **recovered id exact**；compact 原始 float `1e-3` | 卡/技能/角色 id |
| option 其余列（0–8 与 10–29、31–67，共 58 列） | float[K,58] | jsonl: exact；compact: `atol=1e-3, rtol=0` | fp16 噪声 |
| `low_level_action_codes` | tuple[int] | exact | 从 raw payload 读，不在 `Store` 里 |
| `teacher_index` | int | exact | 位置必须一致；同时比 `codes[teacher_index]` |
| `logits` | float[K] | jsonl `atol=1e-5`；compact `atol=1e-3` + argmax exact | 特征一致但 logits 不同 → `MODEL_LOAD_DRIFT`；compact 的 fp16 噪声会传播到 logits，不能用 1e-5 |
| `argmax` / `codes[argmax]` | int | exact | 差 1 个 option 也要报 |
| `value` | float | jsonl `atol=1e-4`；compact `atol=1e-3` | 仅诊断 |
| `winner` / `value_target` | int/float | exact | 先验证重放的是同一局 |

### 3.5 定位第一处分歧的算法

```
# fp16_roundtrip(id) = round(float(np.float16(id/400000)) * 400000)
first_bad = None
for d in sorted(decisions, key=step_index):
    S, L = store_obs[d], live_obs[d]
    S_codes = store_raw_payload[d]["observation"]["low_level_action_codes"]
    # 0) 先确认是同一局（winner/value_target 从 replay record 取）
    if d.winner != replay_winner or d.value_target != replay_value:
        return CONTEXT_MISMATCH(d, "winner/value_target")
    # 1) 形状
    if len(S.tokens) != len(L.tokens): return CONTEXT_MISMATCH(d, "T")
    if len(S.options) != len(L.options): return CONTEXT_MISMATCH(d, "K")
    # 2) 离散块
    if not equal(S.tokens[:, :41], L.tokens[:, :41]): return STATE_MISMATCH(d, first_col)
    if not equal(S.options[:, :9], L.options[:, :9]): return OPTION_SET_MISMATCH(d, first_col)
    # 3) definition 列（最关键）
    ids_s = round(S.tokens[:, 42] * 400000)
    ids_l = round(L.tokens[:, 42] * 400000)
    if not equal(ids_s, ids_l):
        if all(ids_s == fp16_roundtrip(ids_l)):
            return FORMAT_FP16_ID(d, "token", first_row, ids_s, ids_l)
        return LAYOUT_DRIFT(d, "token", first_row, ids_s, ids_l)
    # 同理 option[:, 30]
    # 4) 其余连续列
    if not allclose(S.tokens[:, 43:], L.tokens[:, 43:], atol=...):
        return FEATURE_DRIFT(d, "token", first_col)
    # 5) 动作索引
    if S.teacher_index != L.teacher_index: return ACTION_INDEX_DRIFT(d)
    if S_codes != L.codes: return CODEBOOK_DRIFT(d)
# 6) 特征全过，才比模型
if argmax_s != argmax_l or codes_s != codes_l:
    return MODEL_DRIFT(d, logits_maxdiff, state_dict_hash, encoder_config_diff)
return OK
```

**分类规则**：

- `CONTEXT_MISMATCH`：先查 `seed`/`opponent`/teacher spec 是否与 manifest 一致；最可能是重放构造不同，而不是模型问题。
- `STATE_MISMATCH`：token 类型块不同 → 同一决策序号看到了不同状态；查 `player_view` 是否用了 full_state、或 engine 版本/牌表不同。
- `FORMAT_FP16_ID`：`id_store == fp16_roundtrip(id_live)` 且不相等 → **就是本次 A2 的问题**；继续统计受影响 token/option 的比例。
- `LAYOUT_DRIFT`：id 不满足 fp16 关系 → encoder 词表块/`include_opponent_tag`/`definition_feature_indices` 漂移。
- `FEATURE_DRIFT`：非 id 列不同 → 查 target slot 解析、metadata、dice/hand 顺序、history。
- `ACTION_INDEX_DRIFT`：特征一致但 teacher_index 不同 → 查 `_dedupe_spec_payloads`/`legal_low_level_specs` 的回退路径。
- `MODEL_DRIFT`：特征全过、logits 不同 → 查 state_dict hash、`arch`、`token_def_index`、`from_dict`。
- `NUMERICAL`：logits 只差 <1e-5，argmax 翻转 → 记录，不当作系统性不一致。

**快速二分**：如果一局很长，先比第 1 个决策；OK 则比最后一个；再二分中间。单局 ~5 s，O(log n) 次重放即可定位第一个坏决策。

### 3.6 测试用例（至少 dgeo + dvalin 各一）

| case | replay dir | deck | opponent | seed | checkpoint | 期望 |
|---|---|---|---|---|---|---|
| `dgeo_compact` | `data/replays/dfbc_double_geo_navia_c80` | `double_geo_navia` | `natlan_battleship` | 0 | `A0i.pt` | 第一个 hand-card token 上 `FORMAT_FP16_ID`；id 交集 1/15 |
| `dvalin_compact` | `data/replays/dfbc_dvalin_bonk_c80` | `dvalin_bonk` | `natlan_battleship` | 0 | `A0i.pt` | 第一个 hand-card token 上 `FORMAT_FP16_ID`；id 交集 0/17；play-card option 0/… |
| `jsonl_control` | `data/replays/probe_bc`（superconduct_aggro vs natlan，legacy jsonl） | `superconduct_aggro` | `natlan_battleship` | 500000 | `h4m_bc_d256_e150.pt`（无 id） | 全字段 exact；argmax 一致（证明 jsonl/无 id 路径本身是通的） |
| `a0_control` | 同 `dvalin_compact` | `dvalin_bonk` | `natlan_battleship` | 0 | `h4m_bc_d256_e150.pt`（id_buckets=0） | 所有列都有 fp16 量化噪声（~1e-3）；没有 id 分支放大；logits `atol=1e-3` + argmax 一致 → 证明「放大器是 id 分支」 |
| `a0i_vs_a0i_noid` | 同 `dvalin_compact` | 同上 | 同上 | 0 | `A0i.pt` vs `A0i_noid.pt` | A0i 的 logits 在 card option 上出现大偏移；A0i_noid 回到 A0 附近 |
| `seat_swap` | 同一局 | 同上 | 同上 | 0 | `A0i.pt` | 把 `acting_player` 从 0 改成 1（只改 global token 的两列），argmax 变化 → 量化 #2 的影响 |

**判定**：

- 如果 `dvalin_compact` 的第一处分歧在 token 42 且满足 `id_store == fp16_roundtrip(id_live)`，**fixture 级确认** fp16 机制；不需要训练。
- 如果 `a0_control` 的 logits/argmax 在容差内一致，而 `A0i` 的 argmax 大量翻转，**fixture 级确认** id 分支是放大器；这已经足以否定 A0i 的结论，不需要等重训。
- 如果 `seat_swap` 让 argmax 翻转比例显著（例如 >10%），则 #2 也需要修（至少把 seat 加入采集覆盖或去掉该 one-hot）。

### 3.7 修复方向（仅列，不在本审计实现）

- **数据面**：compact payload 增加 `token_def_ids`/`option_def_ids`（int32，无损），或把 `token_features/option_features` 改成 fp32；**不要**尝试从 fp16 反推 id（152 卡 → 28 值，20 组歧义）。
- **模型面**：id 分支的输入改为「显式 int id」，不要从浮点标量反推；`definition_feature_indices` 的解析结果写进 checkpoint config，加载时校验而不是重算。
- **采集面**：teacher 作为 p0/p1 各采一部分（或对调 deck/opponent 再采一遍），覆盖 `acting_player=1`；评测继续双边。
- **训练面**：`torch.manual_seed(config.seed)` 放到模型构造之前；`--save-best`；`#t>0` 用 `random.Random(seed)` 或 `torch.Generator`。
- **评测面**：`--seed-start` 不要同时进 `seeds` 和 `base_seed`；`neural:` 强制 `--contestant-deck`；在 report 里记录实际 `match_seed`。

---

## 4. 需要补的测试（pytest 名称与断言）

> 这些测试大多可以在**无引擎**下跑；标 `[engine]` 的需要 `run_match`。`xfail(strict=True)` 表示「先钉住 bug，修复后去掉」。

### 4.1 直接钉住 A2 的测试

1. **`tests/test_parity_store_live.py::test_compact_fp16_preserves_card_definition_ids`**
   - 构造 `EncodedObservation`，token 第 42 列 = `211161/400000`、`332032/400000`，option 第 30 列同理；
   - `sample_to_payload(format="compact", compact_dtype="float16")` → `observation_from_payload`；
   - 断言 `definition_ids(tokens[:,42]) == [211161, 332032]`、`definition_ids(options[:,30]) == [...]`。
   - **当前会失败**（211161→211133，332032→332031）。先 `xfail(strict=True)`，修复后去掉。
2. **`tests/test_parity_store_live.py::test_compact_fp16_does_not_collapse_card_buckets`**
   - 对 `expert_vocabulary()[1]` 的 152 张卡，`bucket_ids(definition_ids(fp16_roundtrip(ids)))`；
   - 断言 `len(distinct) >= 0.9 * len(cards)`（当前 28/152，必挂）。
3. **`tests/test_parity_store_live.py::test_store_hand_tokens_decode_to_deck_cards`**
   - 读 `data/replays/dfbc_dvalin_bonk_c80/samples.jsonl` 前 N 条；
   - 对 `token_features[:, 7] == 1`（hand_card）的行，断言 `round(x*400000) in deck_spec("dvalin_bonk").cards`；
   - 当前 0/16,873，必挂；修好后应为 100%。
4. **`tests/test_parity_store_live.py::test_store_play_card_options_decode_to_deck_cards`**
   - 对 `option_features[:, 1] == 1`（action_play_card）的行，断言第 30 列解出的 id 在 deck 牌表；
   - 当前 0/1,684，必挂。
5. **`tests/test_parity_store_live.py::test_store_meta_records_encoder_config`**
   - 断言 `payload["meta"]` 含 `encoder_config == default_encoder().to_dict()`、`engine_lock`、`definition_indices == (42,30)`；
   - 当前 `meta` 没有这些字段，必挂。
6. **`tests/test_parity_store_live.py::test_replay_fingerprint_matches_current_encoder`**
   - 对 A2 的 8 个 dir，`CollectSpec(...).fingerprint() == manifest["fingerprint"]`；
   - 当前 **PASS**（我们已经逐个验证）；这是防未来漂移的护栏。

### 4.2 训练/推理一致性

7. **`tests/test_policy_parity.py::test_neural_policy_argmax_matches_collate`**
   - 固定一个 `EncodedObservation`，`model = load_cvpn_model(ckpt)`；
   - `NeuralPolicy.choose` 返回 `codes[argmax(model(collate([obs]))) ]`（t=0）；
   - 断言相等；同时断言 `last_value` 与直接 `model(...)` 的 value 相等（`atol=1e-6`）。
8. **`tests/test_policy_parity.py::test_teacher_index_roundtrips_to_same_code`**
   - 构造 `context`（或用 replay 的 built），`index = codes.index(code)`，断言 `codes[index] == code`；覆盖 `ACTION/CHOOSE_ACTIVE/REROLL/SELECT/SWITCH` 五类。
9. **`tests/test_policy_parity.py::test_include_training_targets_does_not_change_inputs`**
   - 同一 `context`，`encode_context(..., include_training_targets=True/False)`；
   - 断言 `token_features/option_features/token_mask/option_mask` 完全相同，只有 `privileged_state/belief_target` 不同。
10. **`tests/test_policy_parity.py::test_full_state_for_agents_does_not_change_inputs`** `[engine]`
    - 同一局，`run_match(..., full_state_for_agents=True/False)`，比较 p0 的 encoded token/option；
    - 断言完全相同（验证 §2.1 的结论）。
11. **`tests/test_policy_parity.py::test_option_permutation_is_benign`**
    - 把 `option_features` 和 `codes` 同步打乱，断言 `NeuralPolicy.choose` 返回的 code 不变（per-option scorer 的结构性安全）。

### 4.3 训练/评测协议

12. **`tests/test_eval_seed.py::test_protocol_seed_is_not_double_offset`**
    - `ProtocolSpec(base_seed=900, seeds=(900,), ...).tasks()[0].to_match(spec).seed == 900`（当前是 1800 → 必挂）；
    - 再测 `main` 的 `--seed-start 900 --seeds 10` 产生的 `ProtocolTask.to_match().seed` 是 900..909（不是 1800..1809）。
13. **`tests/test_train_init.py::test_model_init_is_seeded_before_construction`**
    - monkeypatch `train.model.CVPN.__init__` 记录进入时的 `torch.random.get_rng_state()`；
    - 断言 `datafirst_bc_train.main` 在构造模型前已经用 `--seed` 调用了 `torch.manual_seed`（或直接把 seed 调用移到 `model = CVPN(...)` 之前，并用两次 run 的首个 state_dict 逐位相等来验收）。
14. **`tests/test_model_id.py::test_checkpoint_records_resolved_definition_indices`**
    - `model.save(...)` 后 `torch.load(...)["config"]["token_def_index"] == 42`、`option_def_index == 30`（当前是 -1）。
15. **`tests/test_model_id.py::test_compatible_load_rejects_new_id_prefixed_params`**
    - 给 `load_state_dict_compatible` 一个缺 `id_proj.0.weight` 的 state dict，断言它只放行**已知的 6 个** id key；新增一个 `id_proj_norm.weight` 缺失时必须报错（当前按前缀会静默放行）。
16. **`tests/test_model_id.py::test_compatible_load_without_zero_init_is_random`**
    - 断言：`id_buckets>0` + 旧 checkpoint + `load_state_dict_compatible` + **不**调 `zero_init_identity_branch` 时，id 分支输出非 0；调了之后输出严格为 0（钉住「静默随机初始化」的语义）。
17. **`tests/test_observation_symmetry.py::test_acting_player_one_hot_both_seats`**
    - 用同一状态构造 `acting_player=0/1` 两个 context，断言 `global_token[45:47]` 分别为 `(1,0)`/`(0,1)`；
    - 并断言 `collect_samples` 的 store 至少包含一个 `act1=1` 的样本（当前不满足 → xfail）。
18. **`tests/test_observation_symmetry.py::test_token_and_option_masks_are_honored`**
    - 构造 `token_mask`/`option_mask` 中间为 False 的 `EncodedObservation`，`collate` 后断言 False 位置仍是 False（当前 `collate` 只按长度置 True，必挂）。

---

## 5. 不确定项与需要跑的最小验证命令

### 5.1 不确定项（按重要性）

1. **fp16 id 破坏是不是 A0i 崩盘的唯一/主因**：机制已实证（store/live 的 id 不同）；但「id 随机向量 → 胜率」的因果链还需要 §3.6 的 `a0_control`/`a0i_vs_a0i_noid` 或一次重采重训来闭环。seat 不匹配是并存的次要因素；另外 A0i 保存的是 ep149（val_ce 从 ep70 的 0.3747 回升到 0.5316）的**过拟合** checkpoint，不是 best，这也会拉低引擎胜率——但它对 A0i_noid 同样存在，解释不了 id-on/id-off 的 45pp 差距。
2. **能否从现有 compact 语料修复**：不能完整修复——152 张卡只对应 28 个 fp16 值，20 组歧义，只有 8 张卡唯一可反推；必须重采（jsonl/fp32）或改 payload 加 int 列。不要尝试「四舍五入到最近的真卡 id」，那会把 321015/321028/321036 之类混在一起。
3. **A0i_noid 的 0.397 val_acc**：它是同一 trunk 去掉 id 分支的推理 ablation，不是重训；不能当作「无 id 模型」的 val_acc。
4. **seat 不匹配的实际效应量**：需要 §3.6 的 `seat_swap`（或 1 seed × 2 sides 的对比）来量化；训练只 p0 是确定的，但影响可能被 `_owner_label` 的相对化部分抵消。
5. **`datafirst_bc_train` 的未播种 init 对 A2 结论的影响**：A0/A0i 是不同架构，init 方差不会改变「A0i 崩盘」的方向；但同 seed 重跑不可复现，做 A/B 时应先修。
6. **eval seed 双倍偏移是否影响其他已发布数字**：配对 A/B 不受影响；但所有 `--seed-start > 0` 的报告，其「seed 900–909」实际是 1800–1809。需要检查 `reports/explore/` 里所有 `base_seed` 非 0 的记录。
7. **`player_view` 是否在所有引擎版本都 mask 对手手牌**：由 `tests/test_info_leak.py` 覆盖；本审计没有重新验证引擎侧，但训练/推理走同一个 `snapshot_notification`，所以即使 mask 不完整也是**两边一致**的泄漏，不是 store↔engine 不一致。
8. **`cfr_search` 的 `chosen_option` 与 live codes 在 determinization 变化时是否总对齐**：A2 不走；如果重启 search-teacher，需要单独 parity。

### 5.2 最小验证命令（都不训练、不长时间评测）

> 下面这些是**已经在本次审计中跑过**的只读检查，或修复后应跑的最小命令。`python` 指 conda env `gitcg`。

**(a) 复现 fp16 id 破坏（~1 s）**

```bash
python -c "
import numpy as np
from reps.observation_encoder import DEFINITION_SCALE
from envs.observation import expert_vocabulary
chars,cards=expert_vocabulary()
rt=lambda i: round(float(np.float16(i/DEFINITION_SCALE))*DEFINITION_SCALE)
print('cards exact', sum(rt(c)==c for c in cards), '/', len(cards))
print('distinct corrupted card ids', len({rt(c) for c in cards}))
print('examples', [(c, rt(c)) for c in sorted(cards)[:6]])
"
```

预期：`cards exact 2 / 152`，`distinct 28`。

**(b) 复现 store 的 hand-card id 不在 deck 牌表（~10 s，只读 replay）**

```bash
python -c "
import json, numpy as np
from train.data import _unpack_dense
from reps.observation_encoder import DEFINITION_SCALE, definition_feature_indices, _TOKEN_TYPES
from envs.decks import deck_spec
ti,_=definition_feature_indices(); HT=_TOKEN_TYPES.index('hand_card')
cards=set(deck_spec('dvalin_bonk').cards); tot=hit=0; n=0
for line in open('data/replays/dfbc_dvalin_bonk_c80/samples.jsonl'):
    rec=json.loads(line)
    for s in rec['samples']:
        T=np.asarray(_unpack_dense(s['observation']['token_features']),dtype=np.float32)
        for row in T:
            if row[HT]==1.0:
                tot+=1; hit+= round(float(row[ti])*DEFINITION_SCALE) in cards
        n+=1
        if n>=4000: break
    if n>=4000: break
print(tot, hit)
"
```

预期：`16873 0`（前 4000 sample）；全量同趋势。

**(c) 核对 A2 语料 manifest 与当前 encoder 是否漂移（~1 s）**

```bash
python -c "
import json
from train.collect import CollectSpec
for d in ['dfbc_dvalin_bonk_c80','dfbc_double_geo_navia_c80','tgt_dvalin_bonk']:
    m=json.load(open(f'data/replays/{d}/manifest.json')); s=m['spec']
    spec=CollectSpec(deck=s['deck'],opponents=tuple(s['opponents']),seeds=tuple(s['seeds']),
                     teacher_spec=s.get('teacher_spec'),opponent_spec=s.get('opponent_spec'),
                     search=None,format=s.get('format','jsonl'),
                     compact_dtype=s.get('compact_dtype','float16'),
                     include_privileged_state=s.get('include_privileged_state',True))
    print(d, spec.fingerprint()==m['fingerprint'])
"
```

预期：全 `True`（说明本次不是 encoder 漂移）。

**(d) 核对 A0i_noid 没有随机 embedding（~1 s）**

```bash
python -c "
import torch
a=torch.load('data/checkpoints/a2/A0i.pt',map_location='cpu',weights_only=False)
b=torch.load('data/checkpoints/a2/A0i_noid.pt',map_location='cpu',weights_only=False)
print('A0i id_buckets',a['config'].get('id_buckets'),'A0i_noid',b['config'].get('id_buckets'))
print('n state',len(a['state_dict']),len(b['state_dict']))
print('all non-id tensors equal',
      all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in b['state_dict']))
"
```

预期：`1024 0`，`20 14`，`True`。

**(e) 核对未播种的模型初始化（~1 s）**

```bash
python -c "
import torch
from train.model import CVPN, CVPNConfig
cfg=CVPNConfig(token_dim=65,option_dim=68,d_model=32)
torch.manual_seed(0); m1=CVPN(cfg)
torch.manual_seed(0); m2=CVPN(cfg)
print(all(torch.equal(m1.state_dict()[k],m2.state_dict()[k]) for k in m1.state_dict()))
"
```

预期：`True`（这是正确行为）；再看 `scripts/datafirst_bc_train.py:144` 是否在 seed 之前——当前是，所以两次 `--seed 0` 的 run 并不保证相同 init。

**(f) 最小引擎验证（可选，2 局，约 1–2 分钟）**

```bash
python -m scripts.eval_protocol \
  --contestant neural:data/checkpoints/kaggle/h4m_bc_d256_e150.pt \
  --contestant-deck double_geo_navia --opponents natlan_battleship \
  --seeds 1 --seed-start 900 --no-write
```

带 `--contestant-deck` 是必须的（否则默认 `superconduct_aggro`）。这个 run 只用于验证 side=0/1 的胜负拆分（raw row 的 `side` 字段），不要当作强度结论。

**(g) 修复后的 parity harness（待实现）**

```bash
python -m scripts.parity_store_live \
  --replay-dir data/replays/dfbc_dvalin_bonk_c80 \
  --deck dvalin_bonk --opponent natlan_battleship --seed 0 \
  --checkpoint data/checkpoints/a2/A0i.pt --limit 3 \
  --json-out reports/explore/parity/dvalin_seed0.json
```

预期（修复后）：`OK`；当前预期：第一处 `FORMAT_FP16_ID`，`token 42`，`id_store=321094 id_live=321015`（或类似）。

### 5.3 最小可判定实验（如果只允许一个）

**用 jsonl 重采一个 deck、重训 A0i、同 seed 认证**：

- 数据：`dvalin_bonk` × 20 opponents × 20 seeds = 400 局（~0.1 h，本机 4.9k 局/h 口径），`--format jsonl`（或 fp32 compact）。
- 训练：同一 `mixed_four_decks` 的其余 3 个 deck 不变，只把 dvalin 的 400 局加到 store；重训 A0i（150 ep，~75 min / GPU；CPU 更久）。
- 评测：seed 900–909 × 20 opponents × 2 sides = 400 局（~0.1 h）。
- 判据：dvalin 的 A0i 从 0.678 回到 ≥ A0i_noid 的 0.303 且接近 A0 的 0.760；dgeo 不变（因为 dgeo 的卡 id 仍是 fp16 坏的，除非也重采）。如果 dvalin 明显恢复、dgeo 不恢复，**fp16 id 机制被闭环**。
- 更便宜的先验：§3.6 的 `a0_control` + `a0i_vs_a0i_noid`（1 局，秒级），先确认「特征确实不同、A0i 的 argmax 确实翻转」，再决定是否重采。

---

## 6. 附：本次审计实际执行的只读检查

- 读取 `reps/observation_encoder.py`、`reps/features.py`、`reps/event_features.py`、`reps/action_adapter.py`、`reps/action_hierarchy.py`、`reps/id_features.py`、`reps/snapshot.py`。
- 读取 `train/data.py`、`train/model.py`、`train/train_loop.py`、`train/collect.py`、`train/batched_inference.py`、`train/resolver.py`。
- 读取 `envs/match.py`、`envs/policy.py`、`envs/observation.py`、`envs/decks.py`、`envs/rollout.py`、`envs/cfr_search.py`、`envs/cfr_probe.py`。
- 读取 `agents/neural.py`、`scripts/datafirst_bc_store.py`、`scripts/datafirst_bc_train.py`、`scripts/train_bc_scale.py`、`scripts/eval_protocol.py`、`scripts/certify_a2.sh`、`scripts/run_id_sweep.sh`。
- 读取 `reports/explore/a2_id_screening.md`、`reports/train/a2_a0i.json`、A0/A0i/A0i_noid 的 eval JSON 的 `protocol` 块。
- 静态解码 `data/replays/dfbc_*_c80`、`tgt_*` 的 `samples.jsonl`，统计 hand-card/play-card 的 definition id 与 deck 牌表的交集；重算 8 个 manifest 的 fingerprint；对照 `A0i.pt`/`A0i_noid.pt` 的 state_dict。
- **未**运行训练、评测、引擎对局；未修改代码或数据，只写本报告。
