# 预注册：history 通道 + 特权 critic 的 A/B（2026-11-20）

> 状态：**预注册**，写于任何一局新数据采集之前。若结果与本文的判定规则不符，必须在
> `docs/failures/` 记录，不得事后改判据。
> 依据：`reports/explore/model_spec_20261120.md`（当前架构与三个缺口）、
> `reports/explore/reframe_20261120.md`（D30 问题刻画）。

## 1. 假设

当前模型的两个输入缺口是：① 32 个 history token 的通道**从未被填充**；
② `privileged_state`（god's-eye 全状态，含对手手牌/牌库直方图与真实骰子）**落盘但从不进网络**。

三条可检验假设：

- **H-hist**：把公开动作历史接进观测，会改变策略质量。方向未知（既可能补上"对手刚用过什么"的
  信息，也可能因为噪声/分布偏移变差）。
- **H-priv**：把特权信息只接进 **value 头**（PTIE），会改善 PG 的 advantage 基线
  （`adv = 终局收益 − value`），从而改善策略。**这是机制，不是猜测**：
  `scripts/train_bc_scale.py::run_batch` 的 `--loss pg` 分支直接用 value 作 baseline。
- **H-null**：两者都没有效果。这是必须能测出来的对照，不许事后解释成"数据不够"。

## 2. 臂（每臂尽量只改一个因素）

| 臂 | 观测 | value 头输入 | 用途 |
|---|---|---|---|
| **A0**（对照） | 无 history | 公开（现状） | 复现当前 recipe |
| **A1** | 有 history（≤32 token） | 公开 | 隔离 history |
| **A2** | 有 history | 公开 + 特权（PTIE，10% privileged dropout） | 隔离 critic |
| **A3**（可选） | 无 history | 公开 + 特权 | 2×2 的另一格 |

A0 与 A1 用**同一批种子**采集，因此对局完全相同、只有特征不同 → 干净的配对比较。
A1 与 A2 共享同一份数据（特权在采集时已落盘），只有训练时是否喂给 critic 不同。

**critic 的接法是加性偏置，不是 concat**（2026-11-20 定，实现前改的）：
`value = value_head(g + priv_proj(x))`，`priv_null = 0`。理由：concat 会把 `value_head`
的输入从 `d` 拓宽到 `2d`，于是 A2 必须**重新初始化 value 头**，而 `--loss pg` 的
`adv = 收益 − value` 正好吃这个头——A2 会仅因价值头被重置而看起来更差，正好污染我们要检验的东西。
加性接法下 `--init-from` 到特权配置是**逐位无损**的（`g + 0 = g`），A1/A2 从同一个 critic 出发。

## 3. 固定量

| 项 | 值 |
|---|---|
| 牌组 | 主：`dvalin_bonk`（最接近 PASS：0.780 / 专家 0.7425 / Δ **+0.0375** CI[−0.009,+0.084]）；副：`shenhe_double_swirl` |
| 采集 | `train.collect --all-opponents --seed-start 9100 --seed-count 100`（= 2000 局/臂，10 workers） |
| teacher | 现有最认证臂：`rl_dvalin1_dvalin_bonk.pt` / `rl_shenhe1_shenhe_double_swirl.pt`，`#t=1.0` |
| store | `--val-every 10`，compact |
| 训练 | `datafirst_bc_train --init-from <teacher> --loss pg --pg-entropy 0.01 --grad-clip 1.0 --lr 3e-5 --batch-size 256 --epochs 1 --d-model 256 --seed 0`（现行 recipe，不改超参） |
| 硬件 | Kaggle T4×2（`kaggle/train_arm.py`，每卡一 run）；训练失败则本机 CPU 同参数重跑 |
| 评测 | `eval_protocol --contestant neural:<ckpt> --contestant-deck <deck> --seeds 10 --seed-start 9200 --strict-ci --baseline expert:<deck>`（= 400 配对局，种子未用过） |

### 3.1 采集口配对性（已实测，2026-11-20）

A0/A1 的"同种子 → 同一局"依赖一个实现细节：**teacher 本身不吃 history**。
`_CaptureTeacher` 只把 `MatchHistory` 快照用于*编码样本*，它自己不在
`uses_match_history` 名单里，所以 `attach_history` 不会给它——teacher 的打法与 `--no-history` 完全一致。

实测（2 seeds，96 个决策）：

```
samples on/off: 96 96 | identical teacher decisions: True
history tokens on : 94   (前两个决策为 0，其余 >0)
history tokens off: 0
```

即：**特征差异是唯一差异**，同一 seed 下两臂打的是同一局。这条是 A0/A1 配对比较成立的前提。

> 注意（下一轮的口径）：第 2 轮起若用 history 训练过的 checkpoint 当 teacher，采集时 teacher 仍然
> 不吃 history，而评测时会吃。这是结构性分布差，必须在下一轮重新预注册，不能直接沿用本轮结论。

### 3.2 实测：history 直接进 mean-pool 会摧毁策略（2026-11-20，训练前发现）

第一次在真实 store 上跑 A1 时 `val_acc` 只有 0.29。定位到不是训练问题，是**架构问题**：
未训练过的 teacher 在两种输入上的表现（val 2000 条）：

| store | 未训练 teacher 的 val_acc | val_ce |
|---|---:|---:|
| `hh_a0_dvalin_bonk`（无 history） | **0.8710** | 0.3149 |
| `hh_a1_dvalin_bonk`（有 history） | **0.3680** | 28.9867 |

即 **32 行 history 一进入 masked mean-pool，就把池化向量位移到训练分布之外**，策略当场崩溃；
1 epoch、lr 3e-5 的 PG 不可能把它救回来（实测 0.749 → 0.29）。

这正是 §5.1 预注册过的"池化稀释"分支，因此按预注册执行结构改动，而不是改判据。

### 3.3 结构补救：history 单独池化 + 零初始化投影

```
h = masked_mean(tokens[history rows])        # 无 history 时为 0 向量
g = g_public + hist_proj(h)                  # hist_proj = Linear(d,d)，weight/bias 全 0 初始化
value / option logits / belief 仍只用 g
```

三条性质，都是这次改动的验收点：

1. **初始化时逐位等价**：`hist_proj(h) = 0` → g 与公开模型逐位相同，因此
   `--init-from` 一个无 history 的 teacher 是**无损**的（不再有"输入分布位移"这个confound）；
2. **无 history 的输入永远不受影响**：h=0 → 0，A0 臂与旧臂行为不变；
3. **能不能用 history 交给训练决定**：模型自己学 `hist_proj` 的权重，而不是被迫把 32 行平均进去。

A0（对照）不需要这个分支；A1/A2 用它。store 不需要重采——只有模型变。

## 4. 判据（跑之前定死）

主指标 = `paired Δ = 我方 − expert:<deck>`，只算干净局（0 fallback / 0 error / 未截断）。

1. **history 是否保留**
   - 若 `Δ(A1) − Δ(A0)` 的 95% CI 下界 **≥ −0.02**（非劣）且点估计 > 0 → **保留**；
   - 若 CI 下界 < −0.02 → history 默认关闭（`--no-history`），记 `docs/failures/`；
   - 其余情况（CI 含 0、点估计 ≤ 0）→ 保留但标注"无证据"，不得写"有效"。
2. **特权 critic 是否保留**
   - 若 `Δ(A2) − Δ(A1)` 的 95% CI 下界 **≥ −0.02** 且点估计 > 0 → 保留为默认；
   - 若 CI 下界 < −0.02 → 关掉，记失败；
   - 其余 → 保留为可选（`--privileged`），不写"有效"。
3. **是否升级认证**
   - 任一臂 Δ 点估计 **≥ +0.05** 且 n=400 的 CI 下界 > 0 → 用 **800 配对局**（20 seeds）
     在新种子上做正式认证；只有 800 局 `CI_low > 0` 且 floor 合规才写 PASS。
   - 该 deck 此前是 +0.0375 → 目标是突破到 +0.05 以上，而不是"用更大样本把 0.0375 显著化"。

### 4.1 目标可达性预检（写于评测之前，2026-11-20）

`eval_protocol --strict-ci` 的 floor 判据是**每对手 Wilson 下界 ≥0.45**，不是点估计。
用两个已 PASS 的臂反推这条线的实际高度：

| 臂 | pooled | 最弱 matchup（n=40） |
|---|---:|---|
| `final20_dgeo` | 0.880 | `electrocharged_moon` 0.700（Wilson low 0.546） |
| `final20_ungeo` | 0.909 | `unyielding_geo` 0.650（Wilson low 0.495） |

即 **PASS 需要每个 matchup ≈0.65 以上**。而 `rl_dvalin3`（当前最强 dvalin 臂，n=400）：

| matchup | rate (n=20) | Wilson low |
|---|---:|---:|
| `shenhe_double_swirl` | 0.350 | 0.181 |
| `unyielding_geo` | 0.350 | 0.181 |

这两个是对池里最强的两个 deck，差距 ~30pp。**本轮不可能靠 +3~5pp 的特征增益把一个
0.35 的 matchup 推到 0.65**，所以：

- 本轮判定**不是** "dvalin 是否 PASS"，而是 **Δ 相对基线的增量**（`Δ(A1)−Δ(A0)`、`Δ(A2)−Δ(A1)`）；
- 第 3 个 PASS 需要的是**针对特定 matchup 的定向补采/定向强化**（D29 的 floor 修补支线），
  本轮不承诺，也不把它写成"已达成"。

## 5. 反向停止条件

- 任一臂 pooled 比对照低 **>5pp** → 立即回滚该因素，不追加种子；
- 采集出现 fallback>0 或 error → 该局丢弃，若 >5% 则停；
- 训练 1 epoch 后 `val_acc` 比 teacher 低 >5pp（灾难性遗忘）→ 停，查 lr/init。

### 5.1 预注册的"如果失败"分支（防止事后编故事）

CVPN 是 **masked mean-pool**：history 把 T 从 ~29 抬到 ~55，等于把一半池化权重分给了历史行。
所以若 A1 显著差于 A0，**首选解释是池化稀释，不是"history 无用"**。届时的下一步是结构改动
（history 行单独池化后 concat，或改用 `TransformerCVPN`），并作为**新的**预注册重跑，
不得在本文件里改判据。

## 6. 记录义务

- 每臂一个 `reports/train/*.json` + `reports/eval/*.json`，带 engine lock 与 seed 口径；
- 结论写入 `docs/STATUS.md`（只写现在有效的）+ 必要的 `docs/failures/`；
- 若 history 有效，同步更新 `reports/explore/model_spec_20261120.md §2.3/§3.1` 的缺口描述。

---

## 7. 结果（2026-11-20，跑完）

完整数据与归因见 `docs/failures/008-history-priv-neutral.md`。三行结论：

1. **结构分支是必需的**：直接 mean-pool 会把未训练 teacher 的 val_acc 从 0.871 打到 0.368；
   零初始化加性分支把它恢复到 0.871（两种输入逐位等价）。
2. **纯模仿（CE 20 epochs）三臂等价**：val_acc 0.8779 / 0.8769 / 0.8770；引擎配对
   `A1−A0 = +0.0000 [−0.030,+0.030]`、`A2−A1 = +0.0000 [−0.016,+0.016]`，n=400。
3. **PG 1 epoch 方差压过效应**：同 recipe 同数据，seed1(GPU) Δ=+0.015、seed0(CPU) Δ=−0.128。

按 §4 判据：H-hist 与 H-priv 都落在"保留但无证据"一格——**不得写成有效**；
且 §5 反向停止条件没有触发（没有 >5pp 的下降）。

## 8. 与 §4.1 的关系

dvalin 的 floor 仍是同一处结构弱 matchup（`shenhe_double_swirl` / `unyielding_geo` 0.35@20），
与本次特征无关；第三个 PASS 仍未达成，也不在本轮承诺范围内。

---

## 9. 第二轮（hist2）：定向补采修 floor + PG 方差锚（写于跑之前，2026-11-20）

### 9.1 为什么换杠杆

§7/`failures/008` 已判定两个输入通道**零增益**，所以第二轮不再碰它们。dvalin 的唯一硬阻塞是
floor 的两个结构弱 matchup，而**数据侧的差距是可测的**（seeds 9200，n=20）：

| matchup | 我方 `rl_dvalin1` | 脚本专家 `expert:dvalin_bonk` | 差距 |
|---|---:|---:|---:|
| `unyielding_geo` | 0.35 | **0.65** | −30pp |
| `shenhe_double_swirl` | 0.35 | **0.50** | −15pp |

**关键**：差距不是"对手太强"，是我方在该 matchup 上比专家差 15–30pp。所以补采必须用
**脚本专家当 teacher**——用当前模型补采只会再写 200 局"输的走法"（这正是 failures/003 的教训）。

### 9.2 臂

| 臂 | store | 训练 | 目的 |
|---|---|---|---|
| `tg_bc_e20_s0` | base 2000 局 + 定向 400 局（expert teacher，ungeo/shenhe 各 200） | CE 20 ep, lr 1e-4, `--init-from rl_dvalin1` | 稳定模仿，看 floor matchup 是否被我方学到 |
| `tg_pg_ank_s0/s1/s2` | 同上 | PG 1 ep, lr 3e-5, **`--pg-anchor-kl 0.1`**, 3 seeds | 压 seed 方差（无锚时同轮 +0.015 / −0.128） |

### 9.3 判据（跑之前定死）

1. **floor 是否修好**：在 **20 seeds（800 配对局，种子 9600–9619，全新）** 上，
   `unyielding_geo` / `shenhe_double_swirl` 的每对手 rate 与 Wilson low；
   与同一批种子上 `expert:dvalin_bonk` 自己的 Wilson low 对照（D25 teacher-limited 豁免）。
2. **是否可宣称第三个 PASS**：同时满足 ① `pooled ≥ 0.75` 且 cluster CI_low 过线；
   ② 每个**非 teacher-limited** 对手 Wilson low ≥0.45；③ 配对 Δ 的 CI_low > 0。缺一不算。
3. **PG 方差**：三个 seed 的 Δ 极差（max−min）必须 < 4pp 才算"锚生效"；否则锚系数要调，不算负结果。
4. **不达标就停**：若 ungeo 的 rate 仍 <0.45（点估计），补采这条路在 dvalin 上判负，
   记 `failures/`，不再加种子。

---

## 10. 第三轮（hist4）：把 floor 缩到单个 matchup，并按功率改用 40 seeds 认证

### 10.1 n=80/matchup 之后的真实阻塞（2026-11-20，跑之前写死）

把 teacher 在 **3 组种子（9200 / 9600 / 9700）共 80 局/matchup** 上并表，
并用同一批种子上 `expert:dvalin_bonk` 自己的 Wilson low 做 D25 豁免判定：

| matchup | 我方 | 专家 | 专家 Wilson low | 判定 |
|---|---:|---:|---:|---|
| `unyielding_geo` | 0.588（low 0.478） | 0.600 | 0.490 | 专家**不豁免** → 我方 0.478 ≥0.45 **过** |
| `shenhe_double_swirl` | 0.562（low 0.453） | 0.613 | 0.503 | 不豁免 → 我方 0.453 ≥0.45 **过（压线）** |
| `dvalin_bonk`（镜像） | 0.438（low 0.334） | 0.475 | 0.369 | **豁免** |
| `double_geo_navia` | 0.487（low 0.381） | 0.637 | 0.528 | 不豁免 → **唯一真实阻塞** |

结论：**只剩 `double_geo_navia` 一个 matchup**，差距 +15pp（0.487 → ≥0.62 才能过 Wilson low）。

### 10.2 臂与判定

- 补采：`dvalin_bonk` vs `double_geo_navia`，**expert teacher**，200 seeds（9400–9599），`--no-history`；
  与已有 base(2000) + 弱 matchup(600) 合并成 `tgt_dvalin_merged3.pt`（2600 局）。
- 训练：Kaggle `hist4`，BC 20 ep（稳定臂）。
- **认证改用 40 seeds / 1600 配对局（种子 9900–9939，全新）**，理由：floor 是**每 matchup** 的
  Wilson low，40 局/matchup 的标准误约 ±8pp，会让同一个模型在不同种子集上从 0.75 掉到 0.25
  （本轮实测）。80 局/matchup 才有分辨力；这是**提高样本量**，不是改判据。
- 判定：`pooled ≥ 0.75` + 每个**非豁免** matchup Wilson low ≥0.45 + 配对 Δ CI_low > 0。

---

## 11. 第四轮（hist5）：在 hist4 臂上再走一轮自博弈 PG（写于跑之前，2026-11-20）

### 11.1 起点

`tg3_bc_e20_s0`（hist4 定向 BC）在 **1600 配对局（seeds 9900–9939）** 上：
pooled **0.7725**（≥0.75 ✅）、floor 在 D25 下干净 ✅、配对 Δ **+0.019 [−0.003,+0.041]** ❌
——只差 **0.003** 的 CI 下界。这是 dvalin 距 PASS 最近的一次。

### 11.2 臂

| 项 | 设定 |
|---|---|
| 采集 | 以 `tg3_bc_e20_s0` 为 teacher、`#t=1.0`，seeds **10000–10059** × 20 对手（1,200 局），`--no-history --no-privileged-state` |
| 训练 | Kaggle `hist5`：`--loss pg --pg-anchor-kl 0.1 --lr 3e-5 --epochs 1 --no-privileged`，`--init-from tg3_bc_e20_s0.pt`，**3 seeds** |
| 筛选 | **800 配对局**（8 seeds，10100–10107）每 seed；取 Δ 最高 |
| 认证 | **40 seeds / 1600 配对局**（10200–10239，全新） |
| 判据 | Δ CI_low **> 0**，且 pooled ≥0.75，且 D25 floor 干净。三者同时满足才写 PASS |

### 11.3 记录义务

- 3 个 seed 的 Δ 极差必须报告（锚是否真的把方差压住）；
- 若最好 seed 的 Δ CI_low ≤0，则 dvalin 在本轮判"仍不及"，**不再加种子刷显著性**，
  下一次必须先有结构性增益（新数据/新方法），而不是同一臂换种子。
