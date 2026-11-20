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
