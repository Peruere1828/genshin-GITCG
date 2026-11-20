---
title: "008 — 公开动作历史 + 特权 critic 在纯模仿下对强度零增益"
kind: failure
status: negative
freshness: 2026-11-20
machines: [local, kaggle-t4]
authority: evidence
related:
  - "reports/explore/hist_priv_prereg_20261120.md"
  - "reports/explore/model_spec_20261120.md"
  - "PLAN.md#d30"
---

# 008 — history 通道 + 特权 critic：机制修好了，强度没动

## 假设

补上两个输入缺口能让 dvalin 的策略变强：

1. **H-hist**：32 个公开动作历史 token（此前通道存在但所有调用点传 `history=()`）能提供
   公开状态里没有的信息（对手刚用过什么、还剩什么资源）；
2. **H-priv**：把 god's-eye 特权状态只接进 value 头（PTIE），能改善 `--loss pg` 的 advantage
   基线 `adv = 终局收益 − value`，从而改善策略。

## 设置

- 牌组 `dvalin_bonk`（+3.75pp 平台期，最接近第三个 PASS 的臂）；teacher `rl_dvalin1_dvalin_bonk.pt`。
- 采集：同一批 100 seeds（9100–9199）× 20 对手 = 2000 局 **两遍**（`--no-history` 与默认），
  实测两遍的 teacher 决策**逐位相同**（96/96 决策、87,398/87,398 训练样本数一致）→ 特征差异是唯一差异。
- 训练：Kaggle T4×2，9 个 run（`KAGGLE_MODE=hist1`），全部 `--init-from rl_dvalin1`；
  - A0 = 无 history、公开 value（现行 recipe 对照）
  - A1 = history（结构分支）、公开 value
  - A2 = history + 特权 critic
  - 每组 × {PG 1 epoch seed 0/1, CE 20 epochs seed 0}
- 评测：`eval_protocol` 400 配对局，种子 9200–9209（此前从未使用），`--strict-ci --baseline expert:dvalin_bonk`。

## 结果

### 1. 结构上必须先修一次（否则根本没有"history 臂"）

直接把 32 行历史塞进 masked mean-pool，未训练的 teacher 立刻崩：

| 输入 | val_acc | val_ce |
|---|---:|---:|
| 无 history | **0.8710** | 0.3149 |
| 有 history（同一 teacher） | **0.3680** | 28.9867 |

改成"历史行单独池化 + 零初始化加性投影"后，两种输入都回到 **0.8710**（逐位等价）。
这条是**工程结论，不是失败**：任何新输入通道在 mean-pool 架构里都必须走零初始化旁路，否则
warm start 一定崩。

### 2. 纯模仿（CE 20 epochs）：三臂完全等价

| 臂 | val_acc | pooled | 配对 Δ vs 专家 | 95% CI |
|---|---:|---:|---:|---|
| A0 无 history | 0.8779 | 0.6775 | +0.0275 | [−0.020,+0.075] |
| A1 history | 0.8769 | 0.6775 | +0.0275 | [−0.021,+0.076] |
| A2 history+特权 | 0.8770 | 0.6775 | +0.0275 | [−0.021,+0.076] |
| 参照：teacher `rl_dvalin1` | — | 0.6900 | +0.0400 | [−0.006,+0.086] |

配对比较（同 400 局）：

- `A1 − A0` = **+0.0000**，SE 0.0154，CI[−0.030,+0.030]（400 局里只有 ~8 局结果不同）
- `A2 − A1` = **+0.0000**，SE 0.0079，CI[−0.016,+0.016]

即 **history 与特权 critic 在这批数据上对决策几乎零影响**。

### 3. PG 1 epoch：配方本身不稳，无法用来判断特征

| 臂 | val_acc（20 epoch CE → PG 1ep） | 引擎 Δ |
|---|---|---:|
| A0 seed1（GPU） | 0.8196 | +0.015 |
| A0 seed0（CPU） | 0.7490 | **−0.1275** |
| A1 seed0/1 | 0.556 / 0.625 | 未测（比 A0 更差） |
| A2 seed0/1 | 0.352 / 0.469 | 未测 |

同一个 recipe、同一份数据、只换 seed 与设备，引擎 Δ 从 **+0.015 掉到 −0.128**。
所以 400 配对局 + 1 epoch PG 的分辨率不足以比较特征——历史 ladder 的 "+3pp 每轮"也在同一噪声带里。

## 归因

1. **脚本专家在公开状态上近似马尔可夫**：它们的规则读 `player_view`（血量/骰子/手牌张数/
   场上状态），不看"对手上一步做了什么"。所以给模型一段公开动作历史，学不到能改变 argmax 的东西。
   这也是 A1/A2 只翻转 ~8/400 局的直接原因。
2. **特权特征的形式不对**：1022 维 histograms（双方手牌/牌堆的卡牌计数）在 value 头上
   **没有改善泛化**（val_value 0.1244 → 0.1274，反而略差），像是对 god view 过拟合。
   有用的是"对手还剩哪张关键牌/能否开大"这类**结构化**特权量，不是整张直方图。
3. **PG 的噪声大于要测的效应**：1 epoch、lr 3e-5、batch 256、adv 以 value 为基线；
   seed 间 ±10pp。要在这上面比较 2–3pp 的特征效应，必须先降方差（多 seed 平均 / BC 锚 / KL 约束）。

## 结论

- **通道补上了、可训练、不伤害**（三臂 val_acc 差 <0.1pp，CI 全部含 0）；
- **但纯模仿下对强度零增益**，且原因是结构性的（对手马尔可夫 + 特权特征形式不对），
  不是"数据不够"或"训练不够久"；
- dvalin 依旧 **NOT PASS**：floor 的两个结构弱 matchup（`shenhe_double_swirl`、`unyielding_geo`
  各 0.35@20 局）与特征无关，需定向补采，见 `hist_priv_prereg_20261120.md §4.1`。

## 重开条件

**history**：只在对手策略**非马尔可夫**时重开（人类对局、league 自博弈对手、或带"记仇/连招"
状态的脚本）。判据：先证明"该对手的下一步动作分布依赖更早的公开历史"（互信息 > 0），
再用 n≥800 配对局测；纯模仿下不再重复。

**特权 critic**：必须同时满足
1. 换成**结构化**特权输入（对手剩余牌数按"威胁卡"分组、burst 就绪、元素附着链），维度 ≤64；
2. 先过**价值真值测试**（`train/replay_branch` 在固定位置 fork 求经验胜率，
   Brier/AUC 显著优于常数 0.25）——这是 D30 时就欠下的前置；
3. advantage 用**多 seed 平均**或 BC 锚（KL 到 teacher），把 PG 的 seed 方差压到 <2pp 才比较。

**PG 配方本身**：单 seed 单 epoch 的 Δ 不可作为结论；任何新的 PG 结论都必须带
≥3 seeds 的 seed-to-seed 方差。
