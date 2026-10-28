---
title: "006 — A2 identity embeddings（实验无效）"
kind: failure
status: invalid
freshness: 2026-10-27
machines: [local, a800]
authority: evidence
related: ["PLAN.md#d25", "reports/explore/a2_id_screening.md", "reports/explore/p0_parity_20261027.md"]
---

# 006 — A2 identity embeddings（实验无效）

## 假设

把 encoder 的 `definition_id / 400000` 反解成类别，加 token/option id embedding，能提升卡牌/角色表示。

## 设置

- CVPN d256 + `id_buckets=1024`、`id_emb_dim=32`；4-deck mixed store；150 ep。
- 认证：3 deck、seed 900、双向、400 局/deck；A0（无 id）vs A0i（id）。
- 结果：A0i store val_acc 0.866（A0 0.838），引擎 dgeo 0.266（A0 0.720）、ungeo 0.358（0.828）、dvalin 0.678（0.760）。

## 为什么无效

- **数据事故**：compact 语料默认 fp16，`definition_id/400000` 量化后不可逆；152 张卡只剩 28 个不同 id，真实 bucket 142 个、与 store bucket 交集仅 4。
- A2 的 id 分支训练时读假 id、推理时读真 id；卡牌类 token/option 几乎全部命中训练时未更新的 embedding 行。
- 另叠加：p0-only/双边评测、模型 init 不受 seed 控制、val_acc 跨语料横比、只保存 final epoch（best ep70）、eval seed 双倍偏移。
- 因此 **A2 不是“id embedding 有害”的证据**；A0i_noid 的 dgeo 0.741 也不能当通用底座。

## 重开条件（全部满足）

1. exact int id：compact payload 带 `token_definition_ids` / `option_definition_ids`，或整体 fp32。
2. parity PASS（`docs/PLAYBOOK.md` §4），definition id 0 失配。
3. store meta `definition_ids_present == events`；训练脚本拒绝缺 id 的旧语料（已实现）。
4. A0 vs A0i 同 store、同 seed、同 init 纪律；800 screen + 1,600 confirm；只看引擎配对 Δ。
5. 2 个 deck（dgeo/dvalin）先判；无正信号再归档。

## 证据

- `reports/explore/a2_id_screening.md`（顶部 `status: invalid` 更正）
- `reports/explore/literature_review.md` §1
- `reports/explore/lit/f05_train_eval.md` / `f07_code_audit.md`
- `reports/explore/p0_parity_20261027.md`

## 不要重演

- 不要在旧 fp16 store 上重训 id 分支。
- 不要用 mixed val_acc 横比不同 store。
- 不要把 A0i/A0i_noid 当通用底座。
