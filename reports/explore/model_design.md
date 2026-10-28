# 七圣召唤打牌 AI 的「大模型」设计与本机小规模验证（M3 探索）

> **status: superseded（2026-10-27）**：早期 design/小规模验证；后续 Transformer/BC/specialist 结论已变化。
> 当前状态见 `docs/STATUS.md`，有效方法见 `docs/PLAYBOOK.md`。

> 2026-10-11，WSL MX550（fp32，2 GB）。代码：`train/model.py`（`TransformerCVPN`）、
> `scripts/bench_model.py`、`tests/test_model_transformer.py`。
> 原始数据：`reports/explore/model_compare_20261011T13*.json`。

## 0. 结论速览

- 现役 `CVPN`（mean-pool + option 打分）在本地数据量（~2k 样本）下已经很稳。
- 新加的 **token 自注意力 + option 交叉注意力** `TransformerCVPN`（4.3× 参数）在**策略头**上略优
  （val_acc 0.741→**0.758**），但**价值头明显变差**（val_loss 0.856→1.176），且慢 2.5×、显存 265 MB。
- 结论：**本地 CPU 档数据量下，注意力模型不划算**；它的价值要等数据/算力放大后才兑现。
  下一步放大（机会算力）时优先解决**价值头标定**与**数据量**，而不是盲目堆层数（3 层反而更差）。

## 1. 现状模型 vs 设计目标

| | 现役 `CVPN` | `TransformerCVPN`（本次） | 目标（放大档） |
|---|---|---|---|
| 观测编码 | token 平均池化 → 全局向量 | token 自注意力（2 层、4 头、d128）+ 掩码池化 | 更深 transformer + CLS/角色/事件分段 |
| 动作打分 | `concat(全局, option)` MLP | option **交叉注意力**到 token 序列 | 分层动作头（high-level + 参数头） |
| 头 | policy + value | policy + value | + **belief 后验头**（G7）+ 可选 oracle value |
| 参数 | ~100k | ~431k | 数 M–数十 M（bf16，机会算力） |
| 本地成本 | 11 s / 60ep | 28 s / 60ep，265 MB | 需 GPU 放大 |

接口与 `CVPN` 完全一致（`forward(tokens, token_mask, options, option_mask) -> (logits, value)`），
可直接进 `train_bc` / `NeuralPolicy` / 学习曲线。

## 2. 本机小规模验证（同一池、同一留出、同种子）

2067 样本（2 脚本对手 × 20 种子），留出 20%，`precollate=True`，60 epoch：

| 模型 | val_loss | val_acc | 训练耗时 | 参数 |
|------|---------:|--------:|--------:|-----:|
| CVPN d128（现役） | **0.856** | 0.741 | 11.1 s | 100k |
| CVPN d256 | **0.831** | 0.738 | 13.4 s | 200k |
| Transformer 2L d128 | 1.176 | **0.758** | 28.4 s | 431k |
| Transformer 3L d128 | 1.388 | 0.697 | 34.1 s | 564k |

解读：

1. **策略头**：注意力让候选动作直接「看」到 token 序列（而非单一池化向量），acc 有小幅提升；
   但 **3 层过拟合/欠优化**，说明本机数据量不足以喂更深的注意力。
2. **价值头**：4.3× 参数在 2k 样本上把 value 回归带偏（val_loss 明显变差）——价值头是当前短板，
   不是 policy 头。
3. **成本**：显存 265 MB、2.5× 墙钟，2 GB 卡上留给放大档的空间有限。

## 3. 放大档设计（机会算力 / GPU 窗口）

优先级（对应 `train/CONTINUAL_RESOLVING.md` G7 与 PLAN §5.1）：

1. **价值头**：改为分布化/HL-Gauss 回归或共享 trunk + 单独归一化；value 目标从「终局胜率」换成
   **搜索根价值**（S-C search-as-teacher）后再评估——当前 value 目标是弱监督，可能是 transformer
   value 变差的主因。
2. **belief 头（G7）**：以 `reps/belief_groups.py` 的组边缘为 proper-scoring 目标，输出对手手牌/牌堆后验，
   供 `reps/belief.py` 的对手范围采样（替代当前均匀先验）。
3. **观测/动作编码**：token 分段（角色/装备/状态/骰子/公开历史）+ 分层动作头（high-level 分类 +
   目标/骰子参数头），抑制组合爆炸（PLAN M1 硬前置）。
4. **规模**：数据放大到 1e5–1e6 样本后再上 4–8 层、d256–512、bf16（机会算力 80G）；本机只做语义验证。
5. **批推理**：`train/batched_inference.py` + `scripts/bench_ops.py` 的结论（fp32、batch≥8、可 `torch.compile`）。

## 4. LLM 作为「大模型」打牌的另一条线（与上面正交）

「大模型」也可以指**语言模型直接决策**（PLAN WS4/D5）：`common/llm.py`（deepseek）+ `agents/llm_assist.py`
（局内候选生成/校验/兜底）+ `scripts/run_llm_matrix.py`（纯策略 vs LLM 辅助分开报分）已具备骨架。设计要点：

- LLM 不替代策略，而做**候选重排/校验/兜底**，受延迟预算约束（≤10 次/局），全部干预入库供蒸馏；
- 观测喂**掩码公共态 + 候选动作语义 key**（禁止上帝视角，沿用 `reps` 编码）；
- 本机可用 API 小样本验证（`.env` 已配 key）；训练/微调留机会算力（32B bf16/LoRA）。
- 与上面的网络路线共享评测口径（`eval/`）。

## 5. 复现

```bash
python -m scripts.bench_model --games 8  --d-model 128 --n-layers 2 --epochs 30 --device cuda
python -m scripts.bench_model --games 20 --d-model 128 --n-layers 2 --epochs 30 --device cuda
python -m pytest tests/test_model_transformer.py -q
```

## 6. 未做

- 未做超参搜索（lr/warmup/权重衰减）与 value loss 权重调优；未接 belief 头。
- 未在本机微调 LLM（2 GB 不可行）；LLM 路线的本机验证仍走 API（L5.2 矩阵）。
