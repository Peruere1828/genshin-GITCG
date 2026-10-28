# 定向文献调研 brief（2026-10-27）

> **status: archived（2026-10-27）**：7 条调研线已完成；最终汇总/裁决见
> [`reports/explore/literature_review.md`](../literature_review.md)。本文件保留为调研上下文。

> 本文件是多个调研 subagent 的共享上下文。**只读**代码与 `refs/`；不改代码、不跑训练/评测。
> 产出写到 `reports/explore/lit/<topic>.md`，最终由主会话汇总为 `reports/explore/literature_review.md`。

## 1. 项目与当前处境

- 项目：Genshin TCG（七圣召唤）AI，目标是在**固定 20 套脚本专家池**上把纯策略强度做上去。
- 评测口径：**纯策略**（LLM 不参与）；20 个对手 × 双向 × ≥20 seeds，配对种子；`scripts/eval_protocol.py`。
- 硬约束：观测只能来自 `notification.state`（信息隐藏测试是 CI 门禁）；动作抽象必须 100% 覆盖合法动作；
  对局按种子确定性重放；结果逐 deck 报，不能用一个 pooled 数字掩盖长尾。

### 1.1 已有的正/负结果（不要重复推荐已被证伪的路线）

| 路线 | 结果 |
|---|---|
| 20-deck 单模型（707k 样本） | 8 个 deck 里 7 个显著变差（负迁移） |
| 单 deck 专家 BC | 最好只到「与同 deck 专家打平」；多数落后 3–11pp |
| 数据量 8k→48k（同 epoch） | 不显著；**epoch/优化预算**才是杠杆 |
| AWR / Q 学习 / DMC 式离线 RL | 均为负结果（`reports/explore/plan_strength_2day_kaggle.md`） |
| search-as-teacher / MCCFR 蒸馏 | 全池配对 Δ 显著为负；根价值区分度≈0 |
| 逐 deck 迭代 PG（REINFORCE+baseline） | **唯一稳定正结果**：dgeo 0.880（+16.8pp）、ungeo 0.909（+5.5pp），完整门禁 PASS；dvalin 平台 +3pp；SA 失败 |
| A2 identity embeddings | **假阳性典型**：store val_acc 0.866（A0 0.838），引擎胜率 dgeo 0.266（A0 0.720）、ungeo 0.358（0.828）、dvalin 0.678（0.760） |

A2 诊断：同一 trunk 去掉 id 分支后 dgeo 0.741 / ungeo 0.670 / dvalin 0.303——**trunk 与 id 分支都强 deck 依赖**，
且 `store val_acc` 与引擎胜率脱钩。当前首要怀疑：**store→engine 特征/训练-评测不一致**，
其次是 BC 的状态分布偏移（expert-vs-expert 语料，eval 时看到自己的状态分布）。

### 1.2 当前工程事实（用于把论文落到代码）

- 模型：`train/model.py` 的 `CVPN`（d256，2 层 token 编码，option 独立打分 + value head）。
- 观测编码：`reps/observation_encoder.py`（token 65 维 / option 68 维；definition 列 token 42 / option 30）。
- 语料：`train/data.py`（`collect_samples`，expert teacher，`history=()`）；`train/collect.py` 落盘；
  `train.collect --search` 走 `envs/cfr_search.py`；compact 格式由 `train/data.py` 的 dense payload 编解码。
- 在线策略：`agents/neural.py`（`NeuralPolicy.choose`，贪心/温度采样）；spec 路由 `envs/policy.py`。
- 对局：`envs/match.py`（Python 侧预洗牌 + `NO_SHUFFLE=1`）、`envs/rollout.py`（多进程）。
- 认证：`scripts/eval_protocol.py` + `eval/scoreboard.py`（Wilson、cluster CI、配对 Δ、floor、Elo、exploitability 下界）。
- 训练脚本：`scripts/datafirst_bc_train.py`、`scripts/train_bc_scale.py`、`train/train_loop.py`、`train/pipeline.py`。
- 现有专家：`agents/scripted/`（20 套规则）；`eval/opponents.py` 是池注册表。
- 参考实现（只读）：`refs/Rebel_base_RL/.../gitcg_world_model`（SoG 蓝本）、`refs/genius-invokation`（引擎）、`refs/AI`（DouZero 式 DMC）。

### 1.3 算力与成本口径

- 本机 12 核 CPU 是采集/评测主力：约 3,700–4,900 局/h（8–12 worker），训练 CPU d256 约 150–300 样本/s。
- A800 80G 为申请制机会资源，**只做 GPU 训练/推理**，CPU 配额 ~10 核；数据本地采集后传入。
- 一局约 46 个决策；compact 语料约 0.65 MB/局；认证 400 局/组（10 seeds）或 800 局/组（20 seeds）。
- 因此任何建议都必须标注：**新增采集局数、训练 GPU·h、评测局数**，并给出“最小可判定实验”。

## 2. 这次调研的 5 个问题（老师指定）

1. **卡牌/不完全信息自博弈**：DouZero、Suphx、ReBeL、SoG——重点看 self-play 采样、value calibration、
   public belief 表示，而不是网络层数。
2. **离线 RL / 模仿学习分布偏移**：DAgger、IQL/CQL/AWR、return-conditioned——解释 AWR 为何在此没赢，
   DAgger 如何接现有 collector。（引用勘误：Mindermann 2022 是 arXiv:2206.07137；PCGrad 是 arXiv:2001.06782。）
3. **league / population training**：PFSP、PSRO、AlphaZero 对手采样——对应 specialist 阶梯与 anchor 防漂移。
4. **组合动作/候选集建模**：Set Transformer、pointer network、option-set attention——对应 A4 与骰子/多目标动作。
5. **train/eval mismatch**：offline RL distribution shift、feature consistency、calibration——A2 的直接教训。

另外单列两条工程性调研：**specialist portfolio + router**、**worst-matchup 靶向采集 + 对手条件化**。

## 3. 产出契约（每个 topic 一份 markdown）

固定结构，便于汇总：

```markdown
# <topic>
## 0. 一句话结论（可执行）
## 1. 与我们问题的对应关系（不要泛泛复述论文）
## 2.  annotated bibliography（8–15 篇/个实现）
每篇：问题 / 方法核心（含关键公式或伪代码）/ 证据（数据集、基线、效应量）/ 局限 / 代码或复现链接 / 对我们的可迁移点
## 3. 最小可复现 recipe（3–5 个，按性价比排序）
每个：输入输出、关键超参、伪代码、需要改的仓库文件、预估成本、预期可观测信号、失败判据
## 4. 预注册实验（最重要：只给 1–2 个）
假设 / 对照 / 指标（引擎胜率，不是 val_acc）/ seeds 与功效 / 采集与训练预算 / 判定阈值 / 停止条件
## 5. 与现有负结果的关系（为什么这次不会重演）
## 6. 不要做什么（negative guidance）
## 7. 参考（链接、版本/日期、可信度）
```

硬性要求：

- **优先读代码/复现细节**，不要只读 abstract；能拿到 GitHub 的就读关键文件（训练循环、损失、采样、评测）。
- 引用必须给可点击 URL；不确定的年份/数字标注“未核实”，**禁止编造**。
- 区分「论文声称」与「我们可迁移的证据」；给出效应量与适用条件。
- 所有建议必须落到本仓库的具体文件/接口；不能写“加一个 Transformer”这种空话。
- 中文正文，论文标题保留英文。
- 不修改 `refs/`、不跑长任务、不写入 `data/`。
