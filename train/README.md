# train — SoG/ReBeL 训练内核（PLAN.md WS3 / M3）

cvpn_model（bf16 小网络 + deck-id 条件化）、continual_resolving + resolving_agent（公共信念搜索，先实用采样 resolver）、
cvpn_training + sog_pipeline（搜索蒸馏外环 + 门禁）、batched_inference（按作业拉起的 GPU 批推理，无常驻服务）。

## 已实现（L4 玩具验证，2026-10-07）

- `model.py`：`CVPN`（小网络，CPU 可训）= token 平均池化 → 每个合法 option 打分 + 价值头。**按 option 特征打分**，
  与进程全局 low-level action code 解耦（见 AGENTS.md 确定性坑）。
- `data.py`：`collect_samples`——用任意 teacher policy（脚本专家）打对局，把每个决策上下文编码成样本，
  teacher 动作用 **option 位置索引**保存，价值目标=该局结果。
- `train_loop.py`：行为克隆/蒸馏（masked CE + value MSE），train/val 切分与指标。
- `pipeline.py`：一轮「采集 → 训练 → 评测（在真引擎里对脚本对手）→ 门禁 → 落盘」。
  CLI：`python -m train.pipeline --train-seeds 6 --eval-seeds 3` / `--overfit`。
  checkpoint 落 `data/checkpoints/`（gitignore），报告落 `reports/train/`。
- `agents/neural.py`：`NeuralPolicy`——把训练好的 checkpoint 接回 `envs.Policy`，可与脚本专家同台评测/自博弈（纯策略模式）。

实测：`--overfit`（1 局采集、60 epoch）零报错完成，训练集一致率 ~0.74–0.88（多数类 ~0.3），说明网络可学。

## ⚠️ 搜索（continual resolving）的引擎阻塞（M3 关键前置）

**gitcg 0.21.0 pybinding 无法克隆/恢复中局状态**：`State.toJson` 写 `canResume: false`
（`packages/core/src/game.ts`、`cbinding/js/main.ts`），`State(json=...)`+`Game` 往返后
`Game.is_resumable()==False`，对局无法续跑。搜索结果需要从同一状态分叉评估多个候选动作。

因此 M3 的搜索必须先解决其一：
1. 扩展 pybinding / 用 TS server 的 `canResume:true` 暂停路径，暴露可恢复快照；
2. 在活对局的回调内做搜索（但无克隆仍不能分叉）；
3. 用学到的模型做 rollout（研究项）。

在此之前，蒸馏的 teacher 只能是**非搜索**策略：脚本专家（BC，已实现）或网络自身先验（自蒸馏）。
`resolver.py` 提供 `PriorResolver`（先验分布/贪心）与 `SEARCH_BLOCKED_REASON`，真 resolver 落地时替换。

## 待办（算力解锁后）

CVPN 放大（bf16/transformer）、`batched_inference`（按作业拉起，无常驻服务）、真搜索 teacher、
搜索-蒸馏外环（候选 checkpoint 门禁：单测+泄漏+冒烟+评测不劣于）。
