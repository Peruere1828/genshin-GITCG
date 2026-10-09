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
- `replay_branch.py`：replay-branch MC teacher（D12 兜底，见下）——离线从任意决策点重放+注入候选+rollout 求动作价值。

实测：`--overfit`（1 局采集、60 epoch）零报错完成，训练集一致率 ~0.74–0.88（多数类 ~0.3），说明网络可学。

## 搜索引擎：快照能载入，但不能"精确分叉"（L5.4 实测，2026-10-09）

对 `gitcg` 0.21.0 pybinding 的中局快照行为做了系统实测（证据：
`scripts/probe_engine_snapshot.py`、`reports/engine/`；文档/常量见 `envs/snapshot.py`）。
结论比早期记录更精确：

- ✅ **往返无损**：`State(json=snap).json() == snap` 逐字节相等（RNG/id 迭代器、整张实体图都在）。
- ✅ **能载入续跑**：`Game(state=State(json=snap))` + `start()/step()` 能跑到 `FINISHED`，不报错。
- ✅ **克隆确定**：同一快照两次续跑（同策略）逐字节一致。
- ❌ **不是活体决策点的忠实分叉**：续跑从 `state.phase` 重新进入 phase 循环，而引擎的暂停点在
  phase 内部也有（`initHands` 抽牌后、`skill_executor` 加充能后、`gotWinner`），序列化的
  `GameState` 不含被挂起的 async continuation，于是续跑会重放一段 phase。`canResume`
  （TS server 日志 / `Game.is_resumable()`）**并不能可靠标记可重放边界**：126 步的探针对局里，
  除最后 2 个快照外全部与活体轨迹分叉。

因此 continual resolving **不能**通过 pybinding 分叉活体对局。TS server 也不从 JSON 恢复
对局（它始终持有活体 JS 对象，`stateLog` 只用于回放/观战）。真活体分叉需要一个
以非 async-continuation 契约序列化的新引擎桥（D12 主路线，未完成）。

### D12 兜底：replay-branch MC teacher（已实现）

`train/replay_branch.py`：整局对 `(卡组, 引擎种子, 动作序列)` 确定，所以任意决策点都能
**从初始态重放到达**，在那里注入候选动作、再用 rollout 策略打完一局。对 rollout 种子求均值
即为蒙特卡洛动作价值估计，可直接作 M3 的搜索蒸馏 target。

- 代价 `(1 + 候选数 × rollouts)` 局/决策 → **离线** teacher，非局内搜索。
- 选择以**选项位置索引**记录（非裸 low-level code），规避进程全局 codebook 漂移。
- 关键确定性：注入动作 = base 自身选择且 rollout = base 时，逐局复现 base（L5.4 分支续跑确定性验收，
  `tests/test_engine_bridge.py` 断言）。同注入两次逐字节一致。
- `evaluate_decision(...)` 对某决策的全部候选做 MC；`aggregate_option_values(...)` 汇总。
- `resolver.py` 的 `PriorResolver`（先验/贪心，非搜索）仍保留作冷启动；真 resolver 落地时替换接口。

## 待办（算力解锁后）

CVPN 放大（bf16/transformer）、`batched_inference`（按作业拉起，无常驻服务）、真搜索 teacher、
搜索-蒸馏外环（候选 checkpoint 门禁：单测+泄漏+冒烟+评测不劣于）。
真活体分叉的引擎桥（D12 主路线）仍欠：需要以非 async-continuation 契约序列化的 TS server 快照；
在此之前 replay-branch MC（`replay_branch.py`）即 teacher。
