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
  CLI：`python -m train.pipeline --train-seeds 6 --eval-seeds 3` / `--overfit` / `--replays data/replays/<tag>`。
  checkpoint 落 `data/checkpoints/`（gitignore），报告落 `reports/train/`。
- `collect.py` + `collect_worker.py`（L5.6 集群数据平面）：可续跑分片采集（`python -m train.collect`），
  一行一局、manifest 指纹护栏 + engine.lock、逐局流式落盘；`--replays` 供 `pipeline`/`learning_curve` 读盘。
- `agents/neural.py`：`NeuralPolicy`——把训练好的 checkpoint 接回 `envs.Policy`，可与脚本专家同台评测/自博弈（纯策略模式）。
  checkpoint 保存时一并写入 encoder 配置，故 `envs.policy.build_policy("neural:<ckpt>")` 可在 rollout/arena/采集
  等多进程路径中直接用训练好的网络（无需在调用处重建 encoder）。
- `replay_branch.py`：replay-branch MC teacher（D12 兜底，见下）——离线从任意决策点重放+注入候选+rollout 求动作价值。
- `learning_curve.py`（L5.3）：CVPN 学习曲线扫描——采集一份样本池后**先固定验证留出集**，再对每个
  (网络规模 `d_model`, 训练样本量) 组合在同一留出集上训练一个**全新**网络，记录 train/val 损失与一致率曲线
  （点间可比）；可选对脚本对手小样本评测。产物 `reports/train/learning_curve_<ts>.json/.csv`（不含 checkpoint）。CLI：
  `python -m train.learning_curve --games 14 --sizes 128 256 512 1024 --d-models 64 128`。

实测：`--overfit`（1 局采集、60 epoch）零报错完成，训练集一致率 ~0.74–0.88（多数类 ~0.3），说明网络可学。

## 搜索引擎分叉：边界快照即精确分叉（L5.4 实测，2026-10-09，I10/D13）

对 `gitcg` 0.21.0 pybinding 的中局快照行为做了系统实测（证据：
`scripts/probe_engine_snapshot.py`、`scripts/probe_boundary_fork.py`、`reports/engine/`；
契约见 `envs/snapshot.py`）。结论：

- ✅ **往返无损**：`State(json=snap).json() == snap` 逐字节相等（RNG/id 迭代器、整张实体图都在）。
- ✅ **能载入续跑**：`Game(state=State(json=snap))` + `start()/step()` 能跑到 `FINISHED`，不报错。
- ✅ **克隆确定**：同一快照两次续跑（同策略）逐字节一致。
- ✅ **边界快照即精确分叉**：`canResume:true` 暂停点（`Game.is_resumable()`）的快照，续跑喂
  record-replay 决策序列**逐字节复现活体终局**（13/13 + 9/9 复跑）。两条纪律：
  ① 只在 `is_resumable()` 点取快照——`canResume:false` 的 phase 内部点（`initHands` 抽牌后、
  `skill_executor` 中途、`gotWinner`）续跑会重放 phase 工作（0/3 复现），不可分叉；
  ② fork 必须补齐游戏级 attrs（`ATTR_PLAYER_ALWAYS_OMNI_*` 等不在 `GameState` 序列化里）——
  `envs/snapshot.fork_game(game_attrs=...)` 已封装。

早期"不能精确分叉"的结论（I9）源于探针混淆（复现用全新策略 RNG + fork 丢 attrs），已由 I10 修正。
因此 **M3 的 continual resolving 直接用 pybinding 分叉**（O(1) 毫秒级 fork + 注入候选 + rollout），
无需重写 pybinding/TS server/引擎手术（D13）。吞吐瓶颈不在 fork，在 rollout 本身（秒级/局）与
Python 回调开销；需要时再做 JS 侧批量 fork/rollout（归 `batched_inference`）。

### 薄 fork 桥 + 实用搜索 resolver（已实现，L5.4，2026-10-09，I11）

- `envs/fork_bridge.py` + `envs/fork_worker.py`：**子进程 fork 桥**（JSON-lines，沿用 `envs.rollout` 形态）。
  `ForkTask` 描述一次「边界快照 + 双方前缀（位置选项索引）+ 注入候选 + rollout」；`run_fork_task` 是进程内
  参考实现，`ForkBridge` 是子进程池（`evaluate` 批量并行）。**fork 必须在子进程**：在活体 `game.step()`
  回调内再跑引擎会撞 JS 运行时崩溃，故在线搜索不支持 `workers==0`。
- `envs/fork_search.py`：**实用版 continual-resolving**——`ForkSearchPolicy` 在最近 `is_resumable()`
  边界快照 fork + 注入候选 + rollout 求均值；驱动 `run_search_match` 每步前记录边界与双方前缀。
  基座策略每决策恰好调用一次且其选项恒为候选之一（RNG 不漂移，搜索不会忽略专家动作）；
  `rollouts`/`top_k`/`search_every`/`max_searches` 控预算；每次决策落 `SearchDecision`（喂教练/蒸馏）。
- 跨进程按**位置选项索引**重放已验证：新进程 fork 边界快照 + 前缀**逐字节复现活体终局**
  （`tests/test_fork_bridge.py`）。`scripts/run_search_agent.py` 做配对种子 search vs expert 消融评测。

## 集群数据平面（CPU 采集 → 存储 → GPU 训练，L5.6，算力前置）

`train/collect.py`（+ `train/collect_worker.py`）把对局编码成样本写入**可续跑**的 replay 目录：

- 一行 = 一局（含该局全部样本），撕裂的尾行只让那一局视为未完成 → 续跑重放该局，不会污染已完成数据；
- `manifest.json` 钉住 spec 指纹（deck/对手/种子/teacher spec/encoder 配置）+ `engine.lock`，指纹漂移拒续跑；
- 逐局流式落盘（`on_result`），作业被打断也保住已完成部分；
- 采集与训练解耦：`train.pipeline --replays <dir...>` 与 `train.learning_curve --replays <dir...>` 直接读盘，
  CPU 队列采集、GPU 队列训练（PLAN §5.1）。

```bash
python -m train.collect --deck superconduct_aggro --all-opponents \
    --seed-start 0 --seed-count 500 --workers 48 --tag coldstart
python -m train.pipeline --replays data/replays/coldstart --epochs 30 --tag cluster
```

LSF 模板：`scripts/cluster/lsf/collect.lsf`（单节点多 worker）、`collect_array.lsf`（海量数组作业，
每元素独立分片目录）、`train_gpu.lsf`（`--replays` 训练）。

`--teacher-spec`/`--opponent-spec` 支持任意 `build_policy` spec（含 `neural:<ckpt>`），
故同一采集器可做脚本预热、网络自博弈、以及网络 vs 脚本 anchor 数据。

### replay-branch MC teacher（已实现，批量离线/兜底）

`train/replay_branch.py`：整局对 `(卡组, 引擎种子, 动作序列)` 确定，所以任意决策点都能
**从初始态重放到达**，在那里注入候选动作、再用 rollout 策略打完一局。对 rollout 种子求均值
即为蒙特卡洛动作价值估计，可直接作 M3 的搜索蒸馏 target。

- 代价 `(1 + 候选数 × rollouts)` 局/决策（每局 O(深度) 重放，慢于 O(1) fork）→ **离线** teacher、
  跨进程批量标注与 fork 桥的对照验收；局内搜索用 `envs/snapshot.fork_game` 分叉。
- 选择以**选项位置索引**记录（非裸 low-level code），规避进程全局 codebook 漂移。
- 关键确定性：注入动作 = base 自身选择且 rollout = base 时，逐局复现 base（L5.4 分支续跑确定性验收，
  `tests/test_engine_bridge.py` 断言）。同注入两次逐字节一致。
- `evaluate_decision(...)` 对某决策的全部候选做 MC；`aggregate_option_values(...)` 汇总。
- `resolver.py` 的 `PriorResolver`（先验/贪心，非搜索）仍保留作冷启动；真 resolver 落地时替换接口。

## 待办（算力解锁后）

CVPN 放大（bf16/transformer）、`batched_inference`（按作业拉起，无常驻服务）、
搜索-蒸馏外环（候选 checkpoint 门禁：单测+泄漏+冒烟+评测不劣于）。
分叉能力已解锁（D13/I10）且薄 fork 桥 + 实用搜索 resolver 已交付（I11）；
剩余非前置项：JS 侧批量 fork/rollout 吞吐优化、CVPN 网络接入搜索（M3，算力解锁后）。

L5.3（学习曲线扫描）已交付 `train/learning_curve.py`，端到端跑通；本地只做千/万级样本的
正确性与规模-数据关系验证，全量放大仍待 GPU 算力（M3）。
