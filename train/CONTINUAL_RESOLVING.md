# 实用 resolver → GT-CFR continual resolving：差距清单（M3 输入）

> 来源：`refs/Rebel_base_RL/docs/SOG_GT_CFR_CONTINUAL_RESOLVING_CHARTER.md`（目标架构）
> 与 `refs/Rebel_base_RL/research/world_model/src/gitcg_world_model/continual_resolving.py`
> （参考实现，含 `_LocalSampledGTCFRSolver`）。本文是 PLAN.md §10「实用 resolver → GT-CFR
> 差距清单」的交付物，也是 **M3 主线清单（D15）**：无 GPU 期间按 §3 顺序在 CPU 档（本机 + WSL）
> 跑通语义与单测，规模放大留给机会算力；中间验收 M3-CPU 见 PLAN §4.3。

## 0. 结论速览

我们现在的 `envs/fork_search.py` 是**determinization + MC 动作价值**的实用 resolver：
fork 活体边界快照（= 真实的隐藏信息确定化）、注入候选、用 rollout 打到终局、取胜率均值。
它**能跑、能当离线/在线 teacher 的雏形**，但相对 charter 的 GT-CFR 主线上还有结构性差距，
且有两条**违反 charter 不变量**的地方，不作主线求解器：

- 违反 §5.2/§7.2：搜索从活体快照 fork，根节点含**真实隐藏信息**（单一确定化），不是公共信念；
- 违反 §5.3：训练目标是 rollout 终局胜率，不是 solver 的根平均策略/价值。

参考实现（`continual_resolving.py`）已经落地了 sampled GT-CFR、公共信念重建、belief 头部与
search-teacher 目标；差距主要在**信念根化、regret 求解、深度受限 + frontier 价值、search-as-teacher**。

## 1. charter 目标架构（压缩）

exact simulator → public belief state → modified safe continual resolving → **GT-CFR 局部求解** →
深度受限 + 神经 frontier 评估 → CVPN 蒸馏（policy prior + value + belief）→ sound self-play →
exploitability 评估。**网络不是决策者，求解器才是；网络逼近更强的求解器。**

## 2. 我们现状 → 参考实现 → 差距

| # | 能力 | 参考实现（`continual_resolving.py` / `resolving_agent.py`） | 我们的现状 | 差距 |
|---|------|------|------|------|
| G1 | **信念根化搜索** | `build_public_belief_state` + `_build_self/opponent_range` + `_LiveParticleBank`，从 `player_view` 采样隐藏手牌/牌堆/骰子 | `ForkSearchPolicy` fork 活体快照（真实隐藏态单一确定化） | **最高**：必须改为「公共信念 + 采样确定化」根，禁用隐藏真值（§5.2） |
| G2 | **regret 求解器** | `_LocalSampledGTCFRSolver._cfr` + `_regret_matching_distribution` + `root_strategy_sum` 平均策略 + 根价值 | 无 CFR/regret；只对候选做 MC 胜率 | **最高**：加 sampled（外采）MCCFR 脚手架，产出根平均策略 + 根价值 |
| G3 | **信念层** | `PublicBeliefState`/`ExplicitRange`/`range_summary`/`PublicBeliefUpdater` | `reps/belief_groups.py`、`reps/public_state.py` 已移植但**未接入搜索** | 中：把 belief 表示接进搜索根与 update |
| G4 | **深度受限 + frontier 价值** | `depth_remaining` 到 0 调 `_leaf_value`（CVPN value + belief summary） | rollout 打到终局（无截断、无价值评估） | 高：引入深度预算与 CVPN value frontier |
| G5 | **prior 作排序非删除** | `_prioritized_root_indices`/growing-tree 展开，prior 只排序 | `top_k` 固定截断候选（虽已改成按 prior 排序） | 中：改成可增长树展开（§5.4）；`ForkSearchPolicy.prior` 已迈出第一步 |
| G6 | **search-as-teacher** | agent 导出 `search_teacher_policy`/`search_teacher_value`/semantic keys/iterations/depth | `SearchDecision` 只有逐选项 MC 均值；训练是专家 BC + 终局价值 | 高：导出根平均策略 + 根/frontier 价值 + belief 目标，蒸馏回网络 |
| G7 | **网络头部** | CVPN 有 policy/value/**belief 后验**/recurrent/search heads | toy CVPN 仅 policy+value（`train/model.py`） | 中：加 belief 头（含 proper scoring loss）、（可选）oracle value 头 |
| G8 | **query solver** | `query_mode` 更高预算离线求解，与在线 actor 解耦 | 无 | 低（M3 后期）：独立 query 求解器写回更强目标 |
| G9 | **GPU 批推理接入搜索** | `GpuInferenceServer` + `batched_inference` | `train/batched_inference.py` 已交付但**未接进搜索** | 中：搜索叶子批量打分走 batched inference |
| G10 | **exploitability 评估** | —（charter §9 要求 reduced-domain + best-response 探针） | 只用对池胜率 | 中（M3 后）：小规模简化域 + BR 探针 |

## 3. 建议的最小 M3 落地顺序

（CPU 档先跑通语义与单测：小网络、1e3–1e4 样本、fp32；样本量/网络规模放大留给机会算力，PLAN §5.1。）

1. **信念根化（G1+G3）**：即便先上粗糙后验（未见牌上的均匀/`belief_groups` 边缘），把搜索根换成
   `PlayerView` + 采样确定化，消除 §5.2 违反。落点：`envs/fork_search.py`（根化）、`reps/belief_groups.py`。
2. **sampled GT-CFR 脚手架（G2+G4）**：在粒子树上做外采 MCCFR（regret matching + 平均策略），
   固定深度 + CVPN frontier 价值（charter §6.3 明确允许外采 MCCFR 作脚手架，语义仍是 GT-CFR）。
   落点：新增 `train/cfr_solver.py`，叶子走 `train/batched_inference.py`。
3. **search-as-teacher（G6）+ 网络 belief 头（G7）**：把根平均策略/根价值/belief 目标写进 replay，
   训练改成搜索蒸馏（policy CE + value 回归 + belief 评分），专家 BC 降为冷启动/弱辅助。
4. 之后：query solver（G8）、异步批推理（G9）、exploitability 代理（G10）。

## 4. 明确不做（charter 不变量，违反即返工）

- 用隐藏真值做在线动作选择 / 从隐藏真值态发起在线局部求解（§5.2、§7.2）；
- 用 PPO/GAE 作主训练信号（§8、§13.1）；
- 用固定极小 top-k 永久删除合法动作作为核心搜索规则（§5.4、§11）；
- 用逐步策略 rollout 取代 continual resolving（§11）。

## 5. 当前实用 resolver 的正确定位

`envs/fork_search.py` + `train/replay_branch.py` 作为**确定化 MC teacher / 对照基线 / 冷启动目标**保留；
它们的产出（逐选项 MC 价值、`SearchDecision` 日志）可作为 M3 solver 的回归对照与 bootstrap 数据，
但**不是**主线求解器。主线按 §3 顺序演进到信念根化的 sampled GT-CFR。
