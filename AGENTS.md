# AGENTS.md

需求/架构/验收的唯一权威是 [PLAN.md](./PLAN.md)（决策记录 §0、里程碑 M0–M6 §4、本地先行 L0–L4 §4.1）

## 现状

- **WS0/WS1 已有可运行代码**（本地先行 L0/L1/L2 起步；2026-10-07）：
  - `reps/`：从 `refs/Rebel_base_RL/.../gitcg_world_model` **逐模块移植**的表示层（`schema`/`snapshot`/`action_hierarchy`/`action_adapter`/`public_state`/`semantic_priors`/`agents`/`replay` + 对应 toml），导入路径改为 `reps.*`。来源与许可见 `NOTICE.md`。
  - `agents/scripted/`：从 `gitcg_expert_system` 移植的脚本卡组规则（`ExpertRuleAgent` + `deck_rules/` + assets/profiles/registry/codec）。**实测可跑完整对局**。
  - `envs/`：`PolicyPlayer`（把 `gitcg.Player` 五类请求接到 `Policy` 抽象，观测只来自 `notification.state`）+ `run_match`（同步跑完整局）+ `expert_policy`/baseline policy + 牌组装配。
  - `eval/`：`opponents`（20 脚本 + 基线对手池）+ `arena`（contestant×pool×paired-seeds 矩阵 + Wilson CI + Elo + 落盘）+ `ladder` + `stats`。
  - `envs/rollout.py` + `rollout_worker.py`：多进程 rollout（**JSON-lines 子进程，不用 multiprocessing**）+ `benchmark.py`（L0 吞吐基线）。
  - `tests/`：信息隐藏（`public_state` 掩码）、action adapter 覆盖/载荷、同种子重放一致、冒烟对局、stats/ladder/arena；`python -m pytest` 全绿（`-m "not slow"` 跳过慢测）。
  - 实测基线见 `reports/LOCAL_BASELINE.md`（单核 ~819 局/h，M0"1 万局/小时"≈13 核）。
- **多进程坑（重要）**：`multiprocessing` 的 `fork` 会在 `gitcg` 的 C/JS 运行时上**死锁**；`spawn` 又会重导入 `__main__`（pytest/`-m` 下崩）。因此 rollout 走自建子进程协议（`python -m envs.rollout_worker`，stdin/stdout JSON-lines，`cwd=仓库根`+`PYTHONPATH`）。不要改回 ProcessPoolExecutor。
- **对手数是 20 不是 22**：`gitcg_expert_system/deck_rules/` 实际注册 20 套（`registry.RAW_DECKS`）。PLAN §4 措辞"22 套"来自更早版本；以代码为准，后续如需补齐再补。
- **重放确定性坑（重要）**：引擎牌堆洗牌用 JS `Math.random()`，**不受** `ATTR_STATE_CONFIG_RANDOM_SEED` 控制（`packages/core/src/utils.ts:shuffle` 注释自曝）。因此 `envs/match.py` 默认在 Python 侧按种子**预洗牌**并传 `NO_SHUFFLE=1`，骰子/摸牌走引擎种子 RNG → 同种子可完全复现。详见 `NOTICE.md`。DISABLED 之前不要删这个 workaround。
- `train/`/`coach/`/`decklab/`/`orchestrator/` 仍为 README 占位（目标态命令未实现）。`eval/arena --smoke`、`envs/benchmark`、`pytest` 已可用（见上）。
- `refs/` 是三个上游参考仓，**只读**（只抄代码思路 + 逐模块移植，不修改、不在其上 apply 补丁，见 PLAN.md §6.5）：
  - `refs/genius-invokation` — 引擎本体 + Python 绑定 `packages/pybinding`（即 `gitcg`）+ IO 协议 `docs/development/io.md`（含 `notification.state` 的信息隐藏语义）
  - `refs/Rebel_base_RL` — SoG 蓝本 `research/world_model/src/gitcg_world_model/`；22 套脚本专家 `research/world_model/src/gitcg_expert_system/deck_rules/`（评测假想敌）；`packages/pybinding/examples/agent_vs_agent.py` 是 Player 子类的参考写法
  - `refs/AI` — DouZero 式 DMC 基线参考
- `refs/` 在 .gitignore 里（不入库，缺了重新 clone）。
- Git remote：`origin` = GitHub（`https://github.com/Peruere1828/genshin-GITCG.git`）；`njugit` = 南大 GitLab（`git@git.nju.edu.cn:Fmyh1828/genshin-GITCG.git`，SSH）。

## 环境

- conda env：**`gitcg`**（Python 3.12，已激活）；`gitcg` 0.21.0 已从 PyPI 装入并冒烟通过（完整对局可跑到 `GameStatus.FINISHED`）。
- 注意：pip 依赖（cffi/protobuf 等）部分解析自 `~/.local/lib/python3.12/site-packages`（user site），并非 conda env 内；排查依赖版本时留意这一点。
- 根目录 `.env` 含 DEEPSEEK API key（已 gitignore）。密钥永不入库、不出现在日志/报告。

## gitcg API 陷阱（0.21.0 实测）

- protobuf 字段名单复数陷阱：`ActionRequest.action` 是候选动作列表（字段名就叫 `action`），不是 `actions`；响应是 `ActionResponse(chosen_action_index=..., used_dice=[...])`，不是 `index`。
- `Game.status()` 是方法不是属性；主循环是 `game.start()` → `while game.is_running(): game.step()`。
- 回调里抛异常不会正常报 Python traceback，而是被 cffi 吞掉后引擎报 "Cannot destructure property 'value'" 之类的诡异错误并判 IO 失败——回调内出错先怀疑自己的 handler 返回值。
- `State.query(...)` 查询语法在不同 gitcg 版本间不兼容：`refs/Rebel_base_RL` 示例里的 `"my pile cards limit 5"` 在 0.21.0 上直接断言失败，用前先小样本验证。
- **观测只能来自 `notification.state`**（对手手牌/骰子已抹除、牌堆隐藏）；禁止用全量 `GameState` 造观测，信息泄漏测试是 CI 硬门禁（PLAN.md §WS1/§7）。

## 硬性工程约束（PLAN.md 推导，违反即返工）

- 强度验收口径 = **纯策略模式**（LLM 不参与）；LLM 辅助模式分开报分（§6.3）。
- `reps/` 动作抽象是 go/no-go 硬前置：抽象动作须 100% 覆盖合法具体动作 + adapter 往返测试（§6.2）。
- 版本冻结（D4）：每个训练 run 复制 `configs/engine.lock.example` 为 `engine.lock` 并填 commit/卡池版本，随 run 元数据保存；卡池升级 = 显式迁移开新 run。
- 无常驻服务、一切长任务可断点续跑（D7）；本机只放代码与小体积报告。
- 训练产物/数据不入库：`data/`、`*.jsonl`、checkpoint 已 gitignore，只提交 schema/清单；`reports/` 放小体积评测与复盘产物。
- 许可 AGPL-3.0：二次开发含网络服务也须开源；不携带米哈游卡面/素材。

## 测试与流程

- `tests/` 固定三类进 CI：信息泄漏（两份只差对手隐藏信息的 state，编码输出必须一致）、action_adapter 往返、同种子重放一致（见 `tests/README.md`）。目前无 lint/typecheck/CI 配置，新增代码先保证这三类可测。
- WS5 护栏（自动改码前必读，PLAN.md §WS5）：Agent 改动只进隔离分支；合并门禁 = 单测 + 泄漏测试 + 冒烟 + 评测不劣于；checkpoint/数据/配置只追加不覆盖。
- 重大决策（改主线、改验收口径）须在 PLAN.md §0 追加决策记录，不改写历史。
