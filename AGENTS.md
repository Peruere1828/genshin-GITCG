# AGENTS.md

需求/架构/验收的唯一权威是 [PLAN.md](./PLAN.md)（决策记录 §0、里程碑 M0–M6 §4、本地先行 L0–L4 §4.1）。

## 现状

- WS0/WS1 + 本地先行 L0–L4 已有可运行代码（2026-10-07–09，明细见 git log）：
  - `reps/`：从 `refs/Rebel_base_RL/.../gitcg_world_model` 逐模块移植的表示层（schema/snapshot/action_hierarchy/action_adapter/public_state/semantic_priors/agents/replay + 对应 toml），导入路径 `reps.*`；来源与许可见 `NOTICE.md`。
  - `agents/scripted/`：从 `gitcg_expert_system` 移植的 20 套脚本卡组规则，可跑完整对局。
  - `envs/`：`PolicyPlayer`（`gitcg.Player` 五类请求 → `Policy` 抽象）+ `run_match` + expert/baseline policy + 牌组装配 + `rollout`（多进程）+ `snapshot`（快照/分叉工具）。
  - `eval/`：`opponents`（20 脚本 + 基线池）+ `arena`（矩阵 + Wilson CI + Elo + 流式落盘/断点续跑）+ `ladder` + `stats`。
  - `common/llm.py`：OpenAI 兼容 LLM 客户端（仅标准库），降级链 local→API→纯策略；`agents/llm_assist.py`（局内辅助）、`coach/replay.py`（复盘教练）、`decklab/sensitivity.py`（换卡敏感性）。
  - `train/`：CVPN + 采集/训练/评测管线 + `replay_branch.py`（replay-branch MC teacher）+ `resolver.py`；`agents/neural.py` 接回 env。
  - `tests/`：信息泄漏、adapter 覆盖、重放一致、引擎桥/分叉、arena、LLM、train 全绿（`python -m pytest`；`-m "not slow"` 跳慢测）。
  - 实测基线见 `reports/LOCAL_BASELINE.md`（单核 ~819 局/h；12 核并行 ~4,900 局/h）。
- `orchestrator/`（WS5）仍为占位，未实现。
- `refs/` 三个上游参考仓**只读**，只抄思路逐模块移植、不修改、不 apply 补丁（PLAN §6.5），不入库、缺了重新 clone：
  - `refs/genius-invokation` — 引擎 + Python 绑定 `packages/pybinding`（即 `gitcg`）+ IO 协议 `docs/development/io.md`；
  - `refs/Rebel_base_RL` — SoG 蓝本 `gitcg_world_model/`、20 套脚本专家 `gitcg_expert_system/deck_rules/`、`agent_vs_agent.py` 示例；
  - `refs/AI` — DouZero 式 DMC 基线参考。
- Git remote：`origin` = GitHub（`https://github.com/Peruere1828/genshin-GITCG.git`）；`njugit` = 南大 GitLab（`git@git.nju.edu.cn:Fmyh1828/genshin-GITCG.git`，SSH）。

## 环境

- 用 conda env **`gitcg`**（Python 3.12）运行一切命令；`gitcg` 0.21.0 装自 PyPI。
- 排查依赖版本时注意：cffi/protobuf 等部分解析自 user site（`~/.local/lib/python3.12/site-packages`）。
- API key 从根目录 `.env` 读取（已 gitignore）；密钥永不入库、不进日志/报告。

## gitcg 0.21.0 用法

- 取候选动作用 `ActionRequest.action`（候选列表）；回 `ActionResponse(chosen_action_index=..., used_dice=[...])`。
- 用 `game.status()`（方法）；主循环 `game.start()` → `while game.is_running(): game.step()`。
- 回调 handler 只返回合法响应对象；对局异常先查 handler 返回值。
- `State.query(...)` 查询语法跨版本不兼容，用前小样本验证。
- 观测只取 `notification.state`；禁止用全量 `GameState` 造观测（信息泄漏测试是 CI 硬门禁）。

## 工程约定

- rollout 走自建子进程协议（`python -m envs.rollout_worker`，stdin/stdout JSON-lines，`cwd=仓库根`+`PYTHONPATH`）；不用 multiprocessing。
- 长任务结果流式落盘 + 按 `task_index` 断点续跑（arena 已实现）；长作业挂 tmux/setsid 跑。
- 对局确定性：`envs/match.py` 在 Python 侧按种子预洗牌并传 `NO_SHUFFLE=1`；每局开始调用 `reset_default_hierarchical_action_codebook()`。
- 训练标签/embedding 用语义 key（`low_level_semantic_key_for_code`），不用裸 `action_code`（仅单局内稳定）。
- 搜索分叉：只在 `Game.is_resumable()` 边界点取快照，fork 用 `envs/snapshot.fork_game(game_attrs=...)` 补游戏级 attrs；批量 teacher 用 `train/replay_branch.py`。
- 局内 LLM 辅助用 `deepseek-chat`；深度复盘用推理模型 + 大 max_tokens。
- 评测池口径以 `eval.opponents` 注册为准（当前 20 套，PLAN D11）。
- `reports/` 只提交聚合摘要；逐事件/逐干预明细放 `data/` 或走 gitignore。

## 硬性工程约束（PLAN.md 推导，违反即返工）

- 强度验收口径 = **纯策略模式**（LLM 不参与）；LLM 辅助模式分开报分（§6.3）。
- `reps/` 动作抽象是 go/no-go 硬前置：抽象动作须 100% 覆盖合法具体动作 + adapter 往返测试（§6.2）。
- 版本冻结（D4）：每个训练 run 复制 `configs/engine.lock.example` 为 `engine.lock` 并填 commit/卡池版本，随 run 元数据保存；卡池升级 = 显式迁移开新 run。
- 无常驻服务、一切长任务可断点续跑（D7）；本机只放代码与小体积报告。
- 训练产物/数据不入库：`data/`、`*.jsonl`、checkpoint 已 gitignore，只提交 schema/清单。
- 许可 AGPL-3.0：二次开发含网络服务也须开源；不携带米哈游卡面/素材。

## 测试与流程

- 三类测试进 CI 硬门禁：信息泄漏（两份只差对手隐藏信息的 state，编码输出必须一致）、action_adapter 往返、同种子重放一致（见 `tests/README.md`）。
- WS5 护栏（PLAN §WS5）：Agent 改动只进隔离分支；合并门禁 = 单测 + 泄漏测试 + 冒烟 + 评测不劣于；checkpoint/数据/配置只追加不覆盖。
- 重大决策（改主线、改验收口径）在 PLAN.md §0 追加决策记录，不改写历史。
