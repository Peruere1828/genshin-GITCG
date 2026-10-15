# AGENTS.md

需求/架构/验收的唯一权威是 [PLAN.md](./PLAN.md)（决策记录 §0、里程碑 M0–M6 §4、本地先行 L0–L5 §4.1/§4.2、无 GPU 中间验收 M3-CPU §4.3、NPC PvE 门禁 PvE-v1 §4.4）。

## 现状

- WS0/WS1 + 本地先行 L0–L5.6 已有可运行代码（2026-10-07–10，明细见 git log）：
  - `reps/`：从 `refs/Rebel_base_RL/.../gitcg_world_model` 逐模块移植的表示层（schema/snapshot/action_hierarchy/action_adapter/public_state/belief/belief_groups/semantic_priors/agents/replay + 对应 toml），导入路径 `reps.*`；来源与许可见 `NOTICE.md`。`reps/belief.py`（M3 S-A）从掩码公共态构 `PublicBeliefState` 并采样确定化根（`reconstruct_state_json` 改写边界快照隐藏区、隐藏实体 id 合成），供 `envs/fork_search.py` 的 `belief_root` 使用。
  - `agents/scripted/`：从 `gitcg_expert_system` 移植的 20 套脚本卡组规则，可跑完整对局。
  - `envs/`：`PolicyPlayer`（`gitcg.Player` 五类请求 → `Policy` 抽象）+ `run_match` + expert/baseline policy + 牌组装配 + `rollout`（多进程）+ `snapshot`（快照/分叉工具）+ `fork_bridge`/`fork_worker`（L5.4 子进程 fork 桥，JSON-lines）+ `fork_search`（实用版 continual-resolving：`ForkSearchPolicy` + `run_search_match`）。
  - `eval/`：`opponents`（20 脚本 + 基线池）+ `arena`（矩阵 + Wilson CI + Elo + 流式落盘/断点续跑）+ `ladder` + `stats`。
  - `common/llm.py`：OpenAI 兼容 LLM 客户端（仅标准库），降级链 API→纯策略；`agents/llm_assist.py`（局内辅助）、`coach/replay.py`（复盘教练）、`decklab/sensitivity.py`（换卡敏感性）。
  - `scripts/run_llm_matrix.py`：L5.2 全对手池「纯策略 vs LLM 辅助」矩阵（分开报分，逐局流式落盘+断点续跑+spec 指纹护栏）。
  - `scripts/run_search_agent.py`：L5.4 fork 搜索 resolver 对池评测（配对种子 search vs expert 消融，流式落盘+指纹护栏+断点续跑）。
  - `train/`：CVPN + 采集/训练/评测管线 + `collect.py`/`collect_worker.py`（集群可续跑分片采集，CPU 数据平面）+ `replay_branch.py`（replay-branch MC teacher）+ `learning_curve.py`（L5.3 数据量×网络规模扫描）+ `resolver.py`；`agents/neural.py` 接回 env。
  - `tests/`：信息泄漏、adapter 覆盖、重放一致、引擎桥/分叉、arena、LLM、train 全绿（`python -m pytest`；`-m "not slow"` 跳慢测）。
  - 实测基线见 `reports/LOCAL_BASELINE.md`（单核 ~819 局/h；12 核并行 ~4,900 局/h）。
- 算力口径（D14）：默认**本机 12 核 + WSL（4 核/8G/MX550 2G，试小网络训练）**；A800 80G + 128 核为申请制机会资源（里程碑不依赖它，窗口内一键零看管放量）；南大集群暂缓（glibc/环境适配成本高），`scripts/cluster/` 留作将来适配起点。
- 当前规划（D16–D20，2026-10-12 **三线并行**，明细与排产见 PLAN §3/§4.4/§10）：
  - **模型主线**（D15，验收 M3-CPU，PLAN §4.3）：按 `train/CONTINUAL_RESOLVING.md` 差距清单推进——信念根化（G1+G3，S-A ✅）→ 外采 MCCFR 脚手架（G2+G4，S-B ✅ `train/cfr_solver.py`）→ search-as-teacher + belief 头（G6+G7，S-C ✅ `envs/cfr_search.py` + `train/belief_head.py` + `train/distill.py` + `train.collect --search`）。**A3/A4 判定（D22/D23）：M3-CPU 未过（~0.50）、搜索 teacher 无增益且显著有害（全池 Δ=−0.066/−0.114）→ 改排产＝数据先行**（BC/自博弈规模化 + 更大网络顶过专家基线 0.561；搜索重开需价值头校准 + 非任意候选裁剪 + 门控式覆盖）。训练先 CPU 档（d_model 64–128、1e3–1e4 样本、fp32）。
  - **NPC PvE 调试系统**（D17，验收 PvE-v1，PLAN §4.4）：定位＝**调试/训练辅助**；**引擎 overlay 移植**敌人 `.gts`（蒙德/璃月全量，约 13 类 + 16 关）＋重建 `libgitcg.so`；`NpcPolicy`（greedy+rotation）＋ `NpcSpec`/三星判定；门禁＝decklab 组牌 × 16 关 × 多种子，三星 ≥ 60%。已有调研/数据：`reports/explore/npc_*.md`、`configs/npc/`、`configs/pools/`。
  - **Web 人机调试接口**（D18）：引擎 `standalone` overlay，流程＝人选牌 → **AI 看对手牌反选** → 随机先手；人可持 NPC 敌人卡组；定位＝**对 AI 调试接口**（不进口径验收）；素材运行时按 id 拉取、不入库。**C0 ✅**：standalone overlay 可本地起局 + **浏览器引擎 ↔ Python HTTP 桥跑通整局**（`web/ai_rpc_server.py`，`engine-overlay/scripts/standalone.sh`；static-data 走可达 v5 镜像）。**C2 ✅**：模型桥——`--policy` 把五类请求路由到 `envs.policy.build_policy`（`neural:<ckpt>[#t=]` / `heuristic` / `legal_random` / `expert:<slug>`），观测仍只取 `notification.state`（C10），失败回退占位策略；`tests/test_web_rpc_server.py`。**T1/T2/T3/C4 ✅**（2026-10-15）：桥按 `(session, who)` 隔离掩码状态与策略实例（自对弈/多开不串局）、`/notify` 带 `seq` 护栏 + 浏览器 notify 队列（决策不吃陈旧状态）、`--ai-deck`+`/ai-deck` 牌组对齐（开局 `/session` 校验告警）、逐决策 JSONL 落盘 `data/web/<时间戳>_<session>/`（`session.json`/`decisions.jsonl`/`summary.json`/`game_log.json`，端点 `/session` `/game-end` `/game-log` `/results`，协议见 `web/README.md`）。下一步：C1 选牌反选 / C3 NPC 卡组 / 随机先手。
  - **引擎构建策略**（D20）：引擎 pin `v0.21.0`，V8 11.6.189.4 纯 CMake 直编（无 depot_tools），在 **Docker 内一次性首建**、改敌人数据只做**增量**（rolldown + 重链，分钟级）；`node 24 直接用`；失败 fallback＝Node 子进程 + TS core + `custom-data-loader`。overlay 源放 `engine-overlay/`（入库）、工作区放 `build/`（gitignore）。
  - **机会算力**（D21）：A800 窗口**只有 GPU 可用、CPU 配额仅 ~10 核** → **采集/评测放本机/WSL**（不放窗口），A800 只做 **GPU 训练/推理**；数据本地采集后传入、checkpoint 回流本地认证（PLAN §5.1）。
- `orchestrator/`（WS5）仍为占位，未实现。
- `refs/` 三个上游参考仓**只读**，只抄思路逐模块移植、不修改、不 apply 补丁（PLAN §6.5），不入库、缺了重新 clone：
  - `refs/genius-invokation` — 引擎 + Python 绑定 `packages/pybinding`（即 `gitcg`）+ IO 协议 `docs/development/io.md`；
  - `refs/Rebel_base_RL` — SoG 蓝本 `gitcg_world_model/`、20 套脚本专家 `gitcg_expert_system/deck_rules/`、`agent_vs_agent.py` 示例；
  - `refs/AI` — DouZero 式 DMC 基线参考。
- Git remote：`origin` = GitHub（`git@github.com:Peruere1828/genshin-GITCG.git`，SSH，**主远程**：代码提交/分支都走这里）；`njugit` = 南大 GitLab（`git@git.nju.edu.cn:Fmyh1828/genshin-GITCG.git`，SSH，**仅镜像 origin 的 master**，供集群离线取代码；另有 `vendor` 分支存离线依赖 gitcg wheel + 资产缓存）。

## 环境

- 用 conda env **`gitcg`**（Python 3.12）运行一切命令；`gitcg` 0.21.0 装自 PyPI。
- 依赖统一装在 conda env `gitcg`；核对版本时看实际解析路径（cffi/protobuf 可能来自 user site `~/.local/lib/python3.12/site-packages`）。
- API key 从根目录 `.env` 读取（已 gitignore）；密钥永不入库、不进日志/报告。

## gitcg 0.21.0 用法

- 取候选动作用 `ActionRequest.action`（候选列表）；回 `ActionResponse(chosen_action_index=..., used_dice=[...])`。
- 用 `game.status()`（方法）；主循环 `game.start()` → `while game.is_running(): game.step()`。
- 回调 handler 只返回合法响应对象；对局异常先查 handler 返回值。
- `State.query(...)` 查询语法跨版本不兼容，用前小样本验证。
- 观测只取 `notification.state`；禁止用全量 `GameState` 造观测（信息泄漏测试是 CI 硬门禁）。

## 工程约定

- 并发统一走自建子进程协议（`python -m envs.rollout_worker` / `envs.fork_worker`，stdin/stdout JSON-lines，`cwd=仓库根`+`PYTHONPATH`）；父进程并发排空子进程 stdout 再发下一批。
- 长任务结果流式落盘 + 按 `task_index` 断点续跑（arena 已实现）；长作业挂 tmux/setsid 跑。
- 集群数据平面（CPU 采集 → 存储 → GPU 训练）：`python -m train.collect`（可续跑、一行一局、manifest 指纹护栏）写 `data/replays/<tag>/`；采集可加 `--format compact [--no-privileged-state]`（fp16 base64，体积 0.54×/0.45×、编解码 2.6–3×，D23 数据先行 + A6；默认 `jsonl` 不变、读入透明兼容）。训练/学习曲线用 `--replays <dir...>` 读盘。LSF 模板见 `scripts/cluster/lsf/`，海量作业用 `collect_array.lsf`（每数组元素独立分片目录）；集群当前暂缓（D14，启用条件见 `scripts/cluster/README.md`）。
- 对局确定性：`envs/match.py` 在 Python 侧按种子预洗牌并传 `NO_SHUFFLE=1`；每局开始调用 `reset_default_hierarchical_action_codebook()`。
- 训练标签/embedding 用语义 key（`low_level_semantic_key_for_code`），不用裸 `action_code`（仅单局内稳定）。
- policy spec 统一走 `envs.policy.build_policy`：`expert:<slug>` / `neural:<ckpt>[#t=<temp>]` / `heuristic` / `legal_random` / `random`；checkpoint 自带 encoder 配置，多进程 rollout/采集可直接用训练网络。
- 搜索分叉：只在 `Game.is_resumable()` 边界点取快照，fork 用 `envs/snapshot.fork_game(game_attrs=...)` 补游戏级 attrs；批量 teacher 用 `train/replay_branch.py`。**在线搜索用 `envs/fork_bridge.py`（fork 一律跑在子进程，`workers>=1`）**，`run_search_match` 拒 `workers==0`。
- 局内 LLM 辅助用 `deepseek-chat`；深度复盘用推理模型 + 大 max_tokens。
- 评测池口径以 `eval.opponents` 注册为准（当前 20 套，PLAN D11）。
- 引擎 overlay（D20）：`refs/` 只读，我们的敌人 `.gts`／standalone 改动放 `engine-overlay/`（入库），构建时复制进 `build/engine/`（gitignore）；**V8 在 Docker 内构建**，`libgitcg.so` 只建一台（本机/WSL 通用）；开发期用 `custom-data-loader` 免构建验证敌人数据，定稿后增量重链。
- `reports/` 只提交聚合摘要；逐事件/逐干预明细放 `data/` 或走 gitignore。

## 硬性工程约束（PLAN.md 推导，违反即返工）

- 强度验收口径 = **纯策略模式**（LLM 不参与）；LLM 辅助模式分开报分（§6.3）。
- `reps/` 动作抽象是 go/no-go 硬前置：抽象动作须 100% 覆盖合法具体动作 + adapter 往返测试（§6.2）。
- 版本冻结（D4）：每个训练 run 复制 `configs/engine.lock.example` 为 `engine.lock` 并填 commit/卡池版本，随 run 元数据保存；卡池升级 = 显式迁移开新 run。
- 无常驻服务、一切长任务可断点续跑（D14）；本机只放代码与小体积报告。
- 训练产物/数据不入库：`data/`、`*.jsonl`、checkpoint 已 gitignore，只提交 schema/清单。
- 许可 AGPL-3.0：二次开发含网络服务也须开源；不携带米哈游卡面/素材。

## 测试与流程

- 三类测试进 CI 硬门禁：信息泄漏（两份只差对手隐藏信息的 state，编码输出必须一致）、action_adapter 往返、同种子重放一致（见 `tests/README.md`）。
- WS5 护栏（PLAN §WS5）：Agent 改动只进隔离分支；合并门禁 = 单测 + 泄漏测试 + 冒烟 + 评测不劣于；checkpoint/数据/配置只追加不覆盖。
- 重大决策（改主线、改验收口径）在 PLAN.md §0 追加决策记录，不改写历史。

## 文档写作规范

- 仅保留最新情况，删除过时/修正/踩坑描述
