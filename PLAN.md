# 七圣召唤 AI + Agent 工程 — 需求与实现路径（PLAN.md）

> 状态：v1.5（2026-10-12：**三线并行立项**——模型主线（D15）＋ NPC PvE 调试系统（D17，§3 WS7/§4.4）＋ Web 人机调试接口（D18，§3 WS8）；V8 采用 **Docker 内构建**策略（D20）；上一版 v1.4 的算力口径见 D14、无 GPU 主线见 D15、工程约定见 §0.1、本地先行 L0–L5.6 已完成见 §4.2）
> 决策记录见 §0；里程碑按阶段推进、不绑定日期。

---

## 0. 已确认的需求决策

| # | 议题 | 决策 |
|---|------|------|
| D1 | 第一优先目标 | **打赢强规则/脚本对手**。以 `gitcg_expert_system/deck_rules` 的脚本卡组为假想敌（数量口径以 D11 为准），胜率台阶是第一验收标准 |
| D2 | Agent 范围 | 全都要：**对局决策 Agent**（局内分析局面并决策）+ **训练自迭代闭环** + **元环境/卡组构筑 Agent** + **复盘/教练 Agent** + **工程自改进** |
| D3 | 训练主线 | **直接上 SoG/ReBeL 路线**（public belief + continual resolving + 搜索监督蒸馏），以 `Rebel_base_RL/research/world_model` 为架构蓝本。DMC/BC 只作冷启动与 sanity 基线，不作主线 |
| D4 | 卡池/版本 | **跟随最新卡池**。接受追版本成本，用"每次训练冻结引擎 commit + 卡池版本"控制不可复现性 |
| D5 | LLM 使用 | **局内局外都用**。局内做局面解读/候选生成/校验（评测需区分"纯策略"与"LLM 辅助"两种模式）；局外做复盘、调度、卡组思考、代码改进 |
| D6 | LLM 资源 | **混合**：本地开源模型为主（部署位置见 D14），关键步骤（深度复盘、卡组思考）可选商业 API |
| D7 | 算力（已被 D14 修订） | **南大超算为主**（公共共享 CPU 队列跑自博弈/搜索，GPU 队列跑训练/推理），**A800 80G 为间隙弹性资源**（科研任务空窗期用，主要承载本地大 LLM）；工程按"无常驻服务、随时可断点迁移"设计。2026-10-10 起以 D14 为准 |
| D8 | 时间粒度 | 按里程碑（M0–M6），不排日期 |
| D9 | 工程目录 | `~/projects/genshin-GITCG`（本机），代码 git 同步；数据/作业落本地盘或机会算力存储（D14） |
| D10 | 本地先行 | 在批量算力（D14 机会资源）到位前，先在**本地 12 核 CPU + LLM API**上推进"本地先行轨道"（§4.1 L0–L4）：环境/评测/表示层 + LLM Agent 层 + SoG 管线玩具验证；大规模训练留给机会算力 |
| D11 | 验收对手口径（修订 D1/§4 的"22 套"表述） | 验收表述改为"**评测池全部已注册脚本对手**"（池版本化，当前 20 套，见 §0.1 C2）；M0/M2/M4 的数值门槛（≥50%/≥70%/无单套 <45%）不变；新增卡组属**评测池扩容**，不阻塞验收；补齐至 22 套为可选工作项（缺 deck share code） |
| D12 | 搜索分叉路线 | **薄 fork 桥**：引擎 `canResume:true` 边界暂停点的快照即精确分叉，经子进程 fork 桥（JSON-lines、无常驻服务，D7/D14）暴露给搜索（契约见 D13）；验收含"**分支续跑确定性**"测试（同种子分叉续跑 vs 不中断跑终局一致）。批量离线标注用 replay-branch MC（同种子重放+注入动作）当 teacher，**主线不因此切 DMC** |
| D13 | 搜索分叉契约（现行） | 分叉三条纪律：① 只在 `Game.is_resumable()` 边界点取快照（`canResume:true` 13/13 精确复现活体终局，证据 `scripts/probe_boundary_fork.py` / `reports/engine/probe_boundary_fork_*.json`）；② fork 显式补齐游戏级 attrs（`ATTR_PLAYER_ALWAYS_OMNI_*` 等不在 state 序列化内，用 `envs/snapshot.fork_game(game_attrs=...)`）；③ fork 一律在子进程运行（`envs/fork_bridge.py`，`run_search_match` 拒 `workers==0`）。由此搜索直接用 pybinding 分叉（毫秒级 fork + 注入候选 + rollout），无需引擎改造；replay-branch MC（`train/replay_branch.py`）保留为批量离线 teacher 与对照。已知边界：`canResume:false` 的 phase 内部暂停点不分叉（搜索不需要在那里分叉）；`Promise.all` 并发 RPC 交错按引擎版本冻结保证（D4） |
| D14 | 算力口径修订（修订 D7/§5，2026-10-10） | **默认算力 = 本机 12 核 + WSL（4 核 / 8G / MX550 2G）**：本机跑 rollout/采集/评测/CPU 档训练；WSL 当并行采集/评测节点，顺带做小网络（d_model ≤ 64、fp32 小 batch）训练试验。**A800 80G + 128 核为申请制机会资源**（时间受限、需申请）：任何里程碑不依赖它，机会窗口按"训练 > 评测 > 采集 > LLM 分析"排优先级、一键零看管跑完。**南大集群暂缓**（glibc/环境适配成本高），`scripts/cluster/` 保留为将来机会算力的适配起点。工程形态不变：无常驻服务、一切长任务断点续跑、CPU 采集与 GPU 训练解耦（§5） |
| D15 | 无 GPU 期间主线（修订 M3 排期） | **严格按 `train/CONTINUAL_RESOLVING.md` 差距清单推进**：信念根化（G1+G3）→ 外采 MCCFR 脚手架（G2+G4）→ search-as-teacher + belief 头（G6+G7）→ query solver / 批推理 / BR 探针（G8–G10）；先在 CPU 档（d_model 64–128、1e3–1e4 样本）跑通语义，规模随算力放大。M3/M4 数值门槛不变，新增 CPU 中间验收 **M3-CPU**（§4.3）作为 GPU 放大前的 go/no-go |
| D16 | 三方向并行立项（2026-10-12） | 在模型主线外新增两条并行线：**NPC PvE 调试系统**（D17）与 **Web 人机调试接口**（D18）；三线并行推进（模型先出）、模块解耦、各自独立验收。模型主线维持 D15、算力维持 D14 不变；两条新线复用 WS0 引擎与 WS2 评测，不改变 WS1/WS3/WS4 主线 |
| D17 | NPC PvE 调试系统 | 定位＝**调试/训练辅助**（确定性假想敌池、课程加权、回归基线），不追求复刻游戏内体验；实现＝**引擎 overlay 移植**（敌人 `.gts` + 重建 `libgitcg.so`，refs 只读）；范围＝**蒙德/璃月全量**（约 13 类敌人 + 16 关规格）；验收＝**独立 PvE 门禁点**（§4.4）：decklab 自动组牌 × 16 关 × 多种子，**三星达成率 ≥ 60%**（挑战目标 3 条全达成计三星）；报表拆分组牌质量 vs 打牌质量 |
| D18 | Web 人机调试接口 | **直接上引擎 standalone overlay**（不自建轻量 web）；流程＝人选牌 → **AI 看对手牌后反选** → 随机先手；人可持 NPC 敌人卡组；定位＝**对 AI 的调试接口**（明示信息流程，**不进口径验收**）；模型端复用 `envs.policy.build_policy` 全 spec，观测走 `notification.state`（与训练同源）；素材照 refs 做法：运行时按 id 拉取、不入库、缓存 gitignore、不随发布物分发 |
| D19 | 卡组口径 | NPC 评测用 **decklab 自动组牌**（池 C 内，闭环含组牌 Agent）；Web 对局＝人自选 + AI 反选（v1 用对位矩阵 + decklab 固定规则起步） |
| D20 | 引擎构建策略（V8） | **Docker 内构建**：镜像基线 `ubuntu:22.04`（与上游 CI 同 glibc 口径）；**引擎源码 pin 到与 PyPI wheel 一致的 tag `v0.21.0`**（`c94826b`；refs HEAD 更新但 ABI/构建方式不同，不采用）；V8 **一次性首建**、改敌人数据只做**增量**（rolldown 生成 `js_code.cpp` + 重链，分钟级，V8 不重编）；**只在一台构建**（linux x64 `.so` 本机/WSL 通用）；开发期用 `custom-data-loader` **免构建快循环**验证敌人数据；**node 24 直接用**（`engines` 仅警告，无 `engine-strict`）；构建 provenance 记入 `engine.lock.example` 变体；**V8 构建失败 fallback**＝Node 子进程 + TS core + `custom-data-loader` 驱动 NPC 环境（web 线不受影响）。**（2026-10-12 事实修正）**：pin 的 `v0.21.0` 用 `piovium/v8-cmake@14d0f727`——**vendor V8 11.6.189.4 源码、纯 CMake+Ninja 直编，无 depot_tools/`V8_WORKSPACE`/`V8_OFFLINE`、无数 GB 下载**；实测首建 **~20 min**（1232 TU）、增量 **~30 s** 且产物 sha256 确定。先前按 refs HEAD（`v8-cmake-barbara`/V8 14.8/volume 缓存）写的措辞作废；证据与流程见 `engine-overlay/README.md` |
| D21 | 机会算力 CPU 配额事实（修订 D14/§5，2026-10-12 实测） | A800 窗口**只有 GPU 可用**、**CPU 配额仅 ~10 核**（与本机 12 核同级）：**CPU 密集的采集/评测不放窗口**，只在本机/WSL 分片累积；A800 专做 **GPU 训练/推理**（CVPN 放大、蒸馏、batched inference、可选本地 LLM）。数据流：本地 `train.collect` → 传入 A800 → `train.pipeline --replays/--distill` GPU 训练 → checkpoint 回流本地 → `scripts/eval_protocol.py` 认证。§5.1 窗口优先级据此收敛为「训练 > LLM 分析」（采集/评测移出窗口）|
| D22 | M3-CPU 首次认证结果与下一步（2026-10-12，A3，`reports/train/a3_m3_cpu_screening.md`） | **M3-CPU 未通过认证**：pooled 0.460–0.500（400 局恰好压线），**strict-CI（cluster 下界 0.371）/ Wilson 下界 0.451 / M4 floor（9/20 <45%，最弱 `unyielding_geo` 0.05）全 FAIL**。**关键负面发现：搜索蒸馏 ≈ 冷启动 BC**（同数据配对 Δ=−0.0175，CI[−0.084,+0.049]，无正增益；当前 teacher 预算 iter4/depth1/cand2 太弱）。**结论：不满足"放大规模"前置**——直接丢 A800 只放大算力、不改 ~0.50 的结构瓶颈。**下一步**：先做 **teacher 预算阶梯 + 搜索智能体直接强度评测**（`scripts/run_cfr_search_agent.py`），判定"搜索本身能否显著强于 expert/BC"：能则按该预算放量蒸馏；不能则重估排产（先补 BC/自博弈数据、后搜索蒸馏）。认证级评测需 **≥50 seeds/对手（1000+ 局/臂，Wilson ±0.03）** + `--strict-ci` |
| D23 | 搜索路线判定与改排产（2026-10-12，A4，`reports/search/a4_search_budget_ladder.md`） | **搜索 teacher 无可测增益，且在可行预算下显著有害**：全池 n=400 配对 Δ：B 档(8/1/4/8)=**−0.066 [−0.125,−0.008]**、D 档(8/2/4/4)=**−0.114 [−0.169,−0.058]**（CI 全在 0 以下）；搜索覆盖 ~87–89% 专家动作，**win→loss 是 loss→win 的 1.5–2.0×**；d1 档 it8≡it16（逐局逐位相同）→ iterations 非瓶颈；根价值区分度 ≈0。**基线校准：脚本专家 `superconduct_aggro` 对全池 0.561 ＞ 我们的 BC/蒸馏网 ~0.50**（网络尚未追上专家）。**决策**：① **改排产＝数据先行**——先规模化采集 BC/自博弈数据 + 更大网络（GPU 窗口）把强度顶过专家基线；② **不把搜索蒸馏放量到 A800**（软目标劣于硬标签，放量只放大有害信号）；③ 重开搜索的前置＝**校准的公共信念价值头 + 非任意候选裁剪 + 门控式覆盖**（仅当根价值优势 > τ 才覆盖专家），三者达标后再跑阶梯 |
| D24 | 数据先行结论与下一杠杆（2026-10-12，A5/A7，`reports/train/bc_scale_datafirst.md`、`reports/datafirst_bc.md`） | **BC 撞到 teacher 天花板**：规模化 BC（SA 8,061 局/372k 样本 + 4 套 deck×20 池×80 种子 compact）**从未显著超过同 deck 专家**（最优 `double_geo_navia` 打平 +0.007；SA −0.061 显著负）；**真正的杠杆是训练时长/优化**（5→15ep Δ=+0.076 显著），**不是数据量**（8k→48k Δ=+0.023 不显著）。**deck 混淆极大**：同一策略在强 deck（dvalin/unyielding/double_geo）pooled **0.689–0.709（≥0.50 strict-CI PASS）**、弱 deck（SA 0.480 / natlan 0.140）→ **验收门禁必须固定 deck 集、逐 deck 报**。**M4 floor 结构性 FAIL**（长尾岩/雷/草系）。**下一杠杆＝策略改进（自博弈/RL 或修好的搜索），不是更多 BC 数据**。工程：`train.collect` 父进程内存 bug（攒全量样本→swap 死，A8 修）；compact 载荷 **0.65 MB/局**（A800 传输 48k 局 ≈ 31 GB 可装）。|

### 0.1 工程约定（现行做法，2026-10-10 正向化整理；证据指向脚本/测试）

| # | 应该做 | 依据/证据 |
|---|--------|-----------|
| C1 | 参考仓**逐模块移植**代码并自写测试（不整仓 apply 补丁）；移植范围/许可见 `NOTICE.md` | `reps/`、`agents/scripted/`（gitcg 0.21.0 实测可跑完整对局） |
| C2 | 验收口径写"**评测池全部已注册脚本对手**"，数量以 `eval.opponents` 注册为准（当前 20 套，D11） | `eval/opponents.py` |
| C3 | 对局确定性：Python 侧按种子预洗牌 + `NO_SHUFFLE=1`，每局开始 reset 动作 codebook；训练标签/选择记录用 **option 位置索引或语义 key**（low-level action code 仅单局内稳定） | `envs/match.py`、`tests/test_replay_consistency.py` |
| C4 | 并发统一走自建 **JSON-lines 子进程协议**（`python -m envs.rollout_worker` / `envs.fork_worker`）；父进程并发排空子进程 stdout 再发下一批 | `envs/rollout.py`、`tests/test_rollout_subprocess.py` |
| C5 | 搜索分叉**只在 `Game.is_resumable()` 边界取快照**，fork 用 `envs/snapshot.fork_game(game_attrs=...)` 补齐游戏级 attrs，且 fork 一律跑在子进程（`ForkBridge`，`workers>=1`） | `scripts/probe_boundary_fork.py`、`tests/test_fork_bridge.py` |
| C6 | 跨进程重放候选动作按**位置选项索引**（不裸用 action code） | `tests/test_fork_bridge.py`（逐字节复现活体终局） |
| C7 | LLM 选型：局内辅助用非推理低延迟模型（`deepseek-chat`），深度复盘用推理模型；客户端带降级链 local→API→纯策略 | `common/llm.py` |
| C8 | 长任务：逐行流式落盘 + 按 `task_index` 断点续跑 + spec 指纹护栏；作业脱离 harness（tmux/setsid）跑；验收先核对原始记录行数 | `eval/arena.py`、`train/collect.py`、`tests/test_arena.py`、`tests/test_collect.py` |
| C9 | 日志入库：`reports/` 只放聚合摘要，逐事件明细与原始 JSONL 留 `data/`（gitignore） | `.gitignore`、`reports/README.md` |
| C10 | 观测只取 `notification.state`（信息隐藏版）；信息泄漏测试是合并硬门禁 | `tests/test_info_leak.py` |
| C11 | 每个训练 run 复制 `configs/engine.lock.example` 为 `engine.lock`，随 run 元数据保存（D4） | `configs/engine.lock` |
| C12 | 批量动作价值标注用 **replay-branch MC**：从初始态重放到决策点 + 注入候选 + rollout 求均值 | `train/replay_branch.py`、`tests/test_engine_bridge.py` |
| C13 | 集群作业模板（`scripts/cluster/`）留作机会算力适配起点；启用前先解决 glibc ≥ 2.34 环境与离线依赖（vendor 分支） | `scripts/cluster/README.md`（D14） |

---

## 1. 目标与非目标

### 目标（按优先级）
1. **G1 强度**：训练出的策略（网络+搜索，可选 LLM 辅助）在固定 meta 上稳定击败评测池全部脚本专家策略（池版本化，当前 20 套，D11），并在自博弈阶梯上持续爬升。
2. **G2 自迭代**：形成"自博弈 → 评测 → 复盘归因 → 生成课程/数据/代码改动 → 再训练 → 门禁验收"的闭环，且 Agent 能主导其中大部分环节。
3. **G3 Agent 能力**：局内实时决策 Agent、局外复盘教练、卡组构筑顾问三件套可独立使用、可评测。
4. **G4 可复现**：任意一个 checkpoint 的训练数据、引擎版本、卡池版本、配置可追溯重放。

### 非目标（v1 明确不做）
- 不做天梯/真人平台自动对战接入（play.piovium.org 仅取公开复盘数据做 BC，遵守平台条款）；
- 不做卡面/素材渲染与**产品化** UI、不对外提供服务（D16 修订：改为做"**对 AI 调试接口**"的自用 web 人机界面，D18——引擎 standalone overlay，素材运行时按 id 拉取、不入库、不随发布物分发）；
- 不做跨游戏泛化、不追求论文发表（但工程按可发表标准留好实验记录）；
- 不做实时 15 秒限时的极致优化（引擎对局不设人类时限，后期再压延迟）。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  WS5 自迭代闭环 Orchestrator（LLM Agent 主导，带护栏门禁）          │
│  计划 → 实验 → 评测 → 复盘归因 → 课程/数据/补丁 → 门禁 → 合并/回滚   │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
┌────────────▼──────────────┐    ┌──────────────▼─────────────────┐
│  WS4 Agent 层（LLM 混合）   │    │  WS3 SoG/ReBeL 训练内核          │
│  · 局内决策 Agent           │◄──►│  · public belief（belief_groups）│
│    （候选生成/校验/兜底）    │    │  · continual_resolving 搜索      │
│  · 复盘/教练 Agent          │    │  · CVPN 策略价值网络              │
│  · 卡组构筑 Agent           │    │  · 搜索监督蒸馏（sog_pipeline）   │
└────────────┬──────────────┘    │  · batched_inference（GPU 批推理）│
             │                   └──────────────┬─────────────────┘
┌────────────▼──────────────────────────────────▼─────────────────┐
│  WS1 观测/动作表示层                                               │
│  observation_encoder（隐藏信息安全 token 化）+ action_hierarchy      │
│  + action_taxonomy.toml（动作抽象/分层，抑制组合爆炸）               │
└────────────┬────────────────────────────────────────────────────┘
┌────────────▼────────────────────────────────────────────────────┐
│  WS0 环境与工程基座：gitcg (pybinding) → Gym 式 Env / 多进程 rollout │
│  IO 协议 RpcRequest/RpcResponse + notification（已做信息隐藏）       │
└────────────┬────────────────────────────────────────────────────┘
┌────────────▼────────────────────────────────────────────────────┐
│  WS2 评测与基准：arena + 脚本对手池（20 套，可扩容） + TrueSkill/Elo 阶梯 + 复盘归因 │
└─────────────────────────────────────────────────────────────────┘
```

数据流（对局内）：`gitcg notification.state → observation_encoder → public belief → CVPN 先验 + continual resolving 搜索 →（可选 LLM 候选/校验）→ 动作 → 引擎`。
数据流（对局外）：`对局日志 → 复盘 Agent 归因 → 训练数据/课程/补丁 → 门禁评测 → 新 checkpoint`。

> **WS7 NPC PvE 调试系统（D17）** 与 **WS8 Web 人机调试界面（D18）** 为 2026-10-12 新增并行线（见 §3、§4.4）；
> 二者复用 WS0 引擎与 WS2 评测，不改变 WS1/WS3/WS4 主线（D16）。

---

## 3. 工作流分解

### WS0 环境与工程基座
- **交付**：`gitcg`（`genius-invokation/packages/pybinding`，pip 可装）包装成 Gym 式环境；多进程/多机 rollout runner；episode 日志格式（JSONL + 引擎 state 快照）。
- **接口基线**：按官方 `examples/agent_vs_agent.py` 写法实现 `Player` 子类（`on_action / on_reroll_dice / on_choose_active / on_select_card / on_switch_hands / on_notify`）；观测走 `docs/development/io.md` 的 `notification.state`（**已隐藏对手手牌/骰子/牌堆，禁止用全量 state 造观测**）。
- **可参考**：`Rebel_base_RL/research/world_model/src/gitcg_world_model/env.py`（环境封装）、`genius-invokation/packages/test/src/controller.ts`（TS 侧驱动思路）。
- **版本策略（D4 落地）**：`engine.lock` 记录 gitcg 的 git commit + 卡池数据版本；每次训练 run 的元数据里强制带上；升级卡池 = 新 run，旧 run 可用旧 lock 重放。

### WS1 观测/动作表示（决定上限的模块）
- **观测**：以 `observation_encoder.py` 为蓝本做 token 化观测（角色/装备/状态/骰子/手牌数/公开事件历史），显式建"公共信息 + 私有信息"两段式编码；写一个**信息泄漏测试**（用两份只差对手隐藏信息的 state 喂编码器，输出必须一致）。
- **动作**：照抄 `action_hierarchy.py + action_taxonomy.toml` 做动作分层/抽象（目标：把裸动作空间压到每步几十个可选抽象动作 + 参数头）；`action_adapter.py` 负责抽象动作 ↔ 引擎具体动作的双向映射，必须有合法化校验（非法动作回退到合法集）。
- **信念**：`belief_groups.py / belief_groups.toml` 作为 public belief 的起点（对手手牌/牌堆的粒子或分组表示），SoG 路线必需。

### WS2 评测与基准（先于训练建成）
- **对手池**：`gitcg_expert_system/deck_rules/` 评测池全部脚本策略（当前 20 套：超导、冻结、水皇、双岩、那维双岩、Skirk 系等；池版本化，D11）作为固定假想敌；每套都是一个固定 checkpoint。
- **指标**：分卡组胜率矩阵（池内对手 × 我方 N 套 meta 卡组）+ TrueSkill/Elo 阶梯（对固定基线 + 历史 checkpoint 双向）+ 平均回合数/决策延迟；**不指望算 exploitability**，用"对固定基线不劣于 + 阶梯上升"做工程门禁。
- **arena**：`run_expert_arena.py` 思路改造；支持并行批量、固定随机种子批、结果入库（SQLite/DuckDB）与曲线图。
- **复盘归因**：每局输出逐决策记录（观测、候选、搜索值、最终动作、LLM 是否干预、结果），供 WS4 教练 Agent 消费。

### WS3 SoG/ReBeL 训练内核（主线）
以 `research/world_model/src/gitcg_world_model/` 为蓝本，逐件替换/加固（原仓库是实验分支，**只当代码库抄，不当可运行基线**）：
1. `cvpn_model.py`：CVPN（策略+价值网络），几百万~几千万参数小 transformer/token 编码；deck-id 条件化输入。CPU 档先用 d_model 64–128 + fp32（本机/WSL 可训），算力放大后切 bf16 更大模型。
2. `continual_resolving.py + resolving_agent.py`：在线公共信念搜索（continual resolving），按 `train/CONTINUAL_RESOLVING.md` 差距清单演进（D15）：信念根化（公共信念 + 采样确定化）→ 外采 MCCFR 脚手架 → search-as-teacher。搜索分叉用 `canResume:true` 边界快照（`envs/snapshot.fork_game`，契约见 D13），从边界点 fork + 注入候选即可；批量标注用 replay-branch MC（`train/replay_branch.py`）。
3. `cvpn_training.py + sog_pipeline.py`：搜索产出作为 teacher → 蒸馏回网络 → 候选 vs 工作 checkpoint 门禁（不劣于才接受）的外环。
4. `batched_inference.py`：GPU 批推理服务（max_batch_size=128），CPU 侧多进程搜索/环境步进通过它批量打分——**GPU 只做推理/训练，CPU 做搜索**，与我们的算力结构匹配。
5. **冷启动**：`bootstrap_agent.py` + 脚本专家对局 + （可选）piovium/AI 的"LM 打分头初始化"思路做 BC 热启动；`piovium/AI` 的 DMC-Q 基线保留为 sanity 对照（若 SoG 管线长时间不涨，回退验证管线本身）。
- 文档基线：`docs/SOG_GT_CFR_CONTINUAL_RESOLVING_CHARTER.md`、`research/world_model/TRAINING_METHODS.md`、`ACTION_HIERARCHY_GUIDE.md`（注意文档里残留 Windows 路径 `E:\Coding\...`，以实际 repo 相对路径为准）。

### WS4 Agent 层（LLM 混合，D5/D6）
- **局内决策 Agent**（双模式，评测必须分开报分）：
  - *纯策略模式*：CVPN + 搜索直接出动作，LLM 不参与——这是 G1 的验收口径；
  - *LLM 辅助模式*：LLM 做局面解读、从搜索 top-k 候选中校验/重排、搜索不可用时兜底；受**延迟预算**约束（每步 ≤ 数秒），所有 LLM 干预记入对局日志用于后续蒸馏。本地开源模型（机会算力上部署 32B 级 bf16 或 LoRA，D14）为主，商业 API 仅离线用。
- **复盘/教练 Agent**：消费 WS2 的逐决策日志，产出（a）失误归因报告（哪一步搜索值/胜率掉点、是否动作抽象背锅）；（b）课程建议（针对薄弱卡组/局面类型加自博弈权重）；（c）BC 标签（"当时更优动作"）。输出是结构化 JSON，可直接进训练管线。
- **卡组构筑 Agent**：围绕 meta 卡组做构筑搜索（换卡敏感性分析、对池内全部假想敌的对位胜率模拟），v1 只在**固定模板内调 5–15 张 flex 位**，不做自由构筑；结论固化为新 deck-list 进评测池。
- **工程 Agent**：读写训练代码/配置、跑实验、汇总结果（见 WS5 护栏）。

### WS5 自迭代闭环（recursive self-improvement）
一个 run 的标准循环：
```
① 计划：Agent 根据上轮复盘产出本轮 hypothesis（改课程/改动作抽象/改网络/改搜索参数）
② 实验：在分支上落实改动 → 小规模冒烟（千局级）
③ 训练/自博弈：训练（机会算力窗口）+ 自博弈采集（本机/WSL 分片，机会算力放量）
④ 评测：对手池胜率矩阵 + TrueSkill 阶梯 + 回归集
⑤ 复盘：教练 Agent 归因 → 更新失败案例库
⑥ 门禁：通过 → 合并 + 发 checkpoint；不通过 → 保留记录、回滚
```
**护栏（工程自改进的安全边界，必须先于自动改码上线）**：
- Agent 改动只进隔离分支/容器，禁止直改主线；
- 合并门禁 = 单测 + 信息泄漏测试 + 冒烟 + 评测不劣于当前 checkpoint，四项全过才可合并；
- 每个自动改动带 hypothesis 与预期指标，评测后自动判定"证实/证伪"进知识库；
- 一切改动可 `git revert`，checkpoint/数据/配置三件套永不被 Agent 覆盖，只追加。

### WS6 数据
- 自博弈数据（主）：搜索 teacher 标注，按 replay 池管理（照 `cvpn_training.py` 的 bounded replay reuse）。
- 脚本专家数据（冷启动）：池内脚本策略对局 + 教练 Agent 修正标签。
- 公开复盘数据（可选热启动）：play.piovium.org 公开数据做 BC，**先读平台条款，注意人机混用限制**。
- 所有数据带 `engine commit + 卡池版本` 元数据（D4）。

### WS7 NPC PvE 调试系统（D17；定位＝调试/训练辅助）
- **交付**：引擎 overlay（敌人 `.gts` + 重建 `libgitcg.so`）＋ `NpcPolicy`（greedy + rotation，确定性）＋ `NpcSpec`（关卡规格/挑战目标/三星判定）＋ 评测入口。
- **实现口径（D20）**：`refs/` 只读，overlay 复制到 `build/engine/` 后加我们的文件；**开发期用 `custom-data-loader` 免构建快循环验证敌人数据，定稿后增量重建 `.so`**；V8 在 Docker 内一次性首建。
- **PvE 机制近似**：NPC 无手牌/无骰子/玩家恒先手 → 用「`NpcPolicy` 永不打出行动牌」近似（引擎只有全局 `initialHandsCount/initialDiceCount`，无 per-player 版本）。
- **用途**：确定性难度可控的假想敌池（跑回归、课程加权）；**三星达成率作独立门禁**（§4.4）。
- **已有数据**：`configs/npc/`（蒂玛乌斯示例，权威敌人数据）；`reports/explore/npc_pve_doc.md`、`npc_lineups_mond_liyue.md`（蒙德/璃月 16 NPC 阵容）。

### WS8 Web 人机调试界面（D18；定位＝对 AI 调试接口）
- **交付**：引擎 `standalone` overlay（人选牌 → AI 看对手牌反选 → 随机先手）＋ Python 模型 HTTP 桥（复用 `envs.policy.build_policy` 全 spec）＋ 逐决策日志落盘。
- **接口基线**：浏览器引擎 `PlayerIO` ↔ 本地 Python 模型服务；观测＝`notification.state` JSON（与训练管线同源，禁上帝视角，C10）；模型端可切纯策略 / LLM 辅助（验收仍纯策略计，§6.3）。
- **素材合规**：照 refs 做法——运行时按 id 拉取、不入库、缓存 gitignore、不随发布物分发（§8）。
- **用途**：人 vs 模型对局调试；明示信息流程，**不进强度验收口径**。

---

## 4. 里程碑与验收标准

| 里程碑 | 内容 | 验收标准 |
|--------|------|----------|
| **M0 基座** | gitcg 环境封装 + 多进程 rollout + episode 日志 + arena 骨架 + 评测池脚本对手全部可跑 | 脚本互打跑通，吞吐按 L0 实测基线换算（12 worker ~4,900 局/h，见 `reports/LOCAL_BASELINE.md`）；信息泄漏测试通过；同种子重放一致 |
| **M1 观测/动作** | observation_encoder + action_hierarchy/taxonomy + belief_groups 落地 | 任一步合法抽象动作 ≤ 且能覆盖 100% 合法具体动作（adapter 往返测试）；随机策略与脚本对局无非法动作（零回退） |
| **M2 冷启动基线** | 脚本专家 BC / DMC-Q sanity 基线 + 评测阶梯 v1 | BC/DMC 基线对池内全部对手整体胜率 ≥ 50%（分卡组报表出）；阶梯与曲线可自动出图 |
| **M3 SoG 内核 v1** | CVPN + 信念根化 continual resolving（差距清单 G1–G6，D15）+ 搜索蒸馏外环 + batched inference（算力到手后接入）；分 CPU 档（§4.3，先跑通语义）与 GPU 档（放大规模） | 自博弈训练循环稳定跑 ≥ 数十小时不崩；候选 checkpoint 通过门禁率 > 0 且阶梯趋势向上 |
| **M4 强度里程碑 ⭐** | SoG 持续训练 + 课程（按失败案例调自博弈配比） | **纯策略模式对池内全部脚本对手整体胜率 ≥ 70%，且无单套对手胜率 < 45%**；对 M2 基线胜率 ≥ 65% |
| **M5 Agent 层 v1** | 局内 LLM 辅助模式 + 复盘/教练 Agent + 卡组 Agent（模板内 flex） | LLM 辅助模式 ≥ 纯策略模式胜率；教练 Agent 归因与人工抽查一致率达标；构筑 Agent 产出 ≥ 1 套胜率提升的 deck 调整 |
| **M6 自迭代闭环** | WS5 循环全自动跑通 + 护栏门禁 | 无人工干预连续 ≥ 5 个迭代；每迭代有 hypothesis/证实证伪记录；阶梯不回退 |

- M4 是 D1 的主验收；M5/M6 是 D2 的主验收。
- 阶梯固定对照组：对手池全部脚本 + M2 基线 + 历史 checkpoint（防止"打赢了旧自己但打不赢脚本"的假进步）。

### 4.1 本地先行轨道（12 核 CPU + LLM API，批量算力前的推进顺序）

M0→M2 与 M5 的 LLM 部分本地就能做，且**不产生将来要丢弃的工作**——本地阶段的评测口径统一用"小样本 + 置信区间"，代码与数据结构与放大阶段完全一致，只是样本量放大。

| 阶段 | 内容 | 对应里程碑 | 本地验收 |
|------|------|-----------|----------|
| **L0 环境基座** | gitcg 封装、Player 子类、rollout、episode 日志 | M0 前半 | 脚本互打通；**测出吞吐基线（局/小时/核）**，换算出 M0/M4 各需多少核·时（替代原"1 万局/小时"验收） |
| **L1 评测骨架** | arena + 对手池注册表 + 阶梯 + 报表 | M0 后半 | 每对局 50 局小样本（配对种子 + Wilson 置信区间出报表）；隔夜可跑完全阶梯 |
| **L2 表示层** | 观测编码 + 动作抽象 + 信念，全部测试 | M1 | 泄漏测试、adapter 往返、重放一致全绿；覆盖/规模统计达标 |
| **L3 LLM Agent 层** | API 版局内 Agent（候选生成/校验/兜底）+ 复盘教练 + 卡组敏感性 | M5 大半 | LLM Agent 对池内全部对手出小样本胜率矩阵（本身就是一个可交付基线）；教练报告人工抽查通过 |
| **L4 SoG 管线玩具验证** | CVPN + resolver + 蒸馏外环在玩具规模（数千样本、CPU 训练）端到端跑通 | M3 前置 | 一轮"采集→训练→评测→落盘"零报错完成；过拟合小数据集验证网络可学 |

- L3 有一个独立价值：**LLM Agent 本身就是研究产出**（LLM 直接玩七圣召唤 vs 池内脚本的强度测量），其对局日志同时喂教练 Agent 与 BC 冷启动数据。
- 算力放大后回到主轨道：L4 的玩具管线直接放大为 M3 全量训练（GPU 档），M4/M6 按 §4 原验收执行。

### 4.2 本地先行收尾（L5）

L0–L4 已完成。以下为本地双机（D14）上继续推进的可交付产物，不产生废弃工作：

| # | 工作项 | 对应 | 验收 |
|---|--------|------|------|
| L5.1 | 对手池全矩阵底座：20×20×50 局双向（并行口径 ~96 核·时，12 worker 隔夜） | M0/M2 底座 | 胜率矩阵 + Elo + Wilson CI 落盘，含 `engine.lock` 元数据（D4） |
| L5.2 | LLM 小样本胜率矩阵（纯策略 vs LLM 辅助分开报分，§6.3） | L3 独立研究产出 | 对池内全部对手出报表；干预全量记录；先 5 对手×10 局冒烟再放量。**◐ 2026-10-09：全池矩阵 runner `scripts/run_llm_matrix.py` 已交付（流式落盘+断点续跑+spec 指纹护栏，纯策略/LLM 分开聚合，降级比赛单列+连续失败中止），3×3 冒烟跑通（整体 0.722→0.778，54 调用 0 失败）；放量到全 20 套为单命令隔夜任务** |
| L5.3 | CVPN 学习曲线放大（数千~数万样本） | L4→M3 | 训练/验证损失与一致率曲线，确认网络规模-数据量关系。**◐ 2026-10-09：`train/learning_curve.py` 已交付（数据量×网络规模网格、每点全新网络、**固定验证留出集**使点间可比、json/csv 落盘），1465 样本扫 128/256/512/1024 × d_model 64/128：val_loss 单调下降（64 维 2.11→1.11）、val_acc 单调上升（0.50→0.66）；全量放大待 GPU（M3）** |
| L5.4 | **S1 薄 fork 桥**（D12/D13）：中局快照/克隆/分支 | M3 前置 | 分支续跑确定性测试过；Python 侧以 JSON-lines 子进程接入。**✅ 2026-10-09：`canResume:true` 边界快照即精确分叉（13/13，契约 D13），搜索直接用 pybinding 分叉、无需引擎改造；子进程 fork 桥（`envs/fork_bridge.py`/`fork_worker.py`）与实用版 fork 搜索 resolver（`envs/fork_search.py` + `scripts/run_search_agent.py`）已交付，跨进程重放逐字节复现活体终局** |
| L5.5 | （可选）评测池扩容：补齐对手至 22 套（另找 deck share code） | D11 | 新卡组入池并出对位报表，不阻塞任何验收 |
| L5.6 | **集群数据平面**（采集/训练解耦）：可续跑分片采集 + 从 replay 训练/学习曲线 | M3 / §5.1 | 采集中断续跑不重复；CPU 采集与 GPU 训练解耦；LSF 模板齐备（`collect`/`collect_array`/`train_gpu`）。**✅ 2026-10-09** |

**L5 进度（2026-10-08）**：

- **L5.1 ✅ 全矩阵底座**：20×20×50 双向 = **40,000 局，0 error / 0 truncated**（12 核 ~8 小时，一次跑完无断点）。
  产物 `reports/arena/20261008T163940_full_matrix_20x20.json/.pairings.csv`（含 `engine.lock` 元数据，400 对位行 × 100 局）。
  强度基线（Wilson 95% CI）：最强 unyielding_geo 0.831 [0.814, 0.847]、dvalin_bonk 0.733、double_geo_navia 0.723；
  最弱 skirk_chasca_freeze 0.175 [0.159, 0.193]、skirk_ayaka_navia 0.181、ice_water_battleship 0.186；Elo 阶梯同步落盘。
  对手池强度分层清晰（0.175–0.831），作 M2/M4 门禁底座合适。
  **排产口径**：12 worker 并行 **~4,900 局/h**（每局 ~8.8 秒/核，约 50% 并行效率）；40,000 局 ≈ 96 核·时（并行口径）。
- **L5.2 ◐ LLM 小样本矩阵**：5 对手 × 10 局冒烟完成——0 API 失败、0 降级回退，干预改动作率 ~80%，
  预算 6 次/局全部用满；纯策略 vs LLM 辅助 2 胜 1 负 2 平（弱对局提升更明显：double_geo_navia 0.15→0.30）。
  **2026-10-09 续**：全池矩阵 runner `scripts/run_llm_matrix.py` 交付（纯策略/LLM 分开聚合、逐局流式落盘、
  按 `task_index` 断点续跑 + spec 指纹护栏、逐干预明细只在原始流）；3×3 冒烟跑通（整体纯 0.722 / LLM 0.778，
  54 调用 0 失败，改动作 45/54）。放量到全 20 套为单命令隔夜任务（`--seeds 10`）。
- **L5.4 ✅ 引擎桥 + 薄 fork 桥落地（2026-10-09，D12/D13）**：中局快照行为系统实测——往返无损、可续跑、
  克隆确定，且**边界快照即精确分叉**：record-replay 对照实验（`scripts/probe_boundary_fork.py`）
  13/13 个 `canResume:true` 边界快照逐字节复现活体终局（`canResume:false` 的 phase 内部点不分叉，
  搜索不需要在那里分叉）；搜索分叉直接用 pybinding 边界快照，无需引擎改造（D13）。
  已交付：**子进程 fork 桥**（`envs/fork_bridge.py` + `envs/fork_worker.py`，JSON-lines，
  `run_fork_task` 进程内参考实现 + `ForkBridge` 子进程池；跨进程按位置选项索引重放逐字节复现
  活体终局，`tests/test_fork_bridge.py`）；**实用版 continual-resolving 搜索**
  （`envs/fork_search.py`：`ForkSearchPolicy` 在最近 `is_resumable()` 边界 fork+注入候选+rollout，
  基座选项恒为候选之一、basal RNG 不漂移；`run_search_match` 驱动整局；`top_k`/`search_every`/
  `max_searches`/`rollouts` 控预算）；**评测脚本** `scripts/run_search_agent.py`（配对种子
  search vs expert 消融、流式落盘 + 指纹护栏 + 断点续跑）。**纪律（C5）**：fork 一律跑在
  子进程（`run_search_match` 拒绝 `workers==0`）。
  replay-branch MC teacher（`train/replay_branch.py`）保留为批量离线 teacher 与对照。
  剩余可选项：JS 侧批量 fork/rollout 吞吐优化（归 `batched_inference`，非前置）；搜索语义升级到
  信念根化 MCCFR（M3，D15）。
- **L5.3 ◐ CVPN 学习曲线**：`train/learning_curve.py` 交付——采集一份样本池后**先固定验证留出集**，
  再对每个 (网络规模 `d_model`, 训练样本量) 组合在同一留出集上训练全新网络，记录 train/val 损失与一致率
  （json/csv 落盘，不含 checkpoint）。1465 样本（留出 20%）扫 128/256/512/1024 × d_model 64/128：
  val_loss 单调下降（64 维 2.11→1.11）、val_acc 单调上升（0.50→0.66），网络可学、数据有效；
  全量放大并入 M3 GPU 档。测试 `tests/test_learning_curve.py`。
- **L5.6 ✅ 集群数据平面（2026-10-09，采集/训练解耦）**：`train/collect.py` + `train/collect_worker.py`
  ——可续跑分片采集（一行一局、manifest 指纹护栏 + engine.lock、逐局流式落盘）；`train.pipeline --replays`
  与 `train.learning_curve --replays` 从磁盘训练/扫曲线，使 **CPU 队列采集与 GPU 队列训练解耦**（§5.1）。
  LSF 模板补齐 `collect.lsf`（单节点多 worker）、`collect_array.lsf`（海量数组作业，每元素独立分片目录）、
  `train_gpu.lsf`（`--replays` 训练）。测试 `tests/test_collect.py`（序列化无损、撕裂行容错、指纹稳定、
  并行采集 + 续跑不重复）。
- L5.5 未启动（可选）。长作业运行方式见 §0.1 C8（tmux + 流式断点）。

### 4.3 无 GPU 中间验收（M3-CPU，D15）

GPU 放大前的 go/no-go——**「信念根化 MCCFR 搜索 + 小网」纯策略对池内全部对手整体胜率 ≥ 50%（M2 口径）**。
按 `train/CONTINUAL_RESOLVING.md` 差距清单推进，各阶段在 CPU（本机 + WSL）上验收：

| 阶段 | 内容 | 验收 |
|------|------|------|
| S-A 信念根化（G1+G3） | 搜索根换「公共信念 + 采样确定化」（禁用隐藏真值），`reps/belief_groups.py` 接入根与 update | 信息泄漏测试覆盖 belief 根；搜索不再读取隐藏真值 |
| S-B 外采 MCCFR 脚手架（G2+G4） | `train/cfr_solver.py`：regret matching + 根平均策略 + 根价值，深度受限 + CVPN frontier 价值 | 单测（分布合法、平均策略归一、深度截断一致）+ 玩具域收敛 sanity |
| S-C search-as-teacher（G6+G7）**✅ 2026-10-12** | 根平均策略/根价值/belief 目标写进 replay，蒸馏回网络（policy CE + value 回归 + belief 评分），网络加 belief 头 | 一轮采集→训练→评测→落盘零报错 ✅；门禁（候选不劣于）可自动判定 ✅；`train/belief_head.py`+`train/distill.py`+`train.collect --search`+`scripts/run_distill_smoke.py` |
| **M3-CPU 门禁** | 搜索+小网纯策略 vs 评测池 | **整体胜率 ≥ 50%（M2 口径，纯策略模式）** |

训练规模走 CPU 档：d_model 64–128、1e3–1e4 样本、fp32；WSL MX550 小网络训练试验的结论决定训练节点归属（§5.2）。

### 4.4 NPC PvE 调试系统门禁（PvE-v1，D17）

NPC 系统是**新增并行线**（D16），验收为**独立门禁点**，与模型主线（§4.3 M3-CPU）解耦；复用引擎（D20 overlay）与评测（WS2/decklab）：

| 阶段 | 内容 | 验收 |
|------|------|------|
| PvE-B0 构建链 **✅ 2026-10-12** | Docker 内重建 `libgitcg.so`（D20）：pin `v0.21.0`，V8 11.6.189.4 纯 CMake 直编，首建 ~20 min、增量 ~30 s、产物 sha256 确定 | `pytest -m "not slow"` 全绿（162 passed）+ 同种子重放一致 + 跑通一局 ✅；脚手架 `engine-overlay/`，provenance 见其 `README.md` |
| PvE-B1 敌人移植 | 约 13 类敌人 `.gts`（技能/被动/HP/充能；蒙德/璃月全量，从蒂玛乌斯 3 只起步） | 可组牌、跑满、无错误 |
| PvE-B2 策略 **✅ 2026-10-12** | `NpcPolicy`（greedy + rotation，确定性、可复现） | 同种子可复现 ✅：`npc/policy.py`，永不打行动牌、不主动切人，被动/切人由引擎驱动；平局按 (槽位, 技能 id 升序) |
| PvE-B3 规格与判定 **✅ 2026-10-12** | `NpcSpec` + 挑战目标三星判定（7 类）+ 评测入口（可续跑/子进程并行） | 输出胜率 + 三星达成率 ✅：`npc/{spec,policy,objectives,eval}.py`；`envs/match.py` additive 采集 `skill_used`/元素反应事件；冒烟 16 关×3 种子 win_rate 0.979 / 三星率 0.625（Wilson [0.484,0.748]），0 error |
| PvE-B4 关卡数据 | 蒙德/璃月 16 关规格 + 挑战目标原文 | 16 关规格齐全 |
| **PvE-v1 门禁** | decklab 自动组牌（池 C）× 16 关 × 多种子 | **三星达成率 ≥ 60%**；报表拆分组牌质量 vs 打牌质量（D19） |

---

## 5. 算力与部署

**结论（D14，CPU 配额见 D21）：本机 + WSL 双机为默认算力（CPU 采集/评测在此），A800 80G 为申请制机会资源（以 GPU 训练为主），南大集群暂缓。**
依据：RL 网络只有几百万~几千万参数，CPU 档（d_model 64–128、fp32）本地就能训练验证语义；SoG 路线的
真正瓶颈是 CPU 搜索/环境步进——先把信念根化 MCCFR 在 CPU 档跑通（§4.3），规模放大留给机会算力。
所有里程碑不依赖机会资源；工程保持"无常驻服务、断点续跑、CPU 采集与 GPU 训练解耦"，窗口到手即可放量。

| 资源 | 用途 | 关键点 |
|------|------|--------|
| 本机 12 核 / 30G（主力） | rollout/采集/评测 arena、CPU 档训练、fork 搜索、LLM Agent（API） | 12 worker ~4,900 局/h（排产口径，§4.2）；开发与验收全部在此闭环 |
| WSL 4 核 / 8G / MX550 2G（并行节点） | 分片采集/评测；**小网络训练试验**（d_model ≤ 64、fp32 小 batch） | MX550 无 tensor core、显存 2G：试验以"能训 + 速度对照 CPU"为目标，结论决定训练节点归属 |
| A800 80G（机会资源，申请制、时间受限；**CPU 配额仅 ~10 核，见 D21**） | ① CVPN/蒸馏 **GPU 放大训练** + batched inference；② 大 LLM（32B/72B bf16 或 LoRA）做局内辅助/深度复盘 | **只做 GPU 训练/推理**：CPU 密集的采集/评测留本机/WSL（10 核与 12 核同级，不值得占窗口）；数据本地采集后**传入**；一键零看管跑完；LLM 不可用时走商业 API 兜底（D6） |
| 南大集群（暂缓） | — | glibc/环境适配成本高（D14）；`scripts/cluster/` 保留为将来机会算力的适配起点（C13） |

**工程形态约束（由 D7/D14 推出，所有模块必须遵守）**：
- **无常驻服务**：`batched_inference` 等服务按作业拉起/销毁，不假设"某个端口一直开着"；机会算力上的 LLM 服务同理，客户端必须能降级到 API 或纯策略模式。
- **一切长任务可断点**：训练、自博弈、评测全部支持 checkpoint + 重提（作业被排队/中断是常态）。
- 存储：热数据（当前 run 的 replay 池/checkpoint）放本地盘；本机只放代码与小体积报告，大数据留采集机器/机会算力存储（`data/` 已 gitignore）。

### 5.1 机会算力窗口模式（A800 GPU，CPU 配额 ~10 核；申请制、间歇可用；D21）

机会窗口的用法（里程碑验收不变，墙钟与窗口占比成反比；**窗口 CPU 太少，只做 GPU 训练/推理**）：
- 窗口内只跑 **GPU 训练/蒸馏**（+ 可选本地 LLM）；**采集/评测不在窗口做**（D21）；
- 交付节奏：**本地/WSL `train.collect` 分片累积 → 数据传入 A800 → `train.pipeline --replays/--distill` GPU 训练 → checkpoint 回流本地 → `scripts/eval_protocol.py` 认证**；窗口结束即产 checkpoint + 报告；
- 训练输入可断点续跑、一键零看管（§5.1 D7/§5.1）；LLM 不可用时走商业 API 兜底（D6）。

> **A800 窗口实测（2026-10-12，详见 `reports/explore/a800_window_20261012.md`）**：容器 CPU 配额实测 **10 核**（cgroup `cpu.max=1000000/100000`）；10 核饱和采集 ~**2,491 局/h**（120 worker 过度订阅，10 核箱宜 ~10 worker）；**GPU 相对数据平面富余约 100×**（precollated 训练 276k samples/s ⇒ 5 万局 × 50 epoch ≈ **8 min**）；8000 样本上 d512 最优、val_acc 随数据单调上升 → **当前瓶颈是数据而非网络容量**，主线按「本地/WSL 分片采集 → 传入 A800 → GPU 一次性放大/蒸馏」排产。A800 本地存储上限 ~**37 GB**（实测 1.64 MB/局 ≈ 2.3 万局/窗，传入前先压缩）。

### 5.2 本地双机模式（默认：12 核 + WSL 4 核，对应 §4.1 本地先行轨道）

| 负载 | 可行性 | 说明 |
|------|--------|------|
| 环境/rollout/脚本对局 | ✅ 完整 | 纯 CPU；吞吐按 L0 实测基线排产（本机 12 worker ~4,900 局/h，WSL 再加分片） |
| arena / 阶梯 / 报表 | ✅ 完整 | 小样本（50 局/对局）+ Wilson CI，隔夜一轮 |
| 观测/动作/信念 + 测试 | ✅ 完整 | 设计与测试不依赖算力 |
| LLM 局内/复盘/卡组 Agent | ✅ 完整（API 版） | 见下方调用预算约束 |
| SoG 管线（CPU 档） | ✅ 语义验证 + 中等规模 | d_model 64–128、1e3–1e4 样本；M3-CPU 验收（§4.3）在此档完成 |
| CVPN 放大训练 | ⏳ 机会算力 | GPU 档（bf16、更大网络/数据）等机会窗口（§5.1） |
| 大规模自博弈采集 | ⏳ 分片渐进（**本地**） | 本机/WSL 分片累积（窗口 CPU 太少，不放窗口，D21）；`train.collect` 可续跑、断点不重复 |

**WSL/MX550 小网络训练试验**：在 WSL 上以 d_model ≤ 64、fp32、小 batch 训一轮 CVPN，记录吞吐与稳定性，
与本机 CPU 训练对照；试验结论写回本表，决定训练节点归属。

**LLM API 使用约束**：
- 局内调用预算 ≤ 10 次/局，只在关键决策点触发（候选分歧大/搜索值接近/终局附近），评测时全量记录调用与影响；纯策略模式完全不调用（保持 G1 验收口径纯净）；
- 复盘教练按"局级"调用（1–数次/局），批量便宜；卡组分析按"实验级"调用；
- API key 走环境变量/本地密钥管理，**永不入库**（.gitignore 已挡 `*.log`/本地配置）；所有含对局数据的 prompt 注意平台/模型条款；
- LLM 客户端统一走 §5 的降级链接口，后续换本地模型零改动。

## 6. 关键设计决策与备选分支

1. **SoG 为主线，DMC 为备线**：直接上 SoG 的前提是 M1 的观测/动作/信念三件套质量过硬。若 M3 处训练长期不涨，启用备线：先 DMC-Q（piovium/AI 思路）验证管线 → 产出 BC 数据 → 回灌 SoG。切换判据：M3 冒烟后阶梯 2 周量级无上升趋势。**搜索引擎受阻不触发备线切换**（D12/D13）：分叉能力已由边界快照解锁；fork 桥出问题时 teacher 切 replay-branch MC 离线标注，主线仍为 SoG。
2. **动作抽象是硬前置**（不是优化项）：没有它 CVPN 的动作头不可训练，M1 的"抽象动作覆盖全部合法动作"测试是 go/no-go。
3. **纯策略模式为强度验收口径**：LLM 辅助可以更强，但 G1 的验收、训练门禁、阶梯全部以纯策略模式计，避免"LLM 掩盖策略缺陷"。
4. **跟随最新卡池的代价控制**：不追新卡池的中间小版本；卡池升级作为一次显式迁移（更新 engine.lock + 数据重标注策略 + 迁移评测），频率 ≤ 每大版本一次。
5. **逐模块移植 Rebel_base_RL + 自写测试**：参考仓是打补丁式实验分支，只当代码库抄、不当可运行基线——逐模块搬代码 + 自己写测试，不整仓 apply 补丁（C1）。

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 动作空间组合爆炸压不下来 | 训练不收敛 | M1 设 go/no-go；taxonomy 以 Rebel_base_RL 现成的为基础，不够再加参数化动作（目标/骰子选择用独立参数头） |
| 信息泄漏（上帝视角） | 虚高胜率、上线即崩 | 信息泄漏自动化测试进 CI + 门禁；观测只走 `notification.state` |
| Rebel_base_RL 代码不可跑/有隐性 bug | 蓝本不可用 | 逐模块移植 + 重写测试；核心算法对照 charter 文档独立复核 |
| 追卡池版本消耗全部工程量 | 训练中断 | engine.lock 冻结 + 显式迁移节奏（§6.4） |
| CPU 搜索吞吐不足 | 自博弈数据饥饿 | 本机 + WSL 分片横向扩 rollout 并分时累积（机会算力 CPU 配额少，不放采集，D21）；搜索深度/预算自适应（前期浅搜索、后期加深） |
| 薄 fork 桥接入搜索出意外（并发 RPC 交错跨版本不保证） | 搜索 teacher 断供 | 分叉确定性测试常驻 CI（`tests/test_engine_bridge.py`）；teacher 切 replay-branch MC 离线标注（D12/D13），训练外环不空转；引擎升级（D4 迁移）时重跑 `probe_boundary_fork` |
| MX550 / 2G 显存训不动 CVPN | WSL 训练价值有限 | 小网络（d_model ≤ 64、fp32 小 batch）试验定夺训练节点归属（§5.2）；训练默认本机 CPU |
| 机会算力不可预期获得 | 训练放大/本地 LLM 延期 | 训练与自博弈不依赖机会算力（CPU 档先行，D15）；LLM 客户端强制降级链（本地 LLM → API → 纯策略）；机会窗口一键零看管跑完（§5.1） |
| LLM 干预不可评测 | 说不清强弱 | 双模式分开报分 + 干预全量记录 |
| 自迭代闭环改坏工程 | 回归/丢失进度 | §WS5 护栏：分支隔离 + 四项门禁 + 只追加存储 |
| `static-data.piovium.org` 网络不可达（B0 实测） | WS8 standalone 前端构建曾受阻；引擎/数据不受影响 | **已找到可达镜像**：`https://gi-tcg-assets-api-hf.guyutongxue.site`（本项目已在用）**完整实现 v5 API**（`/api/v5/data|datum|image/...` 实测 200，含 webp 图像）→ 构建时与运行时装 `DEFAULT_STATIC_DATA_API_BASE_URL` 指向它即可；引擎构建另用 `--filter=!{packages/assets-manager}` 绕开（记录见 `engine-overlay/README.md` §5） |
| exploitability 算不了 | 无法知道距纳什多近 | 接受工程口径：固定基线 + 阶梯 + 对手池多样性（评测池 20 套，D11 + 自博弈池 + 历史 checkpoint） |

## 8. 许可与合规
- **AGPL-3.0**：genius-invokation / Rebel_base_RL / LPSim 主体均是 AGPL。基于其二次开发并分发（**含以服务形式对外提供**）需以 AGPL 开源。若只想内部研究不对外服务，仍需注意 AGPL 网络条款；建议本项目代码本身按 AGPL-3.0 发布，省去合规判断。
- 卡面/素材版权归米哈游，训练产物与发布物不携带游戏素材。
- play.piovium.org 数据使用前过一遍平台条款（人机混用限制、数据授权）。
- 机会算力使用遵守学校收费/作业规范（公共队列按申请使用，长跑作业用海量作业模式）。

## 9. 目录结构（工程仓 `~/projects/genshin-GITCG`，参考仓 `refs/` 只读不修改）

```
genshin-GITCG/          # = ~/projects/genshin-GITCG
├── PLAN.md                    # 本文档
├── envs/                      # WS0: gitcg 封装、Player 子类、rollout runner
├── reps/                      # WS1: observation_encoder / action_hierarchy / adapter / belief
├── agents/                    # WS4: resolving_agent（纯策略）、llm_assist（局内）、脚本对手适配
├── train/                     # WS3: cvpn_model / cvpn_training / sog_pipeline / batched_inference
├── eval/                      # WS2: arena、阶梯、对手池注册表（20 套，D11）、报表
├── coach/                     # WS4: 复盘归因、课程建议、BC 标签
├── decklab/                   # WS4: 卡组构筑 Agent（模板内 flex）
├── orchestrator/              # WS5: 自迭代循环 + 护栏门禁
├── engine-overlay/            # WS7/WS8: 我们的引擎 overlay 源（敌人 .gts + standalone 改动 + apply 脚本，入库）
├── build/                     # D20: 引擎构建工作区（refs 副本 + V8 workspace + .so，内容不入库）
├── data/                      # 回放池、日志（内容 gitignore，只存 schema 与清单）
├── tests/                     # 信息泄漏测试、adapter 往返、重放一致性等
├── configs/                   # run 配置 + engine.lock + 卡池版本（npc/ 关卡与敌人数据，pools/ 卡池清单）
└── reports/                   # 评测曲线、复盘报告、hypothesis 台账
```

## 10. 下一步行动（三线并行，2026-10-12；D14–D20）

模型主线（A）维持 §4.3（M3-CPU）；NPC（B）与 Web（C）为新增并行线（D16–D20）。阶段划分如下。

**进度（2026-10-12）**：B0 ✅（`engine-overlay/`，pin `v0.21.0`，V8 直编，162 passed）｜**模型主线 S-A/S-B/S-C 全 ✅**｜B1 ✅（17 敌人 `.gts`）｜B2/B3 ✅（`npc/`）｜B4 NPC 数据 ✅｜**C0 ✅**（standalone + 浏览器↔Python 桥）｜**A6 ✅**（紧凑载荷 0.45×/编解码 3×）｜**A5/A7 ✅（数据先行，结论见 D24）**：BC 撞 teacher 天花板、训练时长是杠杆、强 deck 过 ≥0.50 认证线、floor 结构性 FAIL。**下一步（D24）**：策略改进（自博弈/RL 或修好的搜索）；并行修 `train.collect` 内存 bug（A8）+ B5 PvE-v1 门禁。

**阶段 1（立即）**：B0 构建链 go/no-go（Docker 内建 V8，D20）**✅** ‖ A1 S-B MCCFR 脚手架 **✅**

1. **B0 构建链 go/no-go（§4.4 PvE-B0，D20）**：网络可达性（depot_tools/V8/npm）→ `ubuntu:22.04` 镜像（cmake ≥3.25、g++ C++20、ninja、git、python3）→ overlay 复制引擎到 `build/engine/`（refs 只读）+ `pnpm install`（node 24）→ **V8 首建**（`V8_BUILD_JOBS=8`、`V8_WORKSPACE` 挂 volume，建成后 `V8_OFFLINE=ON`）→ `libgitcg.so` 接入 conda `gitcg`；验收＝现有 `pytest -m "not slow"` 全绿 + 同种子重放一致 + 跑通一局；构建 provenance 记入 engine.lock 变体。**失败 fallback**＝Node 子进程 + TS core + `custom-data-loader`。
2. **A1 S-B（G2+G4）**：新增 `train/cfr_solver.py`——外采 MCCFR（regret matching + 根平均策略 + 根价值、深度受限 + CVPN frontier 价值）；根用 `reps/belief.py` 确定化（S-A 已就绪），叶子打分走 `train/resolver.py`。

**阶段 2**：A2 S-C（search-as-teacher + belief 头，G6+G7）‖ B1 敌人 `.gts` 移植（`custom-data-loader` 免构建快循环）+ B2 `NpcPolicy` ‖ C0–C2 Web（standalone overlay + 选牌流程 + 模型 HTTP 桥）

3. **A2 S-C（G6+G7）**：根平均策略/根价值/belief 目标随 `train.collect` 写进 replay，训练改搜索蒸馏（policy CE + value 回归 + belief 评分），网络加 belief 头（`train/model.py`）；
4. **B1/B2**：约 13 类敌人 `.gts`（从蒂玛乌斯 3 只起步）＋ `NpcPolicy`（greedy + rotation）；
5. **C0–C2**：`standalone` overlay 起局 → 选牌流程（人选牌 → AI 看对手牌反选 → 随机先手）→ 模型 HTTP 桥（复用 `build_policy`）。

**阶段 3**：A3 M3-CPU 认证 ‖ B3–B5 PvE 门禁 ‖ C3/C4 NPC 卡组 + 调试闭环

6. **A3 M3-CPU**：「搜索+小网」纯策略对池内全部对手整体胜率 ≥ 50%（M2 口径）——用 `scripts/run_search_agent.py` / `scripts/eval_protocol.py` 出报表；
7. **B3–B5**：`NpcSpec` + 三星判定 + 评测入口；蒙德/璃月 16 关规格与挑战目标原文；**PvE-v1 门禁**（decklab 组牌 × 16 关三星 ≥ 60%，§4.4）；
8. **C3/C4**：人可持 NPC 敌人卡组（`custom-data-loader` 注入）＋ 逐决策日志落盘复盘；
9. **并行收尾**（不阻塞主线）：WSL/MX550 小网络训练试验（结论写回 §5.2）；L5.2 全池 LLM 矩阵放量；L5.3 学习曲线留档；L5.5（可选）评测池扩容；
10. 机会算力窗口到手（§5.1，D21）：本地/WSL `train.collect` 采集 → **传入 A800** → `train.pipeline --replays/--distill` **GPU 档训练放大** → checkpoint 回流 → 本地 `eval_protocol` 认证 → M3 GPU 档 → M4。

---
*本文档随决策变化更新；重大变更（改主线、改验收口径）需在 §0 追加决策记录。*
