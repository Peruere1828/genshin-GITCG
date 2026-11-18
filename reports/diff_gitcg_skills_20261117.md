# gitcg-skills ↔ genshin-GITCG 差异审计报告

- date: 2026-11-17
- 基线: `genshin-GITCG@98b4dfa`（master HEAD，工作区干净）vs `gitcg-skills@2e7c939`
- 方法: gitcg-skills 的 bootstrap 提交 `4d26e8d` 与其 `docs/PROVENANCE.md` 均声明「环境层原样快照自 genshin-GITCG@98b4dfa」，
  故两仓库共享层的一切差异均为 gitcg-skills 侧改动；按 commit（`git log 4d26e8d..2e7c939`）+ 逐文件 diff 分类。
- 口径: **忽略 LLM 自进化内容**（skillloop/、技能库、prompt、LLM 智能体/监控面板等）；聚焦 bug 修复与可移植 feature。
- do_not_edit: 是（生成文档）；讨论更新请新建文件。

---

## 1. 建议移植的 bug 修复（A 类：当前仓库存在同样问题）

### A1. 网页「双方 AI 自对弈」棋盘冻结（observeIo）
- 问题: `engine-overlay/overlay/packages/standalone/src/StandaloneParent.tsx:489` 在自对弈时执行
  `game.players[HUMAN_WHO].io = seatIo(HUMAN_WHO)`，Chessboard 绑定的 `uiIo` 收不到任何通知，
  棋盘永远停在「正在等待…」，对局其实在跑（只是看不见）。
- 修复（gitcg-skills `a948602`）: 同文件 `:101-108` 新增 `observeIo(main, observer)`——
  `notify` 同时喂桥 io 与 uiIo，`rpc` 只走桥（避免观战 UI 抢走决策），`:549` 应用。
- 移植成本: ~28 行一个 helper + 一行替换，无协议改动。

### A2. 「取 AI 牌组」被自动 counter-pick 静默覆盖
- 问题: `App.tsx:219`（`fetchAiDeck` 内 `setOtherDeckEdited(false)`）+ `App.tsx:429-434`
  （`startGame` 在 `!otherDeckEdited()` 时 `chooseCounterPick` 覆盖 `theirs`）→
  桥声明的 AI 牌组（expert 脚本为之编写）被反选结果替换，产生 deck_mismatch、AI 变笨。
- 修复（gitcg-skills `App.tsx:434-440`）: 取牌组后 `setOtherDeckEdited(true)`，让桥声明的牌组粘住。
- 移植成本: 一行。

### A3. 桥地址写死 127.0.0.1 → 局域网不可达
- 问题: `App.tsx:139` `createSignal("http://127.0.0.1:8770")`，LAN 客户端的浏览器访问的是**自己的** loopback；
  `vite.config.ts` 无反代（`-44,50` 无 `proxy`），还带来 CORS 面。
- 修复（gitcg-skills `a948602`）: `vite.config.ts:33-44,62-72` 同源 `/bridge` 反代（dev + preview 都挂），
  `App.tsx:283-286` 默认改 `${window.location.origin}/bridge`；`scripts/serve_web_demo.sh` 单端口部署脚本。
- 移植成本: 小（vite proxy + 一个默认值 + 可选脚本）。

### A4. DeckBuilder 未传 AssetsManager → NPC/引擎角色渲染用官方镜像数据
- 问题: `App.tsx:797` `<DeckBuilder … />` 不传 `assetsManager`（prop 本身已存在：
  `refs/genius-invokation/packages/deck-builder/src/DeckBuilder.tsx:45-51`），
  组牌器/预览显示的是官方静态镜像而非引擎 overlay 数据（NPC 敌人名字/血量错位）。
- 修复（gitcg-skills）: `App.tsx:311-319`（带 customData 的 AssetsManager memo）+ `:1351-1356` 传入；
  新增 `DeckPreview.tsx`（见 B4）。语义定调（`2a2998c`）: **血量语义完全由引擎数据决定**，
  不做网页层 per-seat HP 覆盖（`c66e569` 的临时覆盖已随之删除）。
- 移植成本: 一个 prop。

### A5. common/llm.py 通用健壮性（我们有 LLM 辅助模式，独立受益）
| 改动 | 位置（gitcg-skills/common/llm.py） | 说明 |
|---|---|---|
| 429/5xx 按 `Retry-After` 重试 + 指数退避 | :171-207 | 3 次尝试，仅 HTTPError 429/5xx 重试，其余 URLError/Timeout 独立退避 |
| 空内容守卫 | :236-240 | 截断致 content 为空 → 返回 `None` 并计 `usage.errors`，而非空串 |
| 缓存/成本记账 | :176-183 | `cached_tokens`（prompt_tokens_details）、`cost_usd` |
| 密钥脱敏前缀扩展 | :28-31 | `sk-` / `oc_sk_` |
| User-Agent | :195-199 | 部分 CDN 拒绝 urllib 默认 UA |
- 配套测试: `tests/test_llm_http.py`（152 行，无网络：重试/空内容/记账/头）。
- 需剥离: `x-opencode-session` / `reasoning_effort` / `SKKEY`（他们 provider 特化）、`GITCG_LLM_TRACE`（可选）。

### A6. 数据 bug: shitou `win_within_rounds(44)`
- 现状: `configs/npc/npc_stages_mond_liyue.json:742-745` `n: 44`（eurogamer 原文 "within 44 Rounds"
  疑似笔误，`unverified_reason` 待游戏内核实，`objectives_unverified_reason` 同）。44 回合 ≈ 无约束 → 该目标恒通过。
- 证据: HoYoWiki 冒险挑战为 **4 回合**（gitcg-skills `reports/npc_special_rules.md §5.5`）；
  gitcg-skills 改为 5（D7 统一 5 的考核口径，非 wiki 值）。
- 建议: 按 wiki 改 4（或按考核设计改 5），并把 `verified` 置真/注明出处。

### A7. 数据 bug: fatuipyroagent `hp_card: 9` vs 引擎 11
- 现状: `configs/npc/npc_enemies_all.json` hp_card=9、`npc_stages_mond_liyue.json` lineup hp=9，
  但引擎卡 2301 实际 11 HP；`tests/test_npc_enemies.py:270` 把它记为 "documented deviation"。
- 修复轨迹: gitcg-skills bootstrap `4d26e8d` 先改 9→11（「引擎实际 11」），`2a2998c`（D12）再拆 NPC 变体（见 B2）。
- 影响: PvE 关卡里该敌人实际 11 血而数据宣称 9；镜女 mirrormaiden=10 两边一致。

### A8. 测试稳健性（轻量）
- `tests/test_web_rpc_server.py:948-954`（gitcg-skills）: counter-pick 断言按 `method=="decklab-rule"` 分支
  （decklab 回退合法地产出 `heuristic`）；`web/counterpick.py` 两边一致，旧断言只在 `reports/arena/` 存在时成立。

---

## 2. 新 feature（B 类：可选移植）

### B1. 引擎「冒险挑战特殊规则」（最大项，跨引擎层）
- 修的问题: 16 关 PvE 的特殊规则此前**只解析、从未执行**——10 关缺初始支援牌、blanche 缺开局 +2 充能、
  4 关缺附元素；`timaeus.special_rules=["artifact_cards_free"]` 是误读（真实规则：支援牌 322003 每回合
  一次付「合成材料」免费装圣遗物；wagner 同理 322004）。
- 组成:
  - 数据: `configs/npc/special_rules.json`（528 行；16 关逐字 HoYoWiki 文本 + 出处；
    10 初始支援 / 7 效果 / 3 无规则；coverage 与 contract 均有 verify）
  - Python: `npc/stage_rules.py`（payload 构造 + `--verify`）；`envs/match.py` 增 `stage_rules=` 参数
    （默认 `None` 行为逐字节不变，`ATTR_CREATEPARAM_STAGE_RULES=2002` 防御式查找）；`npc/eval.py` 接入 + `--no-stage-rules` + 指纹
  - 引擎: `engine-overlay/patches/stage-rules.patch`（667 行、9 文件：core 的 state/game/mutator/utils/
    skill_context/mutation/stage_rule_effects.ts + cbinding/js/{constant,main}.ts）；新增字符串
    CreateParam 属性 2002；初始支援走正常支援打出路径（`on staged` 触发、计 4 张上限、removable 保护）；
    `initial_energy` / `element_application` 两种效果
  - **无属性时逐字节一致**（README `engine-overlay/README.md:222-223`：同 fixture 初始 state JSON 与旧 .so 相同 f9929a10…）
  - 测试: `tests/test_stage_rules.py`（244 行；2 个 runner 测试需改指到 `npc/eval.py`）
- 构建可移植性: `engine-overlay/scripts/` 两边完全一致，patch 对干净 v0.21.0 源树可 `git apply`。

### B2. NPC 变体拆分（2301/2202 vs 9301/9202，D12 设计）
- 动机: 全局 `.gts` HP 补丁会**污染玩家牌组里的官方角色卡**——2301/2202 在 `configs/pools/pool_A_v34_all.json`
  是可玩角色（era 牌组 keqing_shuanghuo 即含 2301）。
- 方案: 官方 2301/2202 恢复/保持原版 11/10；新增 NPC 变体 9301（13 HP）/9202（12 HP），独立技能/状态号段
  （93011+/92021+），`npc/spec.py` ENEMY_ENGINE_IDS 指向变体。
- 对我们: 当前未覆盖 2301/2202 的 .gts，**没有污染问题**；但若未来给 PvE 加难度或给这两张卡做
  平衡改动，应照此拆分而非改全局 .gts。
- 配套工具: `configs/npc/hp_table.json`（审计表）+ `scripts/patch_npc_hp.py`（.gts 批量改写 + 同步
  展示配置 + `--restore`）。注意两个瑕疵：`hp_card_before` 取自已改写的实况文件而非基线；
  `--restore` 的 lineup 按位置 zip，顺序变了会错。

### B3. `configs/engine.lock` 增加 `[overlay]` 溯源段
- `source_tag/source_commit`（v0.21.0 / c94826b…）、`overlay_tree_sha256`、`libgitcg_sha256/bytes`、
  `stage_rules_patch` 路径、updated 日期。可复现性最佳实践，照搬即可。

### B4. `DeckPreview.tsx` 组件（190 行）
- 纯展示、无桥调用：角色行（缩略图+名，坏图隐藏）、卡牌按 id 聚合 ×count、`角色 N / 行动牌 M` 尾注；
  注入 AssetsManager + NPC customData 名称回退（NPC id 无官方资产也能显示）。
- 直接可用于我们配置页牌组预览（当前没有预览）。

### B5. `scripts/import_deck.py` + 校验 helpers
- 分享码 → 命名配置牌组：dry-run 默认、逐项 era 诊断（`sinceVersion`、`obtainable`）、
  `validate_deck_rules`（30 张、≤2 复制、天赋仅对应角色、武器类型匹配——引擎不查的组牌规则）。
- 建议抽 `validate_deck_rules`/`decorate_deck`/`_probe_load`（在他们 `scripts/era_decks.py` 内，
  2829 行的矩阵 runner 不必搬）。我们的 `scripts/explore/fetch_deck_corpus.py` 只做语料抓取，无「导入+校验」。

### B6. `scripts/probe_stall.py` 技术 + 长局归因发现
- 技术: 以 `PolicyPlayer` 子类替换抓公共 mutation 流（伤害/技能/declare_end/winner、逐轮 HP 表）做停滞归因。
- 发现（通用）: 15 回合平局非策略/牌库 bug，是元素护罩（Cryo Ward 131031）/元素生命免疫
  （Electro Slime）+ expert 不换克制元素；`ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT`（attr 105）可配回合上限
  但 `run_match` 从不设置（默认 15）。

### B7. 数据事实性更正（便宜，来自 `reports/npc_special_rules.md`）
- 我们 `configs/npc/npc_stages_mond_liyue.json` 的 HoYoWiki 出处链接错误：
  timmie `entry/2492` 实为「关宏」（正确 1335）；timaeus `entry/2547` 实为「完美留影」（正确 196）。
- `artifact_cards_free` 误读（同 B1）。
- 16 关 `challenge_title` 已从 wiki 取到，可补我们空缺的 `stage_title`。
- `removable: true` 全部是**推断**（`removable_verified: false`，未做游戏内溢出实测）——照搬数据时保留该标注。

---

## 3. 刻意分歧（D 类：不建议照搬）

1. **D4 难度标定**（≤6→6，其余 +2，17 个敌人 .gts + 配套 JSON）：他们脚本胜率 99.7% 饱和后为「星级有区分度」上调。
   属难度设计，我们 PvE 难度自行决策；机制（B2 的审计表）可借，数值不必。
2. **D7 `win_within_rounds` 统一 5**：手测「5 回合可 3 星」当作考试线。注意连带的三处目标改动**与其自身
   wiki 审计矛盾**（§5.5）：shitou wiki=4（我们 44 是 bug，但他们 5 也不是 wiki 值）；lan wiki=6 且存疑
   （**我们的 6 与 wiki 一致**，他们改 5 属考核口径）；swan wiki 仅 1 条目标（**我们的数据是对的**，
   他们补 `win_within_rounds(5)` 是加考试项）。结论：只有 shitou 的 44 需修，swan/lan 不要跟改。
3. **torch-free 删除**：`envs/policy.py` `neural:` 分支 → 显式 `KeyError`；删 `train/`、`kaggle/`、
   `envs/{fork,cfr}_*`、26 个训练测试。我们主线是直接训练，全部保留。
4. **NPC 变体 HP 13/12**（D12）是难度值（11+2/10+2），非官方数据。
5. **web 课程流程收敛**：`SHOW_UNFINISHED=false` 隐藏调试入口、human 固定执 NPC 座、`humanSeat:1` 硬编码等。
6. **web/README.md 重写是文档回退**：删掉了 T1/T2/T3（seq/续局/CORS/aborted 归档）、C1 反选矩阵、
   C3 NPC 近似表、端点/字段清单等协议文档。移植 web 改动时**别跟着删**。
7. `tests/test_web_counterpick.py` 模块级 `assert` → `pytest.skip`（他们缺 `reports/arena/` 数据）。

---

## 4. 忽略清单（C 类：LLM 自进化，按要求不展开）

- `skillloop/` 全部、`data/skills/`（技能库/历史）、`reports/{compare,loadouts,evo_*,t5b_*,prompt_*}` 等实验产物。
- web 的 LLM 集成：`LLMPolicy`、`/ai`、`/ai-decisions`、`/ai-prompt-log`、`/ai-library`、决策日志字段、
  `AiDecisionPanel.tsx`（纯监控）——`web/ai_rpc_server.py` 的 32 个 hunk **全部**属此类或 D，无通用 bug 修复；
  `web/ai-http-io.ts` 两边逐字节相同（无 resume/流式修复可捡）。
- `scripts/{probe_render,probe_llm,extract_skills,build_rules_library,calibrate_thresholds,compare_groups,
  estimate_cost,export_prompt_log,plan_loadout,run_all_groups,run_smoke,audit_trajectory}.py`、
  `scripts/era_decks.py` 主体、`configs/{stage_decks,stage_profiles}.json`。
- `common/llm.py` 的 provider 特化部分（见 A5 剥离项）、`docs/failures/{thinking-in-game-cost-blowup,
  dynamic-vs-scripted}`（LLM 成本/对比负结果）。
- `envs/policy.py` 删除 neural（属 D3，不是 bug）。

另：gitcg-skills 侧两个「教训」可作为流程改进吸收（无需移植代码）：
- `docs/failures/record-decisions-judge-blind.md`：`npc/objectives.py` 的 `play_equipment_card`/`play_event_cards`
  从 `MatchRecord.decisions` 计数，`record_decisions=False` 时恒 false。**我们的 `npc/eval.py:157` 显式
  `record_decisions=True` 不受影响**，但 `envs/rollout_worker.py:28` 默认 False 是潜在陷阱；
  教训是「判定器必需输入被当作可选省内存开关，且只有负例测试」——补正例断言。
- fork 进程池死锁（V8）：我们已知并规避（`npc/eval.py:430` 注释、子进程协议），无需动作。

---

## 5. 移植优先级建议

| 优先 | 项 | 理由 | 成本 |
|---|---|---|---|
| 1 | A1+A2+A3+A4（web 四连修 + DeckPreview B4） | 全是当前仓库现存 bug/缺口，逐个一行到几十行 | 半天 |
| 2 | A5（llm.py 健壮性 + test_llm_http.py） | LLM 辅助模式直接受益 | 半天 |
| 3 | A6+A7（shitou 44 / fatuipyroagent 9）+ B7 出处链接 | 数据正确性 | 1 小时 |
| 4 | B1（引擎 stage rules 全套） | PvE 保真度最大项；需引擎重建（28s 增量）+ 测试改指 | 1-2 天 |
| 5 | 按需：B5 导入校验、B6 停滞探针、B2 审计模式、B3 溯源段 | 工具与可复现性 | 各半天内 |

---

# 附录 A — 移植记录（2026-11-17 已落地）

> 本附录记录报告第一、二节的落地状态；工作区 `genshin-GITCG`（未提交 commit）。

## A.1 第一节：bug 修复

| 项 | 状态 | 落点 / 证据 |
|---|---|---|
| A1 自对弈棋盘冻结 | ✅ | `StandaloneParent.tsx` 新增 `observeIo()` 并用于自对弈座位；`tsc --noEmit` exit 0 |
| A2 「取 AI 牌组」被反选覆盖 | ✅ | `App.tsx` `fetchAiDeck` → `setOtherDeckEdited(true)` |
| A3 同源 `/bridge` 反代 | ✅ | `vite.config.ts` dev+preview proxy；`App.tsx` 默认 `${origin}/bridge`；`scripts/serve_web_demo.sh` |
| A4 DeckBuilder 传 AssetsManager | ✅ | `App.tsx` `npcAssetsManager` memo → `DeckBuilder` / `DeckPreview` |
| A5 `common/llm.py` 健壮性 | ✅ | 429/5xx `Retry-After` 重试、空内容守卫、`cached_tokens`/`cost_usd` 记账、脱敏前缀、UA；`tests/test_llm_http.py` 17 passed |
| A6 shitou `win_within_rounds(44)` | ✅ | `npc_stages_mond_liyue.json` → **4**（按 HoYoWiki；不是源仓库的 5） |
| A7 fatuipyroagent `hp_card` 9 vs 引擎 11 | ✅ | 经 B2 变体 **9301=9 HP** 解决：保留本仓库 PvE 数据，不动 `hp_card` |
| A8 web 测试断言 | ✅ | `test_web_rpc_server.py` 按 `method=="decklab-rule"` 分支 |

## A.2 第二节：新 feature

| 项 | 状态 | 落点 / 证据 |
|---|---|---|
| B1 关卡特殊规则 | ✅ | `patches/stage-rules.patch` + `overlay/packages/{core,cbinding}` + `configs/npc/special_rules.json` + `npc/stage_rules.py` + `envs/match.py` + `npc/eval.py`；`test_stage_rules` 10 + `test_stage_rules_engine` 2；smoke：blanche 全体 +2 充能、timmie 支援 322007、chefmao 双方附火；`npc.stage_rules --verify` 16 关 errors=[] |
| B2 NPC 变体拆分 | ✅ | `fatui_pyro_agent_npc.gts`(9301)=**9 HP**、`mirror_maiden_npc.gts`(9202)=**10 HP**（本仓库 PvE 数据）；官方 2301=11 / 2202=10 不覆盖；`engine.lock` 的 `.so` sha `da8e2020…` 与源仓库 `69edc539…` 不同 |
| B3 `engine.lock [overlay]` | ✅ | `overlay_tree_sha256=3c0c1846…`、`libgitcg_sha256=da8e2020…`、`libgitcg_bytes=36,632,368` |
| B4 DeckPreview | ✅ | `DeckPreview.tsx`（与源逐字节一致）+ App 双列预览 |
| B5 `import_deck.py` | ✅ | 自包含（无 `era_decks` 依赖）；`tests/test_import_deck.py` 16 passed；真实分享码 dry-run 解码 3 角色 + 30 牌、规则 OK、`--probe` 版本诊断可用 |
| B6 `probe_stall.py` | ✅ | `PolicyPlayer` 子类抓公共 mutation 流 + `TracePolicy` 快照；`--check-round-cap` 固定「`ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT=105`、默认 15、`run_match` 不设置」；真实对局与 raw 重放均有 trace；`tests/test_probe_stall.py` 5 passed |
| B7 事实性更正 | ✅ | timmie `2492→1335`、timaeus `2547→196`；13 关 `stage_title` 补全（16/16）；`timaeus` 规则误读与 `removable` 推断写入 `docs/howto/npc-pve.md` |

## A.3 刻意未移植 / 改口径

- **不采用**源仓库的 D4 敌人血量标定（≤6→6、其余 +2）与变体 13/12 血量；变体血量取本仓库 PvE 数据 **9/10**。
- **swan（补目标）/ lan（6→5）不改**：与源仓库自己的 HoYoWiki 审计矛盾（§5.5），我们的数据与 wiki 一致。
- **web 关卡规则接线**（`stage-rules.ts` + `eval.rules` 回填）暂不移植：本仓库桥没有 eval 协议/规则数据源；引擎与 Python 侧已完整可用，接桥时再补。
- 未搬：`web/README.md` 重写、`SHOW_UNFINISHED`、`humanSeat:1`、LLM 面板/`/ai*` 等（第三节）。
- 环境修复（非移植）：`gitcg` env 的 `libgitcg.so` 曾被源仓库的 D4 构建覆盖（我们的 `test_npc_enemies` 因此失败）；已换回本仓库构建，随后按 B1/B2 重建安装。

## A.4 验证

- 引擎补丁：`git -C build/engine apply --check` 通过；9 个改动文件与源仓库 overlay 逐字节一致；构建增量重链 ~4s，`.so` 36,632,368 bytes。
- `tsc --noEmit`（`build/engine/packages/standalone`）exit 0。
- `npc.stage_rules --verify`：16 关、`errors=[]`。
- 全量：`pytest -m "not slow"` → **476 passed, 63 deselected, 6 warnings in 68.06s（EXIT=0）**。
