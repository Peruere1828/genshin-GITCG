# 文档审计与整理建议（2026-10-27）

> 范围：仓库内全部项目 `*.md`（含工作区未提交的 `reports/explore/lit/` 与 `reports/explore/literature_review.md`），不含 `refs/` 上游文档、`build/` 下的生成副本，也不计 `.pytest_cache/README.md`（pytest 缓存）。
> 方式：只读；以 `git log -1 --format=%ad` 的最后提交日 + 文档内标注日期交叉判断 freshness；与代码/报告冲突处给 `file:line`。
> 结论口径：**PLAN.md 是需求/验收唯一权威**；本报告只提整理建议，不改任何文档/代码/数据。
>
> **执行状态（2026-10-27）**：部分 P0/P1 已落地——`docs/` 骨架、`AGENTS.md`/`README.md` 重写、
> `PLAN.md §9/§10` 更新、A2 更正头、D25/D26、`docs/EVAL_PROTOCOL.md` 迁移、`docs/failures/` 与
> `docs/environments/` 初版。仍待人工审阅：模块 README 的状态剥离、全部报告的 front-matter、
> `docs/PLAYBOOK.md` 的门禁措辞。以下保留审计快照。

---

## 0. 结论摘要

1. **结构问题大于单篇质量问题**：当前没有 `docs/` 目录，也没有 `STATUS`、`PLAYBOOK`、`failures/`、`environments/` 四类文档；状态、成功经验、失败经验、机器特化全部挤在 `AGENTS.md`、`PLAN.md §10` 和几篇长报告里。
2. **`AGENTS.md` 事实上的内容止于 2026-10-15**（最后提交 10-16），但 10-16 之后项目已经换过主线：从“数据先行/BC 规模”转为“逐 deck 剥削阶梯 + specialist 集合”。`AGENTS.md:20` 仍写“数据先行 + 顶过专家基线 0.561”，与 `reports/EXPLOIT_LADDER.md:15`、`reports/SPECIALISTS.md:8-9` 的当前交付（dgeo 0.880 / ungeo 0.909）冲突。
3. **`PLAN.md §10` 停在 2026-10-12**（`PLAN.md:363` 仍写“策略改进——待选向”），没有追加 10-16 之后的关键决策（PG/剥削阶梯、A2 复核无效、Kaggle 作为训练节点）。本轮 P0 已在 §0 追加 D25/D26；§10 状态表仍待更新。
4. **A2 的结论曾处于“两份同日文档互相矛盾”的状态**：`reports/explore/a2_id_screening.md:8` 写“A2 失败”，`reports/explore/literature_review.md:65,430-435` 写“A2 判负无效（fp16 definition id 事故）”。本轮 P0 已在 A2 报告顶部追加 `status: invalid` 更正头，并在 PLAN §0 追加 D25/D26；AGENTS 仍未提 A2（待文档重组）。
5. **D23 的“搜索显著有害”需要加限定**：`PLAN.md:33` 与 `reports/search/a4_search_budget_ladder.md:22` 的结论基于当时的根价值测量；`reports/explore/literature_review.md:97,246` 指出 `envs/cfr_probe.py` 把负价值截断成 0，可能使“区分度≈0”成为测量假象。**“不放量搜索”的排产决策可以保留，但“搜索有害”这一因果结论应标为待复核。**
6. **数字没有单一事实源**：同一个“专家基线”在不同文档里是 0.519（20 seeds/n=800）与 0.561（A4 n=400）；“敌人数量”有 13/17/19 三种；“A800”有“128 核”和“10 核配额”两种；“测试数”有 186/234/384 三种。读者无法判断哪个是当前口径。
7. **索引已经腐烂**：`tests/README.md:5` 只列 28 个测试，实际有 51 个 `tests/test_*.py`；`scripts/README.md:5` 停留在 10-07，漏掉 10 个以上新脚本；`reports/README.md:5-17` 漏掉 `SPECIALISTS`、`EXPLOIT_LADDER`、`npc/`、`lit/` 等；`kaggle/README.md:13` 的 mode 列表已与实际 kernel 模式脱节。
8. **存在失效命令/失效数字**：`scripts/README.md:3` 与 `README.md:27` 宣称 `run_iteration` 存在，但仓库没有 `scripts/run_iteration.py`，同文件 `:60` 又写“待办”；`scripts/cluster/A800_RUNBOOK.md:22` 写“应 ~186 passed”，已被后续 234/384 的实测覆盖。
9. **成功/失败混在一起**：`reports/explore/plan_strength_2day_kaggle.md` 585 行，既是计划、执行日志、结果、失败记录，又是“下一步”；其中 `:575-584` 还残留 2026-10-19 的“下一步”，放在 10-27 的内容之后，直接与 `:344` 的“20-deck 单模型 7/8 变差”冲突。
10. **机器特化没有落点**：`/home/ubuntuaaa/miniconda3/...`、`/root/blockdata/...`、`$HOME/.conda/envs/fa_env` 散落在报告命令里；A800 的实际工程路径（`a800_window_20261012.md:31` 的 `/root/blockdata/genshin-GITCG`）与 Runbook（`A800_RUNBOOK.md:59` 的 `~/genshin-GITCG`）不一致；本机 6 物理核/12 线程、30 GiB、无 GPU 的事实没有一份可引用的环境文档。

---

## 1. 标签与判断口径

| 标签 | 含义 | 典型读者问题 |
|---|---|---|
| `canonical` | 权威规格/验收/协议，冲突时以它为准 | “到底按哪个口径做？” |
| `status` | 当前状态/路线，允许过期但必须标 freshness | “现在做到哪、下一步是什么？” |
| `howto` | 命令、运维、复现步骤 | “这条命令怎么跑？” |
| `environment` | 机器特化：路径、硬件、网络、吞吐、并发、坑 | “这台机器上怎么跑？” |
| `report` | 一次性实验证据/测量 | “当时测到了什么？” |
| `failure` | 负结果/无效实验，必须带“重开条件” | “为什么失败、什么条件下再做？” |
| `generated` | 脚本生成，禁止手改 | “这个文件能不能编辑？” |
| `archive` | 已被取代，保留供追溯 | “这段历史还有效吗？” |

`freshness` 取“最后一次与代码/数据核对”的日期；没有核对记录的，用报告内标注日期。`machines` 取值：`通用`、`本机`、`WSL`、`A800`、`NJU`、`Kaggle`。

---

## 2. 全量盘点（63 份 Markdown）

### 2.1 根目录（4）

| 文档 | 标签 | freshness | machines | overlap / 备注 |
|---|---|---|---|---|
| `AGENTS.md` | `status`+`howto`+`environment`（混合） | 2026-10-16（内容止于 10-15） | 本机为主，夹带 WSL/A800/NJU | 与 PLAN §3/§4、README、web/README、train/README、scripts/cluster 大量重叠；第 20–24 行是状态，18/30/34–36 是机器特化，第 22 行单行 2072 字符 |
| `PLAN.md` | `canonical` | 2026-10-16（决策止于 10-12） | 通用 | 唯一需求/验收权威；§10 过期，§9 目录树缺 `web/`、`npc/`、`common/`、`kaggle/`、`scripts/` |
| `README.md` | `status`+`howto` | 2026-10-10 | 通用（示例为本机） | 与 AGENTS/PLAN 重叠；`D15` 仍称“当前主线”，目录表缺 5 个目录 |
| `NOTICE.md` | `canonical` | 2026-10-14 | 通用 | 许可/移植范围；与 `reps/README`、`agents/README` 的移植说明重叠，无冲突 |

### 2.2 模块与主题 README（17）

| 文档 | 标签 | freshness | machines | overlap / 备注 |
|---|---|---|---|---|
| `agents/README.md` | `howto`+`status` | 2026-10-07 | 通用 | 仍写“待办 resolving_agent”；实际搜索/resolver 已在 `envs/` + `train/` 落地 |
| `coach/README.md` | `status` | 2026-10-07 | 通用 | 未接训练/搜索值；与 PLAN WS4 重复 |
| `common/README.md` | `howto` | 2026-10-16 | 通用 | 环境变量契约的实际权威；应成为 env 文档的引用源 |
| `configs/README.md` | `howto`+`canonical` | 2026-10-16 | 通用 | `engine.lock` 口径权威；与 PLAN D4 重复 |
| `configs/npc/README.md` | `canonical` | 2026-10-15 | 本机 | **敌人/关卡数据权威**：19 类敌人、16 关、schema 未覆盖项；与 `reports/explore/npc_pve_doc.md` 重叠且后者更旧 |
| `configs/pools/README.md` | `canonical` | 2026-10-11 | 通用 | 池 C（19 角色/107 牌）权威；与 `reports/explore/mond_liyue_pool.md` 重叠 |
| `decklab/README.md` | `howto`+`status` | 2026-10-12 | 通用 | 明确“规则组牌不是 Agent 组牌”；与 D19 目标措辞有差距 |
| `envs/README.md` | `howto`+`status` | 2026-10-10 | 通用 | 与 PLAN WS0、AGENTS 工程约定重复；`rollout.py` 已扩到 `stream_task_shard` 未提 |
| `eval/README.md` | `howto`+`status` | 2026-10-11 | 通用 | 与 `reports/explore/eval_protocol.md` 重复；说 TrueSkill “待办”但已用 Bradley-Terry |
| `npc/README.md` | `canonical`+`howto` | 2026-10-16 | 本机 | PvE 代码入口；与 configs/npc、engine-overlay、reports/npc 三方重叠 |
| `orchestrator/README.md` | `status`（占位） | 2026-10-07 | 通用 | 3 行占位；应明确“未实现”，防止被当作现状 |
| `reps/README.md` | `howto`+`status` | 2026-10-07 | 通用 | 10-07 后新增 `belief.py`、`id_features.py` 等未列；与 NOTICE 重叠 |
| `tests/README.md` | `howto`+`status` | 2026-10-12 | 通用 | **只列 28 个测试，实际 51 个 `test_*.py`**；缺 web/offline-rl/payload/parity 等 |
| `train/README.md` | `howto`+`status` | 2026-10-17 | 本机/A800/WSL | 198 行，`“已实现（L4，2026-10-07）”` 标题下混入 10-17 的 H2 内容；未写 A3/A4 负结果 |
| `train/CONTINUAL_RESOLVING.md` | `canonical`（设计）+`status` | 2026-10-11 | 本机/WSL | 10-11 后未更新；仍把 search-as-teacher 当主线第 3 步，与 D23“暂缓搜索”冲突 |
| `web/README.md` | `canonical`+`howto` | 2026-10-16 | 本机 | WS8 协议权威（端点/隔离/日志/C3 差异）；与 PLAN WS8、AGENTS:22、standalone README 重叠 |
| `scripts/README.md` | `howto`+`status` | 2026-10-11 | 本机/集群 | `“现状 2026-10-07”` 已腐烂；`run_iteration` 不存在；漏掉 `run_cfr_ladder`、`datafirst_bc_*`、`specialist_table`、`kaggle_*` 等 |

### 2.3 运维/环境/构建（6）

| 文档 | 标签 | freshness | machines | overlap / 备注 |
|---|---|---|---|---|
| `scripts/cluster/README.md` | `howto`+`environment` | 2026-10-14 | NJU | 顶部写“A800 80G + 128 核”，与 D21 的 10 核配额冲突；NJU 环境事实仍准确 |
| `scripts/cluster/A800_RUNBOOK.md` | `howto`+`environment` | 2026-10-14 | A800 | 与 `reports/explore/a800_window_20261012.md` 重叠；`~186 passed` 与 `~/genshin-GITCG` 路径需修 |
| `engine-overlay/README.md` | `canonical`+`howto`+`environment` | 2026-10-12 | 本机/WSL | 引擎 pin/V8/构建 provenance 权威；与 overlay README、D20 重叠 |
| `engine-overlay/overlay/README.md` | `canonical`+`howto` | 2026-10-12 | 本机 | 17 个 `.gts` + 2 上游复用 = 19 类；与 configs/npc 的“19 类”需统一措辞 |
| `engine-overlay/overlay/packages/standalone/README.md` | `canonical`+`howto` | 2026-10-16 | 本机 | WS8 overlay 细节权威；与 `web/README.md` 有功能重叠 |
| `kaggle/README.md` | `howto`+`environment` | 2026-10-17 | Kaggle | 只写训练定位与产物；mode 列表停在 `auto/probe/smoke/runs/h4`，缺 `cap/wave1/wave2/groupb/groupc` |

### 2.4 数据目录说明（3）

| 文档 | 标签 | freshness | machines | overlap / 备注 |
|---|---|---|---|---|
| `data/assets/README.md` | `howto` | 2026-10-11 | 通用 | 资产缓存路径/来源权威；与 `agents/README` 的加载说明重叠 |
| `data/web/README.md` | `howto` | 2026-10-15 | 本机 | C4 日志 schema；与 `web/README.md` C4 段重叠 |
| `data/deck_corpus/README.md` | `howto` | 2026-10-11 | 通用 | 语料 schema/provenance；与 `reports/explore/deck_corpus.md` 重叠 |

### 2.5 报告（33）

| 文档 | 标签 | freshness | machines | overlap / 备注 |
|---|---|---|---|---|
| `reports/README.md` | `howto` | 2026-10-11 | 通用 | 索引腐烂：漏 `SPECIALISTS`、`EXPLOIT_LADDER`、`npc/`、`lit/`；无 frontmatter 规范 |
| `reports/LOCAL_BASELINE.md` | `report`+`status` | 2026-10-14 | 本机 | L0/L1/L5 基线；`12 worker 突发 2984` 与 `排产 4900` 并存，需定义稳态/突发 |
| `reports/EXPLOIT_LADDER.md` | `status`+`howto`（成功经验） | 2026-10-26 | 本机/A800 | **当前最有效的强度方法**；应升级为 `docs/PLAYBOOK.md` 的核心条目，报告保留为证据 |
| `reports/SPECIALISTS.md` | `generated` | 2026-10-27 | 本机 | 自动生成（`scripts/specialist_table`）；2 个 full PASS；需标 `do_not_edit` 与生成时间 |
| `reports/datafirst_bc.md` | `report`+`failure` | 2026-10-14 | 本机 | 多 deck BC 认证；结论“BC 从未显著超过专家”；与 `bc_scale_datafirst` 互相印证 |
| `reports/web_standalone_start.md` | `report` | 2026-10-12 | 本机 | C0 阶段小结；后续 C1/C3/T1–T4 已超越，需 `superseded` 标注 |
| `reports/npc/README.md` | `howto` | 2026-10-12 | 本机 | NPC 报告索引；与 `npc/README` 重叠 |
| `reports/npc/pve_v1_gate.md` | `report`+`generated` | 2026-10-14 | 本机 | PvE-v1 PASS 74.7%；但含 swan/石头/特殊规则未核实，PLAN 未带限定 |
| `reports/npc/npc_pve_smoke_v1.md` | `report`（已被 gate 取代） | 2026-10-12 | 本机 | B2/B3 冒烟；被 `pve_v1_gate` 取代，保留作历史 |
| `reports/train/a3_m3_cpu_screening.md` | `report`+`failure` | 2026-10-14 | 本机 | M3-CPU 未过；distill≈BC；D22 证据 |
| `reports/train/a8_collect_streaming_memory.md` | `report`+`howto`（修复） | 2026-10-12 | 本机 | 采集内存修复；应进 PLAYBOOK/howto，不应埋在报告里 |
| `reports/train/bc_scale_datafirst.md` | `report`+`failure` | 2026-10-14 | 本机 | 48k/15ep 仍显著低于专家；0.519 vs 0.561 的混淆源 |
| `reports/train/collect_throughput_notes.md` | `report`+`howto` | 2026-10-14 | 本机/A800 | compact 格式与采集口径；把 0.561 当 BC 天花板，需改 |
| `reports/search/a4_search_budget_ladder.md` | `report`+`failure` | 2026-10-14 | 本机 | D23 核心证据；结论“显著有害”待 cfr_probe 复核 |
| `reports/explore/a2_id_screening.md` | `report`+`failure`（**无效**） | 2026-10-27 | 本机/A800 | 结论“A2 失败”，但 `literature_review` 判定实验无效；需顶部更正 |
| `reports/explore/a800_window_20261012.md` | `report`+`environment` | 2026-10-12 | A800 | CPU 10 核、37G 存储、1.64MB/局；A800 环境事实权威之一 |
| `reports/explore/deck_corpus.md` | `report`+`generated` | 2026-10-11 | 通用 | 73 套公开卡组；由 `render_deck_report.py` 生成 |
| `reports/explore/eval_protocol.md` | `canonical` | 2026-10-11 | 通用 | **强度评测协议权威**；当前放在 reports 下，建议迁 `docs/EVAL_PROTOCOL.md` |
| `reports/explore/literature_review.md` | `report`+`status`（提案） | 2026-10-27（工作区未提交） | 通用 | 对 A2/D23/D24 的交叉裁决；D25/D26 已进 PLAN §0；信息量最高但位置太深 |
| `reports/explore/model_design.md` | `report`（部分被取代） | 2026-10-11 | WSL | MX550/Transformer 早期结论；后续 transformer 长训结论已变化 |
| `reports/explore/mond_liyue_pool.md` | `report` | 2026-10-11 | 通用 | 区域校对；与 `configs/pools` 重叠，后者是选定口径 |
| `reports/explore/npc_lineups_mond_liyue.md` | `report` | 2026-10-11 | 通用 | 16 关阵容来源；数据权威在 `configs/npc` |
| `reports/explore/npc_pve_doc.md` | `report`（自称现行，实际过时） | 2026-10-11 | 本机 | node 26/v8-cmake-barbara 与现状冲突；待办已全部完成 |
| `reports/explore/plan_strength_2day_kaggle.md` | `report`+`status`+`failure`（战役日志） | 2026-10-27 | 本机/Kaggle/A800 | **事实上的当前状态**；但 585 行混排，`:575-584` 残留旧下一步 |
| `reports/explore/strength_levers_and_mx550.md` | `report`（已被取代） | 2026-10-17 | 本机/WSL | 仍把 AWR/Q 当“最便宜杠杆”；10-17 后已被证否 |
| `reports/explore/lit/brief.md` | `report`（调研上下文） | 2026-10-27（未提交） | 通用 | 7 个子报告的共享 brief；汇总后应标 archive |
| `reports/explore/lit/f01_selfplay_value.md` | `report` | 2026-10-27（未提交） | 通用 | DouZero/Suphx/ReBeL/SoG；价值语义与 belief |
| `reports/explore/lit/f02_dagger_offline.md` | `report` | 2026-10-27（未提交） | 通用 | DAgger/AWR/Q/offline RL；AWR 定义级失败证据 |
| `reports/explore/lit/f03_league_portfolio.md` | `report` | 2026-10-27（未提交） | 通用 | league/PFSP/specialist router |
| `reports/explore/lit/f04_set_actions.md` | `report` | 2026-10-27（未提交） | 通用 | 候选集 attention；结论“先 0 成本诊断” |
| `reports/explore/lit/f05_train_eval.md` | `report` | 2026-10-27（未提交） | 通用 | parity harness、校准、评测统计 |
| `reports/explore/lit/f06_targeted_data.md` | `report` | 2026-10-27（未提交） | 通用 | worst-matchup/对手条件化；10 个严格可修 matchup |
| `reports/explore/lit/f07_code_audit.md` | `report` | 2026-10-27（未提交） | 通用 | store→engine 代码审计；A2 fp16 根因 |

> 说明：`build/kaggle/out*/repo/AGENTS.md`、`PLAN.md` 等是 Kaggle bundle 的生成副本（非 git 跟踪），会与主仓文档漂移；审计/搜索时应排除，不能当第二事实源。

---

## 3. 具体问题清单（23 条）

### P0：会误导当前决策

**P0-01｜`AGENTS.md` 主线过期**
- 证据：`AGENTS.md:20` 仍写“数据先行 + BC/自博弈规模化 + 更大网络顶过专家基线 0.561”；`reports/EXPLOIT_LADDER.md:15` 与 `reports/SPECIALISTS.md:8-9` 已给出 dgeo 0.880 / ungeo 0.909 的 full PASS；`reports/explore/plan_strength_2day_kaggle.md:366-405,514-523` 记录了 PG 三轮与 20-seed 复验。
- 影响：新 agent 会按 10-12 的排产继续堆 BC，而不是维护 specialist/做 parity。
- 建议：AGENTS 只留链接；当前状态由 `docs/STATUS.md` 维护；旧决策不改写。

**P0-02｜PLAN §10 与决策记录缺 10-16 之后的内容**
- 证据：`PLAN.md:363` “策略改进（自博弈/RL 或修好的搜索）——待选向”；`PLAN.md:305-311` §6 只有 1–5 项；`reports/explore/literature_review.md:428-450` 已进 PLAN §0（D25/D26）。
- 影响：PLAN 作为唯一权威却落后于实际交付；A2 无效、specialist PG、Kaggle 训练节点均无决策记录。
- 建议：D25/D26 已追加；仍建议追加 D27（逐 deck 剥削阶梯作为当前交付）与 D28（Kaggle 仅训练节点）；§10 只更新“当前状态/下一步”，历史保留。

**P0-03｜A2 报告已追加 invalid（P0 已执行）**
- 证据：`reports/explore/a2_id_screening.md:8` “A2 在胜率口径下失败”；`reports/explore/literature_review.md:65,430-435` “A2 判负无效（fp16 definition id 事故）”。本轮 P0 已在 A2 报告顶部追加更正头。
- 影响：后来者会把 A2 当“id embedding 有害”的证据，重复关闭路线。
- 建议：已执行“A2 顶部追加 `status: invalid`”；STATUS 仍待建，写“A2 invalid，待 P0 parity 后重做”。

**P0-04｜D23 “搜索显著有害”与测量 bug 冲突**
- 证据：`PLAN.md:33`、`reports/search/a4_search_budget_ladder.md:22` 的“显著为负”；`reports/explore/literature_review.md:97,246` 指出 `envs/cfr_probe.py` 把负价值截断，根价值区分度≈0 可能是测量假象；`reports/explore/plan_strength_2day_kaggle.md:186-190` 的 H3b 也复现了同档结果。
- 影响：把“测量坏”误判为“路线死”，或反之贸然重开。
- 建议：保留“暂不把搜索蒸馏放量”的排产决策；把因果结论降级为“在当前测量/预算下无增益”，重开条件写进 `docs/failures/001-search-as-teacher.md`。

**P0-05｜专家基线 0.519 vs 0.561 混用**
- 证据：`AGENTS.md:20` 用 0.561；`reports/train/bc_scale_datafirst.md:16` 说“顶过 0.561 未达成”，`:29` 同种子专家是 0.519（20 seeds/n=800）；`reports/train/collect_throughput_notes.md:102,111,115` 又把 0.561 当 BC 天花板。
- 影响：所有“差几个点”的结论都随基线漂移。
- 建议：`docs/STATUS.md` 建基线表，强制写 `deck / policy / seeds / n / pooled / Δ / floor`；0.519 与 0.561 作为两个不同测量并列，禁止裸写“专家基线”。

**P0-06｜D24 “训练时长是杠杆，不是数据量”已被后续修正**
- 证据：`PLAN.md:34` 的结论；`reports/explore/plan_strength_2day_kaggle.md:531-536` 用稠密语料把 Group C 从 −0.13~−0.31 拉到 −0.08~+0.00（+0.06~+0.32）；`:545-567` 判定“每 matchup 局数是主因”，`:558-561` 明确“密度是主因，任务数只有小残留”。
- 影响：D24 被当成“不要再补数据”，与后来最有效的“按 matchup 补密度”相反。
- 建议：D24 保留为历史决策；追加更正：“当时 8k→48k 的‘总量’不显著，不等于‘每 matchup 密度’不显著”；STATUS 用新结论。

**P0-07｜敌人数量 13/17/19 三种写法**
- 证据：`PLAN.md:27,248` 与 `AGENTS.md:21` 写“约 13 类”；`engine-overlay/README.md:190` 写“17 个 NPC 敌人 `.gts`”；`configs/npc/README.md:10` 写“19 类”；`engine-overlay/overlay/README.md:14,27-29` 写“17 个 + 2 个上游复用 = 19 类”。
- 影响：PvE 范围与验收口径说不清。
- 建议：统一为“19 类敌人（17 个新增 `.gts` + 2 个上游复用）”；PLAN D17 不改历史，但在 §4.4 状态行加注释。

**P0-08｜A800 “128 核”与“10 核配额”并存**
- 证据：`AGENTS.md:18`、`README.md:6`、`scripts/cluster/README.md:3` 写“A800 80G + 128 核”；`PLAN.md:31`、`scripts/cluster/A800_RUNBOOK.md:7`、`reports/explore/a800_window_20261012.md:9,20-25` 写“宿主 128 核、cgroup 10 核”。
- 影响：读者会以为窗口有 128 核可用，错误排产采集/评测。
- 建议：统一写“宿主 128 核 / 容器配额 10 核”；只在 environments/a800.md 解释 `nproc` 与 `cpu.max` 的区别。

**P0-09｜`run_iteration` 是不存在的命令**
- 证据：`scripts/README.md:3` 与 `README.md:27` 宣称 `run_iteration`；同文件 `scripts/README.md:60` 写“待办”；仓库无 `scripts/run_iteration.py`（`ls` 失败）。
- 影响：一键窗口流程无法照文档执行。
- 建议：删除/改成实际入口（`scripts/kaggle_push.sh`、`train.collect` + `train.pipeline` + `eval_protocol` 组合），或实现脚本后再写。

**P0-10｜测试数 186/234/384 三种硬编码**
- 证据：`scripts/cluster/A800_RUNBOOK.md:22` “应 ~186 passed”；`reports/train/a8_collect_streaming_memory.md:148` “234 passed”；`reports/explore/plan_strength_2day_kaggle.md:140` “384 passed”；当前 `tests/test_*.py` 为 51 个。
- 影响：环境检查会误报失败。
- 建议：文档只写命令 `python -m pytest -m "not slow" -q`，不写数字；若必须写，标日期与 commit。

### P1：事实/索引/引用错误

**P1-11｜node 26 / `v8-cmake-barbara` 与现状冲突**
- 证据：`reports/explore/npc_pve_doc.md:47` 写 node 26 + `piovium/v8-cmake-barbara`；`engine-overlay/README.md:11,45,122` 写 node 24 + `piovium/v8-cmake`，pin `v0.21.0`；`engine-overlay/README.md:30-35` 明确 `v8-cmake-barbara` 是 refs HEAD 的路线，不是 v0.21.0。
- 影响：按旧文档构建会走错 V8 路线。
- 建议：`npc_pve_doc.md` 标 `superseded`，把构建事实收敛到 `engine-overlay/README.md` + `versions.env`。

**P1-12｜A800 工程路径不一致**
- 证据：`reports/explore/a800_window_20261012.md:31` 写 `/root/blockdata/genshin-GITCG`、`:32` 数据落 blockdata；`scripts/cluster/A800_RUNBOOK.md:59` 用 `~/genshin-GITCG/data/replays/`；`reports/explore/a2_id_screening.md:90` 用 `/root/blockdata/conda/gitcg/bin/python`。
- 影响：scp/续跑命令照抄会找不到目录。
- 建议：environments/a800.md 定义 `$REPO`、`$PY`、`$DATA`；报告命令只引用变量。

**P1-13｜目录树缺目录**
- 证据：`PLAN.md:339-355` 缺 `web/`、`npc/`、`common/`、`kaggle/`、`scripts/`；`README.md:15-28` 缺 `web/`、`npc/`、`engine-overlay/`、`kaggle/`、`common/`。
- 影响：新人找不到三线并行的代码位置。
- 建议：以实际 `ls` 为准补全；目录说明只保留一行职责。

**P1-14｜`PLAN §6.2/§6.3/§6.5` 是不存在的章节**
- 证据：`AGENTS.md:62-63`、`scripts/README.md:31`、`reports/explore/eval_protocol.md:18`、`engine-overlay/README.md:6`、`engine-overlay/overlay/README.md:5` 引用；`PLAN.md:305-311` §6 只有 1–5 五个编号项，没有子节。
- 影响：交叉引用全部断链。
- 建议：二选一：① 把 §6 改成 6.1–6.5 子节并保留编号；② 把所有引用改成 `PLAN §6 第 3 条` 这类锚点。前者更利于稳定链接。

**P1-15｜索引腐烂（tests/scripts/reports/kaggle）**
- 证据：`tests/README.md:5` 只列 28 个测试，实际 51 个；`scripts/README.md:5` 停在 10-07；`reports/README.md:5-17` 未列 `SPECIALISTS`/`EXPLOIT_LADDER`/`reports/npc/`/`reports/explore/lit/`；`kaggle/README.md:13` 的 mode 列表缺 `cap/wave1/wave2/groupb/groupc`。
- 影响：读者找不到新产物，或按旧 mode 触发事故（`plan_strength:463-464` 已发生过一次）。
- 建议：索引由脚本生成或 CI 校验；每个目录 README 只做“入口 + 链接”，不复制清单。

**P1-16｜`plan_strength` 残留旧“下一步”**
- 证据：`reports/explore/plan_strength_2day_kaggle.md:575-584` 标题为“下一步（2026-10-19 修订）”，却排在 10-27 的内容之后；`:577` 说 all-deck 单模型“正在跑”，与 `:344` “8 个里 7 个变差”冲突。
- 影响：读者读到文末会得到与正文相反的当前状态。
- 建议：把该节移到 `docs/STATUS.md`（重新核对），原报告标 `archived`，保留时间线。

**P1-17｜过时研究备忘仍在推荐已证否方法**
- 证据：`reports/explore/strength_levers_and_mx550.md:15,52,147` 把 AWR/Q/离线 RL 列为“最便宜杠杆”；`reports/explore/plan_strength_2day_kaggle.md:245-246,303` 已归档 AWR/Q；`reports/explore/model_design.md:12` 说 Transformer“不划算”，`:36` 的 2L 0.758 只是 2k 样本，后续 `plan_strength:353,386` 记录 transformer 43ep val_acc 0.737 > CNN 150ep 0.693。
- 影响：读者会把旧结论当现行建议。
- 建议：两篇顶部加 `status: superseded`，指向 `docs/failures/002-awr-q-dmc.md` 与 `docs/STATUS.md`；不删原文。

**P1-18｜PvE-v1 PASS 未带限定条件**
- 证据：`PLAN.md:252` 写“PASS（74.7%）”；`reports/npc/pve_v1_gate.md:65-69` 列出 swan 只有 1 条目标、石头 `win_within_rounds` 疑笔误、蒂玛乌斯/卯师傅特殊规则未在引擎模拟、decklab 非 Agent 组牌；`configs/npc/README.md:46-50` 同样列待核。
- 影响：把“条件性 PASS”当成“PvE 全机制通过”。
- 建议：STATUS 写“PvE-v1 conditional PASS（口径见 gate §未决问题）”；PLAN §4.4 不改历史数字，但加“限定条件”链接。

**P1-19｜WSL 的 d_model ≤64 限制已过时**
- 证据：`PLAN.md:266,296,323` 写 d_model ≤64；`reports/explore/model_design.md:3,36` 实测 d128/2L transformer（431k 参数，265MB）；`reports/explore/strength_levers_and_mx550.md:100-111` 记录 d128 可训、d256 mean-pool 也够。
- 影响：低估 WSL 用途或写错试验配置。
- 建议：改为“d_model ≤128、2L、fp32、batch 32–64；d256 可试但无余量”；标“未做 d256 系统实测”的边界。

**P1-20｜机器路径散落在报告里**
- 证据：`reports/train/a3_m3_cpu_screening.md:28` `/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python`；`reports/explore/a2_id_screening.md:90` `/root/blockdata/conda/gitcg/bin/python`；`scripts/cluster/README.md:17` `$HOME/.conda/envs/fa_env`；`reports/datafirst_bc.md:3` 与 `PLAN.md:265` 的 30G/32GB 表述不统一（实测 `free -h` 为 30GiB）。
- 影响：复制命令跨机失败；无法判断“30G”是内存还是磁盘。
- 建议：建 `docs/environments/*`，报告只写 `$PY`/`$REPO`/`$DATA`；本机统一写“6 物理核/12 线程、30 GiB、无 GPU”。

### P2：结构/维护性

**P2-21｜`build/kaggle/out*/repo/` 的生成副本污染搜索**
- 证据：`build/kaggle/out/repo/AGENTS.md`、`PLAN.md` 等与主仓同名；grep 会命中旧内容（例如同一 `§6.3` 断链）。
- 影响：审计/agent 可能引用旧副本。
- 建议：文档工具默认排除 `build/`；CI/搜索脚本加 `--exclude-dir=build`；不手改这些副本。

**P2-22｜报告没有 front-matter，generated 与手写混排**
- 证据：`reports/SPECIALISTS.md:3` 自述自动生成，但没有 `do_not_edit`/生成时间/输入 glob；`reports/explore/deck_corpus.md:3` 由脚本渲染；`reports/npc/pve_v1_gate.md` 既有生成数字又有人工“未决问题”；`reports/explore/plan_strength...` 是手写长日志。
- 影响：无法机器索引，也无法判断“哪份能引用”。
- 建议：统一 front-matter（见 §6）；generated 文档加 `generated_by/do_not_edit`；手写注解放独立小节并标 `manual_note: true`。

**P2-23｜模块 README 仍是 10-07 快照**
- 证据：`agents/README.md:6`、`coach/README.md:5`、`reps/README.md:7`、`envs/README.md:6` 都以“已实现（2026-10-07）”开头；`envs/README` 未提 `stream_task_shard`，`reps/README` 未提 `belief.py`/`id_features.py`。
- 影响：模块文档与代码脱节，读者只能读源码。
- 建议：模块 README 只保留“接口/文件职责/测试入口”，状态移到 `docs/STATUS.md`；每季度/每次合并前核对一次。

---

## 4. 权威源冲突与建议裁决

| 冲突事实 | 现有说法（位置） | 建议权威 | 处理 |
|---|---|---|---|
| 当前最强策略 | `AGENTS.md:20` 数据先行/0.561 vs `EXPLOIT_LADDER.md:15`/`SPECIALISTS.md:8-9` dgeo 0.880、ungeo 0.909 | `reports/SPECIALISTS.md` + `EXPLOIT_LADDER.md`（有 20-seed 认证） | 写入 STATUS；AGENTS 只链接 |
| A2 是否成立 | `a2_id_screening.md:8` 失败 vs `literature_review.md:65,430-435` 无效 | `literature_review.md`（更新、含代码级根因） | A2 标 `invalid`，待 parity 重做 |
| 搜索是否“有害” | `PLAN.md:33`/`a4:22` 显著负 vs `literature_review.md:97,246` 测量 bug | 暂以“不放大”为排产，因果结论待 P0-F | 追加限定，不改 D23 历史 |
| 敌人数量 | 13（PLAN/AGENTS）/17（engine-overlay）/19（configs/npc） | `configs/npc/README.md:10`（数据）+ `engine-overlay/README.md:190`（实现） | 统一“19 类=17 新增+2 复用” |
| A800 CPU | 128 核（AGENTS/README/cluster README）vs 10 核（D21/A800 报告） | `PLAN.md:31` + `a800_window_20261012.md:9` | 写“宿主 128/配额 10” |
| 当前评测协议 | `eval/README`、`reports/explore/eval_protocol.md`、PLAN §4 | `reports/explore/eval_protocol.md` | 迁 `docs/EVAL_PROTOCOL.md`，其他链接 |
| 测试数 | 186/234/384 | `pytest` 实际输出 | 文档不写数字 |
| `run_iteration` | README/scripts README 宣称存在 vs 同文件 TODO/文件系统不存在 | 文件系统 | 删除引用或实现 |

---

## 5. 目标文档结构（建议）

```
/
├── AGENTS.md                         # 90–110 行：入口 + 通用硬约束 + 文档地图（不再写状态/机器事实）
├── README.md                         # 40–60 行：项目简介 + 三条入口（PLAN/STATUS/PLAYBOOK）+ 通用 quickstart
├── PLAN.md                           # 唯一需求/验收权威；§0 追加式决策，历史不改写
├── NOTICE.md                         # 许可/移植归属（保留）
├── docs/
│   ├── README.md                     # 文档地图与维护规则（读什么、按什么顺序读）
│   ├── STATUS.md                     # 只写“现在”：当前基线、在跑实验、下一步、阻塞（≤200 行）
│   ├── EVAL_PROTOCOL.md              # 从 reports/explore/eval_protocol.md 迁入；强度评测唯一协议
│   ├── PLAYBOOK.md                   # 成功经验：方法、适用条件、证据、命令、停止条件
│   ├── failures/
│   │   ├── README.md                 # 负结果索引：编号/假设/结论/证据/重开条件
│   │   ├── 001-search-as-teacher.md
│   │   ├── 002-awr-q-dmc.md
│   │   ├── 003-matchup-reweight-winonly.md
│   │   ├── 004-all-deck-generalist.md
│   │   ├── 005-dense-bc-super-expert.md
│   │   └── 006-a2-id-embedding-invalid.md
│   ├── environments/
│   │   ├── README.md                 # 机器矩阵 + 通用环境变量/目录/口径
│   │   ├── local-12c.md              # 本机：6C/12T、30GiB、无 GPU、conda gitcg
│   │   ├── wsl-mx550.md              # WSL：4C/8G、MX550 2G、fp32、d≤128
│   │   ├── a800.md                   # A800：80G、宿主 128/配额 10、37G 存储
│   │   ├── nju-cluster.md            # NJU：LSF、CentOS7/glibc、fa_env、无外网
│   │   └── kaggle-t4.md              # Kaggle：T4×2、4 vCPU、12h、只训练
│   ├── decisions/
│   │   └── README.md                 # D1–Dnn 索引：一行摘要 + PLAN §0 链接（不复制正文）
│   └── howto/
│       ├── collect-and-train.md      # train.collect / compact / pipeline / learning_curve
│       ├── engine-overlay.md         # V8/overlay/增量重建
│       ├── web-debug.md              # WS8 起局、反选、日志、故障局
│       └── npc-pve.md                # PvE 关卡/敌人/门禁
├── reports/                          # 一次性证据，保留；只加 front-matter 与索引
│   ├── README.md                     # 索引 + frontmatter 规范 + 归档规则
│   ├── EXPLOIT_LADDER.md             # 保留为证据（内容被 PLAYBOOK 摘要）
│   ├── SPECIALISTS.md                # generated，勿手改
│   ├── explore/...
│   └── ...
└── （现有模块 README）                # 只保留接口/文件职责/测试入口，不再承载状态
```

### 每个文件的职责（核心）

| 文件 | 只写什么 | 不写什么 |
|---|---|---|
| `AGENTS.md` | 入口、权威地图、通用硬约束、通用命令、文档规则 | 当前基线数字、机器路径、一次性实验结论、历史 |
| `README.md` | 项目是什么、三条入口、通用 quickstart、许可链接 | 当前状态细节、机器特化命令 |
| `PLAN.md` | 需求、架构、里程碑、验收阈值、追加式决策 | 当前跑分、操作步骤、机器路径 |
| `docs/STATUS.md` | 当前有效基线（带 n/seeds/deck）、在跑实验、下一步、阻塞 | 历史、失败归因长文、旧“下一步” |
| `docs/PLAYBOOK.md` | 已认证有效方法、适用条件、证据链接、命令、停止条件 | 未验证猜想、一次性结果、失败方法细节 |
| `docs/failures/NNN-*.md` | 假设、设置、结果、归因、重开条件 | 成功方法、当前排产 |
| `docs/environments/*.md` | 路径、硬件、网络、环境变量、吞吐、并发、坑、冒烟命令 | 训练结论、模型强弱 |
| `docs/decisions/README.md` | D 编号索引、状态、PLAN 链接 | 决策全文（避免双份） |
| `reports/**` | 一次实验的原始证据、聚合数字、命令、产物路径 | 当前状态、普适规律（应提炼到 PLAYBOOK/failures） |
| 模块 README | 文件职责、接口、测试入口、命令 | 项目状态、跑分、历史 |

---

## 6. Report front-matter 模板（建议强制）

```yaml
---
title: "A3 — M3-CPU 筛选认证"
kind: report                 # canonical|status|howto|environment|report|failure|generated|archive
status: valid                # valid|invalid|superseded|archived|in-progress
freshness: 2026-10-14        # 最后一次与代码/数据核对
machines: [local]            # local|wsl|a800|nju|kaggle|all
authority: evidence          # canonical|evidence|none
supersedes: []
superseded_by: null
related: ["reports/explore/eval_protocol.md", "PLAN.md#d22"]
run:
  commit: "14f799b"
  engine_lock: "configs/engine.lock"
  seeds: "0-4"
  n: 200
  gate: "pooled>=0.50; strict-ci"
  verdict: "FAIL"
evidence:
  - "reports/explore/a3_screen_v1.json"
  - "data/eval/a3_screen_v1.jsonl"   # gitignored
---
```

生成文档额外字段：

```yaml
generated_by: "python -m scripts.specialist_table"
generated_at: "2026-10-27"
do_not_edit: true
inputs: "reports/explore/*.json"
```

规则：
- `status: invalid` 必须写 `invalid_reason` 与 `reopen_condition`。
- `status: superseded` 必须写 `superseded_by`；不删除原文。
- 报告里的数字必须带 `n`、种子范围、deck、gate；禁止只写 pooled。
- 生成文档禁止手改；人工注解放 `## Manual notes` 并标 `manual_note: true`。

---

## 7. 单一事实源（Single Source of Truth）

| 事实 | 唯一维护位置 | 其他文档 |
|---|---|---|
| 验收阈值（M0–M6/M3-CPU/PvE-v1） | `PLAN.md §4` | 只链接，不复制数字 |
| 决策历史 | `PLAN.md §0`（追加式） | `docs/decisions/README.md` 只做索引 |
| 当前状态/基线/下一步 | `docs/STATUS.md` | AGENTS/README 只链接 |
| 成功方法/适用条件 | `docs/PLAYBOOK.md` | 报告只提供证据 |
| 失败原因/重开条件 | `docs/failures/*` | 报告只提供证据 |
| 评测协议/CI/样本量 | `docs/EVAL_PROTOCOL.md` | eval/README、reports 只链接 |
| 机器路径/环境变量/吞吐/并发 | `docs/environments/*` | 报告命令用 `$PY/$REPO/$DATA` |
| 引擎 pin/V8/overlay provenance | `configs/engine.lock` + `engine-overlay/README.md` | 其他文档引用 |
| 卡池版本 | `configs/pools/README.md` + JSON | 其他文档引用 |
| NPC 敌人/关卡数据 | `configs/npc/README.md` + JSON | `npc/README`、engine-overlay 只引用 |
| Web 协议 | `web/README.md` | PLAN/AGENTS 只链接 |
| 报告证据 | `reports/**` | 不把内容搬进主文档 |

---

## 8. 迁移计划（按优先级，不执行）

| # | 动作 | 源 → 目标 | 优先级 | 风险 | 需人工确认 |
|---|---|---|---|---|---|
| 1 | 建目录骨架 | `docs/{README,STATUS,PLAYBOOK,EVAL_PROTOCOL}.md`、`docs/{failures,environments,decisions,howto}/` | P0 | 低 | 目录命名是否与团队习惯一致 |
| 2 | 写 `STATUS.md` | 从 `AGENTS.md:5-30` + `EXPLOIT_LADDER` + `SPECIALISTS` + `plan_strength:514-573` 提炼 | P0 | 中 | 当前基线以哪套 seeds/n 为准 |
| 3 | 修一阶事实 | 13→19 类、128→宿主128/配额10、`run_iteration`、186→命令、node26→24、`§6.x` 引用 | P0 | 低 | 无（纯事实） |
| 4 | A2 更正 | `a2_id_screening.md` 顶部追加 `status: invalid`；STATUS 写重开条件 | P0 | 中 | 接受 `literature_review` 的 fp16 根因 |
| 5 | 建 `docs/failures/` | 从 `a3`→001、`plan_strength`→002/003/004、`a2`→006、`bc_scale`→005 | P1 | 中 | 失败归因措辞、重开条件 |
| 6 | 建 `docs/PLAYBOOK.md` | 以 `EXPLOIT_LADDER.md` 为核心，补 dense-corpus parity、eval protocol、streaming collect | P1 | 中 | 只收“已认证”的方法，不把 +0.071 的种子运气写进去 |
| 7 | 拆 `AGENTS.md` | 状态→STATUS；环境→environments；web→howto/web-debug；NPC→howto/npc-pve；引擎→howto/engine-overlay | P1 | 中 | AGENTS 行数上限/保留条目 |
| 8 | 拆 `plan_strength_2day_kaggle.md` | 当前状态→STATUS；失败→failures；时间线保留为 `reports/explore/` 归档日志 | P1 | 中 | 是否保留原文件名（建议保留，避免断链） |
| 9 | 迁 `eval_protocol.md` | `reports/explore/eval_protocol.md` → `docs/EVAL_PROTOCOL.md`，原位置留 stub | P1 | 中 | 外部链接可能指向旧路径 |
| 10 | 更新 PLAN | §10 状态表；§0 已追加 D25/D26，D27/D28 待确认（不改 D22–D24） | P1 | 高 | 决策人确认新决策编号与措辞 |
| 11 | 建 `docs/decisions/README.md` | D1–Dnn 索引，链接 PLAN §0 | P1 | 低 | 是否给旧决策标 `superseded/invalidated` |
| 12 | 报告 front-matter | 全部 `reports/**/*.md`；generated 加 `do_not_edit` | P1 | 低 | 旧报告 status 归类 |
| 13 | 重写 `reports/README.md` | 由 front-matter 生成索引（或手写 + CI 校验） | P1 | 低 | 是否引入生成脚本 |
| 14 | 索引修复 | `tests/README` 51 个测试、`scripts/README` 新脚本、`kaggle/README` modes | P1 | 低 | 无 |
| 15 | 过时报告标记 | `model_design`、`strength_levers`、`npc_pve_doc`、`npc_pve_smoke`、`web_standalone_start` 加 `superseded` | P2 | 低 | 是否影响外部引用 |
| 16 | 文档校验 | `scripts/check_docs.py`：frontmatter、死链、`/home/`/`/root/`、`§6.x`、generated 标记 | P2 | 中 | 是否进 CI 硬门禁 |
| 17 | 清理生成副本 | `build/kaggle/out*/repo/*.md` 不纳入文档工具；不改 git 跟踪 | P2 | 低 | 无（build 已 gitignore） |
| 18 | 删除/保留决策 | 不删除任何报告；`orchestrator/README` 保留占位；`run_iteration` 引用删除或实现 | P2 | 低 | 是否实现 `run_iteration` |

### 迁移风险

- **PLAN 历史不可改写**：D22–D24 是当时的决策，正确做法是追加 D25/D26 并在 §10 标“当前状态”，而不是改旧行。
- **A2 改判需要证据链**：`literature_review` 的 fp16 根因来自 f05/f07 的探针；迁移前先复跑一次 `P0-A` parity，避免“用一份未提交报告推翻已提交报告”。
- **`plan_strength` 是唯一时间线**：拆它时保留原文，否则会丢失“哪个结论在哪天被推翻”的历史；只把“当前”抽到 STATUS。
- **重命名会断链**：优先保留原路径 + front-matter，不移动报告；确需移动时加 stub 与 301 式链接说明。
- **PvE PASS 的措辞**：从“PASS”改成“conditional PASS”可能影响对外汇报；需要负责人确认。

---

## 9. 机器特化清单（每台机器真正需要记录什么）

### 9.0 通用变量契约（所有机器）

| 变量/约定 | 用途 | 备注 |
|---|---|---|
| `$REPO` | 仓库根 | 禁止在文档写死 `/home/...`、`/root/...` |
| `$PY` | 该机 conda env 的 python | 本机/A800 为 `gitcg`；NJU 为 `fa_env` |
| `GITCG_DATA_DIR` | 数据根覆盖 | `common/paths.py` 支持 |
| `GITCG_REPORTS_DIR` | 报告根覆盖 | Kaggle/只读输入时用 |
| `GITCG_ASSETS_CACHE` | 资产缓存路径 | 缺缓存时联网；离线必须设 |
| `GITCG_ASSETS_OFFLINE=1` | 禁止联网 | CI/NJU/A800 建议 |
| `NO_SHUFFLE=1` | 禁引擎洗牌 | `envs/match.py` 内部设置；不要手改 |
| `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` | 防止评测 worker 线程超订 | `eval_protocol` 多 worker 必须 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU | Kaggle 双卡分卡 |
| `GITCG_LLM_*` / `DEEPSEEK_*` | LLM 客户端 | 密钥只从 `.env`/环境变量读 |
| `KAGGLE_PROXY` / `HTTPS_PROXY` | Kaggle 上传代理 | 本机 `127.0.0.1:7890` |
| `GITCG_VENDOR_DIR` | NJU 离线依赖 | `scripts/cluster/bootstrap.sh` |
| `PYTHONPATH=$REPO` + `cwd=$REPO` | 子进程协议 | `envs/rollout_worker` / `envs/fork_worker` |

### 9.1 本机（local-12c）

| 项 | 事实/建议 |
|---|---|
| 路径 | `/home/ubuntuaaa/projects/genshin-GITCG` |
| 硬件 | 6 物理核 / 12 线程；30 GiB 内存；swap 8 GiB；**无 GPU** |
| Python | conda env `gitcg`，`/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python`（Python 3.12） |
| 关键包 | `gitcg 0.21.0`；torch CPU（具体版本以 `pip show torch` 为准，报告中出现过 2.14.1+cpu） |
| 吞吐 | 单核 ~819 局/h；12 worker 稳态 ~4,900 局/h（L5.1）；8 worker 采集 3,299 局/h；11 worker 4,020 局/h；评测 ~1,500 局/h；CPU 训练 d256 150–300 样本/s |
| 并发 | 默认 12 worker；不要同时跑“多个 11–12 worker 采集 + 训练 + 评测”（会 swap/OOM）；评测 worker 设 `OMP_NUM_THREADS=1` |
| 已知坑 | cffi 回调吞 `KeyboardInterrupt`（终端人机用 `HumanQuit`）；每局必须 reset codebook；`NO_SHUFFLE=1`；`train.collect` 旧父进程内存问题已由 `stream_task_shard` 修复，但旧命令仍可能用 `--max-tasks` |
| 冒烟 | `python -m pytest -q -m "not slow"`；`python -m envs.benchmark --games 12 --workers 1` |
| 产物 | `reports/` 小体积入库；`data/`、checkpoint、JSONL 不入库 |

### 9.2 WSL（wsl-mx550）

| 项 | 事实/建议 |
|---|---|
| 路径 | `~/projects/genshin-GITCG`（以实际 WSL 用户为准） |
| 硬件 | 4 核 / 8 GiB；MX550 2 GB（Turing cc 7.5，无 Tensor Core，64-bit，~2.7 TFLOPS fp32） |
| Python | conda env `gitcg`；`torch` CUDA 版（以 `scripts/bench_device --devices cuda` 为准） |
| 吞吐 | mean-pool d128 8.6k 样本/s；2L transformer 3.5k 样本/s（precollated）；端到端 epoch 受 4 核/8G/IO 限制 |
| 并发 | 4 worker 采集/评测；训练 batch 32–64；不要开 AMP |
| 已知坑 | fp16/bf16 比 fp32 慢；2 GB 显存只能 d≤128/2L；8 GB 内存限制 precollate 池；`torch.compile` 对 cc7.5 收益未测 |
| 定位 | CI/冒烟 + 并行采集/评测节点；**不是提强度训练节点** |
| 冒烟 | `python -m scripts.bench_device --devices cpu cuda --d-models 64 128 --epochs 12 --batch-size 32 --precollate --games 6` |

### 9.3 A800（a800）

| 项 | 事实/建议 |
|---|---|
| 路径 | 建议统一 `$REPO=/root/blockdata/genshin-GITCG`（报告实际用法）；不要同时写 `~/genshin-GITCG` |
| 硬件 | A800-SXM4-80GB ×1；宿主 `nproc=128`，**cgroup `cpu.max` = 10 核**；内存/存储以窗口为准 |
| 存储 | `/root/blockdata` 可用 ~37G；`/` 30G；`/dev/shm` 50G 易失；`/public` 只读 |
| Python | conda env `gitcg`（`/root/blockdata/conda/gitcg/bin/python`）；torch 2.7.0+cu126（以窗口实测为准） |
| 吞吐 | batch 512 训练 276k 样本/s；推理 201k 样本/s；数据 1.64 MB/局 jsonl、0.65 MB/局 compact |
| 并发 | **只做 GPU 训练/推理**；CPU 采集/评测在本机/WSL；DataLoader 进程数 ≤ 可用核；`--precollate` |
| 已知坑 | `nproc` 显示 128 是宿主，实际 10 核；37G 存储上限；传数据用 scp/rsync；checkpoint 回本地认证；不要在窗口跑 `eval_protocol` |
| 冒烟 | `python -c "import torch; print(torch.cuda.is_available())"`；`cat /sys/fs/cgroup/cpu.max`；`nvidia-smi` |
| 数据流 | 本地 `train.collect` → 传入 `data/replays/` → `train.pipeline --replays --device cuda` → checkpoint 回流 → 本地 `eval_protocol` |

### 9.4 南大集群（nju-cluster）

| 项 | 事实/建议 |
|---|---|
| 调度 | LSF（`bsub < xxx.lsf`），`scripts/cluster/lsf/*.lsf` |
| OS/glibc | CentOS 7（glibc 2.17）；`gitcg` manylinux_2_34 → `module load glibc/2.36-gcc12.1.0` |
| Python | 无 conda 频道；复用 `$HOME/.conda/envs/fa_env`（Python 3.10 + torch 2.6.0）；`tomli` 回退 |
| 网络 | 集群不通外网；代码走 njugit master 镜像；wheel/资产走 `vendor` 分支；pypi 镜像 `mirror.nju.edu.cn` |
| 存储 | `/fsb/home`（Tier1）、`/bbfs/fsb/home`（全闪）、`/bbfs/scratch`（30 天）；节点本地 `/tmp`、`/dev/shm` |
| 队列 | `cpu1` ≤24 核；`7702ib` 128c；`6140ib` 72c；`6330ib` 56c；GPU `945090ib` 5090 32G / `734090ib` 4090 24G；备选 A100/V100 |
| 并发 | 公共共享；按队列规格；GPU 作业 `#BSUB -gpu "num=1"` |
| 已知坑 | 集群暂缓（D14）；glibc/资产离线是启用前置；不要把本机 `conda run -n gitcg` 照搬 |
| 冒烟 | `GITCG_VENDOR_DIR=$HOME/genshin-GITCG-vendor bash scripts/cluster/bootstrap.sh`；`bsub < scripts/cluster/lsf/benchmark.lsf` |

### 9.5 Kaggle（kaggle-t4）

| 项 | 事实/建议 |
|---|---|
| 硬件 | T4×2 16 GB；4 vCPU / 29 GB；`/kaggle/working` 20 GB；单会话 12h；GPU 30h/周 |
| 定位 | **只训练**；无引擎（V8/cffi 在 CPU，Kaggle 4 vCPU 太弱）；采集/评测留本机 |
| 数据 | 数据集 200 GB/个、私有 200 GB；`/kaggle/input/<slug>` 只读；上传走代理（`googleapis.com` 直连超时） |
| Python | kernel 内 torch/numpy 版本需现场核对；`weights_only=False`；`--precollate` 必做 |
| 吞吐 | 334k 样本约 36 s/epoch（~9.2k 样本/s，CPU collate 瓶颈）；A800 276k 样本/s 对照 |
| 并发 | 双卡分 `CUDA_VISIBLE_DEVICES`；单 run 时限（4h/11h）要按 epoch 估算 |
| 已知坑 | P100 已停用且 sm_60 不兼容；上传必须代理；`mode.txt` 白名单曾把 `cap` 静默回落到 `runs`（`plan_strength:463-464`）；数据集版本未 ready 就推 kernel 会空跑 |
| 入口 | `scripts/kaggle_push.sh` / `scripts/kaggle_pull.sh`；`kaggle/train_arm.py`；`kaggle/kernel-metadata.json` |
| 冒烟 | `KAGGLE_MODE=smoke scripts/kaggle_push.sh`；先 `kaggle datasets status` 确认版本 |

---

## 10. `AGENTS.md` 改写提纲

### 10.1 目标

- **90–110 行**（当前 77 行但信息密度过高，单行 2072 字符；目标“短行、单一职责”）。
- 只回答三件事：**先读什么、什么不能违反、去哪个文档找细节**。
- 不再承载：当前状态、基线数字、机器路径、一次性实验结论、历史。

### 10.2 建议结构

```markdown
# AGENTS.md（目标 90–110 行）

> 权威地图：需求/验收 → PLAN.md；当前状态 → docs/STATUS.md；
> 有效方法 → docs/PLAYBOOK.md；失败/重开 → docs/failures/；
> 机器 → docs/environments/；评测 → docs/EVAL_PROTOCOL.md。

## 0. 三条硬约束（8 行）
- 强度验收只用纯策略；LLM 辅助分开报分。
- 观测只取 notification.state；信息泄漏测试是硬门禁。
- 不改 refs/；不覆盖 checkpoint/数据/配置；密钥不入库。

## 1. 先读什么（10 行）
- 新人顺序：PLAN §4 → STATUS → PLAYBOOK → 对应环境文档。
- 任务类型 → 必读文档（训练/评测/Web/NPC/引擎）。

## 2. 通用工作流（15 行）
- 单测；冒烟；采集；训练；评测；断点续跑。
- 统一入口：train.collect / train.pipeline / eval_protocol / run_npc_gate。
- 长任务 setsid/tmux；结果流式落盘。

## 3. 对局与复现（12 行）
- 同种子预洗牌 + NO_SHUFFLE=1；每局 reset codebook。
- 语义 key，不裸 action_code；位置索引跨进程。
- fork 只在 is_resumable 边界；fork 必须子进程。

## 4. 数据与版本（10 行）
- engine.lock 随 run；卡池升级开新 run。
- data/ 与 checkpoint 不入库；reports 只放聚合。
- 采集格式 compact 的指纹规则。

## 5. 文档维护（10 行）
- STATUS 只写现在；历史进 reports/decisions。
- 报告必须 front-matter；generated 勿手改。
- 失败必须写重开条件；成功必须写适用条件。

## 6. 机器入口（6 行）
- 本机 / WSL / A800 / NJU / Kaggle 各一行链接。
- 禁止在 AGENTS 写机器路径。

## 7. 索引（10 行）
- docs/ 目录、模块 README、reports/README。
```

### 10.3 保留/迁移对照

| 当前 `AGENTS.md` | 处理 |
|---|---|
| `:3` 权威声明 | 保留，改成“权威地图” |
| `:5-30` 现状（WS0/WS1、三条线、机会算力、git remote） | 移到 `docs/STATUS.md`；git remote 移到 `docs/environments/local-12c.md` |
| `:18` 本机/WSL/A800 口径 | 移到 `docs/environments/README.md` + 各机文档 |
| `:19-24` 三线状态 | 移到 STATUS；AGENTS 只留一行“当前三线见 STATUS” |
| `:21` NPC 状态 | 移到 `docs/howto/npc-pve.md` + `npc/README` |
| `:22` Web 长状态（2072 字符） | 移到 `docs/howto/web-debug.md` + `web/README` |
| `:23` 引擎构建 | 移到 `docs/howto/engine-overlay.md` + `engine-overlay/README` |
| `:24` A800 机会算力 | 移到 `docs/environments/a800.md` |
| `:26-29` refs 说明 | 保留 3 行；详情在 `NOTICE.md` |
| `:30` git remote | 移到 `docs/environments/local-12c.md` |
| `:32-36` 环境 | 移到 `docs/environments/*`；AGENTS 只留“先读环境文档” |
| `:38-44` gitcg 用法 | 保留为“对局 API 速查”6 行 |
| `:46-58` 工程约定 | 保留 8 条；collect/compact 细节移到 `docs/howto/collect-and-train.md` |
| `:60-67` 硬性约束 | **保留**（这是 AGENTS 的核心） |
| `:69-73` 测试/流程 | 保留 3 条；测试清单移到 `tests/README` |
| `:75-77` 文档写作规范 | 改为“STATUS 只写现在；报告标 status；决策追加不改写” |

### 10.4 建议保留的硬约束（不要移到别处）

1. 强度验收 = 纯策略；LLM 辅助分开报分。
2. 观测只取 `notification.state`；信息泄漏测试是合并硬门禁。
3. `reps/` 动作抽象 100% 覆盖 + adapter 往返是 go/no-go。
4. 同种子重放一致；`NO_SHUFFLE=1`；每局 reset codebook。
5. fork 只在 `is_resumable()` 边界；fork 必须子进程（`workers>=1`）。
6. `engine.lock` 随 run；卡池升级 = 新 run。
7. `data/`/checkpoint 不入库；checkpoint/数据/配置只追加。
8. `refs/` 只读；不 apply 上游补丁。
9. 无常驻服务；长任务可断点续跑。
10. 密钥/`.env` 永不入库、不进日志。

---

## 11. 落地验收清单

- [x] `AGENTS.md` ≤110 行，且不出现 `12 核`、`30G`、`/home/`、`/root/`、`0.561` 等机器/基线数字。
- [x] `docs/STATUS.md` 的每个数字都有 `deck / seeds / n / date`。
- [x] `docs/PLAYBOOK.md` 只收“通过完整门禁”的方法；dvalin 的 +0.071 已标“未复现”，不写入成功方法。
- [x] `docs/failures/README.md` 能回答每个失败的“为什么失败、什么条件下重开”。
- [ ] `reports/**` 全部有 front-matter（部分完成：A2 标 `invalid`，旧设计文档标 `superseded`）。
- [x] `PLAN.md §0` 已追加 D25/D26（2026-10-27），未改写 D22–D24；D27/D28 待确认。
- [x] `PLAN §6.2/§6.3/§6.5` 断链已用 6.1–6.5 子节修复。
- [x] `run_iteration` 引用已删除/改成“未实现”。
- [x] `tests/README`、`scripts/README`、`reports/README`、`kaggle/README` 索引已更新。
- [x] `docs/environments/*` 每份都有冒烟命令、吞吐、并发、已知坑。
- [ ] CI 跑文档校验（frontmatter/死链/绝对路径/generated 标记）。
- [ ] 不存在“某个数字在两个地方各写一遍”的情况（抽查 0.519/0.561、13/17/19、128/10、186/234/384）。

---

## 12. 附：审计边界与后续执行

- **审计时**未修改 `PLAN.md`、`AGENTS.md`、代码/配置/数据；未移动/重命名/删除任何文档。
- **后续 P0（2026-10-27）**已按本报告落地主体：`docs/` 骨架、`AGENTS.md`/`README.md` 重写、
  `PLAN §9/§10`、A2 更正头、D25/D26、`docs/EVAL_PROTOCOL.md` 迁移、`failures/`、`environments/`、
  `decisions/`、`howto/`。报告原文未删除，旧路径留 stub。
- 未运行训练/评测；引用数字均来自现有报告或 P0 的 parity/干净语料实测。
- `reports/explore/lit/` 与 `literature_review.md` 的结论以作者原始报告为准；本审计只做交叉引用。
