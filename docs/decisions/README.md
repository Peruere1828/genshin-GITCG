# decisions — 决策索引

> 决策全文在 [`PLAN.md §0`](../../PLAN.md)；本文件只做一行索引，不复制正文。
> 规则：追加不改写；被取代的决策标 `superseded`/`limited`，不删除。

| # | 议题 | 状态 | 备注 |
|---|---|---|---|
| D1 | 第一优先目标 | active | 打赢脚本对手，胜率台阶为第一验收 |
| D2 | Agent 范围 | active | 对局/自迭代/组牌/复盘/工程改进 |
| D3 | 训练主线 | active | SoG/ReBeL；DMC/BC 只作冷启动 |
| D4 | 卡池/版本 | active | 每次训练冻结 engine commit + 卡池 |
| D5 | LLM 使用 | active | 局内/局外都用；纯策略与 LLM 分开报分 |
| D6 | LLM 资源 | active | 商业 API + 降级链 |
| D8 | 时间粒度 | active | 按里程碑，不排日期 |
| D9 | 工程目录 | active | 本机仓库路径 |
| D10 | 本地先行 | active | L0–L4；大规模训练留机会算力 |
| D11 | 验收对手口径 | active | 当前 20 套脚本池，池版本化 |
| D12 | 搜索分叉路线 | active | 薄 fork 桥；子进程 |
| D13 | 搜索分叉契约 | active | `is_resumable` 边界 + attrs + 子进程 |
| D14 | 算力口径 | active | 本机/WSL 默认；A800 机会资源 |
| D15 | 无 GPU 期间主线 | limited | 被后续 specialist/P0 排产部分取代 |
| D16 | 三方向并行立项 | active | 模型/NPC/Web |
| D17 | NPC PvE 调试系统 | active | 独立 PvE-v1 门禁 |
| D18 | Web 人机调试接口 | active | 调试接口，不进口径验收 |
| D19 | 卡组口径 | active | decklab 组牌；Web 人机反选 |
| D20 | 引擎构建策略 | active | Docker/V8/overlay |
| D21 | 机会算力 CPU 配额 | active | 宿主 128 / 配额 ~10；只 GPU |
| D22 | M3-CPU 首次认证 | historical | 未过；搜索蒸馏≈BC |
| D23 | 搜索路线判定 | **limited** | 因果结论被 D26 限定；不放大仍有效 |
| D24 | 数据先行结论 | **limited** | “总量不显著”≠“每 matchup 密度不显著” |
| D25 | A2 复核与 P0 门禁 | **current** | A2 无效；parity 未过冻结新结构 |
| D26 | 搜索因果限定/价值语义 | **current** | 修 cfr_probe/dice；搜索重开需门禁 |

## 使用

- 写新决策：在 `PLAN.md §0` 追加 D27…，再在本表加一行。
- 改验收口径：必须同时更新 `docs/EVAL_PROTOCOL.md` 与 `docs/STATUS.md`。
- 旧决策被推翻：标 `superseded`/`limited` 并链接新决策，不改原文。
