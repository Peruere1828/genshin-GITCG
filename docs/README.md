# docs — 项目文档地图

> 单一事实源：需求/验收 → `PLAN.md`；当前状态 → `docs/STATUS.md`；
> 有效方法 → `docs/PLAYBOOK.md`；失败/重开 → `docs/failures/`；
> 机器 → `docs/environments/`；评测 → `docs/EVAL_PROTOCOL.md`；
> 一次性证据 → `reports/`。

## 阅读顺序

1. **新人**：`PLAN.md §4` → `docs/STATUS.md` → `docs/PLAYBOOK.md` → 对应环境文档。
2. **接训练/评测**：`docs/EVAL_PROTOCOL.md` → `docs/PLAYBOOK.md` → 环境文档。
3. **查某个负结果**：`docs/failures/README.md`。
4. **查历史决策**：`PLAN.md §0`（追加式）→ `docs/decisions/README.md`（索引）。

## 文件职责

| 文件 | 只写什么 | 不写什么 |
|---|---|---|
| `AGENTS.md` | 入口、权威地图、通用硬约束、通用命令 | 当前基线数字、机器路径、一次性实验结论 |
| `README.md` | 项目是什么、三条入口、通用 quickstart | 当前状态细节、机器特化命令 |
| `PLAN.md` | 需求、架构、里程碑、验收阈值、追加式决策 | 当前跑分、操作步骤、机器路径 |
| `docs/STATUS.md` | 当前有效基线（带 n/seeds/deck/date）、在跑实验、下一步、阻塞 | 历史、失败归因长文、旧“下一步” |
| `docs/PLAYBOOK.md` | 已认证有效方法、适用条件、证据、命令、停止条件 | 未验证猜想、一次性结果、失败方法细节 |
| `docs/failures/NNN-*.md` | 假设、设置、结果、归因、重开条件 | 成功方法、当前排产 |
| `docs/environments/*.md` | 路径、硬件、网络、环境变量、吞吐、并发、坑、冒烟命令 | 训练结论、模型强弱 |
| `docs/EVAL_PROTOCOL.md` | 强度评测唯一协议 | 某个 checkpoint 的跑分 |
| `docs/decisions/README.md` | D 编号索引、状态、PLAN 链接 | 决策全文（避免双份） |
| `reports/**` | 一次实验的原始证据、聚合数字、命令、产物路径 | 当前状态、普适规律 |
| 模块 README | 文件职责、接口、测试入口、命令 | 项目状态、跑分、历史 |

## 维护规则

- **状态只进 `STATUS.md`**；报告里的“下一步”过期后不再修改，只加 `superseded`/`invalid` 标记。
- **成功方法必须通过完整门禁**才进 `PLAYBOOK.md`；一次性 +0.07 的种子运气不写入。
- **失败文档必须回答“为什么失败、什么条件下重开”**，否则不算失败文档。
- **决策追加不改写**：`PLAN.md §0` 只追加 D 编号；旧决策用 `superseded`/`limited` 标注。
- **报告数字必须带口径**：deck / policy / seeds / n / gate / date；禁止只写 pooled。
- **机器路径不写进通用文档**：统一用 `$REPO`、`$PY`、`$DATA`；具体值见 `docs/environments/`。
- **生成文档勿手改**：`reports/SPECIALISTS.md` 等由脚本生成，标 `do_not_edit`。

## 报告 front-matter（建议）

```yaml
---
title: "A3 — M3-CPU 筛选认证"
kind: report                 # canonical|status|howto|environment|report|failure|generated|archive
status: valid                # valid|invalid|superseded|archived|in-progress
freshness: 2026-10-14
machines: [local]
authority: evidence          # canonical|evidence|none
supersedes: []
superseded_by: null
related: ["PLAN.md#d22"]
---
```

`status: invalid` 必须写 `invalid_reason` 与 `reopen_condition`；`status: superseded` 必须写 `superseded_by`。报告原文不删除，只加标记与链接。
