# reports — 一次性实验证据

> **这里不是当前状态**。当前基线/排产见 [`docs/STATUS.md`](../docs/STATUS.md)；
> 有效方法见 [`docs/PLAYBOOK.md`](../docs/PLAYBOOK.md)；失败/重开见 [`docs/failures/`](../docs/failures/)。
> 报告只放聚合证据、命令、产物路径；原始逐局/逐事件数据在 `data/`（gitignore）。

## 当前最重要

| 文件 | 内容 | 状态 |
|---|---|---|
| [`EXPLOIT_LADDER.md`](./EXPLOIT_LADDER.md) | 逐 deck PG 剥削阶梯：dgeo 0.880 / ungeo 0.909 | valid |
| [`SPECIALISTS.md`](./SPECIALISTS.md) | 19 个 deck 的认证最优臂（自动生成） | generated |
| [`explore/p0_parity_20261027.md`](./explore/p0_parity_20261027.md) | P0 parity / exact id / 干净语料 / A2 复核 | valid |
| [`explore/literature_review.md`](./explore/literature_review.md) | 7 条文献线 + D25/D26 提案 | valid |
| [`explore/docs_audit.md`](./explore/docs_audit.md) | 63 份文档盘点 + 整理方案 | valid |
| [`explore/a2_id_screening.md`](./explore/a2_id_screening.md) | A2 原始数字（**结论无效**） | invalid |
| [`explore/eval_protocol.md`](./explore/eval_protocol.md) | 已迁到 `docs/EVAL_PROTOCOL.md` 的 stub | stub |

## 分类

- **训练/数据**：`train/*.json`、`train/*.md`、`datafirst_bc.md`、`train/bc_scale_datafirst.md`。
- **搜索/自博弈**：`search/*.md`、`train/a3_m3_cpu_screening.md`、`EXPLOIT_LADDER.md`。
- **评测**：`explore/*.json`、`arena/*.json`、`eval_protocol_smoke.json`。
- **NPC PvE**：`npc/pve_v1_gate.md`（当前）；B2/B3 冒烟聚合 `npc/npc_pve_smoke_v1.json` 保留为证据。
- **Web**：`web/`（当前协议/状态）；C0 起步报告已删除，内容并入 `web/README.md`。
- **LLM**：`llm/*.json`。
- **引擎/基准**：`engine/*.json`、`benchmarks/*.json`、`LOCAL_BASELINE.md`。
- **文献/审计**：`explore/lit/`、`explore/literature_review.md`、`explore/docs_audit.md`。

## 规则

1. **报告是证据，不是状态**：不要把“下一步”写进报告；状态进 `docs/STATUS.md`。
2. **数字带口径**：deck / policy / seeds / n / gate / date；禁止只写 pooled。
3. **生成文档勿手改**：`SPECIALISTS.md` 等由 `python -m scripts.specialist_table` 生成。
4. **失效不删除**：顶部加 `status: invalid|superseded|archived` + 原因 + 链接；原文保留。
5. **路径保留**：移动 canonical 文档时在原位置留 stub（如 `explore/eval_protocol.md`），避免断链。

## 常用命令

```bash
python -m eval.arena --smoke --seeds 3 --workers 12
python -m scripts.eval_protocol --sample-sizes
python -m scripts.specialist_table                 # 生成 SPECIALISTS.md
python -m scripts.compare_eval_runs <tag_a> <tag_b>
```
