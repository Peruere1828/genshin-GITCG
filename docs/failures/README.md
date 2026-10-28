# failures — 负结果索引

> 每条失败必须回答：**为什么失败**、**什么条件下才重开**。
> 成功方法见 `docs/PLAYBOOK.md`；一次性证据见链接的 `reports/` 原文。

| # | 主题 | 结论 | 状态 | 重开条件 |
|---|---|---|---|---|
| 001 | search-as-teacher / MCCFR 蒸馏 | 当前测量/预算下无增益，且全池配对显著负 | **limited**（D23/D26） | 修价值语义 + parity PASS + 校准价值头 + 非任意候选裁剪 + 门控覆盖 |
| 002 | AWR / Q-DMC / 离线 RL | AWR 权重与动作无关；Q 无支撑约束；DMC 规模差 3–4 个数量级 | **invalid / negative** | AWR 归档；Q 只试 TD3+BC 式 BC 锚；DMC 不在本机预算内 |
| 003 | matchup 重加权 / 胜局筛选 | 用 teacher 胜负定权，抬高“老师也输”的组；胜局筛选 −0.14 | **negative** | 只在采集层用 student-vs-expert 可修 gap，且与 uniform 混合 |
| 004 | 20-deck 单模型 generalist | 8 个 deck 里 7 个显著变差；密度是主因 | **negative** | 不再做共享 generalist；per-deck specialist + router |
| 005 | 稠密 BC 超专家 | BC 从未显著超过同 deck 专家；数据量不是主杠杆 | **limited** | 先做 parity；用 PG/自博弈超专家；BC 只做底座 |
| 006 | A2 identity embeddings | fp16 破坏 definition id，实验无效 | **invalid**（D25） | exact int id + parity PASS 后重做；不是“id 有害” |

## 维护

- 新失败按 `NNN-<topic>.md` 追加；不删除旧文档。
- 失败文档必须带 `status`、`evidence`、`reopen_condition`。
- 如果失败被后续实验推翻，把 `status` 改成 `superseded` 并链接新证据，不改写原文。
