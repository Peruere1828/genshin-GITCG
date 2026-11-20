# AGENTS.md

> 权威地图：需求/验收 → [`PLAN.md`](./PLAN.md)；当前状态 → [`docs/STATUS.md`](./docs/STATUS.md)；
> 有效方法 → [`docs/PLAYBOOK.md`](./docs/PLAYBOOK.md)；失败/重开 → [`docs/failures/`](./docs/failures/)；
> 机器环境 → [`docs/environments/`](./docs/environments/)；评测 → [`docs/EVAL_PROTOCOL.md`](./docs/EVAL_PROTOCOL.md)。

## 0. 三条硬约束

1. **强度验收 = 纯策略**（LLM 不参与）；LLM 辅助模式分开报分。
2. **观测只取 `notification.state`**；全量 `GameState` 不得造观测；信息泄漏测试是 CI 硬门禁。
3. **`refs/` 只读**；不修改、不 apply 补丁；`data/`、checkpoint、`.env`/密钥不入库。

## 1. 先读什么

| 任务 | 必读 |
|---|---|
| 接需求/验收 | `PLAN.md §4` → `docs/STATUS.md` |
| 训练/评测 | `docs/EVAL_PROTOCOL.md` → `docs/PLAYBOOK.md` |
| 查失败原因 | `docs/failures/README.md` |
| 换机器/排产 | `docs/environments/README.md` + 对应机器文档 |
| 改引擎/NPC/Web | `docs/howto/engine-overlay.md` / `npc-pve.md` / `web-debug.md` |

阅读顺序：`PLAN §4` → `STATUS` → `PLAYBOOK` → 环境文档。

## 2. 通用工作流

```bash
export PY=/path/to/conda/envs/gitcg/bin/python
export REPO=/path/to/genshin-GITCG
cd "$REPO"

"$PY" -m pytest -q -m "not slow"      # 快速门禁
"$PY" -m pytest -q                    # 含慢测
"$PY" -m envs.benchmark --games 12 --workers 1
"$PY" -m scripts.eval_protocol --sample-sizes
```

- 长任务挂 `tmux`/`setsid`；结果流式落盘 + 按 `task_index` 断点续跑。

## 3. 数据与版本

- `engine.lock` 随 run 保存；卡池升级 = 显式迁移开新 run。
- 所有模型训练实验和评测实验数据必须落盘。

## 4. 文档维护

- `STATUS` 只写现在；历史进 `reports/`；失败进 `docs/failures/`；成功方法进 `docs/PLAYBOOK.md`。
- 报告必须带口径（deck/policy/seeds/n/gate/date）；自动化生成的文档标 `do_not_edit`。
- 重大决策在 `PLAN.md §0` 追加 D 编号，不改写历史；索引见 `docs/decisions/README.md`。
- 机器特化只写 `docs/environments/*`；通用文档用 `$REPO/$PY/$DATA`。

## 5. 机器入口

- [本机](./docs/environments/local-12c.md)
- [WSL](./docs/environments/wsl-mx550.md)
- [A800](./docs/environments/a800.md)
- [南大集群](./docs/environments/nju-cluster.md)
- [Kaggle](./docs/environments/kaggle-t4.md)

## 6. 索引

- 文档地图：[`docs/README.md`](./docs/README.md)
- 当前基线/排产：[`docs/STATUS.md`](./docs/STATUS.md)
- 成功方法：[`docs/PLAYBOOK.md`](./docs/PLAYBOOK.md)
- 失败/重开：[`docs/failures/README.md`](./docs/failures/README.md)
- 评测协议：[`docs/EVAL_PROTOCOL.md`](./docs/EVAL_PROTOCOL.md)
- 决策索引：[`docs/decisions/README.md`](./docs/decisions/README.md)
- 报告索引：[`reports/README.md`](./reports/README.md)
- 测试索引：[`tests/README.md`](./tests/README.md)
