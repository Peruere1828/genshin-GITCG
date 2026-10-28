# AGENTS.md

> 权威地图：需求/验收 → [`PLAN.md`](./PLAN.md)；当前状态 → [`docs/STATUS.md`](./docs/STATUS.md)；
> 有效方法 → [`docs/PLAYBOOK.md`](./docs/PLAYBOOK.md)；失败/重开 → [`docs/failures/`](./docs/failures/)；
> 机器环境 → [`docs/environments/`](./docs/environments/)；评测 → [`docs/EVAL_PROTOCOL.md`](./docs/EVAL_PROTOCOL.md)。
> 本文件只写入口、通用硬约束和文档地图；不写当前基线数字、机器路径、一次性实验结论。

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
export PY=/path/to/conda/envs/gitcg/bin/python   # 本机/A800=gitcg；NJU=fa_env
export REPO=/path/to/genshin-GITCG
cd "$REPO"

"$PY" -m pytest -q -m "not slow"      # 快速门禁
"$PY" -m pytest -q                     # 含慢测
"$PY" -m envs.benchmark --games 12 --workers 1
"$PY" -m scripts.eval_protocol --sample-sizes
```

- 长任务挂 `tmux`/`setsid`；结果流式落盘 + 按 `task_index` 断点续跑。
- 并发统一走子进程协议（`envs.rollout_worker` / `envs.fork_worker`，stdin/stdout JSON-lines，`cwd=$REPO` + `PYTHONPATH`）。
- 合并门禁：单测 + 信息泄漏 + 冒烟 + 评测不劣于；checkpoint/数据/配置只追加不覆盖。

## 3. 对局与复现

- `envs/match.py` 在 Python 侧按种子预洗牌并设 `NO_SHUFFLE=1`；每局开始 `reset_default_hierarchical_action_codebook()`。
- 训练标签/embedding 用语义 key，不用裸 `action_code`（仅单局内稳定）。
- policy spec 统一走 `envs.policy.build_policy`：`expert:<slug>` / `neural:<ckpt>[#t=<temp>]` / `heuristic` / `legal_random` / `random`。
- 搜索分叉只在 `Game.is_resumable()` 边界取快照；fork 用 `envs/snapshot.fork_game(game_attrs=...)`；在线搜索必须子进程（`workers>=1`）。
- `gitcg 0.21.0`：`ActionRequest.action` 候选列表；`ActionResponse(chosen_action_index=..., used_dice=[...])`；`game.status()` 是方法；`State.query` 跨版本不兼容，用前小样本验证。

## 4. 数据与版本

- `engine.lock` 随 run 保存；卡池升级 = 显式迁移开新 run。
- 采集：`train.collect`（一行一局、manifest 指纹、可续跑）；compact 语料必须带 exact definition id（`token_definition_ids`/`option_definition_ids`）。
- 训练/评测前过 parity：`scripts/parity_store_live.py`；parity 未过，冻结新结构训练。
- checkpoint 选型用 `val_ce` early stop + 小样本配对引擎 screen；**不要用 `val_acc`**。
- 强度认证用 `scripts/eval_protocol.py`（纯策略、配对种子、双方、≥20 seeds）；报告带 deck/seeds/n/gate。

## 5. 文档维护

- `STATUS` 只写现在；历史进 `reports/`；失败进 `docs/failures/`；成功方法进 `docs/PLAYBOOK.md`。
- 报告必须带口径（deck/policy/seeds/n/gate/date）；生成文档标 `do_not_edit`。
- 重大决策在 `PLAN.md §0` 追加 D 编号，不改写历史；索引见 `docs/decisions/README.md`。
- 机器特化只写 `docs/environments/*`；通用文档用 `$REPO/$PY/$DATA`。

## 6. 机器入口

- [本机](./docs/environments/local-12c.md) · [WSL](./docs/environments/wsl-mx550.md) ·
  [A800](./docs/environments/a800.md) · [南大集群](./docs/environments/nju-cluster.md) · [Kaggle](./docs/environments/kaggle-t4.md)
- 采集/评测是 CPU 密集；A800/Kaggle 只做 GPU 训练/推理。

## 7. 索引

- 文档地图：[`docs/README.md`](./docs/README.md)
- 当前基线/排产：[`docs/STATUS.md`](./docs/STATUS.md)
- 成功方法：[`docs/PLAYBOOK.md`](./docs/PLAYBOOK.md)
- 失败/重开：[`docs/failures/README.md`](./docs/failures/README.md)
- 评测协议：[`docs/EVAL_PROTOCOL.md`](./docs/EVAL_PROTOCOL.md)
- 决策索引：[`docs/decisions/README.md`](./docs/decisions/README.md)
- 报告索引：[`reports/README.md`](./reports/README.md)
- 测试索引：[`tests/README.md`](./tests/README.md)
