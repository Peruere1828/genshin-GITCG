# genshin-GITCG

七圣召唤（Genius Invokation TCG）AI + Agent 工程：自训练策略 AI（SoG/ReBeL 路线）+ 局内决策、复盘教练、卡组构筑、自迭代闭环与工程自改进。

## 三条入口

- **需求/验收**：[PLAN.md](./PLAN.md)（决策 §0、里程碑 §4、M3-CPU §4.3、NPC PvE §4.4）
- **当前状态**：[docs/STATUS.md](./docs/STATUS.md)（基线、在跑实验、下一步、阻塞）
- **有效方法**：[docs/PLAYBOOK.md](./docs/PLAYBOOK.md)（逐 deck specialist、BC 底座、评测、parity）

其他入口：失败/重开 [docs/failures/](./docs/failures/)；机器 [docs/environments/](./docs/environments/)；评测协议 [docs/EVAL_PROTOCOL.md](./docs/EVAL_PROTOCOL.md)；文档地图 [docs/README.md](./docs/README.md)。

## 目录

| 目录 | 职责 | 对应 |
|---|---|---|
| `envs/` | gitcg 封装、Player、rollout、快照/分叉 | WS0 |
| `reps/` | 观测编码 / 动作分层 / 信念 | WS1 |
| `eval/` | arena、20 套脚本池、Wilson/Elo | WS2 |
| `train/` | CVPN、采集/训练/蒸馏、batched inference | WS3 |
| `agents/` | 纯策略 agent、LLM 辅助、脚本对手适配 | WS4 |
| `coach/` | 复盘归因、课程建议、BC 标签 | WS4 |
| `decklab/` | 卡组构筑（模板内 flex 位） | WS4 |
| `orchestrator/` | 自迭代闭环（占位，未实现） | WS5 |
| `web/` | 人机调试接口（standalone + HTTP 桥） | WS8 |
| `npc/` | NPC PvE 调试系统 | WS7 |
| `tests/` | 信息泄漏、adapter 往返、重放一致性等 | 全部 |
| `configs/` | run 配置、`engine.lock`、卡池/关卡 | D4 |
| `scripts/` | 采集/训练/评测/认证脚本 | §5.1 |
| `docs/` | 状态、方法、失败、机器、howto | — |
| `data/` `reports/` | 回放/产物（不入库）、聚合报告 | WS6 |

## 起步

```bash
# 需要 conda env `gitcg`（Python 3.12）与 gitcg 0.21.0
python -m pytest -q -m "not slow"                  # 快速门禁
python -m envs.benchmark --games 12 --workers 1    # 单核吞吐
python -m scripts.eval_protocol --sample-sizes     # 评测样本量规划
```

机器路径、吞吐、并发和已知坑见 [docs/environments/](./docs/environments/)。

许可：AGPL-3.0；不携带米哈游卡面/素材。移植范围/归属/洗牌确定性见 [NOTICE.md](./NOTICE.md)。
