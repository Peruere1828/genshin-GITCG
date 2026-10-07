# genshin-GITCG

七圣召唤（Genius Invokation TCG）AI + Agent 工程：自训练策略 AI（SoG/ReBeL 路线）+ 一套 Agent（局内决策、复盘教练、卡组构筑、自迭代闭环、工程自改进）。

- **需求与实现路径**：[PLAN.md](./PLAN.md)（决策记录在 §0，里程碑 M0–M6 在 §4）
- **参考仓**：`refs/`（只读，不修改）：
  - `refs/genius-invokation` — 引擎本体 + `packages/pybinding`（gitcg）+ IO 协议 `docs/development/io.md`
  - `refs/Rebel_base_RL` — SoG 蓝本 `research/world_model/src/gitcg_world_model/` + 脚本专家 `src/gitcg_expert_system/deck_rules/`
  - `refs/AI` — DouZero 式 DMC 基线参考

## 目录

| 目录 | 职责 | 对应 PLAN.md |
|------|------|--------------|
| `envs/` | gitcg Gym 式封装、Player 子类、多进程 rollout | WS0 |
| `reps/` | 观测编码 / 动作分层抽象 / 信念建模 | WS1 |
| `eval/` | arena、22 套脚本对手注册表、TrueSkill 阶梯、报表 | WS2 |
| `train/` | CVPN、continual resolving 蒸馏外环、batched inference | WS3 |
| `agents/` | 纯策略 agent、局内 LLM 辅助、脚本对手适配 | WS4 |
| `coach/` | 复盘归因、课程建议、BC 标签 | WS4 |
| `decklab/` | 卡组构筑（模板内 flex 位） | WS4 |
| `orchestrator/` | 自迭代闭环 + 护栏门禁 | WS5 |
| `tests/` | 信息泄漏测试、adapter 往返、重放一致性（进 CI） | 全部 |
| `configs/` | run 配置、`engine.lock`、卡池版本 | D4 |
| `scripts/` | 一键脚本（`run_iteration` 等，间歇窗口零看管） | §5.1 |
| `data/` `reports/` | 回放池/日志（不入库）、评测与复盘产物 | WS6 |

## 起步

```bash
pip install gitcg          # 或从 refs/genius-invokation/packages/pybinding 本地安装
python -m eval.arena --smoke   # M0 冒烟：脚本对手互打
```

许可：本项目按 AGPL-3.0（上游 genius-invokation / Rebel_base_RL 均为 AGPL）；不携带任何米哈游卡面/素材。
