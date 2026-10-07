# reps — 观测/动作/信念表示（PLAN.md WS1 / M1，go/no-go 关卡）

observation_encoder（token 化，公私信息两段式）、action_hierarchy + action_taxonomy.toml（动作抽象/分层）、
action_adapter（抽象↔具体映射 + 合法化）、belief_groups（public belief）。
验收：抽象动作 100% 覆盖合法具体动作 + adapter 往返测试通过。

## 已实现（2026-10-07）

逐模块移植自 `refs/Rebel_base_RL/.../gitcg_world_model`（AGPL-3.0，见 `NOTICE.md`）：

- `schema.py` / `snapshot.py`：`StateSnapshot` 等数据结构；从 `notification.state`（proto）或 `state.json()` 快照。
- `action_hierarchy.py` + `action_semantic_rules.py/.toml` + `non_action_semantic_rules.py/.toml` + `action_taxonomy.toml`：
  动作分层与 `HierarchicalActionCodebook` / `ActionLegalityEngine`（低层动作 code ↔ 高层抽象类别）。
- `action_adapter.py`：把具体请求（action/choose_active/reroll/select_card/switch_hands）枚举为候选
  `LowLevelActionSpec` + 响应载荷（`BuiltDecisionContext`）。
- `public_state.py`：`mask_state_for_player`（对局信息隐藏）+ public belief 起点。
- `semantic_priors.py` / `agents.py` / `replay.py`：先验、legal-random/heuristic 基线、回放序列化。

待办：`observation_encoder`（token 化观测 + 信息泄漏测试）、`belief_groups`、抽象动作覆盖/往返测试加固。
