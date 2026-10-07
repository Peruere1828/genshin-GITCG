# agents — 对局 Agent（PLAN.md WS4 / M5）

ResolvingAgent（纯策略模式，G1 验收口径）、llm_assist（LLM 辅助模式：候选校验/重排/兜底，延迟预算内，干预全量记录）、
脚本对手适配层。客户端强制降级链：本地 LLM → API → 纯策略。

## 已实现（2026-10-07）

- `scripted/`：从 `refs/Rebel_base_RL/.../gitcg_expert_system` 逐模块移植的脚本对手
  （`ExpertRuleAgent` + `deck_rules/` 20 套 + `registry`/`profiles`/`codec`/`assets`/`models`），AGPL-3.0，见 `NOTICE.md`。
  - `load_deck_profiles()` / `load_deck_profiles_by_slug()` 解码 share code → `DeckProfile`；
    资产（卡名/标签/shareId 映射）首次从公开 assets API 拉取并缓存到 `data/assets/`（可设
    `GITCG_ASSETS_CACHE` / `GITCG_ASSETS_OFFLINE=1`）。
  - 经 `envs.policy.expert_policy(slug)` 接入对局。

待办：`llm_assist`（DeepSeek API 局内辅助，双模式分开报分）、`resolving_agent`（纯策略搜索，依赖 `train/`）。
