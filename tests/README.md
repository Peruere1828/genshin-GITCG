# tests — 关键测试（全部进 CI）

信息泄漏测试（两份只差对手隐藏信息的 state，编码必须一致）、action_adapter 往返、同种子重放一致性、非法动作回退。

## 现状（2026-10-07）

`python -m pytest`（从仓库根跑）；`python -m pytest -m "not slow"` 跳过慢测。

- `test_info_hiding.py`：`public_state.mask_state_for_player` 隐藏对手手牌/骰子/牌堆、保留己方与公开信息、幂等。
- `test_info_leak.py`（slow，**WS1 硬门禁**）：只改对手隐藏信息的两份 context，编码器**输入**（token/option/mask/opponent id）
  必须逐字段一致；改公开信息（对手角色血量）必须可见；belief target 必须随 `full_state` 变化（标签由上帝视角算）。
- `test_action_adapter.py`：每个合法具体动作都有 code + kind-appropriate 载荷；动作空间上限护栏；抽象动作 100% 覆盖合法具体动作（往返语义 key 可恢复）；无非法回退。
- `test_replay_consistency.py`（slow）：同种子两次对局结果一致。
- `test_arena.py`（slow）：arena 顺序与多进程结果一致（验证 codebook reset + 预洗牌确定性）。
- `test_env_smoke.py`：脚本对手对局跑通、无 IO 错误 / 无回退。
- `test_stats.py` / `test_ladder.py`：Wilson CI、Elo。
- `test_llm.py`：LLM 客户端 JSON 解析/脱敏、LLM 辅助策略「非法响应回退 + 预算 + prompt 不泄密」、教练结构化输出（mock 客户端，无网络）。
- `test_decklab.py`：换卡保持 30 张、候选池排除、换卡评测小样本（slow）。
- `test_engine_bridge.py`：L5.4/D12——快照往返无损、续跑克隆确定、`InjectionPolicy` 位置/夹取逻辑（快测）；
  replay-branch 注入 base 选择时逐局复现 base、同注入两次一致、`evaluate_decision` 候选覆盖（slow）。

依赖：需要资产缓存（`data/assets/`）；首次运行会联网拉取，CI 可预置缓存并设 `GITCG_ASSETS_OFFLINE=1`。
