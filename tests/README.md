# tests — 关键测试（全部进 CI）

信息泄漏测试（两份只差对手隐藏信息的 state，编码必须一致）、action_adapter 往返、同种子重放一致性、非法动作回退。

## 现状（2026-10-07）

`python -m pytest`（从仓库根跑）；`python -m pytest -m "not slow"` 跳过慢测。

- `test_info_hiding.py`：`public_state.mask_state_for_player` 隐藏对手手牌/骰子/牌堆、保留己方与公开信息、幂等。
  （观测编码器级"两份只差隐藏信息的 state 编码必须一致"待 `reps/observation_encoder` 落地后补。）
- `test_action_adapter.py`：每个合法具体动作都有 code + kind-appropriate 载荷；动作空间上限护栏；无非法回退。
- `test_replay_consistency.py`（slow）：同种子两次对局结果一致。
- `test_env_smoke.py`：脚本对手对局跑通、无 IO 错误 / 无回退。

依赖：需要资产缓存（`data/assets/`）；首次运行会联网拉取，CI 可预置缓存并设 `GITCG_ASSETS_OFFLINE=1`。
