# scripts — 一键脚本（PLAN.md §5.1 间歇窗口零看管）

`run_iteration`：采集→训练→评测→落盘，可断点续跑、可重提；窗口内优先级 训练 > 评测 > 采集 > LLM 分析。

## 现状（2026-10-07）

- `run_llm_agent.py`：固定对局下「纯策略 vs LLM 辅助」小样本对比，产出
  `reports/llm/*.json`（含 Wilson CI、调用量、干预记录）。用法：
  ```bash
  python -m scripts.run_llm_agent --seeds 3 --budget 6          # LLM 辅助(deepseek-chat)
  python -m scripts.run_llm_agent --seeds 3 --no-llm            # 纯策略
  ```
  网络不可用/无 key 时自动退化为纯策略（不报错）。
- `probe_engine_snapshot.py`（L5.4）：实测 pybinding 中局快照行为（往返无损 / 克隆确定 / 活体复现率），
  产出 `reports/engine/probe_snapshot_*.json`。`--all` 扫全部快照。
  ```bash
  python -m scripts.probe_engine_snapshot --seed 3 --sample 8
  ```
- `probe_boundary_fork.py`（L5.4，D13 依据）：record-replay 判定分叉保真——边界快照
  （`is_resumable()` + attrs 镜像）复现活体终局、phase 内部点不复现，产出
  `reports/engine/probe_boundary_fork_*.json`。
  ```bash
  python -m scripts.probe_boundary_fork --seed 3 --sample 8
  ```
- `run_replay_branch.py`（L5.4/D12）：对某决策做 replay-branch MC 动作价值估计，产出
  `reports/train/replay_branch_*.json`。
  ```bash
  python -m scripts.run_replay_branch --seed 3 --rollouts 4 --top-k 4 --rollout-spec legal_random
  ```
- `run_llm_matrix.py`（L5.2）：把「纯策略 vs LLM 辅助」从小样本单局放大到**全对手池**，两模式分开报分
  （§6.3），结果逐局流式落 `data/llm/<tag>.jsonl`（按 `task_index` 断点续跑 + spec 指纹护栏），
  聚合报告落 `reports/llm/matrix_<ts>_<tag>.json`；逐干预明细只在原始流里。
  ```bash
  python -m scripts.run_llm_matrix --smoke                    # 3 对手 x 3 种子冒烟
  python -m scripts.run_llm_matrix --seeds 10 --budget 6      # 全 20 套对手，隔夜
  python -m scripts.run_llm_matrix --no-llm --seeds 10        # 只跑纯策略（不耗 API）
  ```
  无 key/网络不可用时以退出码报错（纯策略用 `--no-llm`），不会静默退化成错误口径。

待办：`run_iteration`（采集→训练→评测→落盘，断点续跑）。依赖 `train/` 落地。
