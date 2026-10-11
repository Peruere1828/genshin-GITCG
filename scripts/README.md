# scripts — 一键脚本（PLAN.md §5.1 机会窗口零看管）

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
  无 key/网络不可用时 LLM 模式以退出码报错（纯策略用 `--no-llm`）；全部跑完后重跑只重新聚合，不再需要 key。
  客户端失败的比赛逐局标 `degraded` 并在报告 `llm_quality` 中单列（含非降级 `clean` 胜率）；连续 N 次
  （默认 5，`--max-consecutive-errors`）客户端失败即中止待续跑，保证 llm 列始终是 LLM 辅助口径。
- `explore/fetch_deck_corpus.py`（探索任务 1）：抓取 miHoYo「卡牌广场」公开 API 的卡组 +
  手工抄录的公开攻略分享码，用引擎编解码逐条验证（3 角色 + 30 行动牌），可选各跑 1 局
  `run_match`；写 `data/deck_corpus/`（gitignore），`--export-reports` 另存紧凑副本。
  `explore/render_deck_report.py` 渲染 `reports/explore/deck_corpus.md`。
  ```bash
  python -m scripts.explore.fetch_deck_corpus --no-fetch --engine --export-reports
  python -m scripts.explore.render_deck_report
  ```
- `eval_protocol.py`（WS2，2026-10-11）：**强度评测跑分板**——任意 `build_policy` spec 对固定脚本池、
  双侧同种子，逐对手 + 汇总胜率（Wilson + 对手聚类 CI）、固定锚点 Bradley-Terry/Elo 评级、最优反应探针、
  门禁判定；逐局流式落 `data/eval/<tag>.jsonl`（按 `task_index` 断点续跑 + spec 指纹护栏），聚合落
  `reports/explore/`。协议全文见 `reports/explore/eval_protocol.md`。
  ```bash
  python -m scripts.eval_protocol --sample-sizes                 # 样本量规划表（无需引擎）
  python -m scripts.eval_protocol --contestant expert:superconduct_aggro \
      --opponents unyielding_geo natlan_battleship --seeds 2 --workers 4   # 小样本冒烟
  python -m scripts.eval_protocol --contestant neural:<new> --baseline neural:<old> --seeds 20  # 配对门禁
  ```
- `bench_ops.py` / `bench_model.py`（M3 探索）：MX550 算子/训练吞吐基准，以及 mean-pool CVPN vs
  token-transformer CVPN 的同池对比。
  ```bash
  python -m scripts.bench_ops --device cuda --out reports/explore
  python -m scripts.bench_model --games 20 --d-model 128 --device cuda
  ```

待办：`run_iteration`（采集→训练→评测→落盘，断点续跑）。依赖 `train/` 落地。
