# eval — 评测与基准（PLAN.md WS2，先于训练建成）

22 套脚本对手注册表（refs/Rebel_base_RL/.../gitcg_expert_system/deck_rules/）、arena 并行批量对局、
TrueSkill/Elo 阶梯（固定基线 + 历史 checkpoint 双向）、分卡组胜率矩阵、逐决策日志（供 coach/ 消费）。

## 已实现（2026-10-07）

- `opponents.py`：对手池注册表（20 套脚本专家 `expert:<slug>` + heuristic / legal_random 基线）。
- `stats.py`：Wilson 置信区间与 `estimate_rate`（小样本验收口径）。
- `arena.py`：`ArenaSpec`/`run_arena`（contestant × pool × paired seeds，可选 swap），输出
  每对局矩阵（含 CI）、contestant 总胜率、Elo；落盘 `reports/arena/*.json|csv`，原始记录 `data/arena/*.jsonl`。
  CLI：`python -m eval.arena --smoke --seeds 3 --workers 12`。
- `ladder.py`：基于对局记录的 Elo（固定基线 + 历史 checkpoint 的阶梯骨架）。

待办：TrueSkill（或 Bradley-Terry）、历史 checkpoint 双向阶梯、逐决策日志接入 coach、曲线出图。
