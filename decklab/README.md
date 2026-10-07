# decklab — 卡组构筑 Agent（PLAN.md WS4）

固定模板内调 5–15 张 flex 位：换卡敏感性分析、对 22 套假想敌对位胜率模拟。结论固化为新 deck-list 进评测池。

## 已实现（2026-10-07）

- `sensitivity.py`：固定 30 张模板，做「移 1 张 flex 卡 + 加 1 张候选卡」的换卡评测。
  - `mutate_deck`、`candidate_cards`（来自其它脚本卡组的候选池）。
  - `evaluate_swaps`：每张换卡在配对种子上 vs 固定对手池跑小样本，输出对 baseline 的胜率差 + Wilson CI。
  - 由于改牌后无对应脚本专家，己方用固定通用策略（`heuristic`），对手仍用脚本专家。
  - 通过 `MatchTask.deck0_inline` 把改后牌组传给 worker 进程（无需注册）。
- CLI：`python -m decklab.sensitivity --deck superconduct_aggro --seeds 5 --workers 12`。

待办：LLM 提议换卡（结构化候选）、更强己方策略（训练后再做换卡敏感性）、结论固化进评测池。
