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

待办：`run_iteration`（采集→训练→评测→落盘，断点续跑）。依赖 `train/` 落地。
