# envs — 环境与工程基座（PLAN.md WS0 / M0）
gitcg (pybinding) 的 Gym 式封装、Player 子类（on_action/on_reroll_dice/on_choose_active/on_select_card/on_switch_hands/on_notify）、
多进程 rollout runner、episode 日志格式。观测只允许来自 notification.state（信息隐藏版，见 tests/ 的泄漏测试）。
