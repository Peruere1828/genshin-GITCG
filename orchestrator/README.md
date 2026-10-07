# orchestrator — 自迭代闭环（PLAN.md WS5 / M6）
计划(hypothesis) → 实验 → 训练/自博弈 → 评测 → 复盘 → 门禁(单测+泄漏测试+冒烟+评测不劣于) → 合并/回滚。
护栏：Agent 改动只进隔离分支；checkpoint/数据/配置只追加不覆盖。
