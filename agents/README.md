# agents — 对局 Agent（PLAN.md WS4 / M5）
ResolvingAgent（纯策略模式，G1 验收口径）、llm_assist（LLM 辅助模式：候选校验/重排/兜底，延迟预算内，干预全量记录）、
脚本对手适配层。客户端强制降级链：本地 LLM → API → 纯策略。
