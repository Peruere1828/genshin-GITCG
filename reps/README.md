# reps — 观测/动作/信念表示（PLAN.md WS1 / M1，go/no-go 关卡）
observation_encoder（token 化，公私信息两段式）、action_hierarchy + action_taxonomy.toml（动作抽象/分层）、
action_adapter（抽象↔具体映射 + 合法化）、belief_groups（public belief）。
验收：抽象动作 100% 覆盖合法具体动作 + adapter 往返测试通过。
