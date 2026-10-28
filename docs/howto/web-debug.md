# howto — Web 人机调试接口

> 协议权威：`web/README.md`。定位：对 AI 的调试接口，不进口径验收。

## 启动

```bash
# standalone overlay
cd engine-overlay && bash scripts/standalone.sh
# Python HTTP 桥
python web/ai_rpc_server.py --policy neural:<ckpt>#t=0 --ai-deck <deck>
```

- 浏览器引擎 ↔ Python 桥；观测只取 `notification.state`。
- 支持 `expert:<slug>` / `neural:<ckpt>[#t=]` / `heuristic` / `legal_random`。
- 桥按 `(session, who)` 隔离；`/notify` 带 `seq` 护栏；决策不吃陈旧状态。
- 日志落 `data/web/<时间戳>_<session>/`（`session.json`/`decisions.jsonl`/`summary.json`/`game_log.json`）。

## 已知差异

- NPC 牌组 `cards=[]`（无手牌，靠空牌堆）；持 NPC 阵容的座位引擎 `alwaysOmni`（每轮 8 万能骰）。
- 支援区/多轮挑战与游戏内机制不一致；差异逐条记录在 `web/README.md` C3 表与 `semantics`。
- 续局：`/session` 带 `resume` → 桥回 `seq_base`；浏览器从基线继续数。
- 桥 IO 故障局标 `aborted`/`io_error`，不算胜负。

## 冒烟

```bash
python -m pytest tests/test_web_rpc_server.py -q
```
