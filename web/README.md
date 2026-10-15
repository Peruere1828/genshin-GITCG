# web/ — 浏览器 ↔ Python 桥（WS8/D18/C2 + T1/T2/T3/C4）

本目录放 **Web 人机调试界面** 的 Python 侧最小服务，配合
`engine-overlay/overlay/packages/standalone/` 的浏览器端改动使用。

## `ai_rpc_server.py`

纯标准库（`http.server`）实现，把引擎的 `PlayerIO` RPC 用 HTTP 暴露给浏览器：

| 端点 | 请求体 | 返回 |
|---|---|---|
| `POST /rpc` | `{"session","who":0\|1,"request":RpcRequest}` | `RpcResponse` |
| `POST /notify` | `{"session","who","seq","notification":{mutation,state}}` | `{"applied":bool,"seq","state_seq"}` |
| `POST /session` | `{"session","meta":{version,ai_seat,ai_self_play,decks}}` | 会话记录 + `deck_mismatch` |
| `POST /game-end` | `{"session","winner"}` | `{"ok","closed"}` |
| `POST /game-log` | `{"session","log"}` | `{"ok","dir"}`（引擎日志归档） |
| `GET /health` | – | `{"ok":true,"policy","rpc","notify","sessions","stale_notify","ai_deck","log_dir"}` |
| `GET /state?session=&who=` | – | 该座位最近一次掩码 GameState（不带参数＝老行为，返回最近一条） |
| `GET /ai-deck` | – | `{"slug","characters","cards"}`（T3 牌组对齐） |
| `GET /results` | – | `{"log_dir","sessions":[…]}` |

`session` 缺省落到 `"-"`（老客户端 / 直接 curl 调试）。

### 三个隔离维度（T1/T2）

- **座位**：引擎 `exposeState(who,…)` 按座位掩码（自己手牌/骰子可见、对手隐 id），
  所以每个 `(session, who)` 各存一份 `latest_state` 与策略实例——双方 AI 自对弈时
  不再互相覆盖（否则某座位会拿到对手视角的掩码态：自家手牌反被隐藏）。
- **会话**：一局一个 id（浏览器每局 `crypto.randomUUID()`），状态 / 策略 / 决策
  计数 / 日志目录都按 session 隔离，多开标签页不串局。策略 role = `web{who}`
  （非默认会话再叠 `@{sid}`），seed 从 `--seed` 派生 → 同一 session 内可复现。
- **通知序号**：`/notify` 带座位内单调 `seq`，过期通知直接丢弃（计数进
  `/health.stale_notify`）；浏览器侧把 notify 串成队列、发 rpc 前先等队列排空，
  杜绝「决策吃到上一步状态」的竞态。不带 `seq` 的请求照常应用（向后兼容）。

### 决策档位

| 档位 | 触发 | 行为 |
|---|---|---|
| 占位策略（默认） | 不给 `--policy` | 纯标准库；`--strategy greedy\|random` + `--seed` |
| 模型桥 | `--policy SPEC` | `envs.policy.build_policy` 全 spec（`neural:<ckpt>[#t=]` / `heuristic` / `legal_random` / `expert:<slug>` / `random`）；**每个 (会话, 座位) 一个独立实例** |

reps/train/torch 只在请求模型策略时延迟导入，默认路径仍是零依赖。任何加载 /
转换 / `choose()` / 非法动作码失败都写 stderr 并回退占位策略，handler 不 500
也不挂死，**且回退会落进决策日志**（`fallback:true` + `error`）。

### T3 牌组对齐

`--ai-deck <slug>`（不给则从 `--policy expert:<slug>` 推断）声明 AI 座位该用的牌组：

- `GET /ai-deck` 把 `{characters,cards}` 给前端，前端用
  `DEFAULT_ASSETS_MANAGER.encode` 回填成分享码（配置页「取 AI 牌组」）；
- 开局 `POST /session` 里浏览器上报实际牌组，与声明不一致就 stderr 告警 + 会话
  记 `deck_mismatch:true`。

原因：脚本专家规则是按自家牌组写的，牌组不符会**静默变蠢**。

### C4 逐决策日志

`--log-dir`（默认 `data/web/`，已在 gitignore 内）每局一个目录：

```
data/web/20261015T120000Z_<session前24字符>/
  session.json     # 开局元数据（版本 / 牌组 / ai_seat …）
  decisions.jsonl  # 每次决策一行 + 终局 game_end 一行
  summary.json     # 关账汇总（胜负 / 回合数 / 各座位决策数 / 回退数 / 过期通知数）
  game_log.json    # 浏览器上传的引擎日志（含人类这边的决策）
```

决策行字段：`t / iso / type / session / who / method / step / state_seq /
notify_seq / latency_ms / policy / fallback / error / response`。
`--no-log` 关闭；写盘失败只告警、不影响对局。终局优先信浏览器的 `POST
/game-end`，`/notify` 里看到 `winner`（或 phase=gameEnd）也会兜底关账。

## 用法

```sh
# 占位策略（零依赖）：
conda run -n gitcg python web/ai_rpc_server.py --port 8770 --strategy random --seed 1

# 模型桥：用训练好的 CVPN 驱动 AI 座位（checkpoint 自带 encoder 配置）：
conda run -n gitcg python web/ai_rpc_server.py --port 8770 \
    --policy "neural:data/checkpoints/curve_dfbcsa_e15_d128_n48000.pt#t=0.2" \
    --ai-deck superconduct_aggro

# 脚本专家（牌组自动对齐 + 逐决策日志）：
conda run -n gitcg python web/ai_rpc_server.py --port 8770 \
    --policy expert:superconduct_aggro

# 另开一个终端：
bash engine-overlay/scripts/standalone.sh dev   # 会先 apply overlay 再起 dev server
# 浏览器 http://localhost:5173/ 勾「AI 对手（HTTP 桥）」（自动取 AI 牌组回填）+「双方 AI 自对弈」
```

CORS 默认只放行本机 Origin（`localhost`/`127.0.0.1`/`::1`）；如需其他来源
（例如经局域网 IP 访问 dev server）用 `--cors-origin <origin|*>` 显式放行。
预检响应带 `Access-Control-Max-Age`（`_CORS_MAX_AGE_SECONDS`），浏览器缓存
OPTIONS 结果，跨域访问时不再每个请求配一次预检。

无第三方依赖、无 node_modules；缓存只有 `__pycache__/`（根 `.gitignore` 已忽略）。

## 定位

这是**调试/训练辅助接口**，不进口径验收（D18）。C2 已把占位策略接到
`reps/` + `train/` 的 CVPN checkpoint（`--policy neural:...`），即
「HTTP 桥 → 本地模型」；`common/llm.py` 侧作为后续可选分支。

## 测试

`tests/test_web_rpc_server.py`：占位路径、snake→camel 映射、proto-JSON→
StateSnapshot、真实对局抓取的端到端决策、oneof 摊平、回退、子进程直跑、HTTP
路由，以及 T1 座位/会话隔离、T2 过期通知护栏、T3 牌组校验、C4 日志生命周期。

## 待办

- `/notify` 目前是**每步一次的全量状态上报**（一局数千条）。可改增量/节流/合并
  以降请求量；未做（当前不是瓶颈）。
