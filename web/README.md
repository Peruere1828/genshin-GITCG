# web/ — 浏览器 ↔ Python 桥（WS8/D18）

本目录放 **Web 人机调试界面** 的 Python 侧最小服务，配合
`engine-overlay/overlay/packages/standalone/` 的浏览器端改动使用。

## `ai_rpc_server.py`

纯标准库（`http.server`）实现，把引擎的 `PlayerIO` RPC 用 HTTP 暴露给浏览器：

| 端点 | 请求体 | 返回 |
|---|---|---|
| `POST /rpc` | `{"who":0\|1,"request":RpcRequest}` | `RpcResponse` |
| `POST /notify` | `{"who":0\|1,"notification":{mutation,state}}` | `{}` |
| `GET /health` | – | `{"ok":true,"rpc":{...},"notify":N}` |
| `GET /state` | – | `{"state": 最近一次通知的 GameState}` |

内置策略（`--strategy greedy|random`，`--seed`）：
`rerollDice` 不重掷、`switchHands` 不换手、`chooseActive`/`selectCard` 取候选第一、
`action` 优先选「合法且非 declareEnd」的动作。

```sh
conda run -n gitcg python web/ai_rpc_server.py --port 8770 --strategy random --seed 1
# 另开一个终端：
bash engine-overlay/scripts/standalone.sh dev
# 浏览器 http://localhost:5173/ 勾「AI 对手（HTTP 桥）」+「双方 AI 自对弈」
```

无第三方依赖、无 node_modules；缓存只有 `__pycache__/`（根 `.gitignore` 已忽略）。

## 定位

这是**调试/训练辅助接口**，不进口径验收（D18）。下一步把随机策略换成
`reps/` + `train/` 的 CVPN checkpoint 或 `common/llm.py`，即「HTTP 桥 → 本地模型」。
