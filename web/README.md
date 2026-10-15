# web/ — 浏览器 ↔ Python 桥（WS8/D18/C2）

本目录放 **Web 人机调试界面** 的 Python 侧最小服务，配合
`engine-overlay/overlay/packages/standalone/` 的浏览器端改动使用。

## `ai_rpc_server.py`

纯标准库（`http.server`）实现，把引擎的 `PlayerIO` RPC 用 HTTP 暴露给浏览器：

| 端点 | 请求体 | 返回 |
|---|---|---|
| `POST /rpc` | `{"who":0\|1,"request":RpcRequest}` | `RpcResponse` |
| `POST /notify` | `{"who":0\|1,"notification":{mutation,state}}` | `{}` |
| `GET /health` | – | `{"ok":true,"policy":"<spec>","rpc":{...},"notify":N}` |
| `GET /state` | – | `{"state": 最近一次通知的 GameState}` |

决策有两档：

| 档位 | 触发 | 行为 |
|---|---|---|
| 占位策略（默认） | 不给 `--policy` | 纯标准库；`--strategy greedy\|random` + `--seed`。`rerollDice` 不重掷、`switchHands` 不换手、`chooseActive`/`selectCard` 取候选第一、`action` 优先「合法且非 declareEnd」 |
| 模型桥 | `--policy SPEC` | 五种请求都走 `envs.policy.build_policy(SPEC, seed=--seed, role="web")` + `reps.action_adapter.build_decision_context`；`SPEC` 支持 `neural:<ckpt.pt>[#t=<temp>]`、`heuristic`、`legal_random`、`expert:<slug>`、`random` |

模型桥只在 `--policy` 时延迟导入 `reps`/`envs`/`gitcg`（torch 仅 neural 才用），
默认路径仍是零外部依赖。任何加载/转换/`choose()`/非法动作码失败都会写 stderr
警告并回退占位策略，handler 不会 500 或挂死。响应字段用引擎要求的 camelCase
（`chosenActionIndex`/`usedDice`/`activeCharacterId`/`selectedDefinitionId`/
`diceToReroll`/`removedHandIds`）。

```sh
# 占位策略（零依赖）：
conda run -n gitcg python web/ai_rpc_server.py --port 8770 --strategy random --seed 1

# 模型桥：用训练好的 CVPN 驱动 AI 座位（checkpoint 自带 encoder 配置）：
conda run -n gitcg python web/ai_rpc_server.py --port 8770 \
    --policy "neural:data/checkpoints/curve_dfbcsa_e15_d128_n48000.pt#t=0.2"
# 也可用基线/脚本专家：--policy heuristic | legal_random | expert:superconduct_aggro

# 另开一个终端：
bash engine-overlay/scripts/standalone.sh dev   # 会先 apply overlay 再起 dev server
# 浏览器 http://localhost:5173/ 勾「AI 对手（HTTP 桥）」+「双方 AI 自对弈」
```

CORS 默认只放行本机 Origin（`localhost`/`127.0.0.1`/`::1`）；如需其他来源
（例如经局域网 IP 访问 dev server）用 `--cors-origin <origin|*>` 显式放行。

无第三方依赖、无 node_modules；缓存只有 `__pycache__/`（根 `.gitignore` 已忽略）。

## 定位

这是**调试/训练辅助接口**，不进口径验收（D18）。C2 已把占位策略接到
`reps/` + `train/` 的 CVPN checkpoint（`--policy neural:...`），即
「HTTP 桥 → 本地模型」；`common/llm.py` 侧作为后续可选分支。

预检响应带 `Access-Control-Max-Age`（`_CORS_MAX_AGE_SECONDS`），浏览器会缓存
OPTIONS 结果，跨域访问时不再每个请求配一次预检。

## 待办

- `/notify` 目前是**每步一次的全量状态上报**（一局数千条）。可改增量/节流/合并
  以降请求量；未做（当前不是瓶颈）。
