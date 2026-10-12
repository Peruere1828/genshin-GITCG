# WS8 standalone overlay（D18）

本目录是上游 `packages/standalone/` 的 overlay 副本（D20：`overlay/` 是引擎仓根镜像，
由 `engine-overlay/scripts/apply-overlay.sh` 复制进 `build/engine/`）。
`refs/` 只读，我们的改动全部在这里。

## 我们改了什么

| 文件 | 改动 | 目的 |
|---|---|---|
| `vite.config.ts` | 新增 `envPrefix` + `define`，注入 `DEFAULT_STATIC_DATA_API_BASE_URL` / `globalThis.__GI_TCG_ASSETS_BASE_URL__`；`server.host/port` | 让浏览器里的 assets-manager 指向可达镜像；命令行可覆盖 |
| `src/ai-http-io.ts` | **新增**：HTTP 版 `CancellablePlayerIO` | 浏览器引擎 ↔ Python 服务的桥 |
| `src/App.tsx` | 配置页新增「AI 对手（HTTP 桥）」复选框 + 地址输入 + 「双方 AI 自对弈」 | 开局入口 |
| `src/StandaloneParent.tsx` | `aiOpponent`/`aiSelfPlay` props：把某座位 io 换成 HTTP 桥，AI 模式不弹窗 | 接线 |

另有一处 **assets-manager** 的 overlay：
`../assets-manager/src/constants.ts` 把 `DEFAULT_STATIC_DATA_API_BASE_URL` 的兜底默认值
从不可达的 `https://static-data.piovium.org/` 改成本机可达的 v5 镜像
`https://gi-tcg-assets-api-hf.guyutongxue.site/`（仍保留 `import.meta.env` /
`process.env` / `GI_TCG_ASSETS_BASE_URL` 三级覆盖）。

## 构建 & 起局

前置：`build/engine/` 已 materialize（`engine-overlay/scripts/materialize.sh`）；
宿主 node24 + pnpm12（或 B0 的 Docker 镜像）。

```sh
# 1. 应用 overlay（把上面的文件复制进 build/engine）
bash engine-overlay/scripts/standalone.sh apply

# 2. 构建 standalone 的全部 workspace 依赖（web-ui-core / deck-builder /
#    state-editor / detail-log-viewer / data-code-analyzer / assets-manager）
#    —— 注意要用可达镜像，否则 assets-manager 的 build:data 会卡在 static-data.piovium.org
bash engine-overlay/scripts/standalone.sh build

# 3. 起 dev server
bash engine-overlay/scripts/standalone.sh dev
#   ➜  http://localhost:5173/
```

**为什么先 build 依赖**：standalone 的若干 workspace 依赖（`@gi-tcg/web-ui-core` 等）
package.json 只导出 `dist/`，Vite dev 无法直接解析其源码，因此必须先构建。
只有 `@gi-tcg/data` / `@gi-tcg/assets-manager` 有 `development` 条件会走 `src/`。

**镜像 / base URL**：`assets-manager` 走 `${base}/api/v5/{data,datum,image}/...`。

- 构建期：`pnpm --filter "@gi-tcg/assets-manager" build:data` 拉 `names/characters/
  entities/keywords` 等，写入 `packages/assets-manager/src/data/*.json`（随 dist 打包；
  `src/data/` 已在引擎 `.gitignore`）。
- 运行期：`assets-manager` 由此 base URL 拉卡牌图/数据。镜像返回
  `access-control-allow-origin: *`，浏览器可直连。
- 覆盖：`GI_TCG_ASSETS_BASE_URL=https://... bash engine-overlay/scripts/standalone.sh dev`。

## 起局

1. 浏览器打开 `http://localhost:5173/`。
2. **双人类（上游默认）**：直接点「开始对局」，先手方棋盘在弹窗、后手方在本页。
   （需要浏览器允许弹出窗口。）
3. **AI 对手（HTTP 桥）**：勾选「AI 对手（HTTP 桥）」，地址填
   `http://127.0.0.1:8770`；先起 Python 服务：

   ```sh
   conda run -n gitcg python web/ai_rpc_server.py --port 8770 --strategy random --seed 1
   ```

   则**先手方（who=0）由 Python 托管、不弹窗**，本页为后手方真人。
   再勾「双方 AI 自对弈」可让两边都交给 Python，便于零看管跑完整局。
4. 「摆盘器」「导入日志」「查看历史/从此处继续」等上游功能保持可用。

## HTTP 桥协议（最小原型）

浏览器侧 `createAiHttpIo`（`src/ai-http-io.ts`）：

- `rpc(request)` → `POST {base}/rpc`，body `{"who":0|1,"request":RpcRequest}`，期望回
  `RpcResponse`（`{"response":{"$case":..., "value":...}}`）。
- `notify(notification)` → `POST {base}/notify`，body `{"who":..,"notification":{mutation,state}}`
  （fire-and-forget，AI 用来“看盘”）。
- `cancelRpc()` → `AbortController.abort()`。

`RpcRequest/Response` 定义见 `packages/typings/src/gen/rpc.ts`；座位 `action` 请求里带
合法动作列表（`validity===0` 为合法）、每个动作的 `autoSelectedDice`（引擎自动选骰），
所以 `action` 回复 = `{chosenActionIndex, usedDice: action[i].autoSelectedDice}` 即可。

## 已知坑

- **`static-data.piovium.org` 不可达**（本机 connect timeout）：所以必须用镜像 +
  上面的 `constants.ts` 兜底；否则卡牌图/`build:data` 全挂。
- **必须先 build 依赖**（见上），否则 dev 起不来。
- **上游第三方 async-context**：游戏结束时会抛一次非致命
  `ReferenceError: contextStorage is not defined`，来自 esm.sh 的
  `@beaker73/async-context`（其 `eval` 生成的 `finally` 引用了未定义的
  `contextStorage`），**不影响对局结果**，仅控制台噪声。
- 宿主 node24 可用（上游 `engines` 写 node ^26，实测 node v24.15 正常）。

## 下一步（WS8 路线）

1. **人选牌**：在主页面用 `@gi-tcg/deck-builder` 选我方牌组（上游已有组件，入口在
   「启动组牌器」）。
2. **AI 看对手牌反选**：把对手（AI）可见的公共态经 `/notify` 发给 Python，Python
   根据我方牌组反选一套应对牌组，回填到 `deck1`（或开局前的一次性 `/choose-deck`）。
3. **开局**：随机先手；把 `aiOpponent` 从「先手固定」推广到「随机座位」。
4. **模型桥**：把 `web/ai_rpc_server.py` 的随机策略换成调用 `reps/`+`train/` 的
   CVPN checkpoint（或 `common/llm.py`），即「HTTP 桥 → 本地模型」。
5. **NPC 卡组**：允许我方持 NPC 敌人卡组（D18），复用 `configs/npc/`。
