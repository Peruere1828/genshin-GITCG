# WS8/C0 — Web 人机调试界面起步（standalone 构建 + HTTP 桥）

日期：2026-10-12 · 分支：主线工作区 · 口径：调试接口，不进口径验收（D18）

## 结论

- **上游 `standalone` 前端已在本机 overlay 里构建起来并成功起局**（浏览器跑 TS 引擎
  `@gi-tcg/core` + `web-ui-core`），双人类弹窗流程正常。
- **浏览器引擎 ↔ Python 的 HTTP 桥打通**：AI-vs-AI 自对弈在无头 Chromium 里跑到
  `gameEnd`（10 回合，Python 处理 143 次 `action` + 479 条通知），端到端可复现。
- `refs/` 未改；改动只在 `engine-overlay/`（入库）与 `build/`（gitignore）；素材运行时
  按 id 从镜像拉取，不入库。

## 工具链 / pin

| 项 | 值 |
|---|---|
| 引擎源码 | `v0.21.0` / `c94826b4611a19f0057d667f7db3b0f9aaf68c7e`（B0 已 materialize 到 `build/engine`） |
| 宿主 node | v24.15.0（上游 `engines` 写 ^26，实测 24 正常） |
| pnpm | 12.8.1 |
| vite | 8.0.8（引擎 catalog） |
| 静态数据镜像 | `https://gi-tcg-assets-api-hf.guyutongxue.site/`（v5 API，`access-control-allow-origin: *`） |
| overlay hash（本阶段） | `overlay_tree_sha256=1d7db4f0a53fd7d814a5df003c2ffd4300c61c62a3f0afd2d28c3a82292113a6` |

## 起局命令与结果

```sh
# 1. 应用 overlay
bash engine-overlay/scripts/apply-overlay.sh
# 2. 构建 standalone 全部 workspace 依赖（用镜像）
GI_TCG_ASSETS_BASE_URL=https://gi-tcg-assets-api-hf.guyutongxue.site/ \
  DEFAULT_STATIC_DATA_API_BASE_URL=https://gi-tcg-assets-api-hf.guyutongxue.site/ \
  pnpm --filter "@gi-tcg/standalone^..." build      # 在 build/engine 下，退出码 0
# 3. 起 dev
bash engine-overlay/scripts/standalone.sh dev       # http://localhost:5173/
```

结果（无头 Chromium via Playwright）：

| 场景 | 结果 |
|---|---|
| 双人类（上游默认，弹窗） | 首页加载 OK；点「开始对局」弹窗打开；`window.theGame` 就绪；26 个镜像请求（entities + 卡图）；**0 页面错误**；停在 `initHands` 等人类操作（预期） |
| AI 对手 + 双方自对弈 | 无弹窗起局；10 回合跑到 `gameEnd`；`/health` 显示 `{"switchHands":2,"chooseActive":4,"rerollDice":20,"action":143}`、`notify:479`；**0 致命错误** |

## overlay 改动清单

| 路径 | 类型 | 说明 |
|---|---|---|
| `engine-overlay/overlay/packages/standalone/vite.config.ts` | 改 | 注入 assets base URL（`envPrefix`+`define`）、dev server host/port |
| `engine-overlay/overlay/packages/standalone/src/ai-http-io.ts` | 新增 | HTTP 版 `CancellablePlayerIO` |
| `engine-overlay/overlay/packages/standalone/src/App.tsx` | 改 | 「AI 对手（HTTP 桥）」开关/地址 + 「双方 AI 自对弈」 |
| `engine-overlay/overlay/packages/standalone/src/StandaloneParent.tsx` | 改 | `aiOpponent`/`aiSelfPlay` 接线；AI 模式不弹窗 |
| `engine-overlay/overlay/packages/standalone/README.md` | 新增 | 该 overlay 的构建/起局/协议/坑/下一步 |
| `engine-overlay/overlay/packages/assets-manager/src/constants.ts` | 改 | `DEFAULT_STATIC_DATA_API_BASE_URL` 兜底改为可达镜像（三级覆盖保留） |
| `engine-overlay/scripts/standalone.sh` | 新增 | apply/build/dev/preview 一条龙 |
| `engine-overlay/README.md` | 追加 | WS8 段落指向上述内容 |
| `web/ai_rpc_server.py` | 新增 | Python HTTP 桥（纯标准库，随机/贪心策略） |
| `web/README.md` | 新增 | 协议与服务说明 |

## base URL / 素材处理

- 上游 `DEFAULT_STATIC_DATA_API_BASE_URL` 默认 `https://static-data.piovium.org/`，本机
  **不可达**（connect timeout）。处置：
  1. `build:data`（构建期拉 name/category 数据）与运行期取图/取数据都指向镜像；
  2. overlay 把兜底默认值改成镜像，`envPrefix`/`define`/`process.env` 可覆盖；
  3. 镜像 CORS `*`，浏览器直连即可，无需自建反代。
- 素材**不入库**：`build/engine/packages/assets-manager/src/data/`、`dist/`、卡图缓存都在
  `build/`（gitignore）与引擎自身 `.gitignore` 内。

## 遇到的坑

1. **standalone 依赖包只在 `dist/` 导出**（`web-ui-core`/`deck-builder`/`state-editor`/
   `detail-log-viewer`/`data-code-analyzer`）：Vite dev 无法解析源码 → 必须先
   `pnpm --filter "@gi-tcg/standalone^..." build`；用 `GI_TCG_ASSETS_BASE_URL` 让
   `assets-manager` 的 `build:data` 走镜像。
2. **`static-data.piovium.org` 阻断**：见上；`assets-manager` 虽非 `.so` 构建所需，但
   standalone 真依赖它。
3. **`import.meta.env?.X` 的可选链**：Vite `define` 的字符串替换匹配不到
   `import.meta.env?.DEFAULT_...`，故走 `envPrefix` + overlay 兜底默认值双保险。
4. **上游第三方 async-context**：游戏结束时抛一次非致命
   `ReferenceError: contextStorage is not defined`（esm.sh 的 `@beaker73/async-context`
   用 `eval` 生成的 `finally` 引用了未定义的 `contextStorage`）。**不影响对局结果**，
   仅控制台噪声。
5. GitHub 在本网络不可达（不影响：引擎源码 B0 已物化；standalone 依赖走 npm
   registry，可达）。

## HTTP 桥原型状态

- 协议：`POST /rpc`（`RpcRequest`→`RpcResponse`）、`POST /notify`（状态/事件上报）、
  `GET /health`、`GET /state`。已用真实引擎跑通完整对局。
- Python 侧目前是「合法动作随机」的占位策略；`action` 回复用引擎给的
  `autoSelectedDice` 作为 `usedDice`。
- 浏览器侧 io 支持 `cancelRpc`（`AbortController`）与通知转发（AI 看盘）。

## 下一步建议

1. **人选牌**：接 `@gi-tcg/deck-builder` 让我方在起局前选牌组。
2. **AI 看牌反选**：把公共态经 `/notify` 喂给 Python，Python 依我方牌组反选对手牌组。
3. **随机先手**：`aiOpponent` 从固定 who=0 推广到随机座位。
4. **模型桥**：`web/ai_rpc_server.py` 的占位策略替换为 `reps/`+`train/` 的 CVPN
   checkpoint（或 `common/llm.py`）。
5. **NPC 卡组**：允许我方持 `configs/npc/` 的敌人卡组（D18）。
