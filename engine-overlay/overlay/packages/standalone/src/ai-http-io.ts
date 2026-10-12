// Copyright (C) 2024-2025 Guyutongxue
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// --- WS8/D18 overlay 新增：浏览器引擎 ↔ Python 的 HTTP 桥 ---------------------
// 把某个座位的 `CancellablePlayerIO` 实现为对本地 Python 服务的 HTTP 调用：
//   rpc(request)      -> POST {base}/rpc      body {who, request}   -> RpcResponse
//   notify(note)      -> POST {base}/notify   body {who, notification}（fire-and-forget）
//   cancelRpc()       -> AbortController.abort()（回合被取消/终止时）
// 这样 Python 侧（web/ai_rpc_server.py）就能拿到引擎的请求/状态并作出决策，
// 为后续「AI 看对手牌反选」「模型 HTTP 桥」打地基。

import type {
  CancellablePlayerIO,
  Notification,
  RpcRequest,
  RpcResponse,
} from "@gi-tcg/core";

export interface AiHttpIoOptions {
  /** 例如 "http://127.0.0.1:8770"（不要带结尾斜杠） */
  baseUrl: string;
  /** 该 io 服务的座位（0 = 先手弹窗位，1 = 主窗口位） */
  who: 0 | 1;
  /** 是否把引擎通知（对局事件/状态）也 POST 给 Python，默认 true（AI 需要看盘） */
  forwardNotifications?: boolean;
  /** 调试日志回调 */
  onLog?: (direction: "→" | "←" | "!", method: string, payload?: unknown) => void;
}

const trimSlash = (s: string) => s.replace(/\/+$/, "");

export function createAiHttpIo(options: AiHttpIoOptions): CancellablePlayerIO {
  const base = trimSlash(options.baseUrl);
  const { who, forwardNotifications = true, onLog } = options;
  let controller: AbortController | null = null;

  return {
    notify: (notification: Notification) => {
      if (!forwardNotifications) {
        return;
      }
      void fetch(`${base}/notify`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ who, notification }),
        keepalive: true,
      }).catch(() => {
        /* 通知是尽力而为，不阻塞对局 */
      });
    },

    rpc: async (request: RpcRequest): Promise<RpcResponse> => {
      controller?.abort();
      const ctrl = new AbortController();
      controller = ctrl;
      const method = request.request?.$case ?? "unknown";
      onLog?.("→", method, request);
      let resp: Response;
      try {
        resp = await fetch(`${base}/rpc`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ who, request }),
          signal: ctrl.signal,
        });
      } catch (e) {
        onLog?.("!", method, e);
        throw e;
      }
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`AI HTTP ${resp.status} ${resp.statusText}: ${text}`);
      }
      const json = (await resp.json()) as RpcResponse;
      if (!json || typeof json !== "object" || !("response" in json)) {
        throw new Error(`AI HTTP invalid response: ${JSON.stringify(json)}`);
      }
      onLog?.("←", method, json);
      return json;
    },

    cancelRpc: () => {
      controller?.abort();
      controller = null;
    },
  };
}
