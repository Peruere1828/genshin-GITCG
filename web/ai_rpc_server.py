#!/usr/bin/env python3
# web/ai_rpc_server.py
#
# WS8/D18：浏览器引擎 ↔ Python 的 HTTP 桥（最小原型）。
#
# 浏览器侧 `@gi-tcg/standalone` 的 AI 座位把 `PlayerIO.rpc/notify` 转成对
# 本服务的 HTTP POST（见 engine-overlay/overlay/packages/standalone/src/ai-http-io.ts）：
#   POST /rpc      body {"who":0|1, "request": RpcRequest}      -> RpcResponse
#   POST /notify   body {"who":0|1, "notification": {...}}       -> {}（尽力而为）
#   GET  /health   -> {"ok":true,...}
#   GET  /state    -> 最近一次通知里的（掩码后）GameState，供调试
#
# 决策逻辑（纯标准库，先“合法+随机”即可）：
#   rerollDice   -> 不重掷
#   switchHands  -> 不换手
#   chooseActive -> 候选第一只
#   selectCard   -> 候选第一张
#   action       -> 优先选“合法且非 declareEnd”的动作，随机取一个；否则选第一个合法；
#                   usedDice 用该动作的 autoSelectedDice（引擎提示的自动选骰）。
#
# 跑：
#   conda run -n gitcg python web/ai_rpc_server.py --port 8770
# 前端（standalone）里勾选「AI 对手（HTTP 桥）」并填 http://127.0.0.1:8770 即可。

from __future__ import annotations

import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# 引擎枚举：ActionValidity.VALID === 0（见 packages/typings/src/gen/rpc.ts）
VALID = 0


class Policy:
    """最简策略：合法动作里随机挑一个（不主动 declarEnd，除非只剩它）。"""

    def __init__(self, strategy: str = "greedy", seed: int | None = None) -> None:
        self.strategy = strategy
        self.rng = random.Random(seed)

    def reroll_dice(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"diceToReroll": []}

    def switch_hands(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"removedHandIds": []}

    def choose_active(self, payload: dict[str, Any]) -> dict[str, Any]:
        cands = payload.get("candidateIds") or [0]
        return {"activeCharacterId": cands[0]}

    def select_card(self, payload: dict[str, Any]) -> dict[str, Any]:
        cands = payload.get("candidateDefinitionIds") or [0]
        return {"selectedDefinitionId": cands[0]}

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        actions = payload.get("action") or []
        if not actions:
            return {"chosenActionIndex": 0, "usedDice": []}
        valid = [i for i, a in enumerate(actions) if a.get("validity") == VALID]
        if not valid:
            idx = 0
        else:
            # 优先非 declareEnd，让牌局更“活”
            non_end = [
                i for i in valid
                if (actions[i].get("action") or {}).get("$case") != "declareEnd"
            ]
            pool = non_end or valid
            if self.strategy == "random":
                idx = self.rng.choice(pool)
            else:
                idx = pool[0]
        used = list(actions[idx].get("autoSelectedDice") or [])
        return {"chosenActionIndex": idx, "usedDice": used}


class State:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.lock = threading.Lock()
        self.rpc_counts: dict[str, int] = {}
        self.notify_count = 0
        self.latest_state: Any = None
        self.latest_mutation: Any = None


def make_handler(state: State, quiet: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = "giTcgAiBridge/0.1"

        # ---- helpers ---------------------------------------------------
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            if not quiet:
                super().log_message(fmt, *args)

        def _read_json(self) -> Any:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _dispatch_rpc(self, who: int, request: dict[str, Any]) -> dict[str, Any]:
            req = (request or {}).get("request") or {}
            method = req.get("$case")
            value = req.get("value") or {}
            with state.lock:
                state.rpc_counts[method or "?"] = state.rpc_counts.get(method or "?", 0) + 1
            if method == "rerollDice":
                resp_value = state.policy.reroll_dice(value)
            elif method == "switchHands":
                resp_value = state.policy.switch_hands(value)
            elif method == "chooseActive":
                resp_value = state.policy.choose_active(value)
            elif method == "selectCard":
                resp_value = state.policy.select_card(value)
            elif method == "action":
                resp_value = state.policy.action(value)
            else:
                raise ValueError(f"unknown rpc method: {method!r}")
            return {"response": {"$case": method, "value": resp_value}}

        # ---- routes ----------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("content-length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                with state.lock:
                    self._json(
                        {
                            "ok": True,
                            "rpc": dict(state.rpc_counts),
                            "notify": state.notify_count,
                        }
                    )
            elif self.path.startswith("/state"):
                self._json({"state": state.latest_state})
            else:
                self._json({"service": "gi-tcg-ai-bridge", "endpoints": ["/rpc", "/notify", "/health", "/state"]})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_json()
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"bad json: {e}"}, status=400)
                return
            if self.path.startswith("/rpc"):
                try:
                    out = self._dispatch_rpc(int(body.get("who", 0)), body.get("request") or {})
                except Exception as e:  # noqa: BLE001
                    self._json({"error": str(e)}, status=500)
                    return
                self._json(out)
            elif self.path.startswith("/notify"):
                note = body.get("notification") or {}
                with state.lock:
                    state.notify_count += 1
                    if isinstance(note, dict):
                        state.latest_state = note.get("state")
                        state.latest_mutation = note.get("mutation")
                self._json({})
            else:
                self._json({"error": "not found"}, status=404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="GI-TCG browser<->Python AI bridge (minimal)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--strategy", choices=["greedy", "random"], default="random")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    policy = Policy(strategy=args.strategy, seed=args.seed)
    state = State(policy)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args.quiet))
    print(
        f"[ai-bridge] listening on http://{args.host}:{args.port} "
        f"(strategy={args.strategy}, seed={args.seed})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
