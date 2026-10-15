#!/usr/bin/env python3
# web/ai_rpc_server.py
#
# WS8/D18：浏览器引擎 ↔ Python 的 HTTP 桥。
#
# 浏览器侧 `@gi-tcg/standalone` 的 AI 座位把 `PlayerIO.rpc/notify` 转成对
# 本服务的 HTTP POST（见 engine-overlay/overlay/packages/standalone/src/ai-http-io.ts）：
#   POST /rpc      body {"who":0|1, "request": RpcRequest}      -> RpcResponse
#   POST /notify   body {"who":0|1, "notification": {...}}       -> {}（尽力而为）
#   GET  /health   -> {"ok":true,...}
#   GET  /state    -> 最近一次通知里的（掩码后）GameState，供调试
#
# 决策逻辑有两种模式：
#   * 默认（不给 --policy）：纯标准库占位策略，合法动作里随机/贪心挑一个：
#       rerollDice   -> 不重掷
#       switchHands  -> 不换手
#       chooseActive -> 候选第一只
#       selectCard   -> 候选第一张
#       action       -> 优先选“合法且非 declareEnd”的动作；否则第一个合法；
#                       usedDice 用该动作的 autoSelectedDice（引擎提示的自动选骰）。
#   * `--policy SPEC`（C2 模型桥）：把五种请求都路由到仓库已训练策略
#       `envs.policy.build_policy(SPEC, seed=SEED, role="web")` +
#       `reps.action_adapter.build_decision_context`。SPEC 支持
#       `neural:<checkpoint.pt>[#t=<temp>]`、`heuristic`、`legal_random`、
#       `expert:<slug>`、`random`。reps/train/torch 等依赖只在请求模型策略时
#       才延迟导入，默认路径仍是零依赖。任何加载/转换/选择失败都回退占位策略。
#
# 跑：
#   conda run -n gitcg python web/ai_rpc_server.py --port 8770
#   conda run -n gitcg python web/ai_rpc_server.py --port 8770 \
#       --policy "neural:data/checkpoints/curve_dfbcsa_e15_d128_n48000.pt#t=0.2"
# 前端（standalone）里勾选「AI 对手（HTTP 桥）」并填 http://127.0.0.1:8770 即可。

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

# 允许直接 `python web/ai_rpc_server.py` 运行：此时 sys.path[0] 是 web/，仓库根不在
# 路径上，`envs`/`reps`/`train` 无法导入（C2 的延迟导入会全部失败并回退占位）。
# 把仓库根补进 sys.path，使直接运行与 `python -m web.ai_rpc_server` 等价。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 引擎枚举：ActionValidity.VALID === 0（见 packages/typings/src/gen/rpc.ts）
VALID = 0

# 默认只允许来自本机的网页驱动本桥（见 Handler._cors）。
# 这是**有意识**的收敛：桥能替 AI 座位做决策，若允许任意 Origin，
# 用户浏览器里任意一个网页都能 POST /rpc 驱动本地桥。
_LOCAL_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

# 五种 RPC 方法名 -> gitcg `Request` oneof 的 proto 字段名（snake_case）。
_METHOD_TO_FIELD = {
    "rerollDice": "reroll_dice",
    "switchHands": "switch_hands",
    "chooseActive": "choose_active",
    "action": "action",
    "selectCard": "select_card",
}

# 模型策略 seed 派生用的角色名（build_policy 把 seed/role 折进策略 RNG）。
_WEB_ROLE = "web"

# CORS 预检缓存时长（秒）：浏览器据此缓存 OPTIONS 结果，避免跨域访问时
# 每个 /notify、/rpc 都先配一次预检请求。
_CORS_MAX_AGE_SECONDS = 600


def _is_local_origin(origin: str) -> bool:
    """Origin 形如 scheme://host[:port]，判断其 host 是否为本机。"""
    try:
        host = urlparse(origin).hostname
    except ValueError:
        return False
    return host in _LOCAL_ORIGIN_HOSTS


def _warn(message: str) -> None:
    """桥内非致命错误一律写 stderr，绝不冒泡成 500。"""
    print(f"[ai-bridge] {message}", file=sys.stderr, flush=True)


class Policy:
    """占位策略：合法动作里随机/贪心挑一个（不主动 declareEnd，除非只剩它）。"""

    def __init__(self, strategy: str = "greedy", seed: int | None = None) -> None:
        self.strategy = strategy
        self.name = f"placeholder:{strategy}"
        self.rng = random.Random(seed)

    def decide(
        self, who: int, method: str, value: dict[str, Any], latest_state: Any = None
    ) -> dict[str, Any]:
        """把 RPC 方法分派到具体的占位决策（`who`/`latest_state` 占位不用）。"""
        if method == "rerollDice":
            return self.reroll_dice(value)
        if method == "switchHands":
            return self.switch_hands(value)
        if method == "chooseActive":
            return self.choose_active(value)
        if method == "selectCard":
            return self.select_card(value)
        if method == "action":
            return self.action(value)
        raise ValueError(f"unknown rpc method: {method!r}")

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


class RepoPolicy:
    """模型桥：把决策路由到 `envs.policy.build_policy` 产出的真实策略。

    只在 `--policy` 下构造。重型依赖（reps/train/torch/gitcg.proto）都在
    首次决策时延迟导入，默认占位路径保持纯标准库零依赖。任何一步失败
    （无 notification、proto 转换失败、`choose()` 抛错、选中非法动作码）
    都回退到占位策略并写 stderr 警告，保证 handler 永不 500/挂死。
    """

    def __init__(
        self,
        spec: str,
        *,
        seed: int | None,
        role: str,
        fallback: Policy,
    ) -> None:
        self.spec = spec
        self.name = spec
        self.seed = 0 if seed is None else int(seed)
        self.role = role
        self._fallback = fallback
        self._policy: Any = None
        self._load_error: str | None = None
        self._step_index = 0

    def _ensure_policy(self) -> Any:
        """延迟构造仓库策略；失败只记一次错误并静默回退。"""
        if self._policy is None and self._load_error is None:
            try:
                from envs.policy import build_policy

                self._policy = build_policy(self.spec, seed=self.seed, role=self.role)
                _warn(f"策略已加载：{self.spec!r} (seed={self.seed}, role={self.role})")
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                _warn(f"加载策略 {self.spec!r} 失败，全程回退占位策略：{self._load_error}")
        return self._policy

    def decide(
        self, who: int, method: str, value: dict[str, Any], latest_state: Any = None
    ) -> dict[str, Any]:
        try:
            policy = self._ensure_policy()
            if policy is None:
                raise RuntimeError(self._load_error or "policy unavailable")
            if latest_state is None:
                raise RuntimeError("尚未收到任何 notification.state")
            resp = _repo_decide(policy, who, method, value, latest_state, self._step_index)
            self._step_index += 1
            return resp
        except Exception as exc:  # noqa: BLE001
            _warn(f"策略决策失败（{method}），回退占位策略：{type(exc).__name__}: {exc}")
            return self._fallback.decide(who, method, value, latest_state)


def _flatten_ts_oneofs(descriptor: Any, obj: Any) -> Any:
    """把 protobuf-ts 的 JSON 转成 Python ``json_format.ParseDict`` 期望的形式。

    浏览器（``@gi-tcg/core``，protobuf-ts）把 oneof 序列化成
    ``{"<oneofName>": {"$case": "<member>", "value": {...}}}``，而 Python 的
    ``json_format`` 要求直接设 oneof 成员字段（``{"<memberJsonName": {...}}``）。
    同一消息两边表示不同，直接 ParseDict 会报
    ``Message type "Action" has no field named "action"``。

    本函数按消息描述符递归摊平：遇到 ``{oneofName: {$case, value}}`` 就改写成
    ``{memberJsonName: value}``，其余字段原样递归。
    """
    if not isinstance(obj, dict):
        return obj

    fields_by_name = {f.name: f for f in descriptor.fields}
    fields_by_json = {f.json_name: f for f in descriptor.fields}
    oneofs = {o.name: o for o in descriptor.oneofs}

    out: dict[str, Any] = {}
    for key, val in obj.items():
        oneof = oneofs.get(key)
        if oneof is not None and isinstance(val, dict) and "$case" in val:
            member = val["$case"]
            field = next(
                (f for f in oneof.fields if f.name == member or f.json_name == member),
                None,
            )
            if field is not None:
                out[field.json_name] = _flatten_ts_oneofs(
                    field.message_type, val.get("value") or {}
                )
                continue
        field = fields_by_name.get(key) or fields_by_json.get(key)
        if field is not None and field.message_type is not None:
            if field.is_repeated:
                out[key] = (
                    [_flatten_ts_oneofs(field.message_type, v) for v in val]
                    if isinstance(val, list)
                    else val
                )
            else:
                out[key] = _flatten_ts_oneofs(field.message_type, val)
        else:
            out[key] = val
    return out


def _repo_decide(
    policy: Any,
    who: int,
    method: str,
    value: dict[str, Any],
    latest_state: Any,
    step_index: int,
) -> dict[str, Any]:
    """用仓库策略跑一次决策：request JSON -> proto -> 决策上下文 -> 响应 JSON。"""
    from google.protobuf import json_format

    from gitcg.proto import rpc_pb2

    from reps.action_adapter import build_decision_context

    field = _METHOD_TO_FIELD[method]
    # 浏览器 payload 是 protobuf-ts JSON（oneof 为 {$case, value} 嵌套），
    # 先摊平再 ParseDict；否则 Action 等带 oneof 的消息会解析失败。
    flat = _flatten_ts_oneofs(rpc_pb2.Request.DESCRIPTOR, {field: value or {}})
    request_msg = rpc_pb2.Request()
    json_format.ParseDict(flat, request_msg)
    request = getattr(request_msg, field)

    decision_type = _decision_type_map()[method]
    view = _state_snapshot(latest_state)
    built = build_decision_context(
        acting_player=int(who),
        request_type=decision_type,
        request=request,
        full_state=view,
        player_view=view,
        full_state_json=None,
        step_index=int(step_index),
    )
    code = int(policy.choose(built))
    payload = built.payload_by_low_level_code.get(code)
    if payload is None:
        # 策略选了非法/未映射动作码（含 -1）——交给上层回退。
        raise RuntimeError(f"策略返回非法动作码：{code}")
    return _response_from_payload(method, payload)


def _state_snapshot(latest_state: Any) -> Any:
    """把最近一次 notification.state（protobuf-JSON）解成掩码 StateSnapshot。

    硬规则 C10：桥只有掩码后的 `notification.state`，绝不构造全量/god state。
    """
    from google.protobuf import json_format

    from gitcg.proto import state_pb2
    from reps.snapshot import snapshot_notification

    if isinstance(latest_state, str):
        payload: Any = json.loads(latest_state)
    else:
        payload = latest_state
    state = state_pb2.State()
    json_format.ParseDict(payload, state)
    return snapshot_notification(_NotificationLike(state))


class _NotificationLike:
    """`snapshot_notification` 只需要 `.state`，用它包一层 proto State。"""

    __slots__ = ("state",)

    def __init__(self, state: Any) -> None:
        self.state = state


# 方法名 -> reps.schema.DecisionType（延迟导入：默认路径不 import reps）。
def _decision_type_map() -> dict[str, Any]:
    from reps.schema import DecisionType

    return {
        "rerollDice": DecisionType.REROLL_DICE,
        "switchHands": DecisionType.SWITCH_HANDS,
        "chooseActive": DecisionType.CHOOSE_ACTIVE,
        "action": DecisionType.ACTION,
        "selectCard": DecisionType.SELECT_CARD,
    }


def _response_from_payload(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """仓库 payload 是 snake_case，浏览器引擎要 camelCase——逐方法精确映射。"""
    if method == "action":
        return {
            "chosenActionIndex": int(payload.get("chosen_action_index", 0)),
            "usedDice": [int(d) for d in payload.get("used_dice", ())],
        }
    if method == "chooseActive":
        return {"activeCharacterId": int(payload["active_character_id"])}
    if method == "selectCard":
        return {"selectedDefinitionId": int(payload["selected_definition_id"])}
    if method == "rerollDice":
        return {"diceToReroll": [int(d) for d in payload.get("dice_to_reroll", ())]}
    if method == "switchHands":
        return {"removedHandIds": [int(d) for d in payload.get("removed_hand_ids", ())]}
    raise ValueError(f"unknown rpc method: {method!r}")


class State:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.lock = threading.Lock()
        self.rpc_counts: dict[str, int] = {}
        self.notify_count = 0
        self.latest_state: Any = None
        self.latest_mutation: Any = None


def make_handler(state: State, quiet: bool, cors_origin: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "giTcgAiBridge/0.1"

        # ---- helpers ---------------------------------------------------
        def _path(self) -> str:
            """去掉 query string 的路径，用于精确路由匹配。"""
            return urlparse(self.path).path

        def _cors(self) -> None:
            # 默认仅回显本机 Origin（localhost/127.0.0.1/::1）；--cors-origin 可显式覆盖。
            origin = self.headers.get("Origin")
            if cors_origin is not None:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")
            elif origin and _is_local_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            # 让浏览器缓存预检结果（否则每个跨域请求都配一个 OPTIONS）。
            self.send_header("Access-Control-Max-Age", str(_CORS_MAX_AGE_SECONDS))

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
            if method not in _METHOD_TO_FIELD:
                raise ValueError(f"unknown rpc method: {method!r}")
            with state.lock:
                state.rpc_counts[method] = state.rpc_counts.get(method, 0) + 1
                latest_state = state.latest_state
            # 模型策略/占位策略共用 decide()；自身保证不抛异常。
            resp_value = state.policy.decide(who, method, value, latest_state)
            return {"response": {"$case": method, "value": resp_value}}

        # ---- routes ----------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("content-length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self._path()
            if path == "/health":
                with state.lock:
                    self._json(
                        {
                            "ok": True,
                            "policy": getattr(state.policy, "name", "unknown"),
                            "rpc": dict(state.rpc_counts),
                            "notify": state.notify_count,
                        }
                    )
            elif path == "/state":
                with state.lock:
                    self._json({"state": state.latest_state})
            else:
                self._json({"service": "gi-tcg-ai-bridge", "endpoints": ["/rpc", "/notify", "/health", "/state"]})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_json()
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"bad json: {e}"}, status=400)
                return
            path = self._path()
            if path == "/rpc":
                try:
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                    out = self._dispatch_rpc(int(body.get("who", 0)), body.get("request") or {})
                except Exception as e:  # noqa: BLE001
                    self._json({"error": str(e)}, status=500)
                    return
                self._json(out)
            elif path == "/notify":
                # 与 /rpc 一致：任何解析/形状错误都返回错误响应，不让连接静默断开。
                try:
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                    note = body.get("notification")
                    if note is not None and not isinstance(note, dict):
                        raise ValueError("notification must be a JSON object")
                    note = note or {}
                    with state.lock:
                        state.notify_count += 1
                        state.latest_state = note.get("state")
                        state.latest_mutation = note.get("mutation")
                except Exception as e:  # noqa: BLE001
                    self._json({"error": str(e)}, status=400)
                    return
                self._json({})
            else:
                self._json({"error": "not found"}, status=404)

    return Handler


def build_policy_from_args(args: argparse.Namespace) -> Any:
    """按命令行参数组装策略：有 `--policy` 走模型桥，否则纯占位。"""
    placeholder = Policy(strategy=args.strategy, seed=args.seed)
    if not args.policy:
        return placeholder
    return RepoPolicy(
        spec=args.policy,
        seed=args.seed,
        role=_WEB_ROLE,
        fallback=placeholder,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="GI-TCG browser<->Python AI bridge")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--strategy", choices=["greedy", "random"], default="random")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--policy",
        default=None,
        help=(
            "模型桥策略 spec（走 envs.policy.build_policy）；留空则用 --strategy 占位策略。"
            " 例：neural:<ckpt.pt>[#t=<temp>] | heuristic | legal_random | expert:<slug> | random"
        ),
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--cors-origin",
        default=None,
        help="显式设置 Access-Control-Allow-Origin（如 '*'）；默认仅放行本机 Origin",
    )
    args = ap.parse_args()

    policy = build_policy_from_args(args)
    state = State(policy)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(state, args.quiet, args.cors_origin)
    )
    print(
        f"[ai-bridge] listening on http://{args.host}:{args.port} "
        f"(policy={policy.name}, seed={args.seed}, "
        f"cors={'local-only' if args.cors_origin is None else args.cors_origin})",
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
