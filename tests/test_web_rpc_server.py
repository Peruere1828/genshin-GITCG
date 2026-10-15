"""C2 模型桥（web/ai_rpc_server.py）单测。

覆盖三块：
  * 占位策略默认路径（不依赖任何重型依赖）；
  * snake_case payload -> camelCase 响应映射、protobuf-JSON -> StateSnapshot 转换；
  * 模型桥（`--policy`）端到端决策：用一次短种子对局在运行时生成真实
    `notification.state` 与 `action` 请求（protobuf-JSON），再喂给
    `RepoPolicy.decide` / HTTP handler，验证响应形状与回退行为。

不依赖训练 checkpoint、不依赖浏览器；HTTP 用例绑 127.0.0.1 临时端口。
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from web import ai_rpc_server as srv

A, B = "superconduct_aggro", "natlan_battleship"

# 仓库根（本文件在 tests/ 的上一层），用于子进程直接运行 web/ai_rpc_server.py。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# 运行时确定性夹具：跑一小段真实对局，抓 notification / 五类请求的 proto-JSON
# --------------------------------------------------------------------------- #
class _CapturePolicyPlayer:
    """薄包装：正常运行 PolicyPlayer，同时抓首批 notification 与请求 JSON。"""

    def __new__(cls, *args, **kwargs):
        from envs.match import PolicyPlayer
        from google.protobuf import json_format

        class _Player(PolicyPlayer):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.requests: dict[str, dict[str, str]] = {}

            def _capture(self, method: str, request) -> None:
                if method in self.requests:
                    return
                self.requests[method] = {
                    "request_json": json_format.MessageToJson(request),
                    "notification_json": (
                        json_format.MessageToJson(self.notification)
                        if self.notification is not None
                        else ""
                    ),
                }

            def on_reroll_dice(self, request):
                self._capture("rerollDice", request)
                return super().on_reroll_dice(request)

            def on_switch_hands(self, request):
                self._capture("switchHands", request)
                return super().on_switch_hands(request)

            def on_choose_active(self, request):
                self._capture("chooseActive", request)
                return super().on_choose_active(request)

            def on_select_card(self, request):
                self._capture("selectCard", request)
                return super().on_select_card(request)

            def on_action(self, request):
                self._capture("action", request)
                return super().on_action(request)

        return _Player(*args, **kwargs)


@pytest.fixture(scope="module")
def captured() -> dict[str, dict[str, str]]:
    """确定性跑一段对局，返回 {method: {request_json, notification_json}}。"""
    from gitcg import Game, low_level

    from envs.decks import deck_spec
    from envs.match import PolicyPlayer, _build_create_param
    from envs.policy import legal_random_policy
    from reps.action_hierarchy import reset_default_hierarchical_action_codebook

    reset_default_hierarchical_action_codebook()
    game = Game(create_param=_build_create_param(deck_spec(A), deck_spec(B), seed=7))
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_0, 1)
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_1, 1)
    cap = _CapturePolicyPlayer(0, legal_random_policy(seed=1))
    game.set_player(0, cap)
    game.set_player(1, PolicyPlayer(1, legal_random_policy(seed=2)))
    game.start()
    steps = 0
    while game.is_running() and "action" not in cap.requests and steps < 600:
        game.step()
        steps += 1
    assert "action" in cap.requests, "未能在限定步数内抓到 action 请求"
    return cap.requests


def _payload(entry: dict[str, str]) -> dict:
    return json.loads(entry["request_json"])


def _state(entry: dict[str, str]) -> dict:
    return json.loads(entry["notification_json"])["state"]


# --------------------------------------------------------------------------- #
# 默认路径：占位策略
# --------------------------------------------------------------------------- #
def test_placeholder_name_and_action_pick():
    policy = srv.Policy(strategy="greedy", seed=1)
    assert policy.name == "placeholder:greedy"
    out = policy.decide(
        0,
        "action",
        {
            "action": [
                {"validity": 0, "action": {"$case": "declareEnd"}},
                {"validity": 0, "action": {"$case": "useSkill"}, "autoSelectedDice": [8]},
            ]
        },
    )
    # greedy 优先非 declareEnd，并使用引擎提示的 autoSelectedDice
    assert out == {"chosenActionIndex": 1, "usedDice": [8]}


def test_placeholder_other_methods():
    policy = srv.Policy(strategy="random", seed=1)
    assert policy.decide(0, "rerollDice", {}) == {"diceToReroll": []}
    assert policy.decide(0, "switchHands", {}) == {"removedHandIds": []}
    assert policy.decide(
        0, "chooseActive", {"candidateIds": [11, 22]}
    ) == {"activeCharacterId": 11}
    assert policy.decide(
        0, "selectCard", {"candidateDefinitionIds": [7, 9]}
    ) == {"selectedDefinitionId": 7}


# --------------------------------------------------------------------------- #
# snake -> camel 响应映射
# --------------------------------------------------------------------------- #
def test_response_mapping_is_exact():
    assert srv._response_from_payload(
        "rerollDice", {"dice_to_reroll": [1, 2]}
    ) == {"diceToReroll": [1, 2]}
    assert srv._response_from_payload(
        "switchHands", {"removed_hand_ids": [3]}
    ) == {"removedHandIds": [3]}
    assert srv._response_from_payload(
        "chooseActive", {"active_character_id": 5}
    ) == {"activeCharacterId": 5}
    assert srv._response_from_payload(
        "selectCard", {"selected_definition_id": 7}
    ) == {"selectedDefinitionId": 7}
    assert srv._response_from_payload(
        "action", {"chosen_action_index": 2, "used_dice": [8, 4]}
    ) == {"chosenActionIndex": 2, "usedDice": [8, 4]}


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        srv._response_from_payload("nope", {})


# --------------------------------------------------------------------------- #
# protobuf-JSON -> StateSnapshot（只用掩码后的 notification.state）
# --------------------------------------------------------------------------- #
def test_state_snapshot_from_browser_json(captured):
    view = srv._state_snapshot(_state(captured["action"]))
    assert len(view.players) == 2
    assert isinstance(view.round_number, int)
    # snapshot_state 走的是 gitcg 绑定的 store-JSON，不能吃浏览器 proto-JSON：
    # 这里固化“必须走 ParseDict -> snapshot_notification”这一事实。
    from reps.snapshot import snapshot_state

    with pytest.raises(Exception):
        snapshot_state(None, json.dumps(_state(captured["action"])))


# --------------------------------------------------------------------------- #
# 模型桥端到端
# --------------------------------------------------------------------------- #
def test_repo_decide_heuristic_action(captured):
    entry = captured["action"]
    action_count = len(_payload(entry)["action"])
    policy = srv.RepoPolicy(
        "heuristic", seed=3, role="web", fallback=srv.Policy("random", seed=1)
    )
    out = policy.decide(0, "action", _payload(entry), _state(entry))
    assert set(out) == {"chosenActionIndex", "usedDice"}
    assert 0 <= out["chosenActionIndex"] < action_count
    assert all(isinstance(d, int) for d in out["usedDice"])


@pytest.mark.parametrize("spec", ["heuristic", "legal_random", "random"])
def test_build_policy_specs_route(captured, spec):
    entry = captured["action"]
    policy = srv.RepoPolicy(
        spec, seed=5, role="web", fallback=srv.Policy("random", seed=1)
    )
    out = policy.decide(0, "action", _payload(entry), _state(entry))
    assert "chosenActionIndex" in out


def test_repo_decide_all_captured_methods(captured):
    """对局中真实出现的非 action 请求也要能过模型桥。"""
    policy = srv.RepoPolicy(
        "legal_random", seed=9, role="web", fallback=srv.Policy("greedy", seed=1)
    )
    checked = 0
    for method in ("rerollDice", "switchHands", "chooseActive", "selectCard"):
        entry = captured.get(method)
        if entry is None:
            continue
        out = policy.decide(0, method, _payload(entry), _state(entry))
        assert isinstance(out, dict) and out, f"{method} 未返回响应"
        checked += 1
    assert checked >= 1, "对局中应至少出现一个非 action 的请求"


def test_repo_decide_accepts_protobuf_ts_nested_oneof(captured, capsys):
    """浏览器（protobuf-ts）把 oneof 序列化成 ``{oneofName: {$case, value}}``，
    与 Python ``json_format`` 的扁平表示不同——直接 ParseDict 会报
    ``Message type "Action" has no field named "action"``。桥必须能吃下这种形式。

    手搓一个 protobuf-ts 风格的 ActionRequest（``Action.action`` 为嵌套 oneof）。
    用 capsys 断言没有走回退（否则单动作时占位策略也会返回 index 0，无法区分）。
    """
    entry = captured["action"]
    ts_request = {
        "action": [
            {
                "action": {"$case": "declareEnd"},
                "validity": 0,
                "isFast": True,
                "requiredCost": [],
                "autoSelectedDice": [],
                "preview": [],
            }
        ]
    }
    policy = srv.RepoPolicy(
        "legal_random", seed=1, role="web", fallback=srv.Policy("random", seed=1)
    )
    out = policy.decide(0, "action", ts_request, _state(entry))
    assert set(out) == {"chosenActionIndex", "usedDice"}
    assert out["chosenActionIndex"] == 0
    assert "回退" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 回退：加载失败 / choose 抛错 / 非法动作码 / 无通知
# --------------------------------------------------------------------------- #
def test_repo_policy_load_failure_falls_back(captured):
    entry = captured["action"]
    policy = srv.RepoPolicy(
        "definitely-not-a-spec", seed=1, role="web", fallback=srv.Policy("random", seed=1)
    )
    out = policy.decide(0, "action", _payload(entry), _state(entry))
    assert "chosenActionIndex" in out  # 未崩溃，回退占位策略


def test_repo_policy_choose_raises_falls_back(captured):
    entry = captured["action"]

    class _Boom:
        def choose(self, built):
            raise RuntimeError("boom")

    policy = srv.RepoPolicy(
        "heuristic", seed=1, role="web", fallback=srv.Policy("random", seed=1)
    )
    policy._policy = _Boom()  # 绕过加载，注入会抛错的策略
    out = policy.decide(0, "action", _payload(entry), _state(entry))
    assert "chosenActionIndex" in out


def test_repo_policy_illegal_code_falls_back(captured):
    entry = captured["action"]

    class _Illegal:
        def choose(self, built):
            return -1

    policy = srv.RepoPolicy(
        "heuristic", seed=1, role="web", fallback=srv.Policy("random", seed=1)
    )
    policy._policy = _Illegal()
    out = policy.decide(0, "action", _payload(entry), _state(entry))
    assert "chosenActionIndex" in out


def test_repo_policy_without_notification_falls_back():
    policy = srv.RepoPolicy(
        "heuristic", seed=1, role="web", fallback=srv.Policy("greedy", seed=1)
    )
    out = policy.decide(0, "action", {"action": [{"validity": 0}]}, None)
    assert "chosenActionIndex" in out


# --------------------------------------------------------------------------- #
# HTTP 路由：/health（含 policy）/ /notify / /state / /rpc 及回退
# --------------------------------------------------------------------------- #
class _Server:
    def __init__(self, policy) -> None:
        from http.server import ThreadingHTTPServer

        self.state = srv.State(policy)
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), srv.make_handler(self.state, quiet=True)
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method: str, path: str, body: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            headers = {"content-type": "application/json"} if payload else {}
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            return resp.status, data
        finally:
            conn.close()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def http_server(captured):
    server = _Server(
        srv.RepoPolicy(
            "heuristic", seed=2, role="web", fallback=srv.Policy("random", seed=1)
        )
    )
    try:
        yield server
    finally:
        server.close()


def test_http_health_reports_policy(http_server):
    status, data = http_server.request("GET", "/health")
    assert status == 200
    assert data["ok"] is True
    assert data["policy"] == "heuristic"
    assert "rpc" in data and "notify" in data


def test_cors_preflight_caches_with_max_age(http_server):
    """预检响应要带 Access-Control-Max-Age，浏览器才会缓存 OPTIONS 结果，
    否则每个跨域请求都先配一次预检（/notify 一局几千条 → 请求数翻倍）。"""
    conn = http.client.HTTPConnection("127.0.0.1", http_server.port, timeout=10)
    try:
        conn.request("OPTIONS", "/notify", headers={"Origin": "http://172.26.108.168:7001"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 204
        assert resp.getheader("Access-Control-Max-Age") == str(srv._CORS_MAX_AGE_SECONDS)
        assert resp.getheader("Access-Control-Allow-Methods")
        assert resp.getheader("Access-Control-Allow-Headers")
    finally:
        conn.close()


def test_http_notify_then_state_and_rpc(http_server, captured):
    entry = captured["action"]
    notification = json.loads(entry["notification_json"])

    status, _ = http_server.request(
        "POST", "/notify", {"who": 0, "notification": notification}
    )
    assert status == 200

    status, state_resp = http_server.request("GET", "/state")
    assert status == 200 and state_resp["state"] == notification["state"]

    body = {
        "who": 0,
        "request": {"request": {"$case": "action", "value": _payload(entry)}},
    }
    status, resp = http_server.request("POST", "/rpc", body)
    assert status == 200
    assert resp["response"]["$case"] == "action"
    value = resp["response"]["value"]
    assert set(value) == {"chosenActionIndex", "usedDice"}
    assert 0 <= value["chosenActionIndex"] < len(_payload(entry)["action"])


def test_http_rpc_without_notification_falls_back():
    server = _Server(
        srv.RepoPolicy(
            "heuristic", seed=2, role="web", fallback=srv.Policy("greedy", seed=1)
        )
    )
    try:
        body = {"who": 0, "request": {"request": {"$case": "rerollDice", "value": {}}}}
        status, resp = server.request("POST", "/rpc", body)
        assert status == 200
        assert resp == {"response": {"$case": "rerollDice", "value": {"diceToReroll": []}}}
    finally:
        server.close()


def test_http_rpc_unknown_method_errors(captured):
    server = _Server(srv.Policy(strategy="random", seed=1))
    try:
        body = {"who": 0, "request": {"request": {"$case": "bogus", "value": {}}}}
        status, resp = server.request("POST", "/rpc", body)
        assert status == 500
        assert "error" in resp
    finally:
        server.close()


def test_direct_script_run_loads_model_policy(captured, tmp_path):
    """直接 `python web/ai_rpc_server.py` 运行时仓库根要能被加入 sys.path。

    回归 C2 的隐蔽 bug：直接运行脚本时 sys.path[0] 是 web/，`envs` 导入失败，
    模型策略全程静默回退占位（AI 变随机）。本用例以子进程方式直接运行脚本，
    喂入真实 notification + action 请求，确认模型策略真的被加载且未回退。
    """
    # 找空闲端口
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    err_log = tmp_path / "stderr.log"
    out_log = tmp_path / "stdout.log"
    with out_log.open("w") as out, err_log.open("w") as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join("web", "ai_rpc_server.py"),
                "--port",
                str(port),
                "--policy",
                "heuristic",
            ],
            cwd=REPO_ROOT,
            stdout=out,
            stderr=err,
        )
        try:
            # 等端口可连
            deadline = time.time() + 20
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        break
                except OSError:
                    time.sleep(0.1)
            assert proc.poll() is None, "桥进程启动即退出"

            entry = captured["action"]
            notification = json.loads(entry["notification_json"])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.request(
                    "POST",
                    "/notify",
                    body=json.dumps({"who": 0, "notification": notification}),
                    headers={"content-type": "application/json"},
                )
                assert conn.getresponse().status == 200
                conn.request(
                    "POST",
                    "/rpc",
                    body=json.dumps(
                        {
                            "who": 0,
                            "request": {
                                "request": {
                                    "$case": "action",
                                    "value": json.loads(entry["request_json"]),
                                }
                            },
                        }
                    ),
                    headers={"content-type": "application/json"},
                )
                resp = conn.getresponse()
                body = json.loads(resp.read().decode())
                assert resp.status == 200
                assert set(body["response"]["value"]) == {"chosenActionIndex", "usedDice"}
            finally:
                conn.close()
            time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    err_text = err_log.read_text()
    assert "策略已加载" in err_text
    assert "No module named 'envs'" not in err_text
    assert "回退" not in err_text
