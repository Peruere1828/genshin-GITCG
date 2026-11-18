#!/usr/bin/env bash
# scripts/serve_web_demo.sh
#
# 单端口可视化观战：standalone 前端（Vite）+ Python AI 桥。
#
# 浏览器只访问 `http://<本机 IP>:$PORT/`；前端把 `/bridge/*` 反代到本机
# `web/ai_rpc_server.py`（见 engine-overlay/overlay/packages/standalone/vite.config.ts），
# 所以局域网只需要放行一个端口。
#
# 用法（前台，Ctrl-C 一起退出）：
#   bash scripts/serve_web_demo.sh
#
# 后台常驻：
#   setsid nohup bash scripts/serve_web_demo.sh >/tmp/gitcg-web.log 2>&1 < /dev/null &
#
# 环境变量：
#   PORT          前端端口（默认 7001）
#   BRIDGE_PORT   Python 桥端口（默认 8770，仅本机监听，经前端反代）
#   POLICY        AI 座位的策略 spec（默认 heuristic；也可 legal_random /
#                 expert:<slug> / neural:<ckpt>）
#   AI_DECK       可选：桥声明的 AI 牌组 slug（默认空，由 --policy
#                 expert:<slug> 推断或桥的 --ai-deck 决定）
#   SEED          策略种子（默认 20261112）
#   CONDA_ENV     Python 环境（默认 gitcg）
#
# 网页是调试接口：勾「AI 对手（HTTP 桥）」后，对面座位由本机 Python 桥托管；
# 桥起在 127.0.0.1:$BRIDGE_PORT，只经前端 `/bridge` 反代暴露。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-7001}"
BRIDGE_PORT="${BRIDGE_PORT:-8770}"
POLICY="${POLICY:-heuristic}"
AI_DECK="${AI_DECK:-}"
SEED="${SEED:-20261112}"
CONDA_ENV="${CONDA_ENV:-gitcg}"

BRIDGE_PID=""
VITE_PID=""

cleanup() {
  trap - EXIT INT TERM
  # 两个子进程各自 `setsid` 成独立进程组：负 PID 才能连 conda→python、
  # bash→pnpm→vite 的整棵子树一起收掉（只 kill 顶层会留下孤儿占端口）。
  [ -n "$VITE_PID" ] && kill -- -"$VITE_PID" 2>/dev/null || true
  [ -n "$BRIDGE_PID" ] && kill -- -"$BRIDGE_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

AI_DECK_ARG=()
if [ -n "$AI_DECK" ]; then AI_DECK_ARG=(--ai-deck "$AI_DECK"); fi
echo "[web-demo] AI bridge → 127.0.0.1:${BRIDGE_PORT} (env=${CONDA_ENV}, policy=${POLICY}, ai_deck=${AI_DECK:-<由 policy 推断>})"
setsid conda run --no-capture-output -n "$CONDA_ENV" \
  python web/ai_rpc_server.py \
  --host 127.0.0.1 --port "$BRIDGE_PORT" \
  --policy "$POLICY" "${AI_DECK_ARG[@]}" --seed "$SEED" &
BRIDGE_PID=$!

echo "[web-demo] standalone frontend → 0.0.0.0:${PORT} (反代 /bridge → 127.0.0.1:${BRIDGE_PORT})"
setsid env PORT="$PORT" HOST=0.0.0.0 BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}" \
  bash engine-overlay/scripts/standalone.sh dev &
VITE_PID=$!

wait -n "$BRIDGE_PID" "$VITE_PID"
