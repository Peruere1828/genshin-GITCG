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

import getData from "@gi-tcg/data";
import {
  CancellablePlayerIO,
  Deck,
  DetailLogEntry,
  Game,
  GameState,
  GameStateLogEntry,
  GiTcgIoError,
  Version,
  exposeState,
  serializeGameStateLog,
  setAsyncContext,
} from "@gi-tcg/core";

import { createClient, StandaloneChessboard } from "@gi-tcg/web-ui-core";
import "@gi-tcg/web-ui-core/style.css";
import { For, Show, createSignal, onCleanup, onMount } from "solid-js";
import { DetailLogViewer } from "@gi-tcg/detail-log-viewer";
import mergeErrorCause from "merge-error-cause";
import { createAiHttpIo } from "./ai-http-io";

export interface StandaloneParentProps {
  logs?: GameStateLogEntry[];
  /** C1/C4：一局一个会话 id（App 在开局时生成），桥按它隔离状态/策略/日志 */
  sessionId: string;
  /** 本窗口（真人）这边的牌组对象；对面座位用 `otherDeck` */
  myDeck: Deck;
  /** 对面座位的牌组对象（AI 模式＝AI 牌组；双人模式＝弹窗那边） */
  otherDeck: Deck;
  /** 本窗口托管的座位：随机先手时可能是 0（=先手） */
  humanSeat: 0 | 1;
  version: Version;
  /** WS8/D18 overlay：非空时对面座位由该地址的 Python HTTP 服务托管 */
  aiOpponent?: string;
  /** WS8/D18 overlay：非空时本窗口座位也托管（双方 AI 自对弈，便于调试） */
  aiSelfPlay?: boolean;
}

export function StandaloneParent(props: StandaloneParentProps) {
  // 席位派生：seat0 恒为先手（引擎 currentTurn 从 0 开始）。「随机先手」= 随机决定
  // 真人坐哪个座位，对面那个座位交给桥/弹窗。
  const HUMAN_WHO: 0 | 1 = props.humanSeat;
  const OTHER_WHO: 0 | 1 = (1 - props.humanSeat) as 0 | 1;
  // 两个座位各自的牌组：真人坐 seat0 时两张牌对调。
  const SEAT_DECKS: [Deck, Deck] =
    HUMAN_WHO === 0
      ? [props.myDeck, props.otherDeck]
      : [props.otherDeck, props.myDeck];

  const [uiIo, Chessboard] = createClient(HUMAN_WHO, {
    onGiveUp: () => {
      game?.giveUp(HUMAN_WHO);
    },
  });

  const [popupWindow, setPopupWindow] = createSignal<Window | null>(null);

  const showPopup = async () => {
    if (popupWindow() !== null) {
      return;
    }
    const newWindow = window.open(
      window.location.href,
      "_blank",
      "popup=yes, depended=yes, height=750, width=750",
    );
    setPopupWindow(newWindow);
    // 不知道为什么 onbeforeload 不起作用
    const autoGiveupWhenChildCloseInterval = window.setInterval(() => {
      if (newWindow?.closed && autoGiveupWhenChildCloseInterval !== null) {
        window.clearInterval(autoGiveupWhenChildCloseInterval);
        game?.giveUp(OTHER_WHO);
        setPopupWindow(null);
      }
    }, 1000);
    await new Promise<void>((resolve) => {
      const handler = (e: MessageEvent) => {
        const { data } = e;
        if (typeof data !== "object" || data === null) {
          return;
        }
        if (data.giTcg !== "1.0") {
          return;
        }
        if (!("method" in data)) {
          return;
        }
        if (data.method === "initialized") {
          window?.removeEventListener("message", handler);
          resolve();
        }
      };
      window?.addEventListener("message", handler);
    });
  };

  const childIo: CancellablePlayerIO = {
    notify: (...params) => {
      popupWindow()?.postMessage({
        giTcg: "1.0",
        method: "notify",
        params,
      });
    },
    rpc: async (...params): Promise<any> => {
      const id = Math.random().toString(36).slice(2);
      const result = new Promise((resolve) => {
        const handler = (e: MessageEvent) => {
          const { data } = e;
          if (typeof data !== "object" || data === null) {
            return;
          }
          if (data.giTcg !== "1.0") {
            return;
          }
          if (!("method" in data)) {
            return;
          }
          if (data.method === "rpc" && data.id === id) {
            window?.removeEventListener("message", handler);
            resolve(data.result);
          }
        };
        window?.addEventListener("message", handler);
      });
      popupWindow()?.postMessage({
        giTcg: "1.0",
        method: "rpc",
        params,
        id,
      });
      return result;
    },
    cancelRpc: () => {
      popupWindow()?.postMessage({
        giTcg: "1.0",
        method: "cancelRpc",
      });
    },
  };

  const [stateLog, setStateLog] = createSignal<GameStateLogEntry[]>([]);
  const [fromImport, setFromImport] = createSignal(false);
  // -1 代表不显示历史
  const [viewingLogIndex, setViewingLogIndex] = createSignal<number>(-1);
  const [viewingWho, setViewingWho] = createSignal<0 | 1>(HUMAN_WHO);
  const viewingState = (): GameStateLogEntry | undefined => {
    const index = viewingLogIndex();
    return stateLog()[index];
  };
  const enableLog = () => {
    setViewingWho(HUMAN_WHO);
    setViewingLogIndex(stateLog().length - 1);
  };
  const maxLogIndex = () => {
    return stateLog().length - (fromImport() ? 1 : 2);
  };

  const exportLog = () => {
    const logs = serializeGameStateLog(stateLog());
    const blob = new Blob([JSON.stringify({ ...logs, gv: props.version })], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gameLog.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  let game: Game | null = null;
  const pause = async (
    state: GameState,
    mutations: unknown,
    canResume: boolean,
  ) => {
    if (game !== null) {
      setStateLog((logs) => [...logs, { state, canResume }]);
    }
  };

  const onGameError = (e: unknown, from?: Game) => {
    if (!from || from === game) {
      const merged = mergeErrorCause(e);
      console.error(merged);
      alert(
        `游戏出现了内部错误！请点击主窗口下方“导出日志”按钮生成日志文件，并反馈至 GitHub Issue，thanks~\n${merged.message}\n${merged.stack}`,
      );
    }
  };

  // 引擎在 rpc 被取消/中止时（例如「从此处继续」会调 cancelRpc）同样抛
  // GiTcgIoError，那不是桥故障，不该弹窗。
  const isCancellation = (err: unknown): boolean => {
    let cur: unknown = err;
    for (let i = 0; i < 5 && cur instanceof Error; i++) {
      if (cur.name === "AbortError" || /abort|cancel/i.test(cur.message)) {
        return true;
      }
      cur = (cur as { cause?: unknown }).cause;
    }
    return false;
  };

  // WS8/D18 overlay：AI 桥（HTTP）出错时引擎会把该座位判负并结束对局
  // （GiTcgIoError -> gotWinner(flip(who))）。默认完全静默，会让人误以为
  // 是真赢了。这里显式提示，让人知道是桥挂了而不是真赢了。
  const onIoError = (e: GiTcgIoError) => {
    const merged = mergeErrorCause(e);
    if (isCancellation(e)) {
      console.warn("[ai-http] io error (cancelled, ignored)", merged);
      return;
    }
    console.error("[ai-http] io error", merged);
    const isAiSeat =
      !!props.aiOpponent && (e.who === OTHER_WHO || !!props.aiSelfPlay);
    const seat = `${e.who === 0 ? "先手" : "后手"}方`;
    const hint = isAiSeat
      ? "\n若这是 AI 桥：请确认 web/ai_rpc_server.py 正在运行、地址可访问。"
      : "";
    alert(
      `${seat}${isAiSeat ? "（AI 桥）" : ""} IO 出错，本局将判该方落败并结束（并不代表真的分出了胜负）。` +
        `${hint}\n${merged.message}`,
    );
  };

  // AI 桥调试日志：目前只把错误方向（"!"）打到 console；两个座位都接上，
  // 之前 seat 1 的 io 完全没有日志。
  const aiIoLog = (
    dir: "→" | "←" | "!",
    method: string,
    payload?: unknown,
  ) => {
    if (dir === "!") {
      console.warn(`[ai-http] ${method}`, payload);
    }
  };

  // T3：把本局两个座位的牌组（definition id）报给桥。桥用它校验
  // AI 座位的牌组是否等于 `--ai-deck`/反选声明的那套（脚本专家规则按自家牌组写，
  // 牌组不符会静默变蠢），并按实际牌组派生座位策略，写进决策日志。
  const reportSession = async () => {
    if (!props.aiOpponent) {
      return;
    }
    const meta = {
      version: props.version,
      ai_seat: OTHER_WHO,
      human_seat: HUMAN_WHO,
      ai_self_play: !!props.aiSelfPlay,
      decks: {
        "0": SEAT_DECKS[0],
        "1": SEAT_DECKS[1],
      },
    };
    try {
      const resp = await fetch(`${props.aiOpponent}/session`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session: props.sessionId, meta }),
      });
      if (!resp.ok) {
        console.warn(`[ai-http] /session HTTP ${resp.status}`);
        return;
      }
      const out = (await resp.json()) as { deck_mismatch?: boolean };
      if (out?.deck_mismatch) {
        console.warn(
          "[ai-http] AI 座位牌组与桥声明的牌组不一致（详见桥的 stderr）",
        );
      }
    } catch (e) {
      console.warn("[ai-http] /session 上报失败", e);
    }
  };

  // 终局归档（C4）：把引擎日志（含人类这边的决策）和胜负报给桥，与桥侧的
  // 逐决策 JSONL 落到同一目录。没接桥（纯人机对打）时什么都不做。
  const finishGame = async (finished: Game) => {
    const winner = finished.state.winner;
    if (winner === null || winner === undefined) {
      // 被「从此处继续」terminate 掉的那局没有胜负，不是真正的终局。
      console.warn("[ai-http] 对局结束但无 winner，跳过归档");
      return;
    }
    if (!props.aiOpponent) {
      return;
    }
    const logs = stateLog();
    if (logs.length > 0) {
      try {
        const serialized = serializeGameStateLog(logs);
        await fetch(`${props.aiOpponent}/game-log`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            session: props.sessionId,
            log: { ...serialized, gv: props.version },
          }),
        });
      } catch (e) {
        console.warn("[ai-http] /game-log 上传失败", e);
      }
    }
    try {
      await fetch(`${props.aiOpponent}/game-end`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          session: props.sessionId,
          winner,
          rounds: finished.state.roundNumber,
        }),
      });
    } catch (e) {
      console.warn("[ai-http] /game-end 上报失败", e);
    }
  };

  const startEndedGame = (started: Game) => {
    started
      .start()
      .then(() => finishGame(started).catch(() => undefined))
      .catch((e) => onGameError(e, started));
  };

  const createGame = async (state?: GameState) => {
    if (!state) {
      state = Game.createInitialState({
        decks: SEAT_DECKS,
        data: getData(props.version),
        versionBehavior: props.version,
      });
    }
    await setAsyncContext(true);
    const game = new Game(state);
    (window as any).theGame = game;
    game.onPause = pause;
    game.onIoError = onIoError;
    // WS8/D18 overlay：本窗口座位托管真人棋盘；对面座位按模式交给 HTTP 桥或弹窗。
    // 两个 AI 座位共用同一 sessionId（T1：桥内部再按 who 分开）。
    if (props.aiOpponent) {
      game.players[OTHER_WHO].io = createAiHttpIo({
        baseUrl: props.aiOpponent,
        who: OTHER_WHO,
        sessionId: props.sessionId,
        forwardNotifications: true,
        onLog: aiIoLog,
      });
      if (props.aiSelfPlay) {
        game.players[HUMAN_WHO].io = createAiHttpIo({
          baseUrl: props.aiOpponent,
          who: HUMAN_WHO,
          sessionId: props.sessionId,
          forwardNotifications: true,
          onLog: aiIoLog,
        });
      } else {
        game.players[HUMAN_WHO].io = uiIo;
      }
    } else {
      game.players[OTHER_WHO].io = childIo;
      game.players[HUMAN_WHO].io = uiIo;
    }
    return game;
  };

  const startGame = async () => {
    try {
      if (!props.aiOpponent) {
        await showPopup();
      }
      await reportSession();
      const initialGame = await createGame();
      game = initialGame;
      startEndedGame(initialGame);
    } catch (e) {
      onGameError(e);
    }
  };

  const resumeGame = async () => {
    try {
      if (!props.aiOpponent) {
        await showPopup();
      }
      await reportSession();
      const logs = stateLog().slice(0, viewingLogIndex() + 1);
      childIo.cancelRpc?.();
      uiIo.cancelRpc?.();
      game?.terminate();
      const latestState = logs[logs.length - 1];
      const newGame = await createGame(latestState.state);
      game = newGame;
      startEndedGame(newGame);
      setStateLog(logs);
      setViewingLogIndex(-1);
      setFromImport(false);
    } catch (e) {
      onGameError(e);
    }
  };

  const childGiveUpHandler = (e: MessageEvent) => {
    const { data } = e;
    if (typeof data !== "object" || data === null) {
      return;
    }
    if (data.giTcg !== "1.0") {
      return;
    }
    if (!("method" in data)) {
      return;
    }
    if (data.method === "giveUp") {
      // 弹窗托管的是对面那个座位（真人随机先手时可能在 seat0 或 seat1）。
      game?.giveUp(OTHER_WHO);
    }
  };
  let detailDialog!: HTMLDialogElement;
  const [detailLog, setDetailLog] = createSignal<readonly DetailLogEntry[]>([]);
  const showDetail = () => {
    setDetailLog(game?.detailLog ?? []);
    detailDialog.showModal();
    detailDialog.scrollIntoView({ block: "end" });
  };
  const closeDetail = () => {
    detailDialog.close();
    setDetailLog([]);
  };

  onMount(() => {
    window.addEventListener("beforeunload", () => {
      popupWindow()?.close();
    });
    window.addEventListener("message", childGiveUpHandler);
    if (Array.isArray(props.logs)) {
      setStateLog(props.logs);
      setViewingLogIndex(props.logs.length - 1);
      setFromImport(true);
    } else {
      startGame();
    }
  });
  onCleanup(() => {
    window.removeEventListener("message", childGiveUpHandler);
  });

  return (
    <div class="standalone-board-shell">
      <Show
        when={viewingState()}
        fallback={
          <>
            <div class="title-row">
              <span class="title">
                {HUMAN_WHO === 0 ? "先手方棋盘（我）" : "后手方棋盘（我）"}
              </span>
              <button disabled={stateLog().length <= 1} onClick={enableLog}>
                查看历史
              </button>
            </div>
            <Chessboard />
          </>
        }
      >
        {(state) => (
          <>
            <div class="title-row">
              <span class="title">
                {viewingWho() ? "后" : "先"}手方棋盘（查看历史中）
              </span>
              <Show
                when={viewingLogIndex() <= maxLogIndex() && state().canResume}
              >
                <button onClick={resumeGame}>从此处继续</button>
              </Show>
              <button onClick={() => setViewingWho((i) => (1 - i) as 0 | 1)}>
                切换玩家
              </button>
              <button
                disabled={viewingLogIndex() <= 0}
                onClick={() => setViewingLogIndex((i) => i - 1)}
              >
                后退一步
              </button>
              <span>
                {viewingLogIndex() + 1} / {stateLog().length}
              </span>
              <button
                disabled={viewingLogIndex() >= maxLogIndex()}
                onClick={() => setViewingLogIndex((i) => i + 1)}
              >
                前进一步
              </button>
              <button
                disabled={fromImport()}
                onClick={() => setViewingLogIndex(-1)}
              >
                返回游戏
              </button>
            </div>
            <StandaloneChessboard
              class="grayscale"
              who={viewingWho()}
              state={exposeState(viewingWho(), state().state)}
              mutations={[]}
            />
          </>
        )}
      </Show>
      <button disabled={stateLog().length === 0} onClick={exportLog}>
        导出日志
      </button>
      <button onClick={showDetail}>显示细节</button>
      <dialog ref={detailDialog!}>
        <DetailLogViewer logs={detailLog()} />
        <button onClick={closeDetail}>关闭</button>
      </dialog>
    </div>
  );
}
