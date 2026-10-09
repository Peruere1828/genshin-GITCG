"""Thin fork bridge: evaluate candidate actions by forking a boundary snapshot.

PLAN.md D13 / L5.4 ("S1 缩减为薄 fork 桥 ... 子进程暴露给搜索"). ``envs.snapshot``
establishes *what* can be forked (an ``is_resumable()`` boundary snapshot, with
game-level attrs mirrored); this module wraps the fork + replay + rollout into a
single serializable task and runs it either in-process (``run_fork_task``) or
across a pool of fresh interpreter subprocesses (``ForkBridge``) using the same
JSON-lines shape as ``envs.rollout``.

Why subprocesses
----------------
The live game drives its engine from inside ``game.step()`` callbacks; forking a
second engine *inside* that callback is not something the pybinding promises to
support. Running each fork in its own interpreter (a) sidesteps any reentrancy
question, (b) parallelizes the CPU-bound rollouts across cores (PLAN.md §5: CPU
does search, GPU only inference), and (c) keeps the "no long-lived service"
constraint (D7) -- the pool is spawned per run and torn down.

Task shape
----------
A task describes one branch from a boundary snapshot:

* ``prefix0`` / ``prefix1`` -- the positional option indices each player already
  chose *since the snapshot*. Positional indices (not raw action codes) are the
  cross-process stable key: the engine's global action codebook is process-local
  but the ordering of a request's legal options is a pure function of the state.
* ``inject_player`` / ``inject_ordinal`` / ``inject_option`` -- the acting player
  replays its own prefix, then forces ``inject_option`` at its next decision
  (``inject_ordinal == len(prefix)+1``), then hands over to its rollout policy.
  ``inject_player < 0`` means *pure replay* (both players replay their full
  prefix; used by the exact-fork determinism check).
* ``rollout0`` / ``rollout1`` -- policy specs (``envs.policy.build_policy``) for
  decisions after the prefix/injection; ``rollout_seed`` seeds them.

Determinism: with ``inject_player < 0`` and the full recorded prefix, a task
reproduces the live terminal byte-for-byte (``tests/test_fork_bridge.py``),
mirroring the record-replay evidence in ``reports/engine/probe_boundary_fork_*``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from common.seeding import derive_seed
from envs.policy import Policy, build_policy

@dataclass(frozen=True)
class ForkTask:
    """One fork-and-rollout request (see module docstring)."""

    index: int
    snapshot: str
    game_attrs: Mapping[int, int]
    deck0: str
    deck1: str
    prefix0: tuple[int, ...] = ()
    prefix1: tuple[int, ...] = ()
    inject_player: int = -1
    inject_ordinal: int = 0
    inject_option: int = 0
    rollout0: str = "legal_random"
    rollout1: str = "legal_random"
    rollout_seed: int = 0
    max_steps: int = 5000
    tag: str = ""
    capture_terminal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "snapshot": self.snapshot,
            "game_attrs": {str(int(k)): int(v) for k, v in dict(self.game_attrs).items()},
            "deck0": self.deck0,
            "deck1": self.deck1,
            "prefix0": [int(i) for i in self.prefix0],
            "prefix1": [int(i) for i in self.prefix1],
            "inject_player": int(self.inject_player),
            "inject_ordinal": int(self.inject_ordinal),
            "inject_option": int(self.inject_option),
            "rollout0": self.rollout0,
            "rollout1": self.rollout1,
            "rollout_seed": int(self.rollout_seed),
            "max_steps": int(self.max_steps),
            "tag": self.tag,
            "capture_terminal": bool(self.capture_terminal),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ForkTask":
        attrs = payload.get("game_attrs") or {}
        return cls(
            index=int(payload["index"]),
            snapshot=payload["snapshot"],
            game_attrs={int(k): int(v) for k, v in attrs.items()},
            deck0=payload["deck0"],
            deck1=payload["deck1"],
            prefix0=tuple(int(i) for i in payload.get("prefix0", ())),
            prefix1=tuple(int(i) for i in payload.get("prefix1", ())),
            inject_player=int(payload.get("inject_player", -1)),
            inject_ordinal=int(payload.get("inject_ordinal", 0)),
            inject_option=int(payload.get("inject_option", 0)),
            rollout0=payload.get("rollout0", "legal_random"),
            rollout1=payload.get("rollout1", "legal_random"),
            rollout_seed=int(payload.get("rollout_seed", 0)),
            max_steps=int(payload.get("max_steps", 5000)),
            tag=payload.get("tag", ""),
            capture_terminal=bool(payload.get("capture_terminal", False)),
        )


@dataclass(frozen=True)
class ForkResult:
    """Outcome of one branch (winner is ``None`` for a draw)."""

    index: int
    winner: int | None
    rounds: int
    error: str | None = None
    truncated: bool = False
    tag: str = ""
    terminal: str | None = None

    def outcome(self, perspective: int) -> float:
        if self.error is not None:
            return 0.0
        if self.winner is None:
            return 0.5
        return 1.0 if int(self.winner) == int(perspective) else 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ForkResult":
        winner = payload.get("winner")
        return cls(
            index=int(payload["index"]),
            winner=None if winner is None else int(winner),
            rounds=int(payload.get("rounds", 0) or 0),
            error=payload.get("error"),
            truncated=bool(payload.get("truncated", False)),
            tag=payload.get("tag", ""),
            terminal=payload.get("terminal"),
        )


# --------------------------------------------------------------------------- #
# policies used to reproduce a prefix and then hand over to a rollout
# --------------------------------------------------------------------------- #
class PositionalReplayPolicy:
    """Replay previously recorded option indices (positional, cross-process safe)."""

    def __init__(self, indices: Sequence[int], *, name: str = "replay") -> None:
        self.indices = tuple(int(i) for i in indices)
        self.i = 0
        self.name = name
        self.observed: list[int] = []

    def choose(self, built) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        self.observed.append(len(codes))
        if self.i >= len(self.indices):
            raise RuntimeError(
                f"replay exhausted at decision {self.i} "
                f"(no recorded option; {len(codes)} legal options now)"
            )
        idx = self.indices[self.i]
        self.i += 1
        if idx < 0 or idx >= len(codes):
            raise RuntimeError(
                f"replay option {idx} out of range for {len(codes)} legal options"
            )
        return int(codes[idx])


@dataclass
class InjectionPolicy:
    """Replay ``prefix`` for earlier decisions, force one option, then ``rollout``."""

    prefix: Policy
    rollout: Policy
    ordinal: int
    option_index: int
    name: str = "inject"
    _n: int = field(default=0, init=False, repr=False)

    def choose(self, built) -> int:
        self._n += 1
        if self._n == self.ordinal:
            codes = [int(c) for c in built.context.legal_low_level_codes]
            if codes:
                return int(codes[min(self.option_index, len(codes) - 1)])
            return int(self.rollout.choose(built))
        inner = self.prefix if self._n < self.ordinal else self.rollout
        return int(inner.choose(built))


@dataclass
class PrefixRolloutPolicy:
    """Replay ``prefix_len`` decisions, then hand every later decision to ``rollout``."""

    replay: Policy
    rollout: Policy
    prefix_len: int
    name: str = "prefix_rollout"
    _n: int = field(default=0, init=False, repr=False)

    def choose(self, built) -> int:
        self._n += 1
        if self._n <= self.prefix_len:
            return int(self.replay.choose(built))
        return int(self.rollout.choose(built))


def _rollout_policy(spec: str, seed: int, role: str) -> Policy:
    return build_policy(spec, seed=seed, role=role)


def run_fork_task(task: ForkTask) -> dict[str, Any]:
    """Execute one fork task in the current process (worker body / sequential path)."""
    from gitcg import GameStatus

    from envs.match import PolicyPlayer
    from envs.snapshot import capture_snapshot, fork_game
    from reps.action_hierarchy import reset_default_hierarchical_action_codebook

    # The codebook is process-global; reset so this fork's option orderings do not
    # depend on earlier tasks in the same worker (AGENTS.md replay determinism).
    reset_default_hierarchical_action_codebook()

    game = fork_game(task.snapshot, game_attrs=task.game_attrs)
    policies: list[Policy] = []
    for player in (0, 1):
        prefix = task.prefix0 if player == 0 else task.prefix1
        replay = PositionalReplayPolicy(prefix, name=f"replay:p{player}")
        if task.inject_player < 0:
            policies.append(replay)  # pure exact replay mode
            continue
        spec = task.rollout0 if player == 0 else task.rollout1
        rollout = _rollout_policy(
            spec, derive_seed(task.rollout_seed, f"p{player}", "fork-rollout"), f"p{player}"
        )
        if player == task.inject_player:
            policies.append(
                InjectionPolicy(
                    prefix=replay,
                    rollout=rollout,
                    ordinal=int(task.inject_ordinal),
                    option_index=int(task.inject_option),
                )
            )
        else:
            policies.append(PrefixRolloutPolicy(replay=replay, rollout=rollout, prefix_len=len(prefix)))
    game.set_player(0, PolicyPlayer(0, policies[0]))
    game.set_player(1, PolicyPlayer(1, policies[1]))

    error: str | None = None
    truncated = False
    try:
        game.start()
        steps = 0
        while game.is_running():
            if steps >= task.max_steps:
                truncated = True
                break
            game.step()
            steps += 1
    except Exception as exc:  # engine/IO failures are swallowed by cffi otherwise
        error = f"{type(exc).__name__}: {exc}"

    winner: int | None = None
    rounds = 0
    terminal: str | None = None
    try:
        winner = game.winner()
        rounds = int(game.round_number())
        if task.capture_terminal:
            terminal = capture_snapshot(game)
    except Exception:
        pass
    if error is None and game.status() == GameStatus.ABORTED:
        try:
            error = f"aborted: {game.error()}"
        except Exception:
            error = "aborted"

    return {
        "index": task.index,
        "winner": winner,
        "rounds": rounds,
        "error": error,
        "truncated": truncated,
        "tag": task.tag,
        "terminal": terminal,
    }


# --------------------------------------------------------------------------- #
# subprocess pool
# --------------------------------------------------------------------------- #
class _ForkWorker:
    """A persistent ``python -m envs.fork_worker`` interpreter."""

    def __init__(self, python: str, env: dict[str, str], repo_root: str, module: str) -> None:
        self.process = subprocess.Popen(
            [python, "-m", module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=repo_root,
            env=env,
        )
        self._stderr: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line)

    def run(self, tasks: Sequence[ForkTask], *, timeout: float) -> list[dict[str, Any]]:
        if not tasks:
            return []
        assert self.process.stdin is not None and self.process.stdout is not None
        results: list[dict[str, Any]] = []
        errors: list[BaseException] = []

        def _reader() -> None:
            try:
                for _ in range(len(tasks)):
                    line = self.process.stdout.readline()
                    if not line:
                        errors.append(RuntimeError("fork worker closed stdout early"))
                        return
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller
                errors.append(exc)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        for task in tasks:
            self.process.stdin.write(json.dumps(task.as_dict(), ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(f"fork worker timed out after {timeout}s")
        if errors:
            raise RuntimeError(f"fork worker read failed: {errors[0]!r}")
        if len(results) != len(tasks):
            detail = "".join(self._stderr)[-2000:]
            raise RuntimeError(
                f"fork worker returned {len(results)}/{len(tasks)} results:\n{detail}"
            )
        return results

    def close(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self._stderr_thread.join(timeout=1)


class ForkBridge:
    """Pool of fork workers; ``evaluate`` forks + rolls out a batch of candidates.

    ``workers <= 0`` runs the tasks sequentially in-process (used by tests and by
    environments where spawning interpreters is undesirable). The pool is spawned
    lazily on first use and torn down by ``close`` / the context manager.
    """

    def __init__(
        self,
        workers: int | None = None,
        *,
        module: str = "envs.fork_worker",
        timeout: float = 900.0,
        python: str | None = None,
        repo_root: str | None = None,
    ) -> None:
        self.workers = workers
        self.module = module
        self.timeout = float(timeout)
        self._python = python
        self._repo_root = repo_root
        self._pool: list[_ForkWorker] = []
        self._lock = threading.Lock()

    # -- lazy pool -----------------------------------------------------------
    def _resolved_workers(self, task_count: int) -> int:
        configured = self.workers if self.workers is not None else (os.cpu_count() or 1)
        configured = max(0, int(configured))
        if task_count <= 0 or configured == 0:
            return 0
        return max(1, min(configured, task_count))

    def _ensure_pool(self, count: int) -> None:
        if len(self._pool) >= count:
            return
        from common.paths import repo_root

        root = self._repo_root or str(repo_root())
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        python = self._python or sys.executable
        while len(self._pool) < count:
            self._pool.append(_ForkWorker(python, env, root, self.module))

    def evaluate(self, tasks: Iterable[ForkTask]) -> list[ForkResult]:
        """Return one ``ForkResult`` per task (sorted by task index)."""
        tasks = list(tasks)
        if not tasks:
            return []
        resolved = self._resolved_workers(len(tasks))
        if resolved <= 0:
            payloads = [run_fork_task(task) for task in tasks]
            return [ForkResult.from_dict(p) for p in sorted(payloads, key=lambda r: r["index"])]

        with self._lock:
            self._ensure_pool(resolved)
            shards = [tasks[i::resolved] for i in range(resolved)]
            shards = [shard for shard in shards if shard]

            from concurrent.futures import ThreadPoolExecutor

            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                futures = [
                    pool.submit(worker.run, shard, timeout=self.timeout)
                    for worker, shard in zip(self._pool, shards)
                ]
                for future in futures:
                    results.extend(future.result())
        results.sort(key=lambda r: int(r["index"]))
        return [ForkResult.from_dict(p) for p in results]

    def close(self) -> None:
        for worker in self._pool:
            worker.close()
        self._pool = []

    def __enter__(self) -> "ForkBridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def replay_fork_task(
    snapshot: str,
    *,
    index: int,
    deck0: str,
    deck1: str,
    game_attrs: Mapping[int, int] | None = None,
    prefix0: Sequence[int] = (),
    prefix1: Sequence[int] = (),
) -> ForkTask:
    """Build a pure-replay task (no injection) for the exact-fork determinism check."""
    return ForkTask(
        index=index,
        snapshot=snapshot,
        game_attrs=dict(game_attrs or {}),
        deck0=deck0,
        deck1=deck1,
        prefix0=tuple(int(i) for i in prefix0),
        prefix1=tuple(int(i) for i in prefix1),
        inject_player=-1,
        tag="replay",
    )
