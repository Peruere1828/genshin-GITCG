"""Multiprocess rollout: run independent matches across worker subprocesses.

Each worker is a fresh interpreter (``python -m envs.rollout_worker``) speaking a
JSON-lines protocol over pipes. This avoids ``multiprocessing`` fork, which
deadlocks the engine's embedded JS runtime, and spawn's ``__main__`` re-import,
which breaks under pytest / ``-m``. Tasks and results stay plain dicts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from common.lock import runtime_metadata


@dataclass(frozen=True)
class MatchTask:
    """A single match request, addressable by (deck0, deck1, seed).

    ``deck0_inline``/``deck1_inline`` allow evaluating *mutated* decks that have no
    registry entry (deck-building sensitivity); they are plain
    ``{"name", "characters", "cards"}`` dicts so they cross the worker boundary.
    """

    index: int
    deck0: str
    deck1: str
    seed: int
    policy0: str = "expert"
    policy1: str = "expert"
    record_decisions: bool = False
    record_views: bool = False
    tag: str = ""
    deck0_inline: dict[str, Any] | None = None
    deck1_inline: dict[str, Any] | None = None

    def policy_spec(self, role: int) -> str:
        """Resolve ``"expert"`` to ``expert:<same slug as the deck>``."""
        spec = self.policy0 if role == 0 else self.policy1
        deck = self.deck0 if role == 0 else self.deck1
        if spec == "expert":
            return f"expert:{deck}"
        return spec

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "deck0": self.deck0,
            "deck1": self.deck1,
            "seed": self.seed,
            "policy0": self.policy_spec(0),
            "policy1": self.policy_spec(1),
            "record_decisions": self.record_decisions,
            "record_views": self.record_views,
            "tag": self.tag,
            "deck0_inline": self.deck0_inline,
            "deck1_inline": self.deck1_inline,
        }


def _resolve_deck(name: str, inline: dict[str, Any] | None):
    from reps.schema import DeckSpec

    if inline is not None:
        return DeckSpec(
            name=str(inline["name"]),
            characters=tuple(int(c) for c in inline["characters"]),
            cards=tuple(int(c) for c in inline["cards"]),
        )
    from envs.decks import deck_spec

    return deck_spec(name)


def run_task(task: MatchTask) -> dict[str, Any]:
    """Execute one task in the current process (top-level: picklable for pools)."""
    from envs.match import run_match
    from envs.policy import build_policy

    record = run_match(
        _resolve_deck(task.deck0, task.deck0_inline),
        _resolve_deck(task.deck1, task.deck1_inline),
        build_policy(task.policy_spec(0), seed=task.seed, role="p0"),
        build_policy(task.policy_spec(1), seed=task.seed, role="p1"),
        seed=task.seed,
        record_decisions=task.record_decisions,
        record_views=task.record_views,
    )
    payload = record.as_dict(include_decisions=task.record_decisions)
    payload["task_index"] = task.index
    payload["policy0"] = task.policy_spec(0)
    payload["policy1"] = task.policy_spec(1)
    payload["tag"] = task.tag
    return payload


def _mp_context():  # noqa: D401 - kept for reference/back-compat
    """Deprecated: multiprocessing fork deadlocks the gitcg C/JS runtime.

    Kept only so external callers do not break. Use ``run_tasks``, which spawns
    independent worker subprocesses instead.
    """
    return mp.get_context("spawn")


def _run_tasks_sequential(tasks: list[MatchTask]) -> list[dict[str, Any]]:
    return [run_task(task) for task in tasks]


def _run_shard_in_subprocess(
    shard: list[MatchTask],
    *,
    python: str,
    env: dict[str, str],
    repo_root: str,
    worker_module: str = "envs.rollout_worker",
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    import subprocess
    import threading

    process = subprocess.Popen(
        [python, "-m", worker_module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=repo_root,
        env=env,
    )

    stderr_chunks: list[str] = []
    results: list[dict[str, Any]] = []

    def _drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)

    def _drain_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            match = json.loads(line)
            results.append(match)
            if on_result is not None:
                on_result(match)

    # Both pipes must drain *while* tasks are being written: a large shard can
    # fill the worker's stdout pipe (64 KiB) before stdin is fully written, which
    # deadlocked the old write-all-then-read-all version (worker blocks on
    # stdout, parent blocks on stdin).
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()

    assert process.stdin is not None and process.stdout is not None
    for task in shard:
        process.stdin.write(json.dumps(task.as_dict(), ensure_ascii=False) + "\n")
    process.stdin.close()

    stdout_thread.join()
    returncode = process.wait()
    stderr_thread.join(timeout=1)
    if returncode != 0 or len(results) != len(shard):
        detail = "".join(stderr_chunks)[-2000:]
        raise RuntimeError(
            f"rollout worker failed (rc={returncode}, {len(results)}/{len(shard)} results):\n{detail}"
        )
    return results


def run_task_shard(
    shard: list[Any],
    *,
    worker_module: str,
    workers: int | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a list of ``as_dict()``-able tasks through a JSON-lines worker module.

    A thin public wrapper over the (deadlock-safe) subprocess plumbing used by
    rollout, so other task types (e.g. ``train.collect``) reuse it instead of
    re-implementing the two-pipe drain. Results are merged and sorted by ``index``.
    """
    import os as _os
    import sys

    from common.paths import repo_root

    shard = list(shard)
    if not shard:
        return []
    resolved = workers if workers is not None else (_os.cpu_count() or 1)
    resolved = max(1, min(int(resolved), len(shard)))
    env = dict(_os.environ)
    root = str(repo_root())
    env["PYTHONPATH"] = root + _os.pathsep + env.get("PYTHONPATH", "")
    python = sys.executable

    if resolved == 1:
        return _run_shard_in_subprocess(
            shard,
            python=python,
            env=env,
            repo_root=root,
            worker_module=worker_module,
            on_result=on_result,
        )

    from concurrent.futures import ThreadPoolExecutor

    chunks = [shard[i::resolved] for i in range(resolved)]
    chunks = [chunk for chunk in chunks if chunk]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                _run_shard_in_subprocess,
                chunk,
                python=python,
                env=env,
                repo_root=root,
                worker_module=worker_module,
                on_result=on_result,
            )
            for chunk in chunks
        ]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda item: item.get("index", 0))
    return results


def run_tasks(
    tasks: Iterable[MatchTask],
    *,
    workers: int | None = None,
    progress: bool = False,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run tasks, sequentially when ``workers <= 1`` else across worker processes.

    ``on_result`` is invoked from reader threads as each match result arrives,
    so callers can stream results to disk instead of losing everything when a
    long run is interrupted (PLAN.md D7: everything must be resumable).

    Worker processes are spawned as fresh interpreters running
    ``envs.rollout_worker`` over a JSON-lines pipe. This deliberately avoids
    ``multiprocessing`` fork (which deadlocks the engine's embedded JS runtime)
    and spawn's ``__main__`` re-import (which breaks under pytest / ``-m``).
    """
    import os as _os
    import sys

    from common.paths import repo_root

    tasks = list(tasks)
    if not tasks:
        return []
    resolved = workers if workers is not None else (_os.cpu_count() or 1)
    resolved = max(1, min(int(resolved), len(tasks)))
    if resolved == 1:
        results = _run_tasks_sequential(tasks)
        if on_result is not None:
            for match in results:
                on_result(match)
        return results

    shards = [tasks[i::resolved] for i in range(resolved)]
    shards = [shard for shard in shards if shard]
    env = dict(_os.environ)
    root = str(repo_root())
    env["PYTHONPATH"] = root + _os.pathsep + env.get("PYTHONPATH", "")
    python = sys.executable

    from concurrent.futures import ThreadPoolExecutor

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures = [
            pool.submit(
                _run_shard_in_subprocess,
                shard,
                python=python,
                env=env,
                repo_root=root,
                on_result=on_result,
            )
            for shard in shards
        ]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda item: item.get("task_index", 0))
    return results


def make_task_grid(
    deck_pairs: Iterable[tuple[str, str]],
    seeds: Iterable[int],
    *,
    policy0: str = "expert",
    policy1: str = "expert",
    tag: str = "",
    base_seed: int = 0,
) -> list[MatchTask]:
    """Build a deterministic task list for every (deck pair, seed) combination.

    The *same* seed set is reused for every pairing (paired seeds), which keeps
    variance low for A/B comparison (PLAN.md §4.1 L1).
    """
    seed_list = [int(s) for s in seeds]
    tasks: list[MatchTask] = []
    index = 0
    for deck0, deck1 in deck_pairs:
        for raw_seed in seed_list:
            tasks.append(
                MatchTask(
                    index=index,
                    deck0=deck0,
                    deck1=deck1,
                    seed=int(base_seed) + raw_seed,
                    policy0=policy0,
                    policy1=policy1,
                    tag=tag,
                )
            )
            index += 1
    return tasks


def rollout_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return runtime_metadata(extra)
