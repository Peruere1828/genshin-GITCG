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
from typing import Any, Iterable

from common.lock import runtime_metadata


@dataclass(frozen=True)
class MatchTask:
    """A single match request, addressable by (deck0, deck1, seed)."""

    index: int
    deck0: str
    deck1: str
    seed: int
    policy0: str = "expert"
    policy1: str = "expert"
    record_decisions: bool = False
    record_views: bool = False
    tag: str = ""

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
        }


def run_task(task: MatchTask) -> dict[str, Any]:
    """Execute one task in the current process (top-level: picklable for pools)."""
    from envs.decks import deck_spec
    from envs.match import run_match
    from envs.policy import build_policy

    record = run_match(
        deck_spec(task.deck0),
        deck_spec(task.deck1),
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
    shard: list[MatchTask], *, python: str, env: dict[str, str], repo_root: str
) -> list[dict[str, Any]]:
    import subprocess
    import threading

    process = subprocess.Popen(
        [python, "-m", "envs.rollout_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=repo_root,
        env=env,
    )

    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    assert process.stdin is not None and process.stdout is not None
    for task in shard:
        process.stdin.write(json.dumps(task.as_dict(), ensure_ascii=False) + "\n")
    process.stdin.close()

    results: list[dict[str, Any]] = []
    for line in process.stdout:
        line = line.strip()
        if line:
            results.append(json.loads(line))
    returncode = process.wait()
    stderr_thread.join(timeout=1)
    if returncode != 0 or len(results) != len(shard):
        detail = "".join(stderr_chunks)[-2000:]
        raise RuntimeError(
            f"rollout worker failed (rc={returncode}, {len(results)}/{len(shard)} results):\n{detail}"
        )
    return results


def run_tasks(
    tasks: Iterable[MatchTask],
    *,
    workers: int | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run tasks, sequentially when ``workers <= 1`` else across worker processes.

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
        return _run_tasks_sequential(tasks)

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
            pool.submit(_run_shard_in_subprocess, shard, python=python, env=env, repo_root=root)
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
