"""Resumable, sharded replay collection (cluster CPU data plane, PLAN.md WS6/M3).

The cluster split is "CPU nodes collect, GPU nodes train" (PLAN.md §5/§5.1). This
module is the CPU half: it plays games with a teacher policy, encodes every
decision context, and writes the samples to an append-only replay dir that a GPU
job can later read with ``train.data.load_samples``.

Durability (D7):

* one JSONL line = one game (all of its samples), so a torn trailing line simply
  marks that game as not-done and it is replayed on resume;
* a ``manifest.json`` sidecar pins a spec fingerprint (deck/opponents/seeds/teacher
  specs/encoder config) + the engine lock and refuses to resume against drift;
* results stream to disk as they finish (``on_result``), so an interrupted job
  keeps everything already completed.

Usage::

    python -m train.collect --deck superconduct_aggro \
        --opponents natlan_battleship dual_mualani_stacks \
        --seed-start 0 --seed-count 200 --workers 48 --tag coolstart
    # resume = same command again; completed games are skipped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from common.jsonl import dumps, ensure_trailing_newline
from common.lock import load_engine_lock, runtime_metadata
from common.paths import data_dir, ensure_dir
from envs.rollout import run_task_shard

MANIFEST_NAME = "manifest.json"
SAMPLES_NAME = "samples.jsonl"


@dataclass(frozen=True)
class SearchSpec:
    """MCCFR search-teacher knobs for a distillation collection (M3-CPU S-C).

    When a :class:`CollectSpec` carries a ``search`` block the collector runs the
    engine MCCFR adapter (``envs.cfr_search``) instead of the scripted-expert
    behaviour-cloning teacher, and every searched decision is stored with its
    teacher targets (root average strategy / root value / belief labels).
    """

    iterations: int = 8
    depth: int = 1
    candidate_limit: int | None = None
    value_spec: str = ""
    search_every: int = 1
    max_searches: int | None = None
    belief_root: bool = True
    belief_samples: int = 2
    request_types: tuple[str, ...] | None = None
    bridge_workers: int = 2
    # Label-only god view so the encoder fills belief/privileged targets.
    capture_targets: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": int(self.iterations),
            "depth": int(self.depth),
            "candidate_limit": None
            if self.candidate_limit is None
            else int(self.candidate_limit),
            "value_spec": self.value_spec,
            "search_every": int(self.search_every),
            "max_searches": None if self.max_searches is None else int(self.max_searches),
            "belief_root": bool(self.belief_root),
            "belief_samples": int(self.belief_samples),
            "request_types": None if self.request_types is None else list(self.request_types),
            "bridge_workers": int(self.bridge_workers),
            "capture_targets": bool(self.capture_targets),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchSpec":
        request_types = payload.get("request_types")
        return cls(
            iterations=int(payload.get("iterations", 8)),
            depth=int(payload.get("depth", 1)),
            candidate_limit=payload.get("candidate_limit"),
            value_spec=payload.get("value_spec", ""),
            search_every=int(payload.get("search_every", 1)),
            max_searches=payload.get("max_searches"),
            belief_root=bool(payload.get("belief_root", True)),
            belief_samples=int(payload.get("belief_samples", 2)),
            request_types=None if request_types is None else tuple(str(v) for v in request_types),
            bridge_workers=int(payload.get("bridge_workers", 2)),
            capture_targets=bool(payload.get("capture_targets", True)),
        )


@dataclass(frozen=True)
class CollectTask:
    """One game to collect: teacher (player 0) vs opponent (player 1) at ``seed``."""

    index: int
    deck: str
    opponent: str
    seed: int
    teacher_spec: str | None = None
    opponent_spec: str | None = None
    search: SearchSpec | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "deck": self.deck,
            "opponent": self.opponent,
            "seed": int(self.seed),
            "teacher_spec": self.teacher_spec,
            "opponent_spec": self.opponent_spec,
            "search": None if self.search is None else self.search.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CollectTask":
        search = payload.get("search")
        return cls(
            index=int(payload["index"]),
            deck=payload["deck"],
            opponent=payload["opponent"],
            seed=int(payload["seed"]),
            teacher_spec=payload.get("teacher_spec"),
            opponent_spec=payload.get("opponent_spec"),
            search=None if search is None else SearchSpec.from_dict(search),
        )


@lru_cache(maxsize=1)
def _worker_encoder():
    """Encoder cached per process (worker reuse across tasks)."""
    from envs.observation import default_encoder

    return default_encoder()


_WORKER_BRIDGES: dict[int, Any] = {}


def _worker_bridge(workers: int) -> Any:
    """Per-process cached fork bridge for search-teacher collection.

    The collector runs inside a ``train.collect_worker`` subprocess; the bridge
    lazily spawns its own ``envs.fork_worker`` interpreters on first use. Caching
    it here amortises the spawn across a shard instead of once per game.
    """
    key = max(1, int(workers))
    bridge = _WORKER_BRIDGES.get(key)
    if bridge is None:
        from envs.fork_bridge import ForkBridge

        bridge = ForkBridge(workers=key)
        _WORKER_BRIDGES[key] = bridge
    return bridge


def run_search_collect_task(task: CollectTask) -> dict[str, Any]:
    """Play one game with the MCCFR search teacher and return its samples."""
    from train.data import collect_search_samples, sample_to_payload

    search = task.search
    assert search is not None
    error: str | None = None
    payloads: list[dict[str, Any]] = []
    try:
        samples = collect_search_samples(
            task.deck,
            task.opponent,
            [task.seed],
            encoder=_worker_encoder(),
            bridge=_worker_bridge(search.bridge_workers),
            iterations=search.iterations,
            depth=search.depth,
            candidate_limit=search.candidate_limit,
            value_spec=search.value_spec,
            search_every=search.search_every,
            max_searches=search.max_searches,
            belief_root=search.belief_root,
            belief_samples=search.belief_samples,
            request_types=search.request_types,
            capture_targets=search.capture_targets,
        )
        payloads = [sample_to_payload(sample) for sample in samples]
    except Exception as exc:  # engine/IO failures are otherwise swallowed by cffi
        error = f"{type(exc).__name__}: {exc}"
    return {
        "index": task.index,
        "deck": task.deck,
        "opponent": task.opponent,
        "seed": int(task.seed),
        "error": error,
        "samples": payloads,
    }


def run_collect_task(task: CollectTask) -> dict[str, Any]:
    """Play one game and return its encoded samples (worker body / sequential path)."""
    if task.search is not None:
        return run_search_collect_task(task)

    from train.data import collect_samples, sample_to_payload

    error: str | None = None
    payloads: list[dict[str, Any]] = []
    try:
        samples = collect_samples(
            task.deck,
            task.opponent,
            [task.seed],
            encoder=_worker_encoder(),
            teacher_spec=task.teacher_spec,
            opponent_spec=task.opponent_spec,
        )
        payloads = [sample_to_payload(sample) for sample in samples]
    except Exception as exc:  # engine/IO failures are otherwise swallowed by cffi
        error = f"{type(exc).__name__}: {exc}"
    return {
        "index": task.index,
        "deck": task.deck,
        "opponent": task.opponent,
        "seed": int(task.seed),
        "error": error,
        "samples": payloads,
    }


@dataclass(frozen=True)
class CollectSpec:
    deck: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    teacher_spec: str | None = None
    opponent_spec: str | None = None
    search: SearchSpec | None = None

    def tasks(self) -> list[CollectTask]:
        tasks: list[CollectTask] = []
        index = 0
        for opponent in self.opponents:
            for seed in self.seeds:
                tasks.append(
                    CollectTask(
                        index=index,
                        deck=self.deck,
                        opponent=opponent,
                        seed=int(seed),
                        teacher_spec=self.teacher_spec,
                        opponent_spec=self.opponent_spec,
                        search=self.search,
                    )
                )
                index += 1
        return tasks

    def fingerprint(self) -> str:
        from envs.observation import default_encoder

        # ``bridge_workers`` is an execution knob, not a property of the data:
        # exclude it so a resume with a different worker count is not refused.
        search = self.search
        search_spec = None
        if search is not None:
            search_spec = {
                key: value
                for key, value in search.as_dict().items()
                if key != "bridge_workers"
            }
        blob = json.dumps(
            {
                "deck": self.deck,
                "opponents": list(self.opponents),
                "seeds": list(self.seeds),
                "teacher_spec": self.teacher_spec,
                "opponent_spec": self.opponent_spec,
                "search": search_spec,
                "encoder": default_encoder().to_dict(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _scan_samples(path: Path) -> tuple[set[int], int, int, int]:
    """Stream a samples file, returning (done indices, games, samples, errors).

    Streaming matters at scale: the file holds every game's samples and can be
    far larger than memory. A torn trailing line is skipped (its game is redone).
    """
    done: set[int] = set()
    games = samples = errors = 0
    if not path.exists():
        return done, games, samples, errors
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            games += 1
            samples += len(record.get("samples", ()))
            errors += 1 if record.get("error") else 0
            if "index" in record:
                done.add(int(record["index"]))
    return done, games, samples, errors


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def collect(
    spec: CollectSpec,
    *,
    tag: str,
    workers: int | None = None,
    max_tasks: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run (or resume) a collection into ``data/replays/<tag>/`` and return a summary."""
    replay_dir = ensure_dir(data_dir("replays", tag))
    samples_path = replay_dir / SAMPLES_NAME
    manifest_path = replay_dir / MANIFEST_NAME
    fingerprint = spec.fingerprint()

    manifest = _tolerant_read_manifest(manifest_path)
    if manifest and manifest.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"refusing to resume {replay_dir}: spec fingerprint changed "
            f"(delete the dir to start a new run)"
        )
    if not manifest:
        manifest = {
            "tag": tag,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "fingerprint": fingerprint,
            "spec": {
                "deck": spec.deck,
                "opponents": list(spec.opponents),
                "seeds": list(spec.seeds),
                "teacher_spec": spec.teacher_spec,
                "opponent_spec": spec.opponent_spec,
                "search": None if spec.search is None else spec.search.as_dict(),
            },
            "engine_lock": load_engine_lock(),
            "runtime": runtime_metadata(),
        }
        _write_manifest(manifest_path, manifest)

    existing_done, *_ = _scan_samples(samples_path)
    # Isolate a torn trailing line (from a hard kill) so the first append of this
    # run starts a fresh record instead of merging into the partial bytes.
    ensure_trailing_newline(samples_path)
    all_tasks = spec.tasks()
    pending = [task for task in all_tasks if task.index not in existing_done]
    if max_tasks is not None:
        pending = pending[: int(max_tasks)]

    if progress:
        print(
            f"[collect] {len(existing_done)}/{len(all_tasks)} games done, "
            f"{len(pending)} to run ({spec.deck} vs {len(spec.opponents)} opponents, "
            f"workers={workers})"
        )

    lock = threading.Lock()
    written = {"games": 0, "samples": 0, "errors": 0}

    def _on_result(result: dict[str, Any]) -> None:
        with lock:
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(dumps(result) + "\n")
            written["games"] += 1
            written["samples"] += len(result.get("samples", []))
            written["errors"] += 1 if result.get("error") else 0

    if pending:
        run_task_shard(pending, worker_module="train.collect_worker", workers=workers, on_result=_on_result)

    _done, total_games, total_samples, total_errors = _scan_samples(samples_path)
    manifest.update(
        {
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
            "games": total_games,
            "samples": total_samples,
            "errors": total_errors,
            "samples_path": str(samples_path),
        }
    )
    _write_manifest(manifest_path, manifest)
    summary = {
        "tag": tag,
        "replay_dir": str(replay_dir),
        "games": total_games,
        "samples": total_samples,
        "errors": total_errors,
        "ran_this_call": written["games"],
        "manifest": str(manifest_path),
    }
    if progress:
        print(
            f"[collect] total {summary['games']} games / {summary['samples']} samples "
            f"(errors={total_errors}); ran {written['games']} this call -> {replay_dir}"
        )
    return summary


def _tolerant_read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="superconduct_aggro")
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--all-opponents", action="store_true", help="collect vs the full scripted pool")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--teacher-spec", default=None)
    parser.add_argument("--opponent-spec", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--tag", default="collect")
    # S-C search teacher: with ``--search`` the collector runs MCCFR and stores
    # solver targets instead of scripted-expert behaviour cloning.
    parser.add_argument("--search", action="store_true", help="MCCFR search teacher (S-C distillation data)")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--value-spec", default="")
    parser.add_argument("--search-every", type=int, default=1)
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--no-belief-root", action="store_true")
    parser.add_argument("--belief-samples", type=int, default=2)
    parser.add_argument("--bridge-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opponents = args.opponents
    if args.all_opponents:
        from eval.opponents import opponent_names

        opponents = opponent_names()
    if not opponents:
        raise SystemExit("provide --opponents ... or --all-opponents")
    search = None
    if args.search:
        search = SearchSpec(
            iterations=args.iterations,
            depth=args.depth,
            candidate_limit=args.candidate_limit,
            value_spec=args.value_spec,
            search_every=args.search_every,
            max_searches=args.max_searches,
            belief_root=not args.no_belief_root,
            belief_samples=args.belief_samples,
            bridge_workers=args.bridge_workers,
        )
    spec = CollectSpec(
        deck=args.deck,
        opponents=tuple(opponents),
        seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        teacher_spec=args.teacher_spec,
        opponent_spec=args.opponent_spec,
        search=search,
    )
    summary = collect(spec, tag=args.tag, workers=args.workers, max_tasks=args.max_tasks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
