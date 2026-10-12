"""A6 micro-benchmark: jsonl vs compact payload size and (de)serialization speed.

Measures, in-process and single-threaded:

* **size**  -- serialized bytes per decision and per game (record envelope), and
  the compact/jsonl ratio;
* **speed** -- encode (Sample -> JSON line) and decode (JSON line -> Sample)
  throughput in MB/s and samples/s.

Deterministic and CPU-only, so it is valid even while another job is collecting
(size numbers do not depend on machine load; the MB/s numbers are indicative and
should be read as an upper bound under contention).

Prefer ``--replays`` with real data: real observations are mostly 0/1 indicators
(short JSON), so the compact ratio is ~0.54; ``--synthetic`` uses uniform random
floats (long JSON) and overstates the win (~0.16).

Usage::

    python -m scripts.bench_payload_format --replays data/replays/<tag> --games 200 --json out.json
    python -m scripts.bench_payload_format --synthetic --samples 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from train.data import Sample, sample_from_payload, sample_to_payload


def _load_records(source: Path, max_games: int) -> list[list[Sample]]:
    if source.is_dir():
        source = source / "samples.jsonl"
    games: list[list[Sample]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("error"):
                continue
            games.append([sample_from_payload(s) for s in record.get("samples", ())])
            if len(games) >= max_games:
                break
    return games


def _synthetic_games(n_games: int, n_samples: int) -> list[list[Sample]]:
    import numpy as np

    from reps.observation_encoder import BeliefTarget, EncodedObservation

    rng = np.random.default_rng(0)
    games: list[list[Sample]] = []
    for g in range(n_games):
        samples = []
        for i in range(n_samples):
            t = int(rng.integers(8, 16))
            k = int(rng.integers(16, 40))
            vt, vd, vh = 152, 10, 68
            samples.append(
                Sample(
                    observation=EncodedObservation(
                        token_features=tuple(
                            tuple(float(x) for x in rng.random(65)) for _ in range(t)
                        ),
                        token_mask=tuple(bool(x) for x in rng.integers(0, 2, t)),
                        opponent_token_mask=tuple(bool(x) for x in rng.integers(0, 2, t)),
                        option_features=tuple(
                            tuple(float(x) for x in rng.random(vd)) for _ in range(k)
                        ),
                        option_mask=tuple(bool(x) for x in rng.integers(0, 2, k)),
                        low_level_action_codes=tuple(int(x) for x in rng.integers(0, 900, k)),
                        low_to_high_codes=tuple(int(x) for x in rng.integers(0, 60, k)),
                        high_level_mask=tuple(bool(x) for x in rng.integers(0, 2, 51)),
                        privileged_state=tuple(float(x) for x in rng.random(1022)),
                        belief_target=BeliefTarget(
                            hand_histogram=tuple(float(x) for x in rng.random(vt)),
                            hand_size=float(rng.integers(0, 8)),
                            remaining_deck_histogram=tuple(float(x) for x in rng.random(vt)),
                            burst_ready_count=float(rng.integers(0, 3)),
                            group_hand_presence=tuple(float(x) for x in rng.random(vh)),
                            group_remaining_deck_presence=tuple(float(x) for x in rng.random(vh)),
                        ),
                        opponent_tag_id=int(rng.integers(0, 12)),
                        opponent_entity_id=0,
                        opponent_deck_id=0,
                    ),
                    teacher_index=int(rng.integers(0, k)),
                    value_target=float(rng.integers(0, 2)),
                    request_type="action",
                    step_index=i,
                    deck="synthetic",
                    opponent=f"opp{g}",
                )
            )
        games.append(samples)
    return games


def _line(games: list[list[Sample]], fmt: str, dtype: str, priv: bool) -> str:
    record = {
        "index": 0,
        "error": None,
        "samples": [
            sample_to_payload(s, format=fmt, compact_dtype=dtype, include_privileged_state=priv)
            for s in games[0]
        ],
    }
    return json.dumps(record, ensure_ascii=False)


def _time(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _measure(
    games: list[list[Sample]], fmt: str, dtype: str, priv: bool, repeats: int
) -> dict[str, Any]:
    n_samples = sum(len(g) for g in games)
    # encode: every game -> JSON line
    def encode():
        for g in games:
            record = {
                "index": 0,
                "samples": [
                    sample_to_payload(
                        s, format=fmt, compact_dtype=dtype, include_privileged_state=priv
                    )
                    for s in g
                ],
            }
            json.dumps(record, ensure_ascii=False)

    # pre-build the lines once for size + decode
    lines = [
        json.dumps(
            {
                "index": i,
                "samples": [
                    sample_to_payload(
                        s, format=fmt, compact_dtype=dtype, include_privileged_state=priv
                    )
                    for s in g
                ],
            },
            ensure_ascii=False,
        )
        for i, g in enumerate(games)
    ]
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)

    def decode():
        for line in lines:
            record = json.loads(line)
            for payload in record["samples"]:
                sample_from_payload(payload)

    enc = _time(encode, repeats)
    dec = _time(decode, repeats)
    mb = total_bytes / 1e6
    return {
        "format": fmt,
        "dtype": dtype,
        "privileged_state": bool(priv),
        "games": len(games),
        "samples": n_samples,
        "bytes_per_game": total_bytes / max(1, len(games)),
        "bytes_per_sample": total_bytes / max(1, n_samples),
        "total_MB": mb,
        "encode_s": round(enc, 4),
        "decode_s": round(dec, 4),
        "encode_MB/s": round(mb / enc, 1),
        "decode_MB/s": round(mb / dec, 1),
        "encode_samples/s": round(n_samples / enc, 1),
        "decode_samples/s": round(n_samples / dec, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="*", default=None, help="replay dirs or samples.jsonl files")
    p.add_argument("--games", type=int, default=200, help="cap games read per source")
    p.add_argument("--synthetic", action="store_true", help="ignore replays, synthesize samples")
    p.add_argument("--synthetic-games", type=int, default=50)
    p.add_argument("--synthetic-samples", type=int, default=38)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", default=None, help="write the report JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic or not args.replays:
        games = _synthetic_games(args.synthetic_games, args.synthetic_samples)
        source = "synthetic"
    else:
        games = []
        for src in args.replays:
            games.extend(_load_records(Path(src), args.games))
        source = ", ".join(args.replays)
    if not games:
        raise SystemExit("no games loaded")

    configs = [
        ("jsonl", "float16", True),
        ("compact", "float32", True),
        ("compact", "float16", True),
        ("compact", "float16", False),
    ]
    rows = [_measure(games, fmt, dtype, priv, args.repeats) for fmt, dtype, priv in configs]
    base = rows[0]["bytes_per_sample"]

    print(f"source={source} games={len(games)} samples/game={len(games[0])}")
    header = f"{'format':<8} {'dtype':<8} {'priv':<5} {'B/sample':>9} {'B/game':>10} {'MB':>8} {'ratio':>6} {'enc MB/s':>9} {'dec MB/s':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['format']:<8} {row['dtype']:<8} {str(row['privileged_state']):<5} "
            f"{row['bytes_per_sample']:>9.0f} {row['bytes_per_game']:>10.0f} "
            f"{row['total_MB']:>8.2f} {row['bytes_per_sample'] / base:>6.3f} "
            f"{row['encode_MB/s']:>9.1f} {row['decode_MB/s']:>9.1f}"
        )
    report = {
        "source": source,
        "games": len(games),
        "samples_per_game": len(games[0]),
        "rows": rows,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
