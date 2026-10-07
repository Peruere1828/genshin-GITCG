"""Append-only JSONL helpers.

Episode/decision data is written as JSONL with one record per line. Enum and
dataclass values are normalized so records stay JSON-serializable and stable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, bytes):
        return obj.hex()
    if hasattr(obj, "tolist"):  # numpy scalars/arrays
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def dumps(record: Any) -> str:
    return json.dumps(record, default=_default, ensure_ascii=False, sort_keys=True)


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(record))
        handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))
