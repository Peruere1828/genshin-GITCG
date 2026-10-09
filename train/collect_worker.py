"""Worker entrypoint for ``train.collect``: JSON-lines in, JSON-lines out.

Run as ``python -m train.collect_worker``. Reads one ``CollectTask`` dict per line
and writes one task-record dict per line (the encoded samples of that game).
Kept quiet on stdout; failures are reported per-line as ``error`` so the parent
records the game rather than crashing the whole job.
"""

from __future__ import annotations

import json
import sys
import traceback

from train.collect import CollectTask, run_collect_task


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = CollectTask.from_dict(json.loads(line))
            result = run_collect_task(task)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return 1
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
