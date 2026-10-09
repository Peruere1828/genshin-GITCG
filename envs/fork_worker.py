"""Worker entrypoint for ``envs.fork_bridge``: JSON-lines in, JSON-lines out.

Run as ``python -m envs.fork_worker``. Reads one ``ForkTask`` dict per line on
stdin, forks + rolls out the branch, and writes one ``ForkResult`` dict per line
on stdout. The engine must not print to stdout (it would corrupt the protocol);
run with the parent's ``GITCG_WORKER_DEBUG=1`` to forward a traceback to stderr.
"""

from __future__ import annotations

import json
import sys
import traceback

from envs.fork_bridge import ForkTask, run_fork_task


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = ForkTask.from_dict(json.loads(line))
            result = run_fork_task(task)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return 1
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
