"""Batched CVPN inference (PLAN.md WS3 / §5): GPU scores, CPU searches.

The SoG loop is "CPU-heavy search/rollout, GPU-only inference" (PLAN.md §5). When
one process holds the network this is trivial; when many CPU search/rollout
processes share one GPU it is not. This module provides:

* ``BatchScorer`` -- in-process batched scoring (device-aware, optional bf16,
  ``max_batch_size`` chunking). Use it directly when GPU and CPU share a process.
* a JSON-lines **service** (``python -m train.batched_inference --serve``) that
  owns the model and answers batches of encoded observations; launched *by the
  job* and torn down with it -- no long-lived service (D7).
* ``BatchedInferenceService`` -- the client: spawns the service subprocess and
  exposes ``score``/``choose``; context-managed.

Only the encoder runs CPU-side; ``EncodedObservation`` payloads cross the wire
(``train.data`` (de)serialization), so no raw engine state is shipped. Masked
options are returned as ``null`` so the wire stays JSON-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from train.data import observation_from_payload, observation_to_payload


@dataclass
class BatchScorer:
    """Score encoded observations on ``device`` in chunks of ``max_batch_size``."""

    model: Any
    device: str = "cpu"
    max_batch_size: int = 128
    use_bf16: bool = False

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self.model.eval()
        self.model.to(self.device)
        if self.use_bf16:
            self.model.to(torch.bfloat16)

    def _autocast(self):
        torch = self._torch
        if self.use_bf16 and self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def score(
        self, observations: Sequence[Any]
    ) -> tuple[list[list[float | None]], list[float]]:
        """Return ``(logits, values)``; masked option entries are ``None``."""
        from train.model import collate

        torch = self._torch
        logits_out: list[list[float | None]] = []
        values_out: list[float] = []
        observations = list(observations)
        for start in range(0, len(observations), self.max_batch_size):
            chunk = observations[start : start + self.max_batch_size]
            tokens, token_mask, options, option_mask = collate(chunk)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)
            options = options.to(self.device)
            option_mask = option_mask.to(self.device)
            with torch.no_grad(), self._autocast():
                logits, value = self.model(tokens, token_mask, options, option_mask)
            logits = logits.float().cpu()
            values = value.float().cpu()
            for row, mask in zip(logits, option_mask.cpu()):
                logits_out.append(
                    [
                        float(entry) if bool(keep) else None
                        for entry, keep in zip(row.tolist(), mask.tolist())
                    ]
                )
            values_out.extend(float(v) for v in values.tolist())
        return logits_out, values_out

    def choose(self, observation: Any) -> int:
        """Greedy option index for a single observation (0 if none legal)."""
        logits, _values = self.score([observation])
        if not logits or not logits[0]:
            return 0
        row = logits[0]
        best = max(
            (i for i, v in enumerate(row) if v is not None),
            key=lambda i: row[i],
            default=0,
        )
        return int(best)


@dataclass
class ScoredPolicy:
    """A ``Policy`` that encodes CPU-side and delegates scoring to a scorer.

    Works with a local ``BatchScorer`` or a ``BatchedInferenceService`` client, so
    the same policy runs whether the GPU is in-process or behind the job service.
    """

    scorer: Any
    encoder: Any
    name: str = "neural_batch"
    temperature: float = 0.0
    seed: int | None = None
    last_value: float = field(default=0.0, init=False)
    calls: int = field(default=0, init=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose(self, built) -> int:
        codes = list(int(c) for c in built.context.legal_low_level_codes)
        if not codes:
            return -1
        observation = self.encoder.encode_context(
            built.context, history=(), include_training_targets=False
        )
        index = self.choose_index(observation)
        index = max(0, min(index, len(codes) - 1))
        return codes[index]

    def choose_index(self, observation: Any) -> int:
        """Option index for an already-encoded observation (no encoder needed)."""
        if not observation.option_features:
            return 0
        logits, values = self.scorer.score([observation])
        self.calls += 1
        if values:
            self.last_value = float(values[0])
        row = logits[0] if logits else []
        legal = [i for i, v in enumerate(row) if v is not None]
        if not legal:
            return 0
        if self.temperature and self.temperature > 0:
            weights = [pow(2.718281828, float(row[i]) / self.temperature) for i in legal]
            total = sum(weights) or 1.0
            pick = self._rng.random() * total
            running = 0.0
            index = legal[-1]
            for i, weight in zip(legal, weights):
                running += weight
                if pick <= running:
                    index = i
                    break
        else:
            index = max(legal, key=lambda i: row[i])
        return int(index)


# --------------------------------------------------------------------------- #
# JSON-lines service
# --------------------------------------------------------------------------- #
def _handle_request(scorer: BatchScorer, request: dict[str, Any]) -> dict[str, Any]:
    payloads = request.get("observations", ())
    observations = [observation_from_payload(p) for p in payloads]
    logits, values = scorer.score(observations)
    return {"logits": logits, "values": values, "n": len(observations)}


def serve(checkpoint: str, *, device: str, bf16: bool, max_batch_size: int) -> int:
    """Read scoring requests on stdin, write responses on stdout (JSON-lines)."""
    from train.model import CVPN, checkpoint_encoder_config

    model = CVPN.load(checkpoint)
    scorer = BatchScorer(model=model, device=device, max_batch_size=max_batch_size, use_bf16=bf16)
    print(
        f"[batched_inference] serving device={device} bf16={bf16} "
        f"max_batch_size={max_batch_size} encoder_config="
        f"{'yes' if checkpoint_encoder_config(checkpoint) else 'default'}",
        file=sys.stderr,
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = _handle_request(scorer, json.loads(line))
        except Exception as exc:  # noqa: BLE001 - report per request, keep serving
            response = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


class BatchedInferenceService:
    """Client that owns a ``--serve`` subprocess (spawned per job, no daemon)."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda",
        bf16: bool = True,
        max_batch_size: int = 128,
        python: str | None = None,
        repo_root: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        from common.paths import repo_root as _repo_root

        root = repo_root or str(_repo_root())
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            python or sys.executable,
            "-m",
            "train.batched_inference",
            "--serve",
            "--checkpoint",
            checkpoint,
            "--device",
            device,
            "--max-batch-size",
            str(max_batch_size),
        ]
        if bf16:
            command.append("--bf16")
        self.timeout = float(timeout)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=root,
            env=env,
        )
        self._stderr: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line)

    def score(
        self, observations: Sequence[Any]
    ) -> tuple[list[list[float | None]], list[float]]:
        payloads = [observation_to_payload(obs) for obs in observations]
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps({"observations": payloads}) + "\n")
        self._process.stdin.flush()
        reader: list[str | None] = [None]

        def _read() -> None:
            reader[0] = self._process.stdout.readline()  # type: ignore[union-attr]

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        if thread.is_alive() or reader[0] is None:
            raise RuntimeError("batched inference service timed out")
        response = json.loads(reader[0])
        if "error" in response:
            raise RuntimeError(f"batched inference service error: {response['error']}")
        return response["logits"], response["values"]

    def choose(self, observation: Any) -> int:
        logits, _values = self.score([observation])
        row = logits[0] if logits else []
        legal = [i for i, v in enumerate(row) if v is not None]
        if not legal:
            return 0
        return int(max(legal, key=lambda i: row[i]))

    def close(self) -> None:
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        self._stderr_thread.join(timeout=1)

    def __enter__(self) -> "BatchedInferenceService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="run the JSON-lines service")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.serve:
        return serve(
            args.checkpoint,
            device=args.device,
            bf16=args.bf16,
            max_batch_size=args.max_batch_size,
        )
    raise SystemExit("nothing to do: pass --serve")


if __name__ == "__main__":
    raise SystemExit(main())
