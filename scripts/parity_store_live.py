"""Compare a replay/store decision against the live engine encoding (P0).

Usage::

    python -m scripts.parity_store_live \
        --replay-dir data/replays/dfbc_dvalin_bonk_c80 \
        --deck dvalin_bonk --opponent natlan_battleship --seed 0 \
        --checkpoint data/checkpoints/a2/A0i.pt --strict

The script replays the exact same game with the same expert teacher/opponent and
compares, decision by decision:

1. live ``include_training_targets=True`` vs ``False`` (the flag must not change
   policy inputs);
2. replay payload vs live encoding (catches fp16 definition-id loss, layout
   drift and encoder-config drift);
3. ``teacher_index`` / ``legal_low_level_codes`` mapping;
4. optionally, checkpoint logits/argmax on the two feature sets.

Exit code is non-zero with ``--strict`` when any parity issue is found. The
report contains only aggregate counts + the first few divergent cells; raw
features are never written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.observation import default_encoder
from envs.parity import (
    ParityIssue,
    ParityRecorder,
    ParityReport,
    compare_observations,
)
from envs.policy import build_policy
from reps.observation_encoder import TokenObservationEncoder
from train.data import iter_task_records, observation_from_payload


def _load_record(replay_dir: str, deck: str, opponent: str, seed: int) -> dict[str, Any]:
    path = Path(replay_dir) / "samples.jsonl"
    if not path.exists():
        raise SystemExit(f"missing replay file: {path}")
    for record in iter_task_records(path):
        if (
            str(record.get("deck", "")) == deck
            and str(record.get("opponent", "")) == opponent
            and int(record.get("seed", -1)) == int(seed)
        ):
            return record
    raise SystemExit(
        f"no record for deck={deck!r} opponent={opponent!r} seed={seed} in {path}"
    )


def _replay(
    deck: str,
    opponent: str,
    seed: int,
    *,
    teacher_spec: str | None,
    opponent_spec: str | None,
) -> tuple[list[Any], Any]:
    teacher = build_policy(
        teacher_spec or f"expert:{deck}",
        seed=derive_seed(seed, "p0", deck),
        role="p0",
    )
    recorder = ParityRecorder(teacher)
    opp = build_policy(
        opponent_spec or f"expert:{opponent}",
        seed=derive_seed(seed, "p1", opponent),
        role="p1",
    )
    record = run_match(
        deck_spec(deck),
        deck_spec(opponent),
        recorder,
        opp,
        seed=int(seed),
        full_state_for_agents=True,
    )
    return recorder.items, record


def _add_issue(
    report: ParityReport,
    kind: str,
    decision: int,
    step_index: int | None,
    detail: str,
) -> None:
    report.add(
        ParityIssue(
            kind=kind,
            decision=decision,
            step_index=step_index,
            detail=detail,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", default=None, help="optional CVPN checkpoint for logits")
    p.add_argument("--teacher-spec", default=None)
    p.add_argument("--opponent-spec", default=None)
    p.add_argument("--limit", type=int, default=0, help="compare only the first N decisions (0=all)")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--max-issues", type=int, default=20)
    p.add_argument("--encoder", choices=["default", "checkpoint"], default="default")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json-out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = _load_record(args.replay_dir, args.deck, args.opponent, args.seed)
    payloads = list(record.get("samples") or ())
    by_step: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        by_step[int(payload.get("step_index", -1))] = payload

    encoder = default_encoder()
    checkpoint_config = None
    checkpoint = None
    if args.checkpoint:
        from train.model import checkpoint_encoder_config, load_cvpn_model

        checkpoint_config = checkpoint_encoder_config(args.checkpoint)
        checkpoint = load_cvpn_model(args.checkpoint)
    if args.encoder == "checkpoint":
        if checkpoint_config is None:
            raise SystemExit("--encoder checkpoint requires a checkpoint with encoder_config")
        encoder = TokenObservationEncoder.from_dict(checkpoint_config)

    items, match = _replay(
        args.deck,
        args.opponent,
        args.seed,
        teacher_spec=args.teacher_spec,
        opponent_spec=args.opponent_spec,
    )
    limit = len(items) if args.limit <= 0 else min(len(items), int(args.limit))
    report = ParityReport(ok=True, n_decisions=limit)

    if match.error:
        _add_issue(
            report,
            "MATCH_ERROR",
            0,
            None,
            f"live replay failed before parity could be checked: {match.error}",
        )

    for index, item in enumerate(items[:limit]):
        payload = by_step.get(item.step_index)
        if payload is None:
            _add_issue(
                report,
                "MISSING_DECISION",
                index,
                item.step_index,
                "live decision has no matching replay sample at this step_index",
            )
            continue
        corpus_obs = observation_from_payload(payload["observation"])
        live_obs = encoder.encode_context(
            item.built.context, history=(), include_training_targets=False
        )
        live_targets = encoder.encode_context(
            item.built.context, history=(), include_training_targets=True
        )

        # P1: the training-target flag must not change policy inputs.
        flag_report = compare_observations(
            live_targets,
            live_obs,
            decision=index,
            step_index=item.step_index,
            atol=0.0,
            max_issues=args.max_issues,
        )
        for issue in flag_report.issues:
            report.add(
                ParityIssue(
                    kind=f"TARGET_FLAG_{issue.kind}",
                    decision=issue.decision,
                    step_index=issue.step_index,
                    detail=issue.detail,
                    row=issue.row,
                    col=issue.col,
                    expected=issue.expected,
                    actual=issue.actual,
                )
            )

        # P2: replay/store feature vs live feature.
        corpus_report = compare_observations(
            corpus_obs,
            live_obs,
            decision=index,
            step_index=item.step_index,
            atol=args.atol,
            max_issues=args.max_issues,
        )
        for issue in corpus_report.issues:
            report.add(issue)

        # Action index must map to the same legal code.
        codes = tuple(int(v) for v in payload["observation"].get("low_level_action_codes", ()))
        teacher_index = int(payload.get("teacher_index", -1))
        if not (0 <= teacher_index < len(codes)):
            _add_issue(
                report,
                "TEACHER_INDEX",
                index,
                item.step_index,
                f"teacher_index={teacher_index} outside {len(codes)} legal codes",
            )
        elif int(codes[teacher_index]) != int(item.code):
            _add_issue(
                report,
                "TEACHER_CODE",
                index,
                item.step_index,
                f"replay codes[{teacher_index}]={codes[teacher_index]} != live code={item.code}",
            )

        if checkpoint is not None:
            import torch

            from train.model import collate

            token_ok = not len(corpus_obs.token_features) or len(
                corpus_obs.token_features[0]
            ) == int(checkpoint.config.token_dim)
            option_ok = not len(corpus_obs.option_features) or len(
                corpus_obs.option_features[0]
            ) == int(checkpoint.config.option_dim)
            if not token_ok or not option_ok:
                _add_issue(
                    report,
                    "CHECKPOINT_FEATURE_DIM",
                    index,
                    item.step_index,
                    "checkpoint token/option dim does not match the replay feature width",
                )
            else:
                with torch.no_grad():
                    t1, tm1, o1, om1 = collate([corpus_obs])
                    logits_store, _ = checkpoint(t1, tm1, o1, om1)
                    t2, tm2, o2, om2 = collate([live_obs])
                    logits_live, _ = checkpoint(t2, tm2, o2, om2)
                arg_store = int(logits_store.argmax(dim=-1).item())
                arg_live = int(logits_live.argmax(dim=-1).item())
                max_diff = float((logits_store - logits_live).abs().max().item())
                if arg_store != arg_live:
                    _add_issue(
                        report,
                        "MODEL_ARGMAX",
                        index,
                        item.step_index,
                        f"checkpoint argmax store={arg_store} live={arg_live} "
                        f"(max |Δlogit|={max_diff:.3g})",
                    )
                elif max_diff > float(args.atol):
                    _add_issue(
                        report,
                        "MODEL_LOGIT_DRIFT",
                        index,
                        item.step_index,
                        f"same argmax but max |Δlogit|={max_diff:.3g} > {args.atol:g}",
                    )

    output = {
        "replay_dir": str(args.replay_dir),
        "deck": args.deck,
        "opponent": args.opponent,
        "seed": int(args.seed),
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        **report.as_dict(),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
