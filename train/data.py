"""Replay data collection for CVPN training (PLAN.md WS3/WS6).

Collects decision samples by playing games with a *teacher* policy (scripted
expert, or later a search resolver) and encoding each decision context with
``reps.observation_encoder``. The teacher's chosen action is stored as an option
**index** (positional), not a low-level code, so samples are unaffected by the
process-global action-code assignment (AGENTS.md "重放确定性坑").
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import Policy, build_policy
from eval.stats import outcome_score
from reps.observation_encoder import (
    BeliefTarget,
    EncodedObservation,
    TokenObservationEncoder,
)


@dataclass
class Sample:
    observation: EncodedObservation
    teacher_index: int
    value_target: float
    request_type: str
    step_index: int
    deck: str
    opponent: str
    # M3-CPU S-C (G6) search-as-teacher targets. All optional and absent in the
    # legacy hard-BC format, so old replays keep loading (``teacher_index`` +
    # ``value_target`` stay the behaviour-cloning fallback).
    #
    # ``policy_target`` is the solver's **root average strategy** re-indexed to
    # the encoded option order (see ``align_teacher_policy``); it distils a
    # policy CE/KL on the soft teacher distribution instead of the one-hot
    # ``teacher_index``.
    policy_target: tuple[float, ...] | None = None
    # ``action_values`` is the root action-value vector from the *player-0*
    # perspective (zero-sum sign-converted), for diagnostics / optional
    # per-action value auxiliary losses.
    action_values: tuple[float, ...] | None = None
    # ``search_value`` is the solver's root value from the player-0 perspective
    # (the value-regression target; ``value_target`` remains the episode outcome).
    search_value: float | None = None
    # Small provenance block (iterations/depth/particles/root_player) for reports.
    search_meta: dict[str, Any] | None = None

    @property
    def n_options(self) -> int:
        return len(self.observation.option_features)

    @property
    def has_search_target(self) -> bool:
        return self.policy_target is not None or self.search_value is not None


_TARGET_EPS = 1.0e-8


def semantic_keys_for_context(context: Any) -> list[Any]:
    """Semantic action keys for a decision context's legal options (encoder order).

    Mirrors ``envs.cfr_search.CFRSearchPolicy``'s key construction and
    ``reps.action_hierarchy.semantic_action_key_for_spec``: the keys are stable
    across determinizations (unlike raw process-local action codes), so a solver
    strategy keyed by them re-indexes cleanly onto the encoded option order.
    """
    from reps.action_hierarchy import semantic_action_key_for_spec

    specs = tuple(context.legal_low_level_specs)
    keys: list[Any] = []
    for spec in specs:
        keys.append(semantic_action_key_for_spec(spec))
    return keys


def align_teacher_policy(
    strategy: Any, keys: Sequence[Any], n_options: int
) -> tuple[float, ...] | None:
    """Re-index a semantic-key root strategy onto the encoded option order.

    Returns a normalised probability vector, or ``None`` when no probability mass
    lands on the encoded options (the caller then falls back to the hard
    ``teacher_index`` one-hot). Actions not covered by the strategy get 0 mass;
    the prior/strategy never *deletes* an encoded legal action (§5.4) -- a 0 is a
    learned (or unvisited) weight, not a removal from the support.
    """
    n_options = int(n_options)
    if n_options <= 0 or not strategy:
        return None
    values = [max(0.0, float(strategy.get(key, 0.0))) for key in list(keys)[:n_options]]
    values.extend(0.0 for _ in range(n_options - len(values)))
    total = sum(values)
    if total <= _TARGET_EPS:
        return None
    return tuple(value / total for value in values)


def align_teacher_action_values(
    action_values: Any, keys: Sequence[Any], n_options: int
) -> tuple[float, ...] | None:
    """Re-index per-action root values onto the encoded option order (no normalising)."""
    n_options = int(n_options)
    if n_options <= 0 or not action_values:
        return None
    values = [float(action_values.get(key, 0.0)) for key in list(keys)[:n_options]]
    values.extend(0.0 for _ in range(n_options - len(values)))
    return tuple(values)


class _CaptureTeacher:
    """Wrap a policy and record (built_context, chosen_code) for each decision."""

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self.name = f"capture:{getattr(inner, 'name', '?')}"
        self.items: list[tuple[object, int]] = []

    def choose(self, built) -> int:
        code = int(self.inner.choose(built))
        self.items.append((built, code))
        return code


def collect_samples(
    deck: str,
    opponent: str,
    seeds: Sequence[int],
    *,
    encoder: TokenObservationEncoder,
    teacher_spec: str | None = None,
    opponent_spec: str | None = None,
    capture_targets: bool = True,
) -> list[Sample]:
    """Collect teacher-labelled samples for ``deck`` (player 0) vs ``opponent``.

    With ``capture_targets`` (default) the match runs with a god-view
    ``full_state`` so the encoder can fill the *label* fields
    (``belief_target`` / ``privileged_state``) from the true state; agent
    observations still come from the masked ``player_view`` (no leak). Pass
    ``False`` for the legacy zeros-target format.
    """
    samples: list[Sample] = []
    for seed in seeds:
        teacher = build_policy(
            teacher_spec or f"expert:{deck}", seed=derive_seed(seed, "p0", deck), role="p0"
        )
        capture = _CaptureTeacher(teacher)
        opp = build_policy(
            opponent_spec or f"expert:{opponent}",
            seed=derive_seed(seed, "p1", opponent),
            role="p1",
        )
        record = run_match(
            deck_spec(deck),
            deck_spec(opponent),
            capture,
            opp,
            seed=int(seed),
            full_state_for_agents=capture_targets,
        )
        if record.error:
            continue
        value = outcome_score(record.winner, 0)
        for built, code in capture.items:
            context = built.context
            codes = list(int(c) for c in context.legal_low_level_codes)
            if int(code) not in codes:
                continue
            index = codes.index(int(code))
            observation = encoder.encode_context(
                context, history=(), include_training_targets=capture_targets
            )
            if not observation.option_features:
                continue
            samples.append(
                Sample(
                    observation=observation,
                    teacher_index=index,
                    value_target=value,
                    request_type=context.request_type.value,
                    step_index=context.step_index,
                    deck=deck,
                    opponent=opponent,
                )
            )
    return samples


def collect_search_samples(
    deck: str,
    opponent: str,
    seeds: Sequence[int],
    *,
    encoder: TokenObservationEncoder,
    bridge: Any,
    iterations: int = 8,
    depth: int = 1,
    candidate_limit: int | None = None,
    value_spec: str = "",
    search_every: int = 1,
    max_searches: int | None = None,
    belief_root: bool = True,
    belief_samples: int = 2,
    request_types: tuple[str, ...] | None = None,
    capture_targets: bool = True,
) -> list[Sample]:
    """Collect **search-distillation** samples: MCCFR root strategy + value targets.

    For each seed the search player (player 0) solves its decisions with the
    engine MCCFR adapter (``envs.cfr_search.run_cfr_search_match``, S-C/a) on a
    public-belief root, and each searched decision contributes the solver targets
    (root average strategy by semantic key, root value, per-action values) joined
    onto the encoded observation. Non-searched player-0 decisions still emit a
    behaviour-cloning sample (``teacher_index``), so the dataset degrades
    gracefully to BC where the solver was not run.

    ``capture_targets`` runs the match with a god-view ``full_state`` so the
    encoder fills the belief labels (used by the belief head); the search root
    itself still comes from public-belief determinization only (charter §5.2).
    """
    # Imported lazily: ``train.data`` is imported by engine-free training code.
    from envs.cfr_search import run_cfr_search_match

    samples: list[Sample] = []
    for seed in seeds:
        captured: list[list[tuple[Any, int]]] = [[], []]

        def _sink(player: int, built: Any, code: int) -> None:
            captured[int(player)].append((built, int(code)))

        result = run_cfr_search_match(
            deck,
            opponent,
            bridge=bridge,
            seed=int(seed),
            search_players=(0,),
            iterations=iterations,
            depth=depth,
            candidate_limit=candidate_limit,
            value_spec=value_spec,
            search_every=search_every,
            max_searches=max_searches,
            belief_root=belief_root,
            belief_samples=belief_samples,
            request_types=request_types,
            built_sink=_sink,
            full_state_for_agents=capture_targets,
        )
        if result.record.error:
            continue
        value_p0 = outcome_score(result.record.winner, 0)
        searched = {(int(d.player), int(d.ordinal)): d for d in result.searches}
        # Player 0 is the search teacher; index by that player's decision ordinal.
        for ordinal, (built, code) in enumerate(captured[0]):
            context = built.context
            codes = [int(c) for c in context.legal_low_level_codes]
            if not codes:
                continue
            observation = encoder.encode_context(
                context, history=(), include_training_targets=capture_targets
            )
            if not observation.option_features:
                continue
            keys = semantic_keys_for_context(context)
            n_options = len(observation.option_features)
            decision = searched.get((0, ordinal))
            if decision is not None:
                policy_target = align_teacher_policy(decision.strategy, keys, n_options)
                action_values = align_teacher_action_values(
                    decision.action_values, keys, n_options
                )
                if int(decision.player) != 0 and action_values is not None:
                    action_values = tuple(-value for value in action_values)
                root_value = float(decision.value)
                search_value = root_value if int(decision.player) == 0 else -root_value
                teacher_index = int(decision.chosen_option)
                search_meta = {
                    "iterations": int(decision.iterations),
                    "depth": int(decision.depth),
                    "particles": int(decision.particles),
                    "root_player": int(decision.player),
                }
            else:
                policy_target = None
                action_values = None
                search_value = None
                search_meta = None
                teacher_index = codes.index(int(code)) if int(code) in codes else 0
            teacher_index = max(0, min(int(teacher_index), n_options - 1))
            samples.append(
                Sample(
                    observation=observation,
                    teacher_index=teacher_index,
                    value_target=value_p0,
                    request_type=context.request_type.value,
                    step_index=context.step_index,
                    deck=deck,
                    opponent=opponent,
                    policy_target=policy_target,
                    action_values=action_values,
                    search_value=search_value,
                    search_meta=search_meta,
                )
            )
    return samples


def sample_stats(samples: Sequence[Sample]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0}
    sizes = sorted(s.n_options for s in samples)
    wins = sum(1 for s in samples if s.value_target == 1.0)
    searched = sum(1 for s in samples if s.has_search_target)
    return {
        "n": n,
        "mean_options": sum(sizes) / n,
        "max_options": sizes[-1],
        "distinct_request_types": sorted({s.request_type for s in samples}),
        "win_fraction": wins / n,
        # S-C additions (optional; readers of the base keys are unaffected).
        "n_search_targets": searched,
        "search_fraction": searched / n,
    }


# --------------------------------------------------------------------------- #
# serialization (cluster data plane: CPU nodes collect -> storage -> GPU nodes train)
# --------------------------------------------------------------------------- #
# Two on-disk formats share the same line-per-game task-record envelope (A6):
#   * ``jsonl``   -- the historical format: every dense array is a JSON list;
#   * ``compact`` -- dense float arrays are packed into a base64 binary buffer
#     (fp16 by default) carrying an explicit shape, which cuts file size and the
#     parent-side JSON churn. ``observation_from_payload`` auto-detects per field,
#     so readers never need to know which format a file used.
COMPACT_FORMATS = ("jsonl", "compact")
DEFAULT_COMPACT_DTYPE = "float16"
_DENSE_KEY = "__dense__"
_NUMPY_DTYPES = {"float16": np.float16, "float32": np.float32}


def _dtype_array(dtype: str):
    try:
        return _NUMPY_DTYPES[str(dtype)]
    except KeyError as exc:  # pragma: no cover - CLI constrains the choices
        raise ValueError(
            f"unsupported compact dtype {dtype!r}; expected one of {sorted(_NUMPY_DTYPES)}"
        ) from exc


def is_dense_payload(value: Any) -> bool:
    """True when ``value`` is a compact dense-array envelope (base64 + shape)."""
    return isinstance(value, dict) and _DENSE_KEY in value


def _pack_dense(values: Any, dtype: str) -> dict[str, Any]:
    """Pack a 1D/2D numeric sequence into a base64 binary buffer + shape."""
    arr = np.asarray(values, dtype=_dtype_array(dtype))
    return {
        _DENSE_KEY: str(dtype),
        "shape": [int(v) for v in arr.shape],
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _unpack_dense(payload: dict[str, Any]):
    arr = np.frombuffer(
        base64.b64decode(payload["data"]), dtype=_dtype_array(payload[_DENSE_KEY])
    )
    shape = payload.get("shape")
    if shape:
        arr = arr.reshape([int(v) for v in shape])
    return arr


def _is_uniform_2d(rows: Any) -> bool:
    if not isinstance(rows, (list, tuple)):
        return False
    if not rows:
        return True
    if not isinstance(rows[0], (list, tuple)):
        return False
    width = len(rows[0])
    return all(isinstance(row, (list, tuple)) and len(row) == width for row in rows)


def _encode_2d(rows: Any, dtype: str) -> Any:
    """Compact-encode a 2D float matrix (falls back to lists if ragged/empty)."""
    if not rows:
        return []
    if _is_uniform_2d(rows):
        return _pack_dense(rows, dtype)
    return [list(row) for row in rows]


def _encode_1d(values: Any, dtype: str) -> Any:
    """Compact-encode a 1D float vector (empty stays an empty list)."""
    values = list(values)
    if not values:
        return []
    return _pack_dense(values, dtype)


def _as_float_1d(value: Any) -> tuple[float, ...]:
    if is_dense_payload(value):
        return tuple(float(v) for v in _unpack_dense(value).reshape(-1))
    return tuple(float(v) for v in value)


def _as_float_2d(value: Any) -> tuple[tuple[float, ...], ...]:
    if is_dense_payload(value):
        arr = _unpack_dense(value)
        return tuple(tuple(float(v) for v in row) for row in arr.reshape(len(arr), -1))
    return tuple(tuple(float(v) for v in row) for row in value)


def observation_to_payload(
    observation: EncodedObservation,
    *,
    format: str = "jsonl",
    compact_dtype: str = DEFAULT_COMPACT_DTYPE,
    include_privileged_state: bool = True,
) -> dict[str, Any]:
    """JSON-serializable payload for an ``EncodedObservation``.

    ``format="jsonl"`` (default) is the lossless historical representation (dense
    arrays as JSON lists) -- byte-for-byte compatible with pre-A6 replays.
    ``format="compact"`` replaces the dense float fields (``token_features``,
    ``option_features``, the belief histograms and, unless disabled,
    ``privileged_state``) with base64 fp16/fp32 buffers. Int/bool fields and the
    record envelope are unchanged, so the game is still one JSONL line.
    """
    belief = observation.belief_target
    payload: dict[str, Any] = {
        "token_features": [list(row) for row in observation.token_features],
        "token_mask": list(observation.token_mask),
        "opponent_token_mask": list(observation.opponent_token_mask),
        "option_features": [list(row) for row in observation.option_features],
        "option_mask": list(observation.option_mask),
        "low_level_action_codes": list(observation.low_level_action_codes),
        "low_to_high_codes": list(observation.low_to_high_codes),
        "high_level_mask": list(observation.high_level_mask),
        "privileged_state": list(observation.privileged_state),
        "belief_target": {
            "hand_histogram": list(belief.hand_histogram),
            "hand_size": float(belief.hand_size),
            "remaining_deck_histogram": list(belief.remaining_deck_histogram),
            "burst_ready_count": float(belief.burst_ready_count),
            "group_hand_presence": list(belief.group_hand_presence),
            "group_remaining_deck_presence": list(belief.group_remaining_deck_presence),
        },
        "opponent_tag_id": int(observation.opponent_tag_id),
        "opponent_entity_id": int(observation.opponent_entity_id),
        "opponent_deck_id": int(observation.opponent_deck_id),
    }
    if format == "jsonl":
        return payload
    if format != "compact":  # pragma: no cover - validated by callers/CLI
        raise ValueError(f"unknown payload format {format!r}; expected {COMPACT_FORMATS}")

    payload["token_features"] = _encode_2d(observation.token_features, compact_dtype)
    payload["option_features"] = _encode_2d(observation.option_features, compact_dtype)
    if include_privileged_state:
        payload["privileged_state"] = _encode_1d(observation.privileged_state, compact_dtype)
    else:
        payload.pop("privileged_state", None)
    payload["belief_target"] = {
        "hand_histogram": _encode_1d(belief.hand_histogram, compact_dtype),
        "hand_size": float(belief.hand_size),
        "remaining_deck_histogram": _encode_1d(belief.remaining_deck_histogram, compact_dtype),
        "burst_ready_count": float(belief.burst_ready_count),
        "group_hand_presence": _encode_1d(belief.group_hand_presence, compact_dtype),
        "group_remaining_deck_presence": _encode_1d(
            belief.group_remaining_deck_presence, compact_dtype
        ),
    }
    return payload


def observation_from_payload(payload: dict[str, Any]) -> EncodedObservation:
    """Rebuild an ``EncodedObservation`` from either format (auto-detected)."""
    belief = payload["belief_target"]
    return EncodedObservation(
        token_features=_as_float_2d(payload["token_features"]),
        token_mask=tuple(bool(v) for v in payload["token_mask"]),
        opponent_token_mask=tuple(bool(v) for v in payload["opponent_token_mask"]),
        option_features=_as_float_2d(payload["option_features"]),
        option_mask=tuple(bool(v) for v in payload["option_mask"]),
        low_level_action_codes=tuple(int(v) for v in payload["low_level_action_codes"]),
        low_to_high_codes=tuple(int(v) for v in payload["low_to_high_codes"]),
        high_level_mask=tuple(bool(v) for v in payload["high_level_mask"]),
        privileged_state=_as_float_1d(payload.get("privileged_state", ())),
        belief_target=BeliefTarget(
            hand_histogram=_as_float_1d(belief["hand_histogram"]),
            hand_size=float(belief["hand_size"]),
            remaining_deck_histogram=_as_float_1d(belief["remaining_deck_histogram"]),
            burst_ready_count=float(belief["burst_ready_count"]),
            group_hand_presence=_as_float_1d(belief["group_hand_presence"]),
            group_remaining_deck_presence=_as_float_1d(
                belief["group_remaining_deck_presence"]
            ),
        ),
        opponent_tag_id=int(payload["opponent_tag_id"]),
        opponent_entity_id=int(payload["opponent_entity_id"]),
        opponent_deck_id=int(payload["opponent_deck_id"]),
    )


def sample_to_payload(
    sample: Sample,
    *,
    format: str = "jsonl",
    compact_dtype: str = DEFAULT_COMPACT_DTYPE,
    include_privileged_state: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation": observation_to_payload(
            sample.observation,
            format=format,
            compact_dtype=compact_dtype,
            include_privileged_state=include_privileged_state,
        ),
        "teacher_index": int(sample.teacher_index),
        "value_target": float(sample.value_target),
        "request_type": sample.request_type,
        "step_index": int(sample.step_index),
        "deck": sample.deck,
        "opponent": sample.opponent,
    }
    # S-C search targets: emitted only when present, so the base (hard-BC) payload
    # is byte-identical to the pre-S-C format and old readers stay valid.
    if sample.policy_target is not None:
        payload["policy_target"] = [float(v) for v in sample.policy_target]
    if sample.action_values is not None:
        payload["action_values"] = [float(v) for v in sample.action_values]
    if sample.search_value is not None:
        payload["search_value"] = float(sample.search_value)
    if sample.search_meta is not None:
        payload["search_meta"] = dict(sample.search_meta)
    return payload


def sample_from_payload(payload: dict[str, Any]) -> Sample:
    policy_target = payload.get("policy_target")
    action_values = payload.get("action_values")
    search_value = payload.get("search_value")
    search_meta = payload.get("search_meta")
    return Sample(
        observation=observation_from_payload(payload["observation"]),
        teacher_index=int(payload["teacher_index"]),
        value_target=float(payload["value_target"]),
        request_type=payload["request_type"],
        step_index=int(payload["step_index"]),
        deck=payload["deck"],
        opponent=payload["opponent"],
        policy_target=None if policy_target is None else tuple(float(v) for v in policy_target),
        action_values=None if action_values is None else tuple(float(v) for v in action_values),
        search_value=None if search_value is None else float(search_value),
        search_meta=None if search_meta is None else dict(search_meta),
    )


def write_samples_jsonl(
    path: str | Path,
    samples: Sequence[Sample],
    *,
    format: str = "jsonl",
    compact_dtype: str = DEFAULT_COMPACT_DTYPE,
    include_privileged_state: bool = True,
) -> int:
    """Append flat per-sample JSONL records; return the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample_to_payload(
                sample,
                format=format,
                compact_dtype=compact_dtype,
                include_privileged_state=include_privileged_state,
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(samples)


def iter_samples_jsonl(path: str | Path) -> Iterator[Sample]:
    """Yield samples from a flat per-sample JSONL file (ignores torn lines)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield sample_from_payload(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue


def load_samples(path: str | Path) -> list[Sample]:
    """Load samples from a flat JSONL file *or* a replay dir of task records."""
    path = Path(path)
    if path.is_dir():
        return load_replay_samples(path)
    return list(iter_samples_jsonl(path))


# --------------------------------------------------------------------------- #
# task-record format used by the resumable collector (one line = one game)
# --------------------------------------------------------------------------- #
def iter_task_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield complete task records, skipping a torn trailing line and blanks."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn/partial line -> treat that task as not done
            yield record


def load_replay_samples(replay_dir: str | Path) -> list[Sample]:
    """Flatten a collector replay dir into samples (samples of every complete game)."""
    replay_dir = Path(replay_dir)
    samples: list[Sample] = []
    for record in iter_task_records(replay_dir / "samples.jsonl"):
        for payload in record.get("samples", []):
            samples.append(sample_from_payload(payload))
    return samples
