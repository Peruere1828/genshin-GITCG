"""The observation encoder's ``history`` channel is actually wired in.

Engine-free tests: they hand-build masked snapshots and assert that

* history tokens are *appended* to the token list, never interleaved;
* the encoder caps them at ``config.max_history_tokens``;
* a history step carries no hidden information (opponent hand identities in its
  ``pre_state`` must not reach any feature column);
* ``MatchHistory.record`` (training-collection path) and a hand-built
  ``TrajectoryStep`` tuple (inference path) encode identically.
"""

from __future__ import annotations

from types import SimpleNamespace

from reps.history import MatchHistory
from reps.observation_encoder import (
    DEFINITION_SCALE,
    _TOKEN_TYPES,
    TokenObservationEncoder,
    definition_feature_indices,
)
from reps.schema import (
    ActionChoice,
    CharacterSnapshot,
    DecisionContext,
    DecisionType,
    EntitySnapshot,
    LowLevelActionSpec,
    OptionKind,
    PlayerSnapshot,
    StateSnapshot,
    TrajectoryStep,
)

_HISTORY_TYPE_INDEX = _TOKEN_TYPES.index("history")
_OWNER_OFFSET = len(_TOKEN_TYPES)
_OPPONENT_OWNER_INDEX = _OWNER_OFFSET + 1  # _OWNER_TYPES == ("self", "opponent", "neutral")
_DEFINITION_TOKEN_INDEX, _DEFINITION_OPTION_INDEX = definition_feature_indices()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _character(slot: int, definition_id: int, *, active: bool) -> CharacterSnapshot:
    return CharacterSnapshot(
        id=100 + slot,
        definition_id=definition_id,
        health=5,
        max_health=5,
        energy=0,
        max_energy=2,
        defeated=False,
        is_active=active,
    )


def _player(
    *,
    active_definition_id: int,
    hand_ids: tuple[int, ...] = (),
    dice: tuple[int, ...] = (),
) -> PlayerSnapshot:
    characters = (
        _character(0, active_definition_id, active=True),
        _character(1, active_definition_id + 1, active=False),
        _character(2, active_definition_id + 2, active=False),
    )
    hand = tuple(
        EntitySnapshot(id=1000 + offset, definition_id=int(card_id))
        for offset, card_id in enumerate(hand_ids)
    )
    return PlayerSnapshot(
        active_character_id=characters[0].id,
        characters=characters,
        combat_statuses=(),
        summons=(),
        supports=(),
        hand_cards=hand,
        pile_cards=(),
        dice=dice,
        declared_end=False,
        legend_used=False,
    )


def _state(
    *,
    round_number: int = 1,
    opponent_hand_ids: tuple[int, ...] = (),
    self_hand_ids: tuple[int, ...] = (),
) -> StateSnapshot:
    return StateSnapshot(
        phase="action",
        round_number=round_number,
        current_turn=0,
        winner=None,
        players=(
            _player(active_definition_id=200, hand_ids=self_hand_ids),
            _player(active_definition_id=300, hand_ids=opponent_hand_ids),
        ),
    )


def _spec(
    code: int,
    kind: OptionKind = OptionKind.ACTION_USE_SKILL,
    *,
    target_slots: int = 0,
    used_dice: tuple[int, ...] = (),
) -> LowLevelActionSpec:
    from reps.schema import PublicSlotTarget

    return LowLevelActionSpec(
        action_code=int(code),
        request_type=DecisionType.ACTION,
        kind=kind,
        label=f"spec_{code}",
        subject_definition_id=200 + int(code),
        target_slots=tuple(
            PublicSlotTarget(owner="self", zone="character", index=index)
            for index in range(target_slots)
        ),
        used_dice=tuple(int(value) for value in used_dice),
    )


def _context(
    *,
    acting_player: int = 0,
    request_type: DecisionType = DecisionType.ACTION,
    step_index: int = 0,
    specs: tuple[LowLevelActionSpec, ...] = (),
    state: StateSnapshot | None = None,
) -> DecisionContext:
    snapshot = state if state is not None else _state()
    return DecisionContext(
        acting_player=int(acting_player),
        request_type=request_type,
        step_index=int(step_index),
        full_state=snapshot,
        legal_low_level_codes=tuple(spec.action_code for spec in specs),
        legal_low_level_specs=tuple(specs),
        legal_low_level_mask=tuple(True for _ in specs),
        player_view=snapshot,
    )


def _step(
    *,
    acting_player: int = 0,
    round_number: int = 1,
    kind: OptionKind = OptionKind.ACTION_USE_SKILL,
    code: int = 0,
    reward: float = 0.0,
    done: bool = False,
    spec: LowLevelActionSpec | None = None,
    pre_state: StateSnapshot | None = None,
    request_type: DecisionType = DecisionType.ACTION,
    legal_codes: tuple[int, ...] = (0, 1, 2),
) -> TrajectoryStep:
    snapshot = pre_state if pre_state is not None else _state(round_number=round_number)
    chosen = spec if spec is not None else _spec(code, kind, target_slots=1)
    return TrajectoryStep(
        acting_player=int(acting_player),
        request_type=request_type,
        choice=ActionChoice(action_code=int(code)),
        pre_state=snapshot,
        post_state=snapshot,
        reward=float(reward),
        done=bool(done),
        legal_low_level_codes=tuple(int(value) for value in legal_codes),
        chosen_low_level_spec=chosen,
        player_view=snapshot,
    )


def _is_history_token(token: tuple[float, ...]) -> bool:
    return bool(token[_HISTORY_TYPE_INDEX])


def _history_token_count(observation) -> int:
    return sum(1 for token in observation.token_features if _is_history_token(token))


# --------------------------------------------------------------------------- #
# 1. tokens appear (appended, not interleaved)
# --------------------------------------------------------------------------- #
def test_history_tokens_are_appended():
    encoder = TokenObservationEncoder()
    context = _context()

    without = encoder.encode_context(context, history=())
    history = (
        _step(acting_player=0, round_number=1, code=1),
        _step(
            acting_player=1,
            round_number=1,
            kind=OptionKind.ACTION_DECLARE_END,
            code=2,
        ),
        _step(
            acting_player=0,
            round_number=2,
            kind=OptionKind.ACTION_PLAY_CARD,
            code=0,
            reward=0.25,
            done=False,
        ),
    )
    with_history = encoder.encode_context(context, history=history)

    assert len(with_history.token_features) == len(without.token_features) + 3
    assert _history_token_count(with_history) == 3
    # History is appended at the end, so the base tokens are byte-identical.
    assert with_history.token_features[: len(without.token_features)] == (
        without.token_features
    )
    assert with_history.option_features == without.option_features
    assert with_history.token_mask == tuple(True for _ in with_history.token_features)

    appended = with_history.token_features[len(without.token_features) :]
    assert [_is_history_token(token) for token in appended] == [True, True, True]
    owners = [(token[_OWNER_OFFSET], token[_OWNER_OFFSET + 1]) for token in appended]
    assert owners == [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]


# --------------------------------------------------------------------------- #
# 2. truncation to config.max_history_tokens
# --------------------------------------------------------------------------- #
def test_history_is_truncated_to_config_limit():
    encoder = TokenObservationEncoder()
    context = _context()
    history = tuple(
        _step(acting_player=index % 2, round_number=1 + index // 4, code=index % 3)
        for index in range(40)
    )

    observation = encoder.encode_context(context, history=history)

    assert encoder.config.max_history_tokens == 32
    assert _history_token_count(observation) == 32


# --------------------------------------------------------------------------- #
# 3. no leak: opponent hand identities in a step's pre_state never encode
# --------------------------------------------------------------------------- #
def test_history_step_does_not_leak_opponent_hand_identities():
    hidden_ids = (333029, 333008)
    encoder = TokenObservationEncoder()
    # The acting player's masked view: opponent hand cards present but masked.
    context = _context(state=_state(opponent_hand_ids=(0, 0)))
    leaked_step = _step(
        acting_player=1,  # opponent-owned history token
        round_number=1,
        code=0,
        pre_state=_state(opponent_hand_ids=hidden_ids),
    )

    observation = encoder.encode_context(context, history=(leaked_step,))
    history_tokens = [token for token in observation.token_features if _is_history_token(token)]
    assert len(history_tokens) == 1
    leaked_token = history_tokens[0]
    assert leaked_token[_OPPONENT_OWNER_INDEX] == 1.0
    # The history token carries no entity definition id at all.
    assert leaked_token[_DEFINITION_TOKEN_INDEX] == 0.0

    token_definition_ids = {
        int(round(token[_DEFINITION_TOKEN_INDEX] * DEFINITION_SCALE))
        for token in observation.token_features
    }
    option_definition_ids = {
        int(round(row[_DEFINITION_OPTION_INDEX] * DEFINITION_SCALE))
        for row in observation.option_features
    }
    assert not (token_definition_ids & set(hidden_ids))
    assert not (option_definition_ids & set(hidden_ids))


# --------------------------------------------------------------------------- #
# 4. train/inference parity
# --------------------------------------------------------------------------- #
def test_match_history_record_matches_explicit_trajectory_steps():
    encoder = TokenObservationEncoder()
    specs = (
        _spec(5, OptionKind.ACTION_USE_SKILL, target_slots=1, used_dice=(1, 2)),
        _spec(7, OptionKind.ACTION_DECLARE_END),
    )
    first_context = _context(step_index=1, specs=specs, state=_state(round_number=1))
    second_context = _context(step_index=2, specs=specs, state=_state(round_number=2))

    history = MatchHistory()
    history.record(SimpleNamespace(context=first_context), 5)
    history.record(SimpleNamespace(context=second_context), 7)
    assert len(history) == 2

    expected = (
        TrajectoryStep(
            acting_player=first_context.acting_player,
            request_type=first_context.request_type,
            choice=ActionChoice(action_code=5),
            pre_state=first_context.player_view,
            post_state=first_context.player_view,
            reward=0.0,
            done=False,
            legal_low_level_codes=first_context.legal_low_level_codes,
            chosen_low_level_spec=specs[0],
            player_view=first_context.player_view,
        ),
        TrajectoryStep(
            acting_player=second_context.acting_player,
            request_type=second_context.request_type,
            choice=ActionChoice(action_code=7),
            pre_state=second_context.player_view,
            post_state=second_context.player_view,
            reward=0.0,
            done=False,
            legal_low_level_codes=second_context.legal_low_level_codes,
            chosen_low_level_spec=specs[1],
            player_view=second_context.player_view,
        ),
    )
    assert history.snapshot() == expected

    current_context = _context(step_index=3, specs=specs, state=_state(round_number=3))
    recorded = encoder.encode_context(current_context, history=history.snapshot())
    explicit = encoder.encode_context(current_context, history=expected)
    assert recorded.token_features == explicit.token_features
    assert _history_token_count(recorded) == 2
