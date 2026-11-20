"""Producer side of the history-channel marker (checkpoint ↔ data compatibility).

``NeuralPolicy`` only attaches a ``MatchHistory`` when the checkpoint's
``encoder_config["include_history"]`` says the model was trained with the history
token channel. These tests pin the producer side:

* ``store_has_history`` reads the token rows (token-type argmax) rather than a
  manifest, so a corpus collected without ``--history`` is detected as
  history-free even when its metadata says nothing;
* every checkpoint written by the trainer carries the marker;
* the data-first trainer refuses a store/checkpoint history mismatch unless the
  operator explicitly passes ``--allow-history-mismatch``.
"""

from __future__ import annotations

import json

import pytest
import torch

from envs.observation import default_encoder
from reps.observation_encoder import _TOKEN_TYPES, definition_feature_indices
from scripts.datafirst_bc_train import main as datafirst_main
from scripts.train_bc_scale import (
    Config,
    Store,
    store_has_history,
    store_history_flag,
    train,
    with_history_flag,
)
from train.model import CVPN, CVPNConfig, config_for_encoder

_HISTORY_INDEX = _TOKEN_TYPES.index("history")
_N_TYPES = len(_TOKEN_TYPES)
TOKEN_DIM = 65      # default encoder token width
OPTION_DIM = 68     # default encoder option width


def _token_rows(*types: int) -> torch.Tensor:
    """A ``(len(types), TOKEN_DIM)`` token tensor with the given token types."""
    rows = torch.zeros(len(types), TOKEN_DIM)
    for i, kind in enumerate(types):
        rows[i, kind] = 1.0
    return rows


def _store(tokens: list[torch.Tensor], priv: list[torch.Tensor] | None = None) -> Store:
    n = len(tokens)
    return Store(
        tokens=tokens,
        options=[torch.randn(2, OPTION_DIM) for _ in range(n)],
        teacher=[i % 2 for i in range(n)],
        value=[float(i % 2) for i in range(n)],
        b_hand=[torch.empty(0) for _ in range(n)],
        b_deck=[torch.empty(0) for _ in range(n)],
        b_hist=[torch.empty(0) for _ in range(n)],
        b_hsize=[0.0] * n,
        b_burst=[0.0] * n,
        b_mask=[False] * n,
        keys=[("d", "o", i, i) for i in range(n)],
        priv=priv or [],
    )


# ------------------------------------------------------------------- detection


def test_store_has_history_from_token_rows():
    with_history = _store([
        _token_rows(0, 1, 2),
        _token_rows(0, _HISTORY_INDEX),      # one history token is enough
        _token_rows(0),
    ])
    without_history = _store([_token_rows(0, 1, 2), _token_rows(0), _token_rows(0)])

    assert store_has_history(with_history) is True
    assert store_has_history(without_history) is False
    # An empty store is history-free, not an error.
    assert store_has_history(_store([])) is False


def test_history_detection_ignores_scalar_columns():
    """A history-like scalar in a non-type column must not be mistaken for one."""
    rows = torch.zeros(2, TOKEN_DIM)
    rows[0, 0] = 1.0
    rows[1, _N_TYPES + 2] = 1.0   # owner/phase column, not a token type
    assert store_has_history(_store([rows])) is False


def test_store_history_flag_prefers_the_recorded_value():
    history_tokens = _store([_token_rows(0, _HISTORY_INDEX)])
    history_tokens.has_history = True
    assert store_history_flag(history_tokens) is True

    # A recorded False is re-checked against the tokens: a default-constructed
    # store (or an old cache) must not be able to claim "no history" falsely.
    stale = _store([_token_rows(0, _HISTORY_INDEX)])
    stale.has_history = False
    assert store_history_flag(stale) is True
    # Legacy pickles have no field at all -> falls back to scanning too.
    stale.__dict__.pop("has_history")
    assert store_history_flag(stale) is True

    assert store_history_flag(_store([_token_rows(0, 1, 2)])) is False


def test_subset_carries_the_history_marker():
    store = _store([_token_rows(0, _HISTORY_INDEX) for _ in range(3)])
    store.has_history = True
    assert store.subset([0, 2]).has_history is True


# ----------------------------------------------------------- checkpoint stamping


def test_train_writes_include_history_into_encoder_config(tmp_path):
    store = _store([_token_rows(0, _HISTORY_INDEX) for _ in range(4)])
    model = CVPN(CVPNConfig(token_dim=TOKEN_DIM, option_dim=OPTION_DIM, d_model=16))
    config = Config(
        epochs=1, batch_size=2, value_coef=0.0,
        save_best=True, best_path=str(tmp_path / "best.pt"),
    )
    train(store, store, model, config, False, encoder_config={"k": 1})

    stamp = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert stamp["encoder_config"]["include_history"] is True
    assert stamp["encoder_config"]["k"] == 1  # existing keys preserved


def test_train_stamps_a_minimal_encoder_config_when_none(tmp_path):
    store = _store([_token_rows(0, 1, 2) for _ in range(4)])  # history-free
    model = CVPN(CVPNConfig(token_dim=TOKEN_DIM, option_dim=OPTION_DIM, d_model=16))
    config = Config(
        epochs=1, batch_size=2, value_coef=0.0,
        save_best=True, best_path=str(tmp_path / "best.pt"),
    )
    train(store, store, model, config, False, encoder_config=None)

    stamp = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert stamp["encoder_config"] == {"include_history": False}


def test_with_history_flag_never_mutates_the_input():
    source = {"card_vocabulary": (1, 2)}
    out = with_history_flag(source, True)
    assert out == {"card_vocabulary": (1, 2), "include_history": True}
    assert "include_history" not in source


# ------------------------------------------------------------- trainer refusal


def _write_cache(path, store: Store) -> None:
    meta = {
        "events": len(store),
        "definition_ids_present": len(store),
        "definition_indices": [int(v) for v in definition_feature_indices()],
        "encoder_config": default_encoder().to_dict(),
    }
    torch.save({"train": store, "val": store, "meta": meta}, path)


def test_trainer_refuses_history_mismatch(tmp_path):
    store = _store([_token_rows(0, 1, 2) for _ in range(3)])  # history-free
    cache = tmp_path / "store.pt"
    _write_cache(cache, store)

    encoder = default_encoder()
    checkpoint = tmp_path / "expects_history.pt"
    CVPN(config_for_encoder(encoder, d_model=16)).save(
        str(checkpoint), encoder_config={"include_history": True}
    )

    with pytest.raises(SystemExit) as excinfo:
        datafirst_main([
            "--store-cache", str(cache),
            "--init-from", str(checkpoint),
            "--d-model", "16", "--epochs", "1", "--batch-size", "2",
            "--out", str(tmp_path / "out.pt"),
        ])
    message = str(excinfo.value)
    assert "history=True" in message and "history=False" in message
    assert "--allow-history-mismatch" in message


def test_trainer_allow_history_mismatch_escape_hatch(tmp_path):
    store = _store([_token_rows(0, 1, 2) for _ in range(3)])
    cache = tmp_path / "store.pt"
    _write_cache(cache, store)

    checkpoint = tmp_path / "expects_history.pt"
    CVPN(config_for_encoder(default_encoder(), d_model=16)).save(
        str(checkpoint), encoder_config={"include_history": True}
    )
    out = tmp_path / "out.pt"
    report = tmp_path / "report.json"
    code = datafirst_main([
        "--store-cache", str(cache),
        "--init-from", str(checkpoint),
        "--allow-history-mismatch",
        "--d-model", "16", "--epochs", "1", "--batch-size", "2",
        "--out", str(out), "--report", str(report),
    ])
    assert code == 0
    assert json.loads(report.read_text())["has_history"] is False
    stamp = torch.load(out, map_location="cpu", weights_only=False)
    # The store is the source of truth: the mismatched checkpoint does not get to
    # relabel the data's input format.
    assert stamp["encoder_config"]["include_history"] is False
