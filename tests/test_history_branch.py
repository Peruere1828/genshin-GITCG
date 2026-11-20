"""History gets its own pooled branch, added through a zero-initialised projection.

Measured 2026-11-20: appending the encoder's 32 ``history`` token rows to the
tokens and feeding them into the masked mean-pool moved an untouched teacher's
pooled vector off-distribution, collapsing it (val_acc 0.8710 -> 0.3680,
val_ce 0.3149 -> 28.9867). The fix, under test here:

* history rows are pooled separately (never in the public pool);
* ``g = g_public + hist_proj(h)`` with ``hist_proj`` zeroed (weight *and* bias);
* at init that is bitwise the public-only model -- including on history-bearing
  inputs -- so ``--init-from`` a historyless teacher is lossless;
* an input with no history rows yields ``h == 0`` -> ``hist_proj(0) == 0``, so it
  stays identical to the public-only path forever;
* the branch is learnable: a non-zero ``hist_proj`` changes history-bearing
  outputs and leaves historyless ones untouched.
"""

from __future__ import annotations

import pytest
import torch

from envs.observation import default_encoder
from reps.observation_encoder import _TOKEN_TYPES
from train.model import (
    HISTORY_PARAM_PREFIXES,
    CVPN,
    CVPNConfig,
    TransformerCVPN,
    TransformerCVPNConfig,
    config_for_encoder,
    history_token_mask,
    load_state_dict_compatible,
    masked_mean,
    transformer_config_for_encoder,
)

_HISTORY_INDEX = _TOKEN_TYPES.index("history")
_TOKEN_DIM = 12  # >= len(_TOKEN_TYPES) so the token-type one-hot is complete
_OPTION_DIM = 5
_B, _T, _K = 2, 8, 4


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _make_tokens(
    history_rows: list[list[int]],
    pad_rows: list[list[int]],
    *,
    seed: int = 3,
):
    """``(tokens, token_mask)`` with explicit token-type one-hots.

    ``history_rows`` may include padded columns on purpose (the mask test feeds a
    history one-hot on a padded row to prove ``token_mask`` wins).
    """
    torch.manual_seed(seed)
    tokens = torch.zeros(_B, _T, _TOKEN_DIM)
    # Default type = global (index 0), random continuous tail so projections differ.
    tokens[..., 0] = 1.0
    tokens[..., len(_TOKEN_TYPES):] = torch.randn(_B, _T, _TOKEN_DIM - len(_TOKEN_TYPES))
    for row, columns in enumerate(history_rows):
        for column in columns:
            tokens[row, column, : len(_TOKEN_TYPES)] = 0.0
            tokens[row, column, _HISTORY_INDEX] = 1.0
    token_mask = torch.ones(_B, _T, dtype=torch.bool)
    for row, columns in enumerate(pad_rows):
        for column in columns:
            token_mask[row, column] = False
    return tokens, token_mask


def _options(seed: int = 5):
    torch.manual_seed(seed)
    options = torch.randn(_B, _K, _OPTION_DIM)
    option_mask = torch.ones(_B, _K, dtype=torch.bool)
    option_mask[0, _K - 1] = False
    return options, option_mask


def _historyless_batch():
    tokens, token_mask = _make_tokens(
        history_rows=[[], []],
        pad_rows=[[6, 7], [7]],
    )
    return tokens, token_mask


def _history_batch():
    # History appended at the tail of the real part, padding strictly after it.
    tokens, token_mask = _make_tokens(
        history_rows=[[4, 5], [3]],
        pad_rows=[[6, 7], [4, 5, 6, 7]],
    )
    return tokens, token_mask


def _cvpn(history: bool) -> CVPN:
    return CVPN(
        CVPNConfig(
            token_dim=_TOKEN_DIM,
            option_dim=_OPTION_DIM,
            d_model=16,
            history_branch=history,
        )
    ).eval()


def _transformer(history: bool) -> TransformerCVPN:
    return TransformerCVPN(
        TransformerCVPNConfig(
            token_dim=_TOKEN_DIM,
            option_dim=_OPTION_DIM,
            d_model=16,
            n_layers=1,
            n_heads=4,
            dim_feedforward=32,
            history_branch=history,
        )
    ).eval()


# factory, whether the *logits* (not only the value) consume ``g``
_FACTORIES = [
    pytest.param(_cvpn, True, id="cvpn"),
    pytest.param(_transformer, False, id="transformer"),
]


def _forward(model, tokens, token_mask, options, option_mask):
    with torch.no_grad():
        logits, value = model(tokens, token_mask, options, option_mask)
    return logits, value


# --------------------------------------------------------------------------- #
# 1. bitwise init identity (both input kinds)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory,_logits_use_g", _FACTORIES)
def test_init_branch_is_bitwise_identity(factory, _logits_use_g):
    torch.manual_seed(0)
    public = factory(False)
    torch.manual_seed(0)
    history = factory(True)

    missing = load_state_dict_compatible(history, public.state_dict())
    expected_missing = {
        key for key in history.state_dict() if key.startswith(HISTORY_PARAM_PREFIXES)
    }
    assert set(missing) == expected_missing
    assert expected_missing  # the branch really is new
    # Zero weight *and* bias, so ``hist_proj(h) == 0`` for every ``h``.
    assert torch.equal(history.hist_proj.weight, torch.zeros_like(history.hist_proj.weight))
    assert torch.equal(history.hist_proj.bias, torch.zeros_like(history.hist_proj.bias))

    options, option_mask = _options()
    for name, batch in (("historyless", _historyless_batch()), ("history", _history_batch())):
        tokens, token_mask = batch
        logits_public, value_public = _forward(public, tokens, token_mask, options, option_mask)
        logits_hist, value_hist = _forward(history, tokens, token_mask, options, option_mask)
        # Bitwise, not allclose: no history row may perturb the public path at init.
        assert torch.equal(logits_public, logits_hist), name
        assert torch.equal(value_public, value_hist), name


# --------------------------------------------------------------------------- #
# 2. the branch is learnable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory,logits_use_g", _FACTORIES)
def test_branch_can_learn_from_history(factory, logits_use_g):
    model = factory(True).eval()
    options, option_mask = _options()

    tokens_no, mask_no = _historyless_batch()
    tokens_hist, mask_hist = _history_batch()

    logits_no_before, value_no_before = _forward(model, tokens_no, mask_no, options, option_mask)
    logits_hist_before, value_hist_before = _forward(
        model, tokens_hist, mask_hist, options, option_mask
    )

    with torch.no_grad():
        model.hist_proj.weight.fill_(0.05)  # bias stays zero

    logits_no_after, value_no_after = _forward(model, tokens_no, mask_no, options, option_mask)
    logits_hist_after, value_hist_after = _forward(
        model, tokens_hist, mask_hist, options, option_mask
    )

    # No history rows -> h == 0 -> hist_proj(0) == 0: nothing may move.
    assert torch.equal(logits_no_before, logits_no_after)
    assert torch.equal(value_no_before, value_no_after)

    # History rows -> h != 0 -> the branch moves ``g`` (and therefore the value).
    assert not torch.equal(value_hist_before, value_hist_after)
    if logits_use_g:
        # The mean-pool CVPN scores options from ``g``, so logits must move too.
        assert not torch.equal(logits_hist_before, logits_hist_after)
    else:
        # TransformerCVPN scores options by cross-attending over the tokens, not
        # from ``pooled``; history reaches the actor through that attention route
        # and ``hist_proj`` only feeds value/belief.
        assert torch.equal(logits_hist_before, logits_hist_after)


# --------------------------------------------------------------------------- #
# 3. mask correctness: exactly the history rows, never padded ones
# --------------------------------------------------------------------------- #
def test_history_token_mask_selects_history_rows_only():
    # Row 0: history at 3,4; column 5 carries a history one-hot *but is padded*.
    # Row 1: history at 2; columns 3..7 are padding.
    tokens, token_mask = _make_tokens(
        history_rows=[[3, 4, 5], [2]],
        pad_rows=[[5, 6, 7], [3, 4, 5, 6, 7]],
    )

    mask = history_token_mask(tokens, token_mask)

    expected = torch.zeros(_B, _T, dtype=torch.bool)
    expected[0, 3] = expected[0, 4] = True
    expected[1, 2] = True
    assert torch.equal(mask, expected)
    # The padded history one-hot at [0, 5] is not selected, and no padded row is.
    assert not bool(mask[0, 5])
    assert not bool((mask & ~token_mask).any())


def test_masked_mean_empty_mask_is_exactly_zero():
    values = torch.randn(_B, _T, 4)
    empty = torch.zeros(_B, _T, dtype=torch.bool)
    assert torch.equal(masked_mean(values, empty), torch.zeros(_B, 4))


# --------------------------------------------------------------------------- #
# 4. config wiring (off by default)
# --------------------------------------------------------------------------- #
def test_config_wires_history_flag_and_defaults_off():
    encoder = default_encoder()
    assert config_for_encoder(encoder).history_branch is False
    assert config_for_encoder(encoder, history=True).history_branch is True
    assert transformer_config_for_encoder(encoder).history_branch is False
    assert transformer_config_for_encoder(encoder, history=True).history_branch is True
