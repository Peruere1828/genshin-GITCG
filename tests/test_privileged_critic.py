"""D30: privileged (god's-eye) state feeds the critic, never the actor.

The contract under test:

* actor logits are bit-identical whether or not privileged state is supplied
  (they are a function of the public observation alone);
* the value head is *sensitive* to privileged state, and degrades to a learned
  null embedding when the state is unavailable;
* privileged dropout (train time) can force the null embedding, which is what
  calibrates it for inference;
* every pre-D30 checkpoint still loads bit-compatibly (no new weights);
* ``Store.priv`` / ``_collate_privileged`` round-trip, including "absent";
* a public-only checkpoint can warm-start a privileged model via
  ``train.model.load_state_dict_compatible`` (the trainer's ``--init-from`` path).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from envs.observation import default_encoder
from scripts.train_bc_scale import Store, _collate_privileged
from train.model import (
    PRIVILEGED_PARAM_PREFIXES,
    CVPN,
    CVPNConfig,
    TransformerCVPN,
    TransformerCVPNConfig,
    config_for_encoder,
    load_cvpn_model,
    load_state_dict_compatible,
)

OLD_CHECKPOINT = Path("data/checkpoints/rl_dgeo3_double_geo_navia.pt")

# The exact parameter set of a pre-D30 public-only CVPN built from the default
# encoder (no belief head, no identity branch, no privileged branch).
PRE_D30_KEYS = {
    "token_proj.0.weight", "token_proj.0.bias",
    "token_proj.1.weight", "token_proj.1.bias",
    "option_proj.0.weight", "option_proj.0.bias",
    "option_scorer.0.weight", "option_scorer.0.bias",
    "option_scorer.3.weight", "option_scorer.3.bias",
    "value_head.0.weight", "value_head.0.bias",
    "value_head.2.weight", "value_head.2.bias",
}


def _batch(b: int = 3, t: int = 4, k: int = 5, token_dim: int = 6, option_dim: int = 4):
    torch.manual_seed(1234)
    tokens = torch.randn(b, t, token_dim)
    token_mask = torch.ones(b, t, dtype=torch.bool)
    token_mask[0, t - 1] = False
    options = torch.randn(b, k, option_dim)
    option_mask = torch.ones(b, k, dtype=torch.bool)
    option_mask[1, k - 1] = False
    return tokens, token_mask, options, option_mask


def _cvpn(**over) -> CVPN:
    base = dict(token_dim=6, option_dim=4, d_model=16)
    base.update(over)
    return CVPN(CVPNConfig(**base))


def _transformer(**over) -> TransformerCVPN:
    base = dict(
        token_dim=6, option_dim=4, d_model=16, n_layers=1, n_heads=4, dim_feedforward=32
    )
    base.update(over)
    return TransformerCVPN(TransformerCVPNConfig(**base))


def _store(priv: list[torch.Tensor]) -> Store:
    n = 3
    return Store(
        tokens=[torch.randn(2, 6) for _ in range(n)],
        options=[torch.randn(4, 4) for _ in range(n)],
        teacher=[0, 1, 2],
        value=[1.0, 0.0, 1.0],
        b_hand=[torch.empty(0) for _ in range(n)],
        b_deck=[torch.empty(0) for _ in range(n)],
        b_hist=[torch.empty(0) for _ in range(n)],
        b_hsize=[0.0] * n,
        b_burst=[0.0] * n,
        b_mask=[False] * n,
        keys=[("d", "o", i, i) for i in range(n)],
        priv=priv,
    )


# --------------------------------------------------------------------------- actor


@pytest.mark.parametrize("model_factory", [_cvpn, _transformer])
def test_actor_logits_ignore_privileged_state(model_factory):
    torch.manual_seed(7)
    model = model_factory(privileged_dim=8).eval()
    tokens, token_mask, options, option_mask = _batch()
    privileged = torch.randn(tokens.size(0), 8)
    with torch.no_grad():
        logits_none, value_none, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
        logits_priv, value_priv, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=privileged
        )
    # Bitwise, not allclose: privileged state must not touch the actor at all.
    assert torch.equal(logits_none, logits_priv)
    assert torch.isfinite(value_none).all() and torch.isfinite(value_priv).all()


def test_actor_logits_unchanged_by_privileged_branch_existence():
    """Same public weights -> identical logits with and without the branch."""
    torch.manual_seed(11)
    public = _cvpn().eval()
    privileged = _cvpn(privileged_dim=8).eval()
    # The critic is additive, so every non-privileged key transfers 1:1.
    shared = {
        key: value
        for key, value in public.state_dict().items()
        if not key.startswith(PRIVILEGED_PARAM_PREFIXES)
    }
    privileged.load_state_dict(shared, strict=False)
    tokens, token_mask, options, option_mask = _batch()
    with torch.no_grad():
        logits_public, value_public = public(tokens, token_mask, options, option_mask)
        logits_priv, value_priv, _ = privileged.forward_all(
            tokens, token_mask, options, option_mask, privileged=torch.randn(3, 8)
        )
    assert torch.equal(logits_public, logits_priv)
    # Additive conditioning + zero priv_null: the unprivileged critic is exact.
    with torch.no_grad():
        _, value_null, _ = privileged.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
    assert torch.equal(value_public, value_null)
    assert not torch.equal(value_public, value_priv)


# --------------------------------------------------------------------------- value


def test_value_depends_on_privileged_state_and_none_is_finite():
    torch.manual_seed(3)
    model = _cvpn(privileged_dim=8).eval()
    tokens, token_mask, options, option_mask = _batch()
    a = torch.randn(3, 8)
    b = torch.randn(3, 8)
    with torch.no_grad():
        _, value_none, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
        _, value_a, _ = model.forward_all(tokens, token_mask, options, option_mask, privileged=a)
        _, value_b, _ = model.forward_all(tokens, token_mask, options, option_mask, privileged=b)
    assert torch.isfinite(value_none).all()
    assert not torch.equal(value_a, value_b)
    assert not torch.equal(value_a, value_none)


def test_privileged_dropout_one_equals_the_null_embedding_path():
    torch.manual_seed(5)
    model = _cvpn(privileged_dim=8, privileged_dropout=1.0)
    model.train()
    tokens, token_mask, options, option_mask = _batch()
    privileged = torch.randn(3, 8)
    with torch.no_grad():
        _, value_null, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
        _, value_dropped, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=privileged
        )
    assert torch.equal(value_null, value_dropped)


def test_privileged_argument_on_public_model_is_rejected():
    model = _cvpn().eval()  # privileged_dim == 0
    tokens, token_mask, options, option_mask = _batch()
    with pytest.raises(ValueError):
        model.forward_all(
            tokens, token_mask, options, option_mask, privileged=torch.zeros(3, 8)
        )


# --------------------------------------------------------------- checkpoint compat


def test_default_config_keeps_the_pre_d30_parameter_set():
    encoder = default_encoder()
    model = CVPN(config_for_encoder(encoder))
    assert model.config.privileged_dim == 0
    assert set(model.state_dict()) == PRE_D30_KEYS
    assert not any(k.startswith(PRIVILEGED_PARAM_PREFIXES) for k in model.state_dict())


def test_default_forward_is_unchanged_by_the_optional_argument():
    torch.manual_seed(9)
    model = _cvpn().eval()
    tokens, token_mask, options, option_mask = _batch()
    with torch.no_grad():
        logits_a, value_a = model(tokens, token_mask, options, option_mask)
        logits_b, value_b, _ = model.forward_all(tokens, token_mask, options, option_mask)
    assert torch.equal(logits_a, logits_b) and torch.equal(value_a, value_b)


@pytest.mark.skipif(not OLD_CHECKPOINT.exists(), reason="pre-D30 checkpoint not present")
def test_existing_checkpoint_still_loads_and_forwards():
    model = CVPN.load(str(OLD_CHECKPOINT))  # old config has no privileged_dim key
    assert isinstance(model, CVPN) and not model.has_privileged
    assert model.config.privileged_dim == 0
    tokens = torch.randn(1, 3, model.config.token_dim)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    options = torch.randn(1, 2, model.config.option_dim)
    option_mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        logits, value = model(tokens, token_mask, options, option_mask)
    assert logits.shape == (1, 2) and value.shape == (1,)
    assert isinstance(load_cvpn_model(str(OLD_CHECKPOINT)), CVPN)


# ---------------------------------------------------------------- store round-trip


def test_collate_privileged_stacks_and_reports_absence():
    priv = [torch.randn(8) for _ in range(3)]
    batch = _collate_privileged(_store(priv), [0, 1, 2], "cpu")
    assert batch is not None
    assert batch.shape == (3, 8) and batch.dtype == torch.float32
    assert torch.allclose(batch[1], priv[1])

    assert _collate_privileged(_store([]), [0, 1, 2], "cpu") is None
    assert _collate_privileged(_store([torch.empty(0) for _ in range(3)]), [0, 1, 2], "cpu") is None


def test_store_reorder_supports_privileged_vectors():
    from scripts.train_bc_scale import _reorder

    priv = [torch.full((4,), float(i)) for i in range(3)]
    store = _store(priv)
    _reorder(store, [2, 0, 1])
    assert [float(v[0]) for v in store.priv] == [2.0, 0.0, 1.0]
    assert store.n_privileged() == 3


# ------------------------------------------------------------------- init-from path


def test_init_from_public_checkpoint_into_privileged_model(tmp_path):
    """``--init-from`` target: public weights transfer, value stays bitwise exact."""
    torch.manual_seed(0)
    source = _cvpn().eval()
    target = _cvpn(privileged_dim=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        missing = load_state_dict_compatible(target, source.state_dict())
    # No surgery, no warnings: only the new privileged branch is "missing".
    assert not caught, [str(w.message) for w in caught]
    assert set(missing) == {"priv_null"} | {
        k for k in target.state_dict() if k.startswith("priv_proj.")
    }
    # (a) the null embedding is exactly zero ...
    assert torch.equal(target.priv_null, torch.zeros_like(target.priv_null))
    # ... and every shared tensor (trunk, actor, value head) transferred exactly.
    for key in PRE_D30_KEYS:
        assert torch.equal(target.state_dict()[key], source.state_dict()[key]), key

    tokens, token_mask, options, option_mask = _batch()
    target.eval()
    with torch.no_grad():
        _, value_target, _ = target.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
        _, value_source, _ = source.forward_all(tokens, token_mask, options, option_mask)
    # (b) ``value_head(g + priv_null)`` with ``priv_null == 0`` is the source value,
    # bitwise -- the A1/A2 comparison has no "value function was reset" confound.
    assert torch.equal(value_target, value_source)


def test_null_path_matches_a_zeroed_null_embedding():
    """Dropping every row equals the None path (and a public clone's value)."""
    torch.manual_seed(21)
    public = _cvpn().eval()
    model = _cvpn(privileged_dim=8, privileged_dropout=1.0)
    model.load_state_dict(
        {
            key: value
            for key, value in public.state_dict().items()
            if not key.startswith(PRIVILEGED_PARAM_PREFIXES)
        },
        strict=False,
    )
    model.train()  # dropout is only active in train mode
    tokens, token_mask, options, option_mask = _batch()
    with torch.no_grad():
        _, value_public = public(tokens, token_mask, options, option_mask)
        _, value_none, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=None
        )
        _, value_dropped, _ = model.forward_all(
            tokens, token_mask, options, option_mask, privileged=torch.randn(3, 8)
        )
    assert torch.equal(value_none, value_dropped)
    assert torch.equal(value_none, value_public)


def test_compatible_load_still_rejects_other_gaps():
    source = _cvpn().eval()
    target = _cvpn(privileged_dim=5)
    state = dict(source.state_dict())
    state.pop("token_proj.0.weight")
    with pytest.raises(RuntimeError, match="token_proj.0.weight"):
        load_state_dict_compatible(target, state)
