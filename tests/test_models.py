"""Focused model regression tests."""

from __future__ import annotations

import pytest
import torch

import nonlinearrnnscanbeparallel.models  # noqa: F401
from nonlinearrnnscanbeparallel.data.datamodule import SortedGraphConnectivityDataset
from nonlinearrnnscanbeparallel.models.registry import get_model, list_models
from nonlinearrnnscanbeparallel.models.rkan import RKANLayer


def test_models_registered() -> None:
    names = list_models()
    assert "mlp_rnn" in names
    assert "rkan_rnn" in names
    assert "m2rnn" in names
    assert "min_gru" in names
    assert "min_lstm" in names


def test_mlp_rnn_forward_backward() -> None:
    model = get_model(
        "mlp_rnn",
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 10, 64)
    state = model.init_state(2, x.device)
    logits, new_state = model(x, state)
    assert logits.shape == (2, 10, 2)
    assert len(new_state) == 2
    loss = logits.sum()
    loss.backward()


def test_rkan_rnn_forward_backward() -> None:
    model = get_model(
        "rkan_rnn",
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 10, 64)
    state = model.init_state(2, x.device)
    logits, new_state = model(x, state)
    assert logits.shape == (2, 10, 2)
    assert len(new_state) == 2
    loss = logits.sum()
    loss.backward()


def test_m2rnn_forward_backward() -> None:
    model = get_model(
        "m2rnn",
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 10, 64)
    state = model.init_state(2, x.device)
    logits, new_state = model(x, state)
    assert logits.shape == (2, 10, 2)
    assert len(new_state) == 2
    loss = logits.sum()
    loss.backward()


def test_min_gru_forward_backward() -> None:
    model = get_model(
        "min_gru",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 8, 32)
    state = model.init_state(2, x.device)
    assert all(state_value.hidden.min() > 0 for state_value in state)
    logits, new_state = model(x, state)
    assert logits.shape == (2, 8, 2)
    assert len(new_state) == 2
    loss = logits.sum()
    loss.backward()


def test_min_lstm_forward_backward() -> None:
    model = get_model(
        "min_lstm",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 8, 32)
    state = model.init_state(2, x.device)
    assert all(state_value.hidden.min() > 0 for state_value in state)
    logits, new_state = model(x, state)
    assert logits.shape == (2, 8, 2)
    assert len(new_state) == 2
    loss = logits.sum()
    loss.backward()


def test_m2rnn_state_shape() -> None:
    """Matrix-valued state must be [B, N, K, V]."""
    model = get_model(
        "m2rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        key_dim=16,
        value_dim=16,
        dropout=0.0,
        num_classes=2,
    )
    state = model.init_state(2, torch.device("cpu"))
    assert len(state) == 1
    assert state[0].hidden.shape == (2, 2, 16, 16)


def test_m2rnn_forget_gate_in_range() -> None:
    """Forget gate ψ must map to [0, 1]."""
    from nonlinearrnnscanbeparallel.models.m2rnn import psi

    x = torch.randn(100, 4)
    alpha = torch.randn(4)
    beta = torch.randn(4)
    f = psi(x, alpha, beta)
    assert f.min() >= 0.0
    assert f.max() <= 1.0


def test_dataset_deterministic_by_index() -> None:
    """__getitem__(idx) must be a pure function of idx (sorted graph connectivity)."""
    ds = SortedGraphConnectivityDataset(num_samples=100, max_nodes=16, max_seq_len=66, seed=42)
    s1 = ds[3]
    s2 = ds[3]
    assert torch.all(s1.input_ids == s2.input_ids)
    assert s1.labels.tolist() == s2.labels.tolist()
    assert s1.num_nodes == s2.num_nodes
    assert s1.source == s2.source
    assert s1.target == s2.target


def test_dataset_reproducible_across_instances() -> None:
    """Same seed => identical samples regardless of how many times we index."""
    ds_a = SortedGraphConnectivityDataset(num_samples=100, max_nodes=16, seed=42)
    ds_b = SortedGraphConnectivityDataset(num_samples=100, max_nodes=16, seed=42)
    sa = ds_a[7]
    sb = ds_b[7]
    assert torch.all(sa.input_ids == sb.input_ids)
    assert sa.labels.tolist() == sb.labels.tolist()


def test_dataset_disjoint_splits() -> None:
    """Train/val splits must have disjoint index sets and correct sizes."""
    from torch.utils.data import random_split

    full = SortedGraphConnectivityDataset(num_samples=40, max_nodes=8, seed=42)
    train, val = random_split(full, [30, 10], generator=torch.Generator().manual_seed(42))
    train_idx = {int(i) for i in train.indices}
    val_idx = {int(i) for i in val.indices}
    assert train_idx.isdisjoint(val_idx), "train and val must be disjoint"
    assert len(train_idx) == 30
    assert len(val_idx) == 10
    assert train_idx | val_idx == set(range(len(full))), "splits must cover full set"


def test_dataset_preserves_eos() -> None:
    """EOS must be present and label points to it for all samples."""
    ds = SortedGraphConnectivityDataset(num_samples=50, max_nodes=8, max_seq_len=66, seed=7)
    for idx in range(len(ds)):
        sample = ds[idx]
        eos_token = int(sample.num_nodes) + 3
        # EOS token must appear in the sequence.
        assert (sample.input_ids == eos_token).any(), (
            f"idx={idx}: EOS token {eos_token} missing; num_nodes={sample.num_nodes}"
        )
        # Label must be at the EOS position (not -100).
        eos_positions = (sample.labels != -100).nonzero(as_tuple=True)[0]
        assert len(eos_positions) == 1, f"idx={idx}: expected exactly 1 label position"
        eos_idx = int(eos_positions[0].item())
        assert sample.input_ids[eos_idx] == eos_token, (
            f"idx={idx}: label at position {eos_idx} is not EOS"
        )


def test_dataset_rejects_tight_seq_len() -> None:
    """max_seq_len < 2*max_nodes+2 must raise at construction."""
    with pytest.raises(ValueError):
        SortedGraphConnectivityDataset(num_samples=10, max_nodes=32, max_seq_len=65)


# rKAN regression tests.


def test_rkan_jacobi_trainable_and_grads() -> None:
    """alpha_raw/beta_raw must be trainable and receive gradients."""
    layer = RKANLayer(in_dim=4, out_dim=8, degree=3, num_basis=4, rkan_type="jacobi")
    alpha_param = layer.jacobi.alpha_raw
    beta_param = layer.jacobi.beta_raw
    assert isinstance(alpha_param, torch.nn.Parameter)
    assert isinstance(beta_param, torch.nn.Parameter)

    x = torch.randn(2, 4, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert alpha_param.grad is not None
    assert beta_param.grad is not None
    assert alpha_param.grad.abs().sum() > 0
    assert beta_param.grad.abs().sum() > 0


def test_rkan_coordinatewise() -> None:
    """Perturbing one input coordinate changes the output (not summed/collapsed)."""
    layer = RKANLayer(in_dim=3, out_dim=2, degree=3, num_basis=4, rkan_type="jacobi")
    x = torch.zeros(1, 3)
    out0 = layer(x)
    x_pert = x.clone()
    x_pert[0, 1] = 5.0
    out1 = layer(x_pert)
    assert not torch.allclose(out0, out1)


def test_rkan_pade_forward_backward() -> None:
    """Padé mode must execute (not silently ignored)."""
    layer = RKANLayer(in_dim=4, out_dim=4, degree=3, num_basis=4, rkan_type="pade")
    x = torch.randn(2, 4)
    out = layer(x)
    assert out.shape == (2, 4)
    loss = out.sum()
    loss.backward()
    assert layer.theta_e.grad is not None
    assert layer.theta_d.grad is not None


# One task path drives all five registered models.


def _task_smoke(model_name: str) -> None:
    """A single RNNTask must drive each registered model through BPTT chunking.

    Uses `_forward_chunked` + CrossEntropyLoss directly to exercise the
    truncated-state path (max_seq_len=4 < 10-token sequence) without needing
    a live Trainer/hook context.
    """
    from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask

    model_spec = {
        "name": model_name,
        "vocab_size": 36,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "num_classes": 2,
        "max_seq_len": 4,  # Smaller than T=10 to force multi-chunk BPTT.
        "key_dim": 8,
        "value_dim": 8,
    }
    task = RNNTask(model_spec=model_spec)
    x = torch.randint(0, 36, (2, 10))
    labels = torch.full((2, 10), -100, dtype=torch.long)
    labels[:, -1] = 0
    logits, _ = task._forward_chunked(x)
    loss = task.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor)


def test_rnntask_smoke_mlp_rnn() -> None:
    _task_smoke("mlp_rnn")


def test_rnntask_smoke_rkan_rnn() -> None:
    _task_smoke("rkan_rnn")


def test_rnntask_smoke_m2rnn() -> None:
    _task_smoke("m2rnn")


def test_rnntask_smoke_min_gru() -> None:
    _task_smoke("min_gru")


def test_rnntask_smoke_min_lstm() -> None:
    _task_smoke("min_lstm")


# Single-token step() path must match full forward.


def test_mlp_rnn_step_rollout_matches_forward() -> None:
    """Autoregressive step() rollout must equal forward() for MLP-RNN."""
    model = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 6, 32)
    state = model.init_state(2, x.device)
    logits, _ = model(x, state)

    step_logits = []
    step_state = model.init_state(2, x.device)
    for t in range(6):
        out_t, step_state = model.step(x[:, t, :], step_state)
        step_logits.append(out_t)
    rollout = torch.stack(step_logits, dim=1)
    assert torch.allclose(logits, rollout, atol=1e-5)


def test_rkan_rnn_uses_rkan_heads() -> None:
    """rKAN-RNN recurrence heads must be rKAN, not MLP (all MLPs replaced)."""
    from nonlinearrnnscanbeparallel.models.mlp_rnn import MLPHead
    from nonlinearrnnscanbeparallel.models.rkan import RKANHead

    model = get_model(
        "rkan_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    for layer_pair in model.layers:
        rnn_layer, _ = layer_pair
        for head in rnn_layer.heads:
            assert isinstance(head, RKANHead)
            assert not isinstance(head, MLPHead)


def test_rkan_rnn_step_rollout_matches_forward() -> None:
    """Autoregressive step() rollout must equal forward() for rKAN-RNN."""
    model = get_model(
        "rkan_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 6, 32)
    state = model.init_state(2, x.device)
    logits, _ = model(x, state)

    step_logits = []
    step_state = model.init_state(2, x.device)
    for t in range(6):
        out_t, step_state = model.step(x[:, t, :], step_state)
        step_logits.append(out_t)
    rollout = torch.stack(step_logits, dim=1)
    assert torch.allclose(logits, rollout, atol=1e-5)


def test_min_gru_step_rollout_matches_forward() -> None:
    """Autoregressive step() rollout must equal parallel minGRU forward()."""
    model = get_model(
        "min_gru",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 6, 32)
    state = model.init_state(2, x.device)
    logits, _ = model(x, state)

    step_logits = []
    step_state = model.init_state(2, x.device)
    for t in range(6):
        out_t, step_state = model.step(x[:, t, :], step_state)
        step_logits.append(out_t)
    rollout = torch.stack(step_logits, dim=1)
    assert torch.allclose(logits, rollout, atol=1e-5)


def test_min_lstm_step_rollout_matches_forward() -> None:
    """Autoregressive step() rollout must equal parallel minLSTM forward()."""
    model = get_model(
        "min_lstm",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 6, 32)
    state = model.init_state(2, x.device)
    logits, _ = model(x, state)

    step_logits = []
    step_state = model.init_state(2, x.device)
    for t in range(6):
        out_t, step_state = model.step(x[:, t, :], step_state)
        step_logits.append(out_t)
    rollout = torch.stack(step_logits, dim=1)
    assert torch.allclose(logits, rollout, atol=1e-5)


def test_m2rnn_step_first_token_matches_forward() -> None:
    """M²RNN step() is exact for the first token (zero left conv context)."""
    model = get_model(
        "m2rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        key_dim=8,
        value_dim=8,
        dropout=0.0,
        num_classes=2,
    )
    x = torch.randn(2, 5, 32)
    state = model.init_state(2, x.device)
    logits, _ = model(x, state)

    step_state = model.init_state(2, x.device)
    out_0, _ = model.step(x[:, 0, :], step_state)
    assert torch.allclose(logits[:, 0, :], out_0, atol=1e-5)


# rKAN mapping and Jacobi basis tests.


def test_rational_mapping_bounded_and_rejects_unknown() -> None:
    """All six φ mappings land in [-1, 1] with positive ι; unknown type raises."""
    from nonlinearrnnscanbeparallel.models.rkan import RationalMapping

    x = torch.rand(4, 8)  # Positive, like sigmoid outputs fed to the mapping.
    for mapping_type in (
        "linear",
        "logarithmic_semi",
        "algebraic_semi",
        "exponential_semi",
        "logarithmic_infinite",
        "algebraic_infinite",
    ):
        out = RationalMapping(mapping_type=mapping_type, iota=2.0)(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    mapping = RationalMapping(iota=2.0)
    assert mapping.iota.item() > 0

    with pytest.raises(ValueError):
        RationalMapping(mapping_type="bogus")


def test_jacobi_j0_unit_and_param_constraints() -> None:
    """J_0 is the unit constant; effective α/β stay > -1 and init to config."""
    from nonlinearrnnscanbeparallel.models.rkan import JacobiPolynomial

    poly = JacobiPolynomial(degree=3, alpha=0.5, beta=-0.5)
    vals = poly(torch.linspace(-1.0, 1.0, 11))
    assert torch.allclose(vals[..., 0], torch.ones(11), atol=1e-5)
    assert poly.alpha.item() > -1.0
    assert poly.beta.item() > -1.0
    assert poly.alpha.item() == pytest.approx(0.5)
    assert poly.beta.item() == pytest.approx(-0.5)


def test_inv_helpers_roundtrip() -> None:
    """Init helpers invert ELU/SoftPlus so effective values equal config defaults."""
    from nonlinearrnnscanbeparallel.models.rkan import _inv_elu, _inv_softplus

    for y in (2.0, 0.0, -0.5):
        assert torch.nn.functional.elu(torch.tensor(_inv_elu(y))).item() == pytest.approx(y)
    for y in (0.5, 1.0, 3.0):
        assert torch.nn.functional.softplus(torch.tensor(_inv_softplus(y))).item() == pytest.approx(
            y
        )


# Registry and training-path contract tests.


def test_registry_rejects_unknown_and_duplicate() -> None:
    """Unknown names and duplicate registration fail loudly."""
    from nonlinearrnnscanbeparallel.models.registry import get_model, register_model

    with pytest.raises(ValueError):
        get_model("no_such_model")

    class _Dummy:
        pass

    with pytest.raises(ValueError):
        register_model("mlp_rnn")(_Dummy)


def test_bptt_chunked_matches_full_forward() -> None:
    """Truncated BPTT chunking must not change logits (state detach only)."""
    from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask

    task = RNNTask(
        model_spec={
            "name": "mlp_rnn",
            "vocab_size": 36,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "num_classes": 2,
            "max_seq_len": 4,
        }
    )
    task.eval()
    x = torch.randint(0, 36, (2, 10))
    full, _ = task.forward(x)
    chunked, _ = task._forward_chunked(x)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_optimizer_excludes_embedding_from_decay() -> None:
    """2D matrices decay; embeddings and 1D params do not."""
    from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask

    task = RNNTask(
        model_spec={
            "name": "mlp_rnn",
            "vocab_size": 36,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "num_classes": 2,
        }
    )
    groups = task.configure_optimizers()["optimizer"].param_groups
    by_id = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert by_id[id(task.input_embed.weight)] == 0.0
    assert any(g["weight_decay"] > 0 for g in groups)


def test_psi_alpha1_is_sigmoid() -> None:
    """Paper note: effective α=1 reduces ψ to σ(-x - β)."""
    from nonlinearrnnscanbeparallel.models.m2rnn import psi

    x = torch.randn(4, 4)
    beta = torch.randn(4)
    alpha_raw = torch.zeros(4)  # effective α = exp(0) = 1
    assert torch.allclose(psi(x, alpha_raw, beta), torch.sigmoid(-x - beta), atol=1e-6)
