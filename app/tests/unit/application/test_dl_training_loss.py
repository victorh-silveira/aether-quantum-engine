"""Testes unitarios para dl_training_loss.py com cobertura 100%."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from src.application.services.deep_learning.dl_training_loss import (
    anti_collapse_loss_knobs,
    aux_regression_weight,
    calculate_masked_loss,
    model_core,
)


def test_aux_regression_weight_reads_settings():
    with patch(
        "src.application.services.deep_learning.dl_training_loss._read_dl_settings",
        return_value={"aux_regression_weight": 0.25},
    ):
        assert aux_regression_weight() == 0.25


def test_aux_regression_weight_raises_if_missing():
    with (
        patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value={}),
        pytest.raises(ValueError, match="aux_regression_weight obrigatorio"),
    ):
        aux_regression_weight()


def test_anti_collapse_loss_knobs_returns_defaults_or_settings():
    with patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value={}):
        m_w, e_w, floor = anti_collapse_loss_knobs()
        assert m_w == 0.15
        assert e_w == 0.10
        assert floor == 0.08

    custom = {
        "confidence_margin_penalty_weight": 0.3,
        "entropy_penalty_weight": 0.2,
        "confidence_margin_floor": 0.12,
    }
    with patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value=custom):
        m_w, e_w, floor = anti_collapse_loss_knobs()
        assert m_w == 0.3
        assert e_w == 0.2
        assert floor == 0.12


def test_model_core_unwraps_inner():
    wrapped = MagicMock()
    wrapped.inner = nn.Linear(4, 1)
    assert model_core(wrapped) is wrapped.inner
    raw = nn.Linear(4, 1)
    assert model_core(raw) is raw


def test_calculate_masked_loss_plain_bce():
    model = nn.Linear(4, 1)
    x = np.random.randn(8, 4).astype(np.float32)
    y = np.array([1.0, 0.0] * 4, dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    weights = [1.0] * 8
    device = torch.device("cpu")

    loss = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        confidence_margin_weight=0.0,
        entropy_penalty_weight=0.0,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_calculate_masked_loss_with_focal_margin_and_entropy():
    model = nn.Linear(4, 1)
    x = np.random.randn(8, 4).astype(np.float32)
    y = np.array([1.0, 0.0] * 4, dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    weights = [1.0] * 8
    device = torch.device("cpu")

    loss = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.1,
        focal_gamma=2.0,
        confidence_margin_weight=0.25,
        entropy_penalty_weight=0.15,
        confidence_margin_floor=0.10,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_calculate_masked_loss_with_aux_regression():
    class DummyDualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 1)
            self.regression_head = nn.Linear(4, 1)

        def forward(self, x, *, logits: bool = False, return_aux: bool = False):
            _ = logits
            raw = self.head(x).squeeze(-1)
            aux = self.regression_head(x).squeeze(-1)
            if return_aux:
                return raw, aux
            return raw

    model = DummyDualHead()
    x = np.random.randn(8, 4).astype(np.float32)
    y = np.array([1.0, 0.0] * 4, dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    weights = [1.0] * 8
    delta = np.random.randn(8).astype(np.float32)
    device = torch.device("cpu")

    loss = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        delta_batch=delta,
        aux_regression_weight_val=0.2,
        confidence_margin_weight=0.1,
        entropy_penalty_weight=0.1,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_calculate_masked_loss_with_aux_regression_type_error():
    class DummyDualHeadNoKwargs(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 1)
            self.regression_head = nn.Linear(4, 1)

        def forward(self, x):
            raw = self.head(x).squeeze(-1)
            aux = self.regression_head(x).squeeze(-1)
            return raw, aux

    model = DummyDualHeadNoKwargs()
    x = np.random.randn(8, 4).astype(np.float32)
    y = np.array([1.0, 0.0] * 4, dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    weights = [1.0] * 8
    delta = np.random.randn(8).astype(np.float32)
    device = torch.device("cpu")

    loss = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        delta_batch=delta,
        aux_regression_weight_val=0.2,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_calculate_masked_loss_tuple_logits_without_aux():
    class DummyTupleModel(nn.Module):
        def forward(self, x, *, logits: bool = False):
            _ = logits
            return torch.zeros(len(x), 1), "extra"

    model = DummyTupleModel()
    x = np.random.randn(4, 2).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    weights = [1.0] * 4
    device = torch.device("cpu")

    loss = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        delta_batch=None,
        aux_regression_weight_val=0.1,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_binary_option_asymmetric_loss():
    from src.application.services.deep_learning.dl_training_loss import (
        BinaryOptionAsymmetricLoss,
        dl_asymmetric_loss_enabled,
        dl_payout_rate,
    )

    loss_fn = BinaryOptionAsymmetricLoss(payout_rate=0.85)
    logits = torch.tensor([2.0, -2.0])
    targets = torch.tensor([1.0, 0.0])
    out = loss_fn(logits, targets)
    assert out.shape == torch.Size([2])
    assert torch.all(torch.isfinite(out))

    with patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value={}):
        assert dl_payout_rate() == 0.85
        assert dl_asymmetric_loss_enabled() is True

    custom = {"loss_payout_rate": 0.78, "asymmetric_payout_loss": False}
    with patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value=custom):
        assert dl_payout_rate() == 0.78
        assert dl_asymmetric_loss_enabled() is False

    invalid = {"loss_payout_rate": "invalid"}
    with patch("src.application.services.deep_learning.dl_training_loss._read_dl_settings", return_value=invalid):
        assert dl_payout_rate() == 0.85


def test_calculate_masked_loss_asymmetric_flag():
    model = nn.Linear(4, 1)
    x = np.random.randn(4, 4).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    weights = [1.0] * 4
    device = torch.device("cpu")

    loss_asym = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        asymmetric_payout_loss=True,
        payout_rate=0.85,
    )
    assert isinstance(loss_asym, torch.Tensor)
    assert torch.isfinite(loss_asym)

    loss_sym = calculate_masked_loss(
        model,
        x,
        y,
        mask,
        weights,
        device,
        label_smoothing=0.0,
        focal_gamma=0.0,
        asymmetric_payout_loss=False,
    )
    assert isinstance(loss_sym, torch.Tensor)
    assert torch.isfinite(loss_sym)
