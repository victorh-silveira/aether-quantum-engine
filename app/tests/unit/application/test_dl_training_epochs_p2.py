"""Testes do seletor de checkpoint TCN (ACC peak + ramo sharp por nitidez)."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_checkpoint_keeps_peak_acc_when_later_loss_improves():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    loss1, acc1, _sa, _sl, _sv, state_peak, _ss1, improved1 = checkpoint_if_improved(
        model,
        val_loss=0.65,
        val_acc=0.55,
        val_sharpness=0.05,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert improved1 is True
    assert state_peak is not None
    peak_key = next(iter(state_peak))
    peak_tensor = state_peak[peak_key].clone()
    with torch.no_grad():
        for tensor in model.parameters():
            tensor.add_(0.5)
    loss2, acc2, _sa2, _sl2, _sv2, state_loss_only, _ss2, improved2 = checkpoint_if_improved(
        model,
        val_loss=0.50,
        val_acc=0.51,
        val_sharpness=0.05,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=loss1,
        best_val_acc=acc1,
        best_sharp_acc=acc1,
        best_sharp_loss=0.65,
        best_sharp_value=0.05,
    )
    assert improved2 is True
    assert state_loss_only is None
    assert acc2 == pytest.approx(0.55)
    assert loss2 == pytest.approx(0.50)
    assert torch.equal(state_peak[peak_key], peak_tensor)


def test_prefer_sharp_when_acc_clears_soft_floor():
    from src.application.services.deep_learning.dl_training_checkpoint import (
        checkpoint_if_improved,
        prefer_sharp_checkpoint,
    )

    model = create_direction_model(arch="tcn")
    _l, _a, sa, sl, sv, dull_state, sharp_state, _imp = checkpoint_if_improved(
        model,
        val_loss=0.65,
        val_acc=0.56,
        val_sharpness=0.02,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert dull_state is not None
    assert sharp_state is not None
    _l2, _a2, sa2, sl2, sv2, _s2, sharp_state2, _imp2 = checkpoint_if_improved(
        model,
        val_loss=0.55,
        val_acc=0.54,
        val_sharpness=0.04,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=_l,
        best_val_acc=_a,
        best_sharp_acc=sa,
        best_sharp_loss=sl,
        best_sharp_value=sv,
    )
    assert sharp_state2 is not None
    assert sv2 == pytest.approx(0.04)
    assert prefer_sharp_checkpoint(dull_state, sharp_state2, best_acc=0.56, sharp_acc=0.54) is sharp_state2
    assert prefer_sharp_checkpoint(None, sharp_state2, best_acc=0.54, sharp_acc=0.54) is sharp_state2
    assert sa2 == pytest.approx(0.54)
    assert sl2 == pytest.approx(0.55)


def test_sharp_checkpoint_prefers_higher_sharpness():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    _l, _a, sa, sl, sv, _bs, first, _i = checkpoint_if_improved(
        model,
        val_loss=0.60,
        val_acc=0.61,
        val_sharpness=0.03,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert first is not None
    _l2, _a2, sa2, sl2, sv2, _bs2, second, _i2 = checkpoint_if_improved(
        model,
        val_loss=0.40,
        val_acc=0.54,
        val_sharpness=0.045,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=_l,
        best_val_acc=_a,
        best_sharp_acc=sa,
        best_sharp_loss=sl,
        best_sharp_value=sv,
    )
    assert second is not None
    assert sv2 == pytest.approx(0.045)
    assert sa2 == pytest.approx(0.54)


def test_sharp_checkpoint_same_sharp_same_acc_prefers_lower_loss():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    _l, _a, sa, sl, sv, _bs, first, _i = checkpoint_if_improved(
        model,
        val_loss=0.60,
        val_acc=0.55,
        val_sharpness=0.04,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert first is not None
    _l2, _a2, sa2, sl2, sv2, _bs2, second, _i2 = checkpoint_if_improved(
        model,
        val_loss=0.40,
        val_acc=0.55,
        val_sharpness=0.04,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=_l,
        best_val_acc=_a,
        best_sharp_acc=sa,
        best_sharp_loss=sl,
        best_sharp_value=sv,
    )
    assert second is not None
    assert sl2 == pytest.approx(0.40)
    assert sa2 == pytest.approx(0.55)
    assert sv2 == pytest.approx(0.04)


def test_checkpoint_saves_acc_peak_even_with_majority_collapse():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    _l, _a, _sa, _sl, _sv, state, sharp, improved = checkpoint_if_improved(
        model,
        val_loss=0.40,
        val_acc=0.58,
        val_sharpness=0.05,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
        collapse_hit=True,
    )
    assert state is not None
    assert sharp is None
    assert improved is True
    assert _a == pytest.approx(0.58)


def test_checkpoint_keeps_high_acc_even_with_elevated_ce():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    _l, acc, _sa, _sl, _sv, state, sharp, improved = checkpoint_if_improved(
        model,
        val_loss=0.80,
        val_acc=0.55,
        val_sharpness=0.05,
        min_sharpness=0.05,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert improved is True
    assert state is not None
    assert sharp is not None
    assert acc == pytest.approx(0.55)


def test_fit_training_epochs_checkpoint_on_val_acc_only():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")
    acc_values = iter([0.5, 0.52, 0.52, 0.52])

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            return_value=0.65,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.val_collapse_hit",
            side_effect=lambda *_a, **_k: (next(acc_values), 0.02, False),
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs._mean_epoch_loss",
            return_value=(0.5, 1),
        ),
    ):
        avg, state, ran = fit_training_epochs(
            model,
            x[:16],
            y[:16],
            mask[:16],
            [1.0] * 16,
            x[16:],
            y[16:],
            mask[16:],
            device,
            epochs=4,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=10,
            min_epochs=1,
            min_oos_sharpness=0.05,
            min_val_accuracy=0.53,
        )
    assert math.isfinite(avg)
    assert ran >= 1
    assert state is None or isinstance(state, dict)

    from src.application.services.deep_learning.dl_training_checkpoint import prefer_sharp_checkpoint

    st_peak = {"w": 1}
    st_sharp = {"w": 2}
    res1 = prefer_sharp_checkpoint(st_peak, st_sharp, best_acc=0.50, sharp_acc=0.50, min_val_accuracy=0.55)
    assert res1 == st_sharp
    res2 = prefer_sharp_checkpoint(st_peak, st_sharp, best_acc=0.54, sharp_acc=0.48, min_val_accuracy=0.55)
    assert res2 == st_peak
