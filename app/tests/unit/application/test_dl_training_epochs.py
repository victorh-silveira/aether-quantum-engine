import math
from unittest.mock import patch

import numpy as np
import torch

from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_fit_training_epochs_early_stopping():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")
    val_losses = iter([0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89])

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            side_effect=lambda *_args, **_kwargs: next(val_losses),
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.model_accuracy",
            return_value=0.5,
        ),
    ):
        avg, state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=50,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=3,
        )
    assert ran == 5
    assert state is not None
    assert math.isfinite(avg)


def test_fit_training_epochs_respects_min_epochs():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            return_value=1.0,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.model_accuracy",
            return_value=0.5,
        ),
    ):
        _avg, _state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=20,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=2,
            min_epochs=5,
        )
    assert ran == 6


def test_fit_training_epochs_disabled_runs_all_epochs():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            return_value=1.0,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.model_accuracy",
            return_value=0.5,
        ),
    ):
        _avg, _state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=7,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=0,
        )
    assert ran == 7


def test_fit_training_epochs_skips_non_finite_loss():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(8, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    device = torch.device("cpu")
    finite = torch.tensor(0.5, requires_grad=True)
    calls = {"n": 0}

    def loss_side_effect(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return finite
        return torch.tensor(float("nan"))

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._masked_loss",
            side_effect=loss_side_effect,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            return_value=0.5,
        ),
    ):
        avg, state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 8,
            x,
            y,
            mask,
            device,
            epochs=2,
            batch_size=4,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
        )
    assert ran == 2
    assert state is not None
    assert math.isfinite(avg)


def test_fit_training_epochs_reduce_on_plateau_scheduler():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")
    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            return_value=0.7,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.model_accuracy",
            return_value=0.55,
        ),
    ):
        avg, state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=3,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            lr_scheduler="reduce_on_plateau",
        )
    assert ran == 3
    assert state is not None
    assert math.isfinite(avg)


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
            return_value=0.7,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_epochs.model_accuracy",
            side_effect=lambda *_args, **_kwargs: next(acc_values),
        ),
    ):
        avg, state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=4,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=2,
        )
    assert ran >= 2
    assert state is not None
    assert math.isfinite(avg)
