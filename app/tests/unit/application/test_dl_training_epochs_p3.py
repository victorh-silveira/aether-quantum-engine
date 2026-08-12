from unittest.mock import patch

import numpy as np
import torch

from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_fit_training_epochs_val_loss_uses_plain_bce_not_focal():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(24, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0] * 12, dtype=np.float32)
    mask = np.ones(24, dtype=np.float32)
    device = torch.device("cpu")
    seen: list[float] = []

    def _capture_val(*_args, **kwargs):
        seen.append(float(kwargs.get("focal_gamma", -1.0)))
        return 0.60

    with (
        patch(
            "src.application.services.deep_learning.dl_training_epochs._validation_loss",
            side_effect=_capture_val,
        ),
        patch(
            "src.application.services.deep_learning.dl_training_checkpoint.model_accuracy",
            return_value=0.56,
        ),
    ):
        _avg, state, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0] * 24,
            x,
            y,
            mask,
            device,
            epochs=2,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=1.0,
            early_stopping_patience=6,
            min_val_accuracy=0.53,
        )
    assert ran == 2
    assert state is not None
    assert seen
    assert all(gamma == 0.0 for gamma in seen)
