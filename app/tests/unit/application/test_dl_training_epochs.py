import math
from unittest.mock import patch

import numpy as np
import torch

from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


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

    with patch(
        "src.application.services.deep_learning.dl_training_epochs._masked_loss",
        side_effect=loss_side_effect,
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
