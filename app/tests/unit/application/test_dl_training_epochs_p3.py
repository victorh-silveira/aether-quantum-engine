from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


@pytest.mark.parametrize(
    ("validation", "expected_weight"),
    [([(0.52, 0.02, True), (0.51, 0.02, False)], 2.0), ([(0.52, 0.02, True), (0.50, 0.02, True)], 1.0)],
)
def test_training_prefers_noncollapsed_epoch_with_collapsed_fallback(validation, expected_weight):
    from src.application.services.deep_learning import dl_training_epochs as epochs

    model = torch.nn.Linear(1, 1)
    initial_weight = model.weight.item()
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    mask = np.ones(2, dtype=np.float32)

    def advance(*_args, **_kwargs):
        with torch.no_grad():
            model.weight.add_(1.0)
        return 0.6, 1

    with (
        patch.object(epochs, "_mean_epoch_loss", side_effect=advance),
        patch.object(epochs, "_validation_loss", return_value=0.6),
        patch.object(epochs, "val_collapse_hit", side_effect=validation),
        patch.object(epochs, "_build_lr_scheduler", return_value=("cosine", MagicMock())),
    ):
        _, chosen, _ = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0, 1.0],
            x,
            y,
            mask,
            torch.device("cpu"),
            epochs=2,
            batch_size=2,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            min_val_accuracy=0.5,
        )
    assert chosen is not None
    assert chosen["weight"].item() == pytest.approx(expected_weight + initial_weight)


def test_sharp_only_improvement_resets_early_stopping_patience():
    """Melhora apenas de sharpness reinicia paciencia, independentemente do treino aleatorio."""
    from src.application.services.deep_learning import dl_training_epochs as epochs

    model = torch.nn.Linear(1, 1)
    state = model.state_dict()
    unchanged = (0.6, 0.55, 0.56, 0.61, 0.1, None, None)
    checkpoints = [
        (*unchanged[:5], state, None),
        unchanged,
        (*unchanged[:5], None, state),
        unchanged,
        unchanged,
        unchanged,
    ]
    x, y, mask = np.zeros((2, 1), dtype=np.float32), np.array([0.0, 1.0]), np.ones(2)
    with (
        patch.object(epochs, "_mean_epoch_loss", return_value=(0.6, 1)),
        patch.object(epochs, "_validation_loss", return_value=0.6),
        patch.object(epochs, "val_collapse_hit", return_value=(0.55, 0.1, False)),
        patch.object(epochs, "checkpoint_if_improved", side_effect=checkpoints) as checkpoint,
        patch.object(epochs, "_build_lr_scheduler", return_value=("cosine", MagicMock())),
    ):
        loss, chosen, ran = fit_training_epochs(
            model,
            x,
            y,
            mask,
            [1.0, 1.0],
            x,
            y,
            mask,
            torch.device("cpu"),
            epochs=10,
            batch_size=2,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            early_stopping_patience=3,
            min_val_accuracy=0.53,
        )
    assert ran == checkpoint.call_count == 6
    assert abs(loss - 0.6) < 1e-9
    assert chosen is state


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
