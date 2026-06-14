import tempfile
from pathlib import Path

import torch

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.model import create_direction_model, load_model_checkpoint


def test_load_model_checkpoint_rejects_incompatible_architecture():
    model = create_direction_model(arch="tcn", input_dim=FEATURE_DIM)
    legacy = {"state_dict": model.state_dict()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_arch.pth"
        torch.save(
            {
                **legacy,
                "norm_mean": [0.0] * FEATURE_DIM,
                "norm_std": [1.0] * FEATURE_DIM,
                "feature_dim": FEATURE_DIM - 1,
                "arch": "mlp",
                "last_candle_epoch": 1,
                "lookback": 32,
            },
            path,
        )
        assert load_model_checkpoint(path) is None
