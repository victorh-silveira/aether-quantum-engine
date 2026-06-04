from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_training import train_model_walkforward
from src.application.services.deep_learning.model import create_direction_model


def test_train_reextract_when_pair_mask_too_sparse():
    n = 90
    lookback = 20
    x = np.zeros((n, lookback, FEATURE_DIM), dtype=np.float32)
    y = np.ones(n, dtype=np.float32)
    mask_empty = np.zeros(n, dtype=np.float32)
    mask_full = np.ones(n, dtype=np.float32)
    prices = np.linspace(100.0, 120.0, n + lookback + 5)
    model = create_direction_model(input_dim=FEATURE_DIM)
    with patch(
        "src.application.services.deep_learning.dl_training.extract_sequences",
        side_effect=[
            (x, y, mask_empty),
            (x, y, mask_full),
        ],
    ):
        result = train_model_walkforward(
            model,
            prices,
            lookback,
            1,
            0.001,
            20,
            require_pair_label=True,
            sym_is_bull=True,
        )
    assert result is not None
