"""Consistencia das features M5 entre treino longo e inferencia curta."""

import json

import numpy as np

from aether_paths import repo_path
from src.application.services.deep_learning.dl_feature_matrix import build_sequence_tensor
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_training_gate import min_dl_inference_len


def test_runtime_inference_history_preserves_training_features():
    settings = json.loads(repo_path("config", "settings.json").read_text(encoding="utf-8"))
    params = parse_dl_params(settings["deep_learning"], settings["data_handler"], settings["risk_management"]["params"])
    infer_bars = params["inference_history_bars"]
    assert infer_bars == 768
    assert min_dl_inference_len(params) == infer_bars

    rng = np.random.default_rng(20261005)
    close = 5000.0 + np.cumsum(rng.normal(0.0, 1.8, 1200))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    full = build_sequence_tensor(close, 32, len(close) - 1, granularity=300, open_=open_, high=high, low=low)
    tail = build_sequence_tensor(
        close[-infer_bars:],
        32,
        infer_bars - 1,
        granularity=300,
        open_=open_[-infer_bars:],
        high=high[-infer_bars:],
        low=low[-infer_bars:],
    )
    np.testing.assert_allclose(tail, full, rtol=0.0, atol=1e-6)
