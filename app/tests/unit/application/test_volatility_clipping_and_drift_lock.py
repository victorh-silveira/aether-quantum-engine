import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.meta_classifier_features import _base_feature_vector, extract_meta_feature_vector


def test_volatility_clipping_in_precompute_price_series():
    # Cria uma série de preços com 1024 elementos
    prices = np.ones(1024, dtype=np.float64) * 100.0
    # Adiciona um estouro de volatilidade no final
    prices[-1] = 150.0
    high = prices + 1.0
    low = prices - 1.0

    series = precompute_price_series(
        prices,
        granularity=15,
        symbol="R_10",
        high=high,
        low=low,
    )

    # Verifica se os z-scores de bb_width e atr_norm estão dentro do range [-3.0, 3.0]
    assert np.all(series["bb_width"] >= -3.0)
    assert np.all(series["bb_width"] <= 3.0)
    assert np.all(series["atr_norm"] >= -3.0)
    assert np.all(series["atr_norm"] <= 3.0)


def test_meta_classifier_feature_clipping():
    # Caso 1: A partir do cached feature_vector
    feature_vector = [0.0] * 14
    feature_vector[5] = 5.5  # bb_log_width
    feature_vector[9] = -4.2  # atr_norm
    metrics = {"feature_vector": feature_vector}

    v = _base_feature_vector(metrics)
    assert v[5] == 3.0
    assert v[9] == -3.0

    # Caso 2: A partir dos indicators
    metrics_ind = {
        "indicators": {
            "bb_width": 4.16,
            "atr_norm": -5.0,
        },
        "raw_prob": 0.6,
    }
    v_ind = _base_feature_vector(metrics_ind)
    assert v_ind[4] == 3.0
    assert v_ind[5] == -3.0

    # Testar também extract_meta_feature_vector
    meta_v = extract_meta_feature_vector(metrics_ind)
    assert meta_v[4] == 3.0
    assert meta_v[5] == -3.0
