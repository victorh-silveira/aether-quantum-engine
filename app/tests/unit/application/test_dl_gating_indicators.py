from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_gating import check_indicator_gating_bounds
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_check_indicator_gating_bounds_disabled():
    # Se estiver desativado, retorna None
    res = check_indicator_gating_bounds({"hurst": 0.3}, {"enabled": False, "hurst_min": 0.4})
    assert res is None


def test_check_indicator_gating_bounds_hurst():
    cfg = {"enabled": True, "hurst_min": 0.45, "hurst_max": 0.65}
    # Dentro dos limites
    assert check_indicator_gating_bounds({"hurst": 0.5}, cfg) is None
    # Fora dos limites (abaixo)
    assert check_indicator_gating_bounds({"hurst": 0.35}, cfg) == "indicator_hurst"
    # Fora dos limites (acima)
    assert check_indicator_gating_bounds({"hurst": 0.7}, cfg) == "indicator_hurst"


def test_check_indicator_gating_bounds_adx():
    cfg = {"enabled": True, "adx_min": 0.20}
    # Dentro dos limites
    assert check_indicator_gating_bounds({"adx": 0.25}, cfg) is None
    # Fora dos limites
    assert check_indicator_gating_bounds({"adx": 0.15}, cfg) == "indicator_adx"


def test_check_indicator_gating_bounds_vol_ratio():
    cfg = {"enabled": True, "vol_ratio_min": 0.5, "vol_ratio_max": 2.0}
    # Dentro dos limites
    assert check_indicator_gating_bounds({"vol_ratio_short_long": 1.0}, cfg) is None
    # Fora dos limites (abaixo)
    assert check_indicator_gating_bounds({"vol_ratio_short_long": 0.3}, cfg) == "indicator_vol_ratio"
    # Fora dos limites (acima)
    assert check_indicator_gating_bounds({"vol_ratio_short_long": 2.5}, cfg) == "indicator_vol_ratio"


def test_check_indicator_gating_bounds_cmo():
    cfg = {"enabled": True, "cmo_min": -0.5, "cmo_max": 0.5}
    # Dentro dos limites
    assert check_indicator_gating_bounds({"cmo": 0.0}, cfg) is None
    # Fora dos limites (abaixo)
    assert check_indicator_gating_bounds({"cmo": -0.6}, cfg) == "indicator_cmo"
    # Fora dos limites (acima)
    assert check_indicator_gating_bounds({"cmo": 0.7}, cfg) == "indicator_cmo"


def test_check_indicator_gating_bounds_keltner():
    cfg = {"enabled": True, "keltner_pct_b_min": 0.1, "keltner_pct_b_max": 0.9}
    # Dentro dos limites
    assert check_indicator_gating_bounds({"keltner_pct_b": 0.5}, cfg) is None
    # Fora dos limites (abaixo)
    assert check_indicator_gating_bounds({"keltner_pct_b": 0.05}, cfg) == "indicator_keltner"
    # Fora dos limites (acima)
    assert check_indicator_gating_bounds({"keltner_pct_b": 0.95}, cfg) == "indicator_keltner"


def test_predict_symbol_decision_indicator_gating():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
            "indicator_gating": {
                "enabled": True,
                "hurst_min": 0.55,
            },
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}

    # Vamos mockar o predict_next_direction para retornar um CALL forte
    # E mockar o precompute_price_series para retornar valores de Hurst baixos e altos
    mock_series_low_hurst = {
        "hurst": np.array([0.45]),
        "adx": np.array([0.25]),
        "vol_ratio_short_long": np.array([1.0]),
        "cmo": np.array([0.0]),
        "keltner_pct_b": np.array([0.5]),
    }

    mock_series_high_hurst = {
        "hurst": np.array([0.60]),
        "adx": np.array([0.25]),
        "vol_ratio_short_long": np.array([1.0]),
        "cmo": np.array([0.0]),
        "keltner_pct_b": np.array([0.5]),
    }

    # Teste 1: Hurst abaixo do mínimo (deve bloquear com indicator_hurst)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.80, 0.80),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict.precompute_price_series",
            return_value=mock_series_low_hurst,
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "indicator_hurst"

    # Teste 2: Hurst acima do mínimo (deve permitir)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.80, 0.80),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict.precompute_price_series",
            return_value=mock_series_high_hurst,
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["gate_reason"] is None
