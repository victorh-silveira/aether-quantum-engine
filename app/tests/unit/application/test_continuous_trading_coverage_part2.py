"""Parte 2 dos testes unitarios para cobertura de decisao de trading continuo."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.domain.models.trade import TradeDirection


def test_predict_symbol_decision_sync_path():
    """Verifica predicao DL sincrona local direta."""
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"enabled": True},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.65,
        "val_brier": 0.22,
        "deploy_ok": True,
        "deploy_win_rate": 0.58,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.7, 0.7),
        ),
        patch("src.application.services.deep_learning.dl_predict_build.precompute_price_series") as mock_series,
    ):
        mock_series.return_value = {
            "bb_width": np.array([0.1]),
            "atr_norm": np.array([0.1]),
            "adx": np.array([0.2]),
            "vol_ratio_short_long": np.array([1.0]),
            "implied_vol_ratio": np.array([1.0]),
            "hurst": np.array([0.5]),
            "cmo": np.array([0.0]),
            "keltner_pct_b": np.array([0.5]),
            "rsi": np.array([0.5]),
            "macd": np.array([0.0]),
            "macd_signal": np.array([0.0]),
            "di_diff": np.array([0.0]),
        }
        entry = predict_symbol_decision(
            orch,
            "R_10",
            runtime["model"],
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
            force_local=True,
        )
    assert entry["direction"] == TradeDirection.CALL


def test_predict_symbol_decision_sync_always_eager():
    """Path sync local ignora cache e sempre re-infere."""
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"enabled": True},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    orch._active_cycle_id = 4
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.65,
        "val_brier": 0.22,
        "deploy_ok": True,
        "deploy_win_rate": 0.58,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.PUT, 0.4, 0.35),
        ),
        patch("src.application.services.deep_learning.dl_predict_build.precompute_price_series") as mock_series,
    ):
        mock_series.return_value = {
            "bb_width": np.array([0.1]),
            "atr_norm": np.array([0.1]),
            "adx": np.array([0.2]),
            "vol_ratio_short_long": np.array([1.0]),
            "implied_vol_ratio": np.array([1.0]),
            "hurst": np.array([0.5]),
            "cmo": np.array([0.0]),
            "keltner_pct_b": np.array([0.5]),
            "rsi": np.array([0.5]),
            "macd": np.array([0.0]),
            "macd_signal": np.array([0.0]),
            "di_diff": np.array([0.0]),
        }
        entry = predict_symbol_decision(
            orch,
            "R_10",
            runtime["model"],
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
            force_local=True,
        )
    assert entry["direction"] == TradeDirection.PUT


def test_predict_symbol_decision_async_via_asyncio_run():
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"enabled": True},
        "infra": {"triton": {"enabled": False}},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    orch._active_cycle_id = 5
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.65,
        "val_brier": 0.22,
        "deploy_ok": True,
        "deploy_win_rate": 0.58,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.66, 0.6),
        ),
        patch("src.application.services.deep_learning.dl_predict_build.precompute_price_series") as mock_series,
    ):
        mock_series.return_value = {
            "bb_width": np.array([0.1]),
            "atr_norm": np.array([0.1]),
            "adx": np.array([0.2]),
            "vol_ratio_short_long": np.array([1.0]),
            "implied_vol_ratio": np.array([1.0]),
            "hurst": np.array([0.5]),
            "cmo": np.array([0.0]),
            "keltner_pct_b": np.array([0.5]),
            "rsi": np.array([0.5]),
            "macd": np.array([0.0]),
            "macd_signal": np.array([0.0]),
            "di_diff": np.array([0.0]),
        }
        entry = predict_symbol_decision(
            orch,
            "R_10",
            runtime["model"],
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
            force_local=False,
        )
    assert entry["direction"] == TradeDirection.CALL
