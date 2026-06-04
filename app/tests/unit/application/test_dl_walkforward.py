from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bridge_helpers import parse_dl_params
from src.application.services.deep_learning.dl_sim_backtest import direction_wins, run_dl_walkforward
from src.domain.models.trade import TradeDirection


def test_run_dl_walkforward_offline():
    prices = np.sin(np.linspace(0, 20, 260)) + 10.0
    params = parse_dl_params(
        {
            "arch": "tcn",
            "lookback": 20,
            "training_epochs": 2,
            "learning_rate": 0.001,
            "validation_bars": 15,
            "early_stopping_patience": 1,
            "label_min_move_pct": 0.0,
            "min_conviction_execute": 0.0,
            "min_edge_margin": 0.0,
            "min_val_accuracy": 0.0,
            "min_direction_margin": 0.0,
            "require_regime_alignment": False,
            "max_raw_saturation": 1.0,
            "max_val_brier_execute": 1.0,
        }
    )
    result = run_dl_walkforward(prices, params, retrain_every=60)
    assert len(result.trades) >= 1
    assert result.win_rate >= 0.0
    assert result.max_drawdown >= 0.0
    assert result.val_brier <= 1.0


def test_run_dl_walkforward_skips_when_execute_false():
    prices = np.sin(np.linspace(0, 20, 260)) + 10.0
    params = parse_dl_params(
        {
            "arch": "tcn",
            "lookback": 20,
            "training_epochs": 2,
            "learning_rate": 0.001,
            "validation_bars": 15,
            "early_stopping_patience": 1,
            "label_min_move_pct": 0.0,
            "min_conviction_execute": 0.99,
            "min_edge_margin": 0.99,
            "min_val_accuracy": 0.99,
            "min_direction_margin": 0.5,
            "require_regime_alignment": False,
            "max_raw_saturation": 1.0,
            "max_val_brier_execute": 1.0,
        }
    )
    with patch(
        "src.application.services.deep_learning.dl_sim_backtest.predict_symbol_decision",
        return_value={"direction": TradeDirection.CALL, "metrics": {"execute": False}},
    ):
        result = run_dl_walkforward(prices, params, retrain_every=60)
    assert len(result.trades) == 0


def test_direction_wins_boundary_and_short_series():
    prices = np.array([10.0, 11.0])
    assert direction_wins(TradeDirection.CALL, prices, 1) is False
    assert direction_wins(TradeDirection.CALL, prices, 0)
    params = parse_dl_params({"lookback": 20, "validation_bars": 15})
    with pytest.raises(RuntimeError):
        run_dl_walkforward(prices, params)
