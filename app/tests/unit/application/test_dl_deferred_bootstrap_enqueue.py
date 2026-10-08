from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_deferred_train import try_enqueue_next_bootstrap_training
from tests.market_symbols import ALT_SYMBOL


def test_try_enqueue_next_bootstrap_training_schedules_first_pending(orch_ready):
    orch = orch_ready
    n = 3000
    ohlc = tuple(np.linspace(1.0, 2.0, n) for _ in range(4))
    runtime = {"session_trained": False}
    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train._ordered_bootstrap_symbols",
            return_value=[ALT_SYMBOL],
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train._bootstrap_training_context",
            return_value=(
                {"deploy_gate": {"enabled": False}},
                {"lookback": 32, "training_history_bars": 100},
                100,
                60,
                runtime,
                ohlc[0],
                ohlc[1],
                ohlc[2],
                ohlc[3],
                None,
            ),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.runtime_in_training",
            return_value=True,
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        try_enqueue_next_bootstrap_training(orch)
    mock_enqueue.assert_called_once()
    assert mock_enqueue.call_args.args[1] == ALT_SYMBOL


def test_try_enqueue_next_bootstrap_training_skips_when_not_bootstrap_reason(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train._ordered_bootstrap_symbols",
            return_value=["R_10"],
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train._bootstrap_training_context",
            return_value=({}, {}, 100, 60, {}, np.linspace(1.0, 2.0, 200), None, None, None, None),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.runtime_in_training",
            return_value=True,
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.should_retrain_symbol",
            return_value=(True, "new_candle"),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        try_enqueue_next_bootstrap_training(orch)
    mock_enqueue.assert_not_called()


def test_try_enqueue_next_bootstrap_training_stops_on_short_history(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train._ordered_bootstrap_symbols",
            return_value=["R_10"],
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train._bootstrap_training_context",
            return_value=({}, {}, 3000, 60, {}, np.linspace(1.0, 2.0, 50), None, None, None, None),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.runtime_in_training",
            return_value=True,
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        try_enqueue_next_bootstrap_training(orch)
    mock_enqueue.assert_not_called()


def test_try_enqueue_skips_symbol_already_trained(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train._ordered_bootstrap_symbols",
            return_value=["R_10", "R_50"],
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train._bootstrap_training_context",
            return_value=({}, {}, 100, 60, {}, np.linspace(1.0, 2.0, 200), None, None, None, None),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.runtime_in_training",
            side_effect=[False, True],
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        try_enqueue_next_bootstrap_training(orch)
    mock_enqueue.assert_called_once()
