from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bootstrap_train import (
    _bootstrap_training_context,
    run_dl_training_session,
)


@pytest.mark.asyncio
async def test_run_dl_training_session_fails_closed_on_export_error(orch_ready):
    orch = orch_ready
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        new_callable=AsyncMock,
        return_value="fail",
    ):
        assert await run_dl_training_session(orch) is False


def test_bootstrap_training_context_loads_micro_timeframe(orch_ready):
    orch = orch_ready
    orch.config.setdefault("deep_learning", {})["train_timeframe"] = "micro"
    orch.config["deep_learning"]["online_training"] = False
    orch.config["deep_learning"]["training_history_bars"] = 2000
    orch.config["deep_learning"]["lookback"] = 720
    n = 2000
    series = np.linspace(1.0, 2.0, n)

    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.load_symbol_close_ohlc",
            return_value=(series, series, series, series),
        ) as mock_load,
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.load_symbol_microstructure",
            return_value=None,
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.get_symbol_runtime",
            return_value={},
        ),
    ):
        _dl, params, min_len, _g, _rt, prices, *_rest = _bootstrap_training_context(orch, "OTC_SPC")

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs.get("timeframe") == "micro"
    assert params["train_timeframe"] == "micro"
    assert min_len >= 2000
    assert len(prices) == 2000
