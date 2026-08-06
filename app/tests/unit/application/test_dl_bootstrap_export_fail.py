from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bootstrap_train import (
    _train_bootstrap_symbol,
    run_initial_bootstrap_training,
)


@pytest.mark.asyncio
async def test_train_bootstrap_symbol_returns_false_when_history_short(orch_ready):
    orch = orch_ready
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._bootstrap_training_context",
        return_value=({}, {"lookback": 32}, 3000, 60, {}, np.linspace(1.0, 2.0, 100), None, None, None, None),
    ):
        status = await _train_bootstrap_symbol(orch, "OTC_SPC")
    assert status == "wait"


@pytest.mark.asyncio
async def test_train_bootstrap_symbol_fails_when_export_not_ok(orch_ready):
    orch = orch_ready
    n = 3000
    ohlc = tuple(np.linspace(1.0, 2.0, n) for _ in range(4))
    runtime: dict = {}

    async def fake_thread(*_args, **_kwargs):
        runtime["export_ok"] = False

    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._bootstrap_training_context",
            return_value=(
                {},
                {"lookback": 32},
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
            "src.application.services.deep_learning.dl_bootstrap_train.asyncio.to_thread",
            side_effect=fake_thread,
        ),
    ):
        status = await _train_bootstrap_symbol(orch, "OTC_SPC")
    assert status == "fail"


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_stops_on_export_fail(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            side_effect=[["OTC_SPC", "R_50"], ["R_50"], []],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            new_callable=AsyncMock,
            return_value="fail",
        ) as mock_train,
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.runtime_in_training",
            return_value=True,
        ),
    ):
        await run_initial_bootstrap_training(orch)
    assert mock_train.await_count == 2
