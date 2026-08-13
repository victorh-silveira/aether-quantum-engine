from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bootstrap_train import (
    _bootstrap_training_context,
    _train_deploy_attempts,
    run_dl_training_session,
)


def test_train_deploy_attempts_clamps_and_rejects_bad():
    assert _train_deploy_attempts({"train_deploy_retries": 3}) == 3
    assert _train_deploy_attempts({"train_deploy_retries": 99}) == 8
    assert _train_deploy_attempts({"train_deploy_retries": "x"}) == 3
    assert _train_deploy_attempts(None) == 3


def test_reseed_for_attempt_calls_cuda_when_available():
    from src.application.services.deep_learning.dl_bootstrap_train import _reseed_for_attempt

    with (
        patch("src.application.services.deep_learning.dl_bootstrap_train.torch.manual_seed") as seed,
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.torch.cuda.is_available",
            return_value=True,
        ),
        patch("src.application.services.deep_learning.dl_bootstrap_train.torch.cuda.manual_seed_all") as cuda_seed,
    ):
        _reseed_for_attempt(2)
    seed.assert_called_once()
    cuda_seed.assert_called_once()


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
        _dl, params, min_len, _g, _rt, prices, *_rest = _bootstrap_training_context(orch, "R_10")

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs.get("timeframe") == "micro"
    assert params["train_timeframe"] == "micro"
    assert min_len >= 2000
    assert len(prices) == 2000


@pytest.mark.asyncio
async def test_run_dl_training_session_sequences_all_symbols(orch_ready):
    orch = orch_ready
    trained: list[str] = []

    async def fake_train(_orch, symbol):
        trained.append(symbol)
        return "ok"

    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert trained == [str(s) for s in orch.symbols]


@pytest.mark.asyncio
async def test_run_dl_training_session_noop_without_symbols(orch_ready):
    orch = orch_ready
    orch.symbols = []
    orch.config.setdefault("deep_learning", {})["train_symbols"] = []
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        new_callable=AsyncMock,
    ) as mock_train:
        assert await run_dl_training_session(orch) is True
    mock_train.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_dl_training_session_waits_for_history(orch_ready):
    orch = orch_ready
    orch.symbols = ["R_10", "R_50"]
    orch.config.setdefault("deep_learning", {})["train_symbols"] = ["R_10", "R_50"]
    attempts = {"R_10": 0, "R_50": 0}

    async def fake_train(_orch, symbol):
        attempts[symbol] += 1
        return "ok" if attempts[symbol] >= 2 else "wait"

    orch.stream.ensure_cluster_history = AsyncMock()
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert attempts["R_10"] >= 2
    assert attempts["R_50"] >= 2
    orch.stream.ensure_cluster_history.assert_awaited()


@pytest.mark.asyncio
async def test_run_dl_training_session_stops_after_max_wait_rounds(orch_ready):
    orch = orch_ready
    orch.config.setdefault("deep_learning", {})["bootstrap_max_wait_rounds"] = 1
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        new_callable=AsyncMock,
        return_value="wait",
    ):
        assert await run_dl_training_session(orch) is False


@pytest.mark.asyncio
async def test_run_dl_training_session_skips_completed_symbols(orch_ready):
    orch = orch_ready
    orch.symbols = ["R_10", "R_50"]
    orch.config.setdefault("deep_learning", {})["train_symbols"] = ["R_10", "R_50"]
    calls: list[str] = []
    r25_attempts = {"n": 0}

    async def fake_train(_orch, symbol):
        calls.append(symbol)
        if symbol == "R_10":
            return "ok"
        r25_attempts["n"] += 1
        return "ok" if r25_attempts["n"] >= 2 else "wait"

    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert calls[0] == "R_10" and calls.count("R_10") == 1 and calls.count("R_50") >= 2
