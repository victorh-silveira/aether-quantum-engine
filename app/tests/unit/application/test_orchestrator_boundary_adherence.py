from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.execution_quality_skip_yield import quality_skip_yield_seconds
from src.application.services.orchestrator.orchestrator_data_signature import (
    resolve_signature_boundary_seconds,
    seconds_until_next_signature_boundary,
)
from src.application.services.orchestrator.trading_cycle_entry import (
    run_trading_cycle_if_ready,
    trading_cycle_entry_allowed,
)
from src.application.services.orchestrator.trading_cycle_entry_guards import _cycle_cadence_elapsed


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"
GUARDS_MODULE = "src.application.services.orchestrator.trading_cycle_entry_guards"


def test_resolve_signature_boundary_seconds_falls_back_to_cycle_interval(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch.config["orchestrator"].pop("signature_boundary_seconds", None)
    assert resolve_signature_boundary_seconds(orch) == 180


def test_seconds_until_next_signature_boundary_with_180s_cadence(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    delay = seconds_until_next_signature_boundary(orch, now=1000.0)
    assert delay == pytest.approx(80.0)


def test_trading_cycle_entry_blocked_between_macro_boundaries(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch._last_cluster_cycle_end = 1000.0
    with patch(f"{GUARDS_MODULE}.time.time", return_value=1060.0):
        assert trading_cycle_entry_allowed(orch) is False


def test_trading_cycle_entry_allowed_after_macro_cadence_elapsed(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch._last_cluster_cycle_end = 1000.0
    with patch(f"{GUARDS_MODULE}.time.time", return_value=1185.0):
        assert trading_cycle_entry_allowed(orch) is True


@pytest.mark.asyncio
async def test_no_inference_on_intermediate_60s_ticks_under_180s_cadence(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch._last_cluster_cycle_end = 1000.0
    orch._last_epoch = 1060
    orch.stream.is_synchronized = True
    orch.ws.is_running = True
    orch.state.active_contracts = {}
    with (
        patch(f"{GUARDS_MODULE}.time.time", return_value=1060.0),
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.process_redis_settlement_queue", new_callable=AsyncMock),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is False
    collect_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_inference_runs_only_after_full_cadence_window(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch._last_cluster_cycle_end = 1000.0
    orch._last_epoch = 1180
    orch.stream.is_synchronized = True
    orch.ws.is_running = True
    orch.state.active_contracts = {}
    decisions = {"RDBULL": {"metrics": {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.08}}}
    with (
        patch(f"{GUARDS_MODULE}.time.time", return_value=1185.0),
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=decisions,
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.process_redis_settlement_queue", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.trading_cycle_warm_up_suspended", return_value=None),
    ):
        orch.executor.execute_cluster = AsyncMock()
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_awaited_once()


def test_cycle_cadence_elapsed_false_when_cadence_disabled(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch._last_cluster_cycle_end = 1000.0
    assert _cycle_cadence_elapsed(orch) is False


def test_quality_skip_yield_always_zero(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 180
    orch._last_cluster_cycle_end = 820.0
    assert quality_skip_yield_seconds(orch) == 0.0


def test_resolve_signature_boundary_seconds_invalid_cadence_fallback(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["signature_boundary_seconds"] = "bad"
    orch.config["orchestrator"]["cycle_interval_seconds"] = "bad"
    assert resolve_signature_boundary_seconds(orch) == 60
