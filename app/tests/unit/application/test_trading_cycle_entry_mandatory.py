from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"
GUARDS_MODULE = "src.application.services.orchestrator.trading_cycle_entry_guards"


@pytest.mark.asyncio
async def test_trading_cycle_logs_quality_guard_and_executes_in_mandatory_mode(orch_ready, caplog):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    orch.executor.execute_cluster = AsyncMock()
    weak_decisions = {
        "R_10": {
            "metrics": {
                "calibrated_prob": 0.55,
            }
        },
    }
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        caplog.at_level("INFO", logger="AETH"),
    ):
        await run_trading_cycle_if_ready(orch)
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert not guard_logs
    orch.executor.execute_cluster.assert_awaited_once()
    assert orch._last_cycle_cluster_executed is True


@pytest.mark.asyncio
async def test_trading_cycle_marks_cadence_after_mandatory_quality_telemetry(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 60
    weak_decisions = {"R_10": {"metrics": {"calibrated_prob": 0.55}}}
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        patch(f"{GUARDS_MODULE}.time.time", return_value=100.0),
    ):
        await run_trading_cycle_if_ready(orch)
    assert orch._last_cluster_cycle_end == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_trading_cycle_advances_epoch_when_mandatory_executes_weak_signal(orch_ready):
    orch = orch_ready
    orch._last_epoch = 500
    orch._last_processed_epoch = 0
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    weak_decisions = {"R_10": {"metrics": {"calibrated_prob": 0.55, "deploy_ok": True}}}
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock) as mark_mock,
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
    ):
        await run_trading_cycle_if_ready(orch)
    mark_mock.assert_awaited_once()
    assert orch._last_processed_epoch == 500


@pytest.mark.asyncio
async def test_trading_cycle_skips_epoch_advance_when_non_mandatory_quality_suspends(orch_ready):
    orch = orch_ready
    orch._last_epoch = 500
    orch._last_processed_epoch = 0
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    weak_decisions = {"R_10": {"metrics": {"calibrated_prob": 0.55}}}
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock) as mark_mock,
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await run_trading_cycle_if_ready(orch)
    mark_mock.assert_awaited_once()
    assert orch._last_processed_epoch == 500


@pytest.mark.asyncio
async def test_trading_cycle_executes_cluster_on_mandatory(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    weak_decisions = {"R_10": {"metrics": {"calibrated_prob": 0.55, "deploy_ok": True}}}
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
    ):
        await run_trading_cycle_if_ready(orch)
    orch.executor.execute_cluster.assert_awaited_once()


def test_check_and_clear_boundary_cooldown_resets_on_new_boundary(orch_ready):
    from src.application.services.orchestrator.trading_cycle_entry_guards import (
        check_and_clear_boundary_cooldown,
        trading_cycle_entry_allowed,
    )

    orch = orch_ready
    orch.config["pullback_strategy"] = {"enabled": True, "entry_window_seconds": 60.0}
    orch._cooldown_until = 1700001000.0
    orch._last_m5_cycle_boundary = 1700000000

    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.m5_boundary_epoch",
            return_value=1700000900,
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.at_m5_open_window",
            return_value=True,
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry_guards.time.time", return_value=1700000910.0),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards._require_signature_boundary",
            return_value=False,
        ),
    ):
        cleared = check_and_clear_boundary_cooldown(orch)
        assert cleared is True
        assert orch._cooldown_until == 0.0
        assert trading_cycle_entry_allowed(orch) is True


def test_check_and_clear_boundary_cooldown_keeps_cooldown_when_same_boundary(orch_ready):
    from src.application.services.orchestrator.trading_cycle_entry_guards import (
        check_and_clear_boundary_cooldown,
        trading_cycle_entry_allowed,
    )

    orch = orch_ready
    orch.config["pullback_strategy"] = {"enabled": True, "entry_window_seconds": 60.0}
    orch._cooldown_until = 1700001000.0
    orch._last_m5_cycle_boundary = 1700000900

    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.m5_boundary_epoch",
            return_value=1700000900,
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.at_m5_open_window",
            return_value=True,
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry_guards.time.time", return_value=1700000910.0),
    ):
        cleared = check_and_clear_boundary_cooldown(orch)
        assert cleared is False
        assert orch._cooldown_until == 1700001000.0
        assert trading_cycle_entry_allowed(orch) is False


@pytest.mark.asyncio
async def test_trading_cycle_updates_m5_boundary_when_setup_active_and_open_window(orch_ready):
    orch = orch_ready
    orch._last_epoch = 1700000910
    orch._last_processed_epoch = 0
    orch._last_m5_cycle_boundary = 0
    orch.config["pullback_strategy"] = {"enabled": True, "entry_window_seconds": 60.0}
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.at_m5_open_window", return_value=True),
        patch(f"{TRADING_CYCLE_MODULE}.m5_boundary_epoch", return_value=1700000900),
        patch(f"{TRADING_CYCLE_MODULE}.seconds_until_next_signature_boundary", return_value=3.0),
    ):
        await run_trading_cycle_if_ready(orch)
    assert orch._last_m5_cycle_boundary == 1700000900
    assert orch._last_processed_epoch == 1700000910
