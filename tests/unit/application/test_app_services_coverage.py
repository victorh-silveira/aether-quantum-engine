"""Testes unitarios para fechar 100% de cobertura em servicos de aplicacao."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services import infra_timing_config as itc
from src.application.services.log_dedupe import (
    LogDeduper,
    clear_log_channel,
    log_debug_if_changed,
    log_info_if_changed,
    log_warning_if_changed,
)
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from src.application.services.orchestrator.metrics_utils import neutral_metrics
from src.application.services.orchestrator.orchestrator_atomic_state import (
    _caller_function_name,
    orchestrator_atomic_state_context,
    orchestrator_balance_snapshot,
)
from src.application.services.orchestrator.session_persistence_barrier import (
    consume_linear_reset_flag,
    linear_reset_occurred,
)
from src.application.services.orchestrator.session_target_bootstrap import (
    bootstrap_active_session_targets,
    clear_current_session_redis_keys,
    restore_current_session_targets,
)
from src.application.services.settle_log_dedupe import log_settle_info_if_changed


def test_infra_timing_config_overrides_and_invalid_settings():
    with patch("src.application.services.infra_timing_config._load_settings", return_value=["bad"]):
        assert isinstance(itc._load_settings(), list)
    with patch("pathlib.Path.open", create=True) as m_open:
        m_open.return_value.__enter__.return_value.read.return_value = "[]"
        with (
            patch("src.application.services.infra_timing_config.json.load", return_value=["bad"]),
            pytest.raises(ValueError, match="settings.json invalido"),
        ):
            itc._load_settings()

    sr = itc.resolve_stream_reconnect_config({"stream_reconnect": {"max_attempts": 5}})
    assert sr["max_attempts"] == 5
    sr_flat = itc.resolve_stream_reconnect_config({"max_attempts": 7})
    assert sr_flat["max_attempts"] == 7

    hf = itc.resolve_history_fetch_config({"history_fetch": {"chunk": 500}})
    assert hf["chunk"] == 500
    hf_flat = itc.resolve_history_fetch_config({"chunk": 250})
    assert hf_flat["chunk"] == 250


def test_log_dedupe_and_settle_log_dedupe():
    owner = SimpleNamespace()
    logger = MagicMock(spec=logging.Logger)
    logger.quiet_channels = set()

    assert clear_log_channel(owner, "missing") is None
    log_info_if_changed(owner, logger, "ch", "v1", "msg %s", 1)
    log_info_if_changed(owner, logger, "ch", "v1", "msg %s", 1)
    log_debug_if_changed(owner, logger, "ch2", "v1", "dbg %s", 2)
    log_warning_if_changed(owner, logger, "ch3", "v1", "warn %s", 3)
    assert clear_log_channel(owner, "ch") == "v1"

    deduper = LogDeduper(owner)
    deduper.log_quality_guard_cycle_minute(logger, cycle_id=1, minute_bucket="12:00", message="qg")
    deduper.log_quality_starvation_escape(logger, skipped_cycles=2, min_margin=0.01)

    log_settle_info_if_changed(owner, logger, "SETTLE_WIN", "sw", "c1", "ok")
    assert logger.info.call_count >= 3


def test_config_symbols_metrics_and_barrier():
    anchor, syms = normalize_symbols_and_anchor({"anchor": "R_10", "symbols": ["R_25"]})
    assert anchor == "R_10" and syms == ["R_10", "R_25"]
    assert neutral_metrics()["direction"] == "WAIT"
    assert linear_reset_occurred(None) is False
    assert consume_linear_reset_flag(None) is False


@pytest.mark.asyncio
async def test_session_target_bootstrap_and_atomic_state():
    orch = SimpleNamespace(
        _session_targets_bootstrapped=True,
        state_mgr=SimpleNamespace(state=SimpleNamespace(initial_balance=0.0), read_cached_balance=lambda: 0.0),
        risk_manager=SimpleNamespace(initial_bankroll=0.0, reset_session=MagicMock()),
        state_store=SimpleNamespace(get_string=AsyncMock(return_value="100.0"), delete_key=AsyncMock()),
        state=SimpleNamespace(balance=250.0),
        logger=MagicMock(),
    )
    await bootstrap_active_session_targets(orch, 100.0)
    await restore_current_session_targets(orch)
    assert orch.risk_manager.initial_bankroll == 100.0

    orch.state_store.get_string = AsyncMock(return_value=None)
    await restore_current_session_targets(orch)
    orch.state_store.get_string = AsyncMock(return_value="bad-float")
    await restore_current_session_targets(orch)
    orch._session_targets_bootstrapped = False
    await restore_current_session_targets(orch)
    orch.state_store = None
    await clear_current_session_redis_keys(orch)
    orch.state_store = SimpleNamespace(delete_key=AsyncMock())
    await clear_current_session_redis_keys(orch)

    async with orchestrator_atomic_state_context(orch):
        assert orchestrator_balance_snapshot(orch) == 250.0
    assert _caller_function_name() != ""
    with patch(
        "src.application.services.orchestrator.orchestrator_atomic_state.inspect.currentframe", return_value=None
    ):
        assert _caller_function_name() == "<unknown>"
    with patch("src.application.services.orchestrator.orchestrator_atomic_state.inspect.currentframe") as mf:
        mf.return_value = SimpleNamespace(f_back=None)
        assert _caller_function_name() == "<unknown>"


@pytest.mark.asyncio
async def test_engine_mode_persistence_reconnect_and_cycle_entry_edges():
    from src.application.services.orchestrator.engine_mode import training_enabled
    from src.application.services.orchestrator.orchestrator_persistence import persist_full_state_unlocked
    from src.application.services.orchestrator.reconnect_cycle_release import release_trading_cycle_after_reconnect
    from src.application.services.orchestrator.trading_cycle_entry import (
        _closed_m5_candles,
        run_trading_cycle_if_ready,
    )
    from src.application.services.orchestrator.trading_cycle_entry_guards import (
        check_and_clear_boundary_cooldown,
        cycle_cadence_seconds,
        mark_cycle_attempt_complete,
        trading_cycle_entry_allowed,
    )
    from src.application.services.quant_metrics_collector import QuantMetricsCollector

    assert training_enabled() is False
    qmc = QuantMetricsCollector()
    assert qmc.has_observations() is False

    # persist_full_state_unlocked sem save_state_bundle
    from src.infrastructure.state.state_manager import StateManager

    sm_inst = StateManager()
    orch_p = SimpleNamespace(
        state=SimpleNamespace(get_state=AsyncMock(return_value={}), balance=100.0),
        risk_manager=SimpleNamespace(total_session_profit=1.0, initial_bankroll=99.0, get_state=lambda: {}),
        state_store=SimpleNamespace(
            save_snapshot=AsyncMock(),
            set_hash=AsyncMock(),
            set_string=AsyncMock(),
        ),
        persistence=SimpleNamespace(save=MagicMock()),
        state_mgr=SimpleNamespace(mirror_balance=MagicMock(), state=sm_inst.state),
        stream=SimpleNamespace(candles={}),
        anchor="R_10",
        symbols=["R_10"],
        config={"stream": {"micro_granularity": 300}},
        _last_evaluated_boundary_epoch=0,
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_persistence.get_data_state_signature", return_value="sig"
    ):
        await persist_full_state_unlocked(orch_p)
    orch_p.state_store.save_snapshot.assert_awaited_once()

    # reconnect_cycle_release
    orch_r = SimpleNamespace(config={}, logger=MagicMock())
    release_trading_cycle_after_reconnect(orch_r)
    assert orch_r.last_data_signature == ""

    # trading_cycle_entry_guards
    assert cycle_cadence_seconds(orch_p) > 0
    mark_cycle_attempt_complete(orch_p)
    orch_p._cooldown_until = 1.0
    assert check_and_clear_boundary_cooldown(orch_p) is True
    assert check_and_clear_boundary_cooldown(orch_p) is False
    assert trading_cycle_entry_allowed(SimpleNamespace(running=False)) is False
    assert trading_cycle_entry_allowed(SimpleNamespace(running=True, _reconciliation_pending=True)) is False
    assert (
        trading_cycle_entry_allowed(SimpleNamespace(running=True, state=SimpleNamespace(active_contracts={1: {}})))
        is False
    )

    # _closed_m5_candles & _execute_laya_cycle edges
    orch_c = SimpleNamespace(
        anchor="R_10",
        _active_cycle_id=1,
        _cycle_seq=0,
        _last_cluster_cycle_end=0.0,
        _last_evaluated_boundary_epoch=0,
        last_data_signature="",
        is_trading=False,
        running=True,
        _reconciliation_pending=False,
        _api_maintenance_until=0.0,
        _session_persistence_write_active=False,
        config={"stream": {"micro_granularity": 300}},
        stream=SimpleNamespace(get_closed_candles=lambda s, g: [], is_synchronized=False),
        ws=SimpleNamespace(is_running=False),
        state=SimpleNamespace(balance=0.0, active_contracts={}),
        laya=SimpleNamespace(decide=AsyncMock()),
        executor=SimpleNamespace(execute_laya_order=AsyncMock()),
        logger=MagicMock(),
    )
    assert _closed_m5_candles(orch_c, "R_10") == []
    with pytest.raises(AttributeError):
        _closed_m5_candles(SimpleNamespace(stream=SimpleNamespace()), "R_10")
    with (
        patch("src.application.services.orchestrator.trading_cycle_entry.process_redis_settlement_queue", AsyncMock()),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.trading_cycle_entry_allowed", return_value=True
        ),
    ):
        assert await run_trading_cycle_if_ready(orch_c) is False
