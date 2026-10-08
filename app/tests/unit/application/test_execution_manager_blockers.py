import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.api_maintenance_guard import (
    orchestrator_api_maintenance_until,
)
from src.application.services.orchestrator.execution_orders import place_order
from src.domain.models.trade import TradeDirection


_API_MAINTENANCE_FALLBACK_SECONDS = float(resolve_orchestrator_timing_config()["api_maintenance_fallback_seconds"])


def test_candidate_block_reason_from_skip_reason_only():
    from src.application.services.orchestrator.execution_blockers import _candidate_block_reason

    assert _candidate_block_reason({"skip_reason": "doji"}) == "doji"
    assert _candidate_block_reason({"skip_reason": "neg_edge"}) == "neg_edge"
    assert (
        _candidate_block_reason({"skip_reason": "neg_edge", "cal_side_edge": 0.015, "min_edge_floor": 0.025})
        == "min_edge"
    )
    assert (
        _candidate_block_reason({"skip_reason": "neg_edge", "cal_side_edge": "bad", "min_edge_floor": 0.025})
        == "neg_edge"
    )
    assert _candidate_block_reason({"skip_reason": "chop_congestion"}) == "chop_congestion"
    assert _candidate_block_reason({"skip_reason": "wick_rejection_call"}) == "wick_rejection_call"
    assert _candidate_block_reason({"skip_reason": "wick_rejection_put"}) == "wick_rejection_put"
    assert _candidate_block_reason({"skip_reason": "exhaustion_call"}) == "exhaustion_call"
    assert _candidate_block_reason({"skip_reason": "exhaustion_put"}) == "exhaustion_put"
    assert _candidate_block_reason({"deploy_ok": False}) == "deploy"
    assert _candidate_block_reason({"execution_candidate_ready": True}) == "ready_not_selected"
    assert _candidate_block_reason({"signal_status": "SIGNAL_SUSPENDED"}) == "SIGNAL_SUSPENDED"
    assert _candidate_block_reason({"skip_reason": "not_a_gate"}) is None


def test_log_execution_blockers_force_trade_early_return():
    from types import SimpleNamespace

    from src.application.services.orchestrator.execution_blockers import log_execution_blockers

    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
        _active_cycle_id=1,
    )
    executor = SimpleNamespace(orch=orch)
    # deve retornar None imediatamente sem erro
    assert log_execution_blockers(executor, {}) is None


def test_log_execution_blockers_groups_training_symbols(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 6
        orch.symbols = ["R_10", "R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_10": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
                    "R_50": {
                        "direction": None,
                        "metrics": {"gate_reason": "direction_margin", "execute": False},
                    },
                },
            )
        calls = [str(c) for c in mock_info.call_args_list]
        assert any("DL_TREINO" in c and "R_10" in c for c in calls)
        assert not any("EXEC_NONE" in c for c in calls)


def test_log_execution_blockers_training_dedupe_and_completion(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        orch.symbols = ["R_10"]
        training = {"R_10": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}}
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor.logger, "debug") as mock_debug,
        ):
            orch.executor._log_execution_blockers(training)
            orch.executor._log_execution_blockers(training)
        assert sum("DL_TREINO" in str(c) for c in mock_info.call_args_list) == 1
        assert any("DL_TREINO" in str(c) for c in mock_debug.call_args_list)
        trained = {"R_10": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False}}}
        with patch.object(orch.executor.logger, "info") as mock_info_done:
            orch.executor._log_execution_blockers(trained)
        assert any("concluido" in str(c) for c in mock_info_done.call_args_list)


def test_log_execution_blockers_skips_symbol_without_decision_entry(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["R_10", "R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_10": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}},
            )
        calls = [str(c) for c in mock_info.call_args_list]
        assert any("EXEC_EMPTY" in c and "R_10:data" in c for c in calls)


def test_log_execution_blockers_reports_signal_skip_reasons(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 12
        orch.symbols = ["R_10", "R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_10": {
                        "direction": None,
                        "metrics": {
                            "gate_reason": "acc_floor",
                            "skip_reason": "acc_floor",
                            "signal_status": "SKIP:acc_floor",
                            "val_accuracy": 0.524,
                        },
                    },
                    "R_50": {
                        "direction": None,
                        "metrics": {
                            "gate_reason": "exec_vs_candle",
                            "skip_reason": "exec_vs_candle",
                            "tcn_direction": "PUT",
                            "closed_micro_candle_dir": "CALL",
                        },
                    },
                },
            )
        joined = " ".join(str(c) for c in mock_info.call_args_list)
        assert "R_10:acc_floor" in joined
        assert "R_50:exec_vs_candle" in joined
        assert "no_candidate" not in joined


def test_log_execution_blockers_recovery_empty_pool(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 11
        orch.risk_manager.consecutive_losses_linear = 2
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers({}, pending=38.56)
        calls = [str(c) for c in mock_info.call_args_list]
        assert any("EXEC_EMPTY" in c and "38.56" in c for c in calls)


@pytest.mark.asyncio
async def test_execute_orders_maintenance_error_schedules_api_hibernation(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        error = RuntimeError("Market is closed for maintenance")
        orch.executor._place_order = AsyncMock(side_effect=error)
        loop_start = asyncio.get_running_loop().time()
        with patch.object(orch.executor.logger, "warning") as mock_warn:
            count = await orch.executor._execute_orders(
                [("R_10", TradeDirection.CALL, {"conviction": 0.8})],
                0.0,
                100.0,
            )
        assert count == 0
        mock_warn.assert_called_once()
        assert orch._api_maintenance_until - loop_start == pytest.approx(
            _API_MAINTENANCE_FALLBACK_SECONDS,
            abs=0.05,
        )


@pytest.mark.asyncio
async def test_execute_orders_generic_closed_error_emits_warning_without_hibernation(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 9
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        orch.executor._place_order = AsyncMock(side_effect=RuntimeError("Trading session closed"))
        with patch.object(orch.executor.logger, "warning") as mock_warn:
            count = await orch.executor._execute_orders(
                [("R_10", TradeDirection.CALL, {"conviction": 0.8})],
                0.0,
                100.0,
            )
        assert count == 0
        mock_warn.assert_called_once()
        assert orchestrator_api_maintenance_until(orch) == 0.0


@pytest.mark.asyncio
async def test_place_order_maintenance_error_schedules_hibernation_before_raise(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 8
        orch.trade_handler.buy_with_parameters = AsyncMock(
            side_effect=RuntimeError("Erro na proposta: Market is closed")
        )
        loop_start = asyncio.get_running_loop().time()
        with pytest.raises(RuntimeError):
            await place_order(orch.executor, "R_10", TradeDirection.CALL, 10.0)
        assert orch._api_maintenance_until == pytest.approx(loop_start + _API_MAINTENANCE_FALLBACK_SECONDS, abs=1.0)
