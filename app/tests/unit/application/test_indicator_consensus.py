"""Testes do consenso entre indicador publico e motor automatico."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services import indicator_consensus as consensus
from src.application.services.deep_learning.dl_bridge_results import collect_symbol_results
from src.application.services.orchestrator.config_symbols import (
    approved_indicator_symbols,
    normalize_symbols_and_anchor,
    resolve_dl_train_symbols,
)
from src.domain.models.indicator_consensus import ApprovedSignal, agrees_with_execution, approved_signal_from_row
from src.domain.models.trade import TradeDirection
from src.infrastructure.storage import indicator_signal_reader as reader


def _row(**changes):
    row = {
        "symbol": "R_10",
        "side": "CALL",
        "reason": "ok",
        "probability": 0.64,
        "model_version": "v1",
        "candle_epoch": 300,
        "model_active": True,
    }
    return {**row, **changes}


@pytest.mark.parametrize(
    "change",
    [
        {"reason": "experimental"},
        {"model_active": False},
        {"side": "SEM SINAL"},
        {"candle_epoch": 0},
        {"model_version": ""},
        {"probability": None},
        {"probability": float("nan")},
        {"probability": 1.1},
    ],
)
def test_signal_requires_approved_current_version_and_candle(change):
    assert approved_signal_from_row(_row(**change), expected_epoch=300) is None


def test_signal_matching_requires_same_symbol_side_and_epoch():
    signal = approved_signal_from_row(_row(), expected_epoch=300)
    assert signal == ApprovedSignal("R_10", "CALL", 300, 0.64, "v1")
    assert agrees_with_execution(signal, "R_10", "CALL", 300)
    assert not agrees_with_execution(signal, "R_25", "CALL", 300)
    assert not agrees_with_execution(signal, "R_10", "PUT", 300)
    assert not agrees_with_execution(signal, "R_10", "CALL", 600)
    assert not agrees_with_execution(None, "R_10", "CALL", 300)


@pytest.mark.asyncio
async def test_reader_uses_latest_rows_and_closes_connection(monkeypatch):
    connection = SimpleNamespace(fetch=AsyncMock(return_value=[_row(), _row(symbol="R_25", reason="experimental")]))
    connection.close = AsyncMock()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(reader.asyncpg, "connect", connect)
    signals = await reader.read_approved_m5_signals("dsn", ["R_10", "R_25"], expected_epoch=300)
    assert list(signals) == ["R_10"]
    assert "DISTINCT ON (s.symbol)" in connection.fetch.await_args.args[0]
    assert connection.fetch.await_args.args[1] == ["R_10", "R_25"]
    connection.close.assert_awaited_once()
    assert await reader.read_approved_m5_signals("dsn", [], expected_epoch=300) == {}


@pytest.mark.asyncio
async def test_reader_closes_connection_on_query_failure(monkeypatch):
    connection = SimpleNamespace(fetch=AsyncMock(side_effect=RuntimeError("banco")), close=AsyncMock())
    monkeypatch.setattr(reader.asyncpg, "connect", AsyncMock(return_value=connection))
    with pytest.raises(RuntimeError, match="banco"):
        await reader.read_approved_m5_signals("dsn", ["R_10"], expected_epoch=300)
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cycle_does_not_reuse_old_signal_after_database_failure(monkeypatch):
    orch = SimpleNamespace(
        config={"indicator_consensus": {"enabled": True}},
        symbols=["R_10"],
        _indicator_signals={"R_10": ApprovedSignal("R_10", "CALL", 0, 0.6, "old")},
    )
    monkeypatch.setattr(consensus.time, "time", lambda: 620)
    fetch = AsyncMock(side_effect=[{"R_10": ApprovedSignal("R_10", "CALL", 300, 0.64, "v1")}, OSError("off")])
    monkeypatch.setattr(consensus, "read_approved_m5_signals", fetch)
    await consensus.refresh_indicator_consensus(orch)
    assert orch._indicator_epoch == 300
    metrics = {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}
    assert consensus.candidate_agrees_with_indicator(orch, "R_10", "CALL", metrics)
    assert metrics["indicator_model_version"] == "v1"
    assert not consensus.candidate_agrees_with_indicator(orch, "R_10", "PUT", metrics)
    assert not consensus.candidate_agrees_with_indicator(orch, "R_25", "CALL", metrics)
    assert not consensus.candidate_agrees_with_indicator(orch, "R_10", "CALL", {"model_input_candle_epoch": 0})
    assert not consensus.candidate_agrees_with_indicator(
        orch, "R_10", "CALL", {"model_input_candle_epoch": 300, "tcn_direction": "PUT"}
    )
    assert not consensus.candidate_agrees_with_indicator(
        orch, "R_10", "CALL", {"model_input_candle_epoch": 0, "tcn_direction": "CALL"}
    )
    await consensus.refresh_indicator_consensus(orch)
    assert orch._indicator_signals == {}
    assert not consensus.candidate_agrees_with_indicator(orch, "R_10", "CALL", metrics)
    orch.config = {}
    assert not consensus.consensus_enabled(orch.config)
    await consensus.refresh_indicator_consensus(orch)
    assert consensus.candidate_agrees_with_indicator(orch, "R_10", "PUT", {})


def test_auto_symbols_only_accepts_approved_m5(tmp_path: Path, monkeypatch):
    models = tmp_path / "indicator/data/indicator-models"
    models.mkdir(parents=True)
    approved = {
        "symbol": "R_10",
        "period": 300,
        "status": "aprovado",
        "validation_accuracy": 0.55,
        "baseline_accuracy": 0.5,
        "validation_brier": 0.24,
        "baseline_brier": 0.25,
    }
    (models / "R_10_300.json").write_text(json.dumps(approved))
    (models / "R_25_300.json").write_text(json.dumps({**approved, "symbol": "R_25", "status": "experimental"}))
    (models / "FOREX_300.json").write_text(json.dumps({**approved, "symbol": "FOREX"}))
    (models / "1HZ25V_300.json").write_text("invalid")
    assert approved_indicator_symbols(models) == ["R_10"]
    from src.application.services.orchestrator import config_symbols

    monkeypatch.setattr(config_symbols, "_REPOSITORY_ROOT", tmp_path)
    config = {"anchor": "1HZ75V", "symbols": ["1HZ75V"], "indicator_consensus": {"enabled": True, "auto_symbols": True}}
    assert normalize_symbols_and_anchor(config) == ("1HZ75V", ["1HZ75V", "R_10"])
    assert resolve_dl_train_symbols(config) == ["1HZ75V", "R_10"]


def test_consensus_filters_conflicting_and_duplicate_symbols():
    orch = SimpleNamespace(
        config={"indicator_consensus": {"enabled": True}, "orchestrator": {"execution": {"multi_symbol_orders": True}}},
        state=SimpleNamespace(active_contracts={7: object()}),
        risk_manager=SimpleNamespace(contract_to_symbol={7: "R_25"}),
        _indicator_epoch=300,
        _indicator_signals={
            "R_10": ApprovedSignal("R_10", "CALL", 300, 0.64, "v1"),
            "R_25": ApprovedSignal("R_25", "PUT", 300, 0.63, "v2"),
        },
    )
    candidates = [
        ("R_10", TradeDirection.CALL, {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}),
        ("R_10", TradeDirection.CALL, {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}),
        ("R_25", TradeDirection.PUT, {"model_input_candle_epoch": 300, "tcn_direction": "PUT"}),
        ("R_30", TradeDirection.CALL, {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}),
    ]
    assert [item[0] for item in consensus.filter_indicator_candidates(orch, candidates)] == ["R_10"]
    assert consensus.multi_symbol_orders_enabled(orch, recovery_active=False)
    assert not consensus.multi_symbol_orders_enabled(orch, recovery_active=True)
    orch.config = {}
    assert consensus.filter_indicator_candidates(orch, candidates) == candidates
    assert not consensus.multi_symbol_orders_enabled(orch, recovery_active=False)


def test_cluster_budget_reserves_confirmed_stakes_without_overdraft():
    assert consensus.available_cluster_bankroll(100.0, 40.0, 95.0, multi_symbol=True) == 60.0
    assert consensus.available_cluster_bankroll(100.0, 40.0, 45.0, multi_symbol=True) == 45.0
    assert consensus.available_cluster_bankroll(100.0, 110.0, 45.0, multi_symbol=True) == 0.0
    assert consensus.available_cluster_bankroll(100.0, 110.0, 45.0, multi_symbol=False) == 100.0


def test_signal_epoch_is_rechecked_before_purchase(monkeypatch):
    monkeypatch.setattr(consensus.time, "time", lambda: 620)
    assert consensus.signal_epoch_current({"indicator_consensus": {"enabled": True}}, {"indicator_candle_epoch": 300})
    assert not consensus.signal_epoch_current({"indicator_consensus": {"enabled": True}}, {"indicator_candle_epoch": 0})
    monkeypatch.setattr(consensus.time, "time", lambda: 900)
    assert not consensus.signal_epoch_current(
        {"indicator_consensus": {"enabled": True}}, {"indicator_candle_epoch": 300}
    )
    assert consensus.signal_epoch_current({}, {})


def test_failed_symbol_does_not_discard_other_dl_decisions():
    good = {"decision": "CALL"}
    decisions, trained, reason = collect_symbol_results(
        ["R_10", "R_25", "R_30"],
        [(good, "trained"), RuntimeError("offline"), ({"decision": "PUT"}, "")],
        logging.getLogger("test"),
    )
    assert decisions["R_10"] == good
    assert decisions["R_25"]["direction"] is None
    assert decisions["R_25"]["metrics"]["gate_reason"] == "data"
    assert decisions["R_30"]["decision"] == "PUT"
    assert trained == ["R_10"]
    assert reason == "trained"
    with pytest.raises(asyncio.CancelledError):
        collect_symbol_results(["R_10"], [asyncio.CancelledError()], logging.getLogger("test"))


def test_first_symbol_stake_reason_does_not_veto_other_indices():
    from src.application.services.orchestrator.execution_manager import ExecutionManager

    risk = MagicMock()
    risk.kelly_config = {}
    risk.stake_block_reason.return_value = "weak_model"
    orch = SimpleNamespace(
        config={
            "indicator_consensus": {"enabled": True},
            "orchestrator": {"execution": {"multi_symbol_orders": True}},
            "deep_learning": {},
        },
        risk_manager=risk,
        _active_cycle_id=1,
    )
    manager = ExecutionManager(orch)
    orders = [("R_10", TradeDirection.CALL, {}), ("R_25", TradeDirection.PUT, {})]
    assert manager._cluster_stake_block(orders, 100.0) is None
    risk.stake_block_reason.return_value = "stop_win"
    assert manager._cluster_stake_block(orders, 100.0) == "stop_win"


def test_pending_training_on_one_index_keeps_other_indices_eligible():
    from src.application.services.orchestrator.execution_manager import ExecutionManager

    orch = SimpleNamespace(
        config={"indicator_consensus": {"enabled": True}},
        _active_cycle_id=1,
        _dl_training_symbols={"R_25"},
        _dl_training_phase=False,
    )
    assert ExecutionManager(orch)._training_phase_gate() is False
    assert orch._dl_training_phase is True


def test_collects_two_approved_indices_in_same_cluster(monkeypatch):
    from src.application.services.orchestrator import execution_collect as collect

    orch = SimpleNamespace(
        config={"indicator_consensus": {"enabled": True}, "orchestrator": {"execution": {"multi_symbol_orders": True}}},
        state=SimpleNamespace(active_contracts={}),
        risk_manager=SimpleNamespace(
            pending_loss={}, consecutive_losses_linear=0, total_session_profit=0.0, contract_to_symbol={}
        ),
        _active_cycle_id=1,
        _indicator_epoch=300,
        _indicator_signals={
            "R_10": ApprovedSignal("R_10", "CALL", 300, 0.64, "v1"),
            "R_25": ApprovedSignal("R_25", "PUT", 300, 0.60, "v2"),
            "R_30": ApprovedSignal("R_30", "PUT", 300, 0.62, "v3"),
        },
    )
    candidates = [
        ("R_25", TradeDirection.PUT, {"model_input_candle_epoch": 300, "tcn_direction": "PUT"}),
        ("R_10", TradeDirection.CALL, {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}),
        ("R_30", TradeDirection.CALL, {"model_input_candle_epoch": 300, "tcn_direction": "CALL"}),
    ]
    monkeypatch.setattr(collect, "pending_recovery_active", lambda *_: False)
    monkeypatch.setattr(
        collect,
        "extract_collect_params",
        lambda *_args, **_kwargs: ({}, frozenset(), 0.0, 0.0, 0.0, None, orch.config["orchestrator"]["execution"]),
    )
    monkeypatch.setattr(collect, "gather_cluster_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(collect, "filter_loss_protection_candidates", lambda values, **_kwargs: values)
    monkeypatch.setattr(collect, "filter_recovery_hurst_candidates", lambda values, **_kwargs: values)
    monkeypatch.setattr(collect, "mandatory_fallback_if_empty", lambda _manager, _decisions, values, **_kwargs: values)
    monkeypatch.setattr(collect, "apply_cointegration_redirect", lambda values, _risk: values)
    manager = SimpleNamespace(orch=orch, _mandatory_trade_each_cycle=lambda: False)
    orders = collect.collect_cluster_orders(manager, {})
    assert [item[0] for item in orders] == ["R_10", "R_25"]
    assert orders[0][2]["indicator_model_version"] == "v1"
    assert orders[1][2]["indicator_model_version"] == "v2"
    orch._indicator_signals = {}
    assert collect.collect_cluster_orders(manager, {}) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("expire_second", [False, True])
async def test_two_indices_purchase_sequentially_with_reserved_balance(monkeypatch, expire_second):
    from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders

    monkeypatch.setattr(consensus.time, "time", lambda: 620)
    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    risk.active_contract_ids = []
    risk.calculate_stake.return_value = 60.0
    state = SimpleNamespace(balance=100.0, add_contract=AsyncMock())
    orch = SimpleNamespace(
        config={
            "indicator_consensus": {"enabled": True},
            "orchestrator": {"execution": {"multi_symbol_orders": True}},
            "deep_learning": {},
            "risk_management": {"params": {"stake_min": 1.0, "duration": 300}},
        },
        _active_cycle_id=8,
        _contract_cycle={},
        risk_manager=risk,
        state=state,
        business_metrics=None,
        market_writer=None,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(
        side_effect=[SimpleNamespace(contract_id=1, buy_price=60.0), SimpleNamespace(contract_id=2, buy_price=40.0)]
    )
    orders = [
        ("R_10", TradeDirection.CALL, {"indicator_candle_epoch": 300}),
        ("R_25", TradeDirection.PUT, {"indicator_candle_epoch": 300}),
    ]
    with patch("src.application.services.orchestrator.execution_manager_execute.store_contract_audit"):
        if expire_second:
            with patch(
                "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current",
                side_effect=[True, True, False],
            ):
                count = await execute_cluster_orders(executor, orders, 0.0, 100.0)
        else:
            count = await execute_cluster_orders(executor, orders, 0.0, 100.0)

    assert count == (1 if expire_second else 2)
    assert [call.args[0] for call in executor._place_order.await_args_list] == (
        ["R_10"] if expire_second else ["R_10", "R_25"]
    )
    assert [call.args[2] for call in executor._place_order.await_args_list] == (
        [60.0] if expire_second else [60.0, 40.0]
    )
    assert risk.calculate_stake.call_args_list[1].args[0] == 40.0
    assert risk.active_contract_ids == ([1] if expire_second else [1, 2])
    assert orch._contract_cycle == ({1: 8} if expire_second else {1: 8, 2: 8})


@pytest.mark.asyncio
async def test_expired_signal_never_reaches_stake_calculation():
    from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders

    risk = MagicMock()
    orch = SimpleNamespace(
        config={"indicator_consensus": {"enabled": True}, "orchestrator": {"execution": {}}},
        risk_manager=risk,
    )
    executor = SimpleNamespace(orch=orch)
    with patch(
        "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=False
    ):
        assert await execute_cluster_orders(executor, [("R_10", TradeDirection.CALL, {})], 0.0, 100.0) == 0
    risk.calculate_stake.assert_not_called()


@pytest.mark.asyncio
async def test_multisymbol_order_below_broker_minimum_is_not_sent():
    from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders

    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    risk.calculate_stake.return_value = 0.25
    orch = SimpleNamespace(
        config={
            "indicator_consensus": {"enabled": True},
            "orchestrator": {"execution": {"multi_symbol_orders": True}},
            "deep_learning": {},
            "risk_management": {"params": {"stake_min": 1.0}},
        },
        risk_manager=risk,
        state=SimpleNamespace(balance=100.0),
        _active_cycle_id=1,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    with patch(
        "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
    ):
        assert await execute_cluster_orders(executor, [("R_10", TradeDirection.CALL, {})], 0.0, 100.0) == 0
    executor._place_order.assert_not_called()


@pytest.mark.asyncio
async def test_confirmed_purchase_is_counted_when_metrics_fail():
    from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders

    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    risk.active_contract_ids = []
    risk.calculate_stake.return_value = 10.0
    metrics = MagicMock()
    metrics.update_active_contracts_count.side_effect = RuntimeError("telemetry offline")
    orch = SimpleNamespace(
        config={"deep_learning": {}, "risk_management": {"params": {"duration": 300}}},
        _active_cycle_id=9,
        _contract_cycle={},
        risk_manager=risk,
        state=SimpleNamespace(balance=100.0, add_contract=AsyncMock()),
        business_metrics=metrics,
        market_writer=None,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(return_value=SimpleNamespace(contract_id=3, buy_price=10.0))
    with patch("src.application.services.orchestrator.execution_manager_execute.store_contract_audit"):
        count = await execute_cluster_orders(executor, [("R_10", TradeDirection.CALL, {})], 0.0, 100.0)
    assert count == 1
    assert risk.active_contract_ids == [3]
    executor.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_settlement_reconciles_both_active_index_contracts():
    from src.application.services.orchestrator import execution_settlement

    orch = SimpleNamespace(
        ws=SimpleNamespace(is_running=True),
        state=SimpleNamespace(active_contracts={1: object(), 2: object()}),
    )
    executor = SimpleNamespace(orch=orch)
    with (
        patch.object(execution_settlement, "reconcile_single_contract", new_callable=AsyncMock) as reconcile,
        patch.object(execution_settlement.asyncio, "sleep", new_callable=AsyncMock),
    ):
        assert await execution_settlement.reconcile_contracts(executor)
    assert [call.args[1] for call in reconcile.await_args_list] == [1, 2]
