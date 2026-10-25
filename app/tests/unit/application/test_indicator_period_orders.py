"""Selecao M5 com periodos de pesquisa excluidos da compra."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator import indicator_period_orders as orders
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.execution_order_identity import set_indicator_duration
from src.domain.models.indicator_consensus import ApprovedSignal
from src.domain.models.trade import TradeDirection
from src.domain.risk.broker_ev_evidence import BrokerEvEvidence
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter


def _orch(mode="demo"):
    writer = TimescaleMarketWriter(dsn="postgresql://teste")
    return SimpleNamespace(
        config={
            "indicator_consensus": {"enabled": True, "multi_period_trading": True},
            "trading": {"mode": mode},
            "risk_management": {"params": {"stake_min": 1.0}},
            "orchestrator": {
                "execution": {"quote_probability_haircut": 0.0, "max_orders_per_cycle": 3, "multi_symbol_orders": True}
            },
        },
        state=SimpleNamespace(active_contracts={}, balance=1000.0),
        risk_manager=SimpleNamespace(contract_to_symbol={}),
        market_writer=writer,
        trade_handler=SimpleNamespace(fetch_proposal_snapshot=AsyncMock()),
        _indicator_period_signals={
            ("R_10", 900): ApprovedSignal("R_10", "CALL", 900, None, "pullback-1-v15", 900),
            ("R_10", 3600): ApprovedSignal("R_10", "PUT", 0, None, "pullback-1-v60", 3600),
            ("R_25", 900): ApprovedSignal("R_25", "PUT", 900, None, "pullback-1-v25", 900),
            ("R_10", 300): ApprovedSignal("R_10", "PUT", 900, None, "pullback-1-v5", 300),
        },
        _indicator_broker_evidence={},
        _indicator_real_evidence={},
    )


def test_period_candidates_isolate_open_symbol_and_mode(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    candidate = ("R_25", TradeDirection.CALL, {})
    assert len(orders.period_candidates(orch, [candidate])) == 2
    orch.config["pullback_strategy"] = {"enabled": True, "promoted_symbol": "R_100"}
    assert orders.period_candidates(orch, [candidate]) == [candidate]
    orch.config["pullback_strategy"] = {"excluded_symbols": ["R_10"]}
    assert orders.period_candidates(orch, [candidate]) == [candidate]
    orch.config.pop("pullback_strategy")
    orch.state.active_contracts = {7: object()}
    orch.risk_manager.contract_to_symbol = {7: "R_10"}
    assert orders.period_candidates(orch, [candidate]) == [candidate]
    orch.market_writer._last_error = RuntimeError("offline")
    assert orders.period_candidates(orch, [candidate]) == []


def test_pullback_duration_rejects_m15_even_if_called_directly():
    with pytest.raises(RuntimeError, match="somente em M5"):
        set_indicator_duration(
            {},
            {
                "indicator_controls_execution": True,
                "indicator_model_version": "pullback-1-v2",
                "indicator_period_seconds": 900,
            },
        )


def test_period_candidates_requires_promotion_for_real(monkeypatch):
    orch = _orch("real")
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    assert orders.period_candidates(orch, []) == []
    orch._indicator_broker_evidence[("R_10", 300, "pullback-1-v5")] = BrokerEvEvidence(200, (0.2,) * 200)
    assert [(symbol, metrics["duration"]) for symbol, _, metrics in orders.period_candidates(orch, [])] == [("R_10", 5)]
    orch._indicator_real_evidence[("R_10", 300, "pullback-1-v5")] = BrokerEvEvidence(200, (-0.2,) * 200)
    assert orders.period_candidates(orch, []) == []


def test_period_candidates_ignore_legacy_version(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch._indicator_period_signals = {("R_25", 900): ApprovedSignal("R_25", "PUT", 900, 0.7, "old", 900)}
    assert orders.period_candidates(orch, []) == []
    orch._indicator_period_signals = {("R_25", 300): ApprovedSignal("R_25", "PUT", 900, None, "old", 300)}
    assert orders.period_candidates(orch, []) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probability", "mode", "expected"), [(0.65, "real", True), (0.51, "real", False), (0.51, "demo", True)]
)
async def test_prequote_applies_ev_to_real_and_allows_demo_exploration(monkeypatch, probability, mode, expected):
    orch = _orch(mode)
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    recorded = AsyncMock()
    monkeypatch.setattr(orders, "record_indicator_quote", recorded)
    orch.trade_handler.fetch_proposal_snapshot.return_value = {
        "proposal_id": "q1",
        "price": 1.0,
        "payout": 1.8,
        "rate": 0.8,
    }
    metrics = {
        "indicator_period_seconds": 300,
        "indicator_model_version": "v15",
        "indicator_candle_epoch": 900,
        "trade_score": probability,
        "indicator_demo_pilot": mode == "demo",
    }
    result = await orders._prequote(orch, ("R_10", TradeDirection.CALL, metrics), asyncio.Semaphore(1))
    assert (result is not None) is expected
    assert orch.trade_handler.fetch_proposal_snapshot.await_args.kwargs["params"]["duration"] == 5
    recorded.assert_awaited_once()


@pytest.mark.asyncio
async def test_prequote_rejects_missing_quote_expired_signal_and_exploration_cap(monkeypatch):
    orch = _orch()
    current = True
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: current)
    metrics = {
        "indicator_period_seconds": 300,
        "indicator_model_version": "v60",
        "indicator_candle_epoch": 0,
        "trade_score": 0.51,
        "indicator_demo_pilot": True,
        "indicator_exploratory_purchased": 200,
    }
    order = ("R_10", TradeDirection.CALL, metrics)
    current = False
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) is None
    current = True
    orch.trade_handler.fetch_proposal_snapshot.return_value = None
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) is None
    orch.trade_handler.fetch_proposal_snapshot.return_value = {
        "proposal_id": "",
        "price": 1.0,
        "payout": 1.8,
        "rate": 0.8,
    }
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) is None
    orch.trade_handler.fetch_proposal_snapshot.return_value["proposal_id"] = "q1"
    orch.trade_handler.fetch_proposal_snapshot.return_value["rate"] = -1.0
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) is None
    orch.trade_handler.fetch_proposal_snapshot.return_value["rate"] = 0.8
    monkeypatch.setattr(orders, "record_indicator_quote", AsyncMock())
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) == order
    metrics["indicator_demo_pilot"] = False
    assert await orders._prequote(orch, order, asyncio.Semaphore(1)) is None


@pytest.mark.asyncio
async def test_prequote_never_requests_m15_or_unaffordable_m5(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    metrics = {"indicator_period_seconds": 900, "indicator_demo_pilot": True}
    assert await orders._prequote(orch, ("R_25", TradeDirection.PUT, metrics), asyncio.Semaphore(1)) is None
    metrics["indicator_period_seconds"] = 300
    orch.state.balance = 10
    orch.config["risk_management"]["params"]["stake_min"] = 1
    assert await orders._prequote(orch, ("R_25", TradeDirection.PUT, metrics), asyncio.Semaphore(1)) is None
    orch.trade_handler.fetch_proposal_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_period_ranking_selects_higher_demo_lower_bound_then_ev(monkeypatch):
    orch = _orch("real")
    monkeypatch.setattr(
        orders,
        "period_candidates",
        lambda orch, base: [
            ("R_10", TradeDirection.CALL, {"indicator_broker_lower_95": 0.1}),
            ("R_10", TradeDirection.PUT, {"indicator_broker_lower_95": 0.2}),
            ("R_25", TradeDirection.CALL, {"indicator_broker_lower_95": None}),
        ],
    )

    async def quote(orch, order, semaphore):
        order[2]["indicator_prequote_ev"] = 0.9 if order[1] == TradeDirection.CALL else 0.1
        return order

    monkeypatch.setattr(orders, "_prequote", quote)
    monkeypatch.setattr(orders.time, "time", lambda: 900)
    selected = await orders.select_period_orders(orch, [])
    assert [(symbol, direction) for symbol, direction, _ in selected] == [
        ("R_10", TradeDirection.PUT),
        ("R_25", TradeDirection.CALL),
    ]


@pytest.mark.asyncio
async def test_period_ranking_handles_no_candidate_failed_quote_and_timeout(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(orders, "period_candidates", lambda orch, base: [])
    assert await orders.select_period_orders(orch, []) == []
    monkeypatch.setattr(orders, "period_candidates", lambda orch, base: [("R_10", TradeDirection.CALL, {})])
    monkeypatch.setattr(orders.time, "time", lambda: 900)

    async def fail(orch, order, semaphore):
        raise RuntimeError("sem cotacao")

    monkeypatch.setattr(orders, "_prequote", fail)
    assert await orders.select_period_orders(orch, []) == []

    async def slow(orch, order, semaphore):
        await asyncio.sleep(1)
        return order

    monkeypatch.setattr(orders, "_prequote", slow)

    async def timed_out(tasks, timeout):
        return set(), set(tasks)

    monkeypatch.setattr(orders.asyncio, "wait", timed_out)
    assert await orders.select_period_orders(orch, []) == []


@pytest.mark.asyncio
async def test_execution_manager_invokes_period_selection(monkeypatch):
    from src.application.services.orchestrator import execution_manager

    orch = _orch()
    orch.config["indicator_consensus"]["multi_period_trading"] = True
    orch.state.balance = 100.0
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.decay_proposal_skip_cycles = MagicMock()
    orch._active_cycle_id = 1
    orch.mark_cluster_cycle_complete = MagicMock()
    executor = ExecutionManager(orch)
    executor._training_phase_gate = MagicMock(return_value=False)
    executor._collect_orders = MagicMock(return_value=[])
    executor._log_execution_blockers = MagicMock()
    select = AsyncMock(return_value=[])
    monkeypatch.setattr(execution_manager, "select_period_orders", select)
    monkeypatch.setattr(execution_manager, "prepare_recovery_skip_counter", AsyncMock())
    monkeypatch.setattr(execution_manager, "clear_cuda_cache", MagicMock())
    assert await executor.execute_cluster({}) == 0
    select.assert_awaited_once_with(orch, [])


@pytest.mark.asyncio
async def test_execution_manager_logs_selected_setup_assets(monkeypatch):
    from src.application.services.orchestrator import execution_manager

    orch = _orch()
    orch.config["pullback_strategy"] = {"enabled": True}
    orch.state.balance = 100.0
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.decay_proposal_skip_cycles = MagicMock()
    orch._active_cycle_id = 1
    orch.mark_cluster_cycle_complete = MagicMock()
    executor = ExecutionManager(orch)
    executor.logger = MagicMock()
    executor._execute_orders = AsyncMock(return_value=0)
    selected = [("1HZ25V", TradeDirection.PUT, {"indicator_period_seconds": 900})]
    monkeypatch.setattr(execution_manager, "select_period_orders", AsyncMock(return_value=selected))
    monkeypatch.setattr(execution_manager, "prepare_recovery_skip_counter", AsyncMock())
    assert await executor.execute_cluster({}) == 0
    assert any("1HZ25V:15m:PUT" in call.args for call in executor.logger.info.call_args_list)


@pytest.mark.asyncio
async def test_period_ranking_respects_single_order_when_multi_disabled(monkeypatch):
    orch = _orch("real")
    orch.config["orchestrator"]["execution"]["multi_symbol_orders"] = False
    monkeypatch.setattr(
        orders,
        "period_candidates",
        lambda orch, base: [
            ("R_10", TradeDirection.PUT, {"indicator_broker_lower_95": 0.2}),
            ("R_25", TradeDirection.CALL, {"indicator_broker_lower_95": 0.1}),
        ],
    )

    async def quote(orch, order, semaphore):
        order[2]["indicator_prequote_ev"] = 0.1
        return order

    monkeypatch.setattr(orders, "_prequote", quote)
    monkeypatch.setattr(orders.time, "time", lambda: 900)
    selected = await orders.select_period_orders(orch, [])
    assert [(symbol, direction) for symbol, direction, _ in selected] == [
        ("R_10", TradeDirection.PUT),
    ]


def test_period_candidates_uses_bayesian_probability_in_demo(monkeypatch):
    orch = _orch("demo")
    orch.config["pullback_strategy"] = {"invert_signal": True}
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch._indicator_broker_evidence[("R_10", 300, "pullback-1-v5")] = BrokerEvEvidence(10, (1.0, 1.0, 1.0, -1.0))
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][1] == TradeDirection.CALL
    assert candidates[0][2]["indicator_signal_side"] == "PUT"


def test_period_candidates_dynamic_rotation(monkeypatch):
    orch = _orch("demo")
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "R_100",
        "dynamic_rotation": True,
    }
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][0] == "R_10"

    orch._indicator_period_signals[("R_100", 300)] = ApprovedSignal("R_100", "CALL", 900, None, "pullback-1-v100", 300)
    candidates_with_anchor = orders.period_candidates(orch, [])
    assert len(candidates_with_anchor) == 1
    assert candidates_with_anchor[0][0] == "R_100"


def test_period_candidates_symbol_loss_quarantine(monkeypatch):
    orch = _orch("demo")
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch._symbol_loss_quarantine = {"R_10": 9999999999.0}
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 0


def test_period_candidates_stream_fallback_reverts_to_trend_on_flat_market(monkeypatch):
    orch = _orch("demo")
    orch._indicator_period_signals = {}
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "OTC_SPC",
        "period_seconds": 900,
        "stream_fallback": True,
    }
    dummy_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": i * 900})()
        for i in range(60)
    ]
    stream = type("Stream", (), {"micro_candles": {"OTC_SPC": dummy_candles}})()
    orch.stream = stream
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    monkeypatch.setattr(orders, "resolve_organic_direction", lambda candles, base: ("SKIP", "flat"))
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][1] in (TradeDirection.CALL, TradeDirection.PUT)


def test_period_candidates_no_stream_fallback_by_default(monkeypatch):
    orch = _orch("demo")
    orch._indicator_period_signals = {}
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "OTC_SPC",
        "period_seconds": 900,
        "stream_fallback": False,
    }
    dummy_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": i * 900})()
        for i in range(60)
    ]
    stream = type("Stream", (), {"micro_candles": {"OTC_SPC": dummy_candles}})()
    orch.stream = stream
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 0


def test_period_candidates_organic_direction_and_skip(monkeypatch):
    orch = _orch("demo")
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "organic_direction": True,
    }
    dummy_c = [type("Candle", (), {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5})() for _ in range(50)]
    orch.stream = type("Stream", (), {"micro_candles": {"R_10": dummy_c}})()

    monkeypatch.setattr(orders, "resolve_organic_direction", lambda candles, side: ("PUT", "fade"))
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][1] == TradeDirection.PUT

    monkeypatch.setattr(orders, "resolve_organic_direction", lambda candles, side: ("SKIP", "skip"))
    candidates_skipped = orders.period_candidates(orch, [])
    assert len(candidates_skipped) == 0


def test_period_candidates_stream_fallback_inverted(monkeypatch):
    orch = _orch("demo")
    orch._indicator_period_signals = {}
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "OTC_SPC",
        "period_seconds": 900,
        "stream_fallback": True,
        "invert_signal": True,
    }
    dummy_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": i * 900})()
        for i in range(60)
    ]
    stream = type("Stream", (), {"micro_candles": {"OTC_SPC": dummy_candles}})()
    orch.stream = stream
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    monkeypatch.setattr(orders, "resolve_organic_direction", lambda candles, base: ("CALL", "tendencia"))
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][1] == TradeDirection.PUT


@pytest.mark.asyncio
async def test_prequote_edge_none_or_break_even_none(monkeypatch):
    orch = _orch("demo")
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    orch.trade_handler.fetch_proposal_snapshot.return_value = {
        "proposal_id": "p1",
        "price": 10.0,
        "payout": 18.0,
        "rate": 0.8,
    }
    metrics = {
        "trade_score": 0.55,
        "indicator_period_seconds": 300,
        "indicator_model_version": "v1",
        "indicator_candle_epoch": 900,
        "indicator_demo_pilot": False,
    }
    order = ("R_10", TradeDirection.CALL, metrics)
    monkeypatch.setattr(orders, "unit_ev", lambda prob, rate: None)
    res = await orders._prequote(orch, order, asyncio.Semaphore(1))
    assert res is None


def test_period_candidates_stream_fallback_rotates_to_secondary_symbol(monkeypatch):
    import time

    orch = _orch("demo")
    orch._indicator_period_signals = {}
    orch.symbols = ["OTC_SSMI", "OTC_SX5E"]
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "OTC_SSMI",
        "period_seconds": 900,
        "stream_fallback": True,
        "dynamic_rotation": True,
    }
    now_ts = time.time()
    stale_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": now_ts - 5000})()
        for _ in range(60)
    ]
    fresh_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": now_ts - 100})()
        for _ in range(60)
    ]
    stream = type("Stream", (), {"micro_candles": {"OTC_SSMI": stale_candles, "OTC_SX5E": fresh_candles}})()
    orch.stream = stream
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][0] == "OTC_SX5E"


def test_period_candidates_stream_fallback_skips_excluded_active_and_quarantine(monkeypatch):
    orch = _orch("demo")
    orch._indicator_period_signals = {}
    orch.symbols = ["OTC_SSMI", "OTC_EXC", "OTC_ACT", "OTC_QUA", "OTC_OK"]
    orch.state.active_contracts = {99: object()}
    orch.risk_manager.contract_to_symbol = {99: "OTC_ACT"}
    orch._symbol_loss_quarantine = {"OTC_QUA": 9999999999.0}
    orch.config["pullback_strategy"] = {
        "enabled": True,
        "promoted_symbol": "OTC_SSMI",
        "excluded_symbols": ["OTC_EXC"],
        "period_seconds": 900,
        "stream_fallback": True,
        "dynamic_rotation": True,
    }
    dummy_candles = [
        type("Candle", (), {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "epoch": i * 900})()
        for i in range(60)
    ]
    stream = type("Stream", (), {"micro_candles": {"OTC_OK": dummy_candles}})()
    orch.stream = stream
    monkeypatch.setattr(orders, "signal_epoch_current", lambda config, metrics: True)
    candidates = orders.period_candidates(orch, [])
    assert len(candidates) == 1
    assert candidates[0][0] == "OTC_OK"


def test_closed_candles_slice_drops_forming_bar():
    bars = [type("Candle", (), {"epoch": 1700000000 + i * 900})() for i in range(55)]
    cur_open = int(bars[-1].epoch)
    sliced = orders._closed_candles_slice(bars, 900, float(cur_open + 5))
    assert len(sliced) == 54
