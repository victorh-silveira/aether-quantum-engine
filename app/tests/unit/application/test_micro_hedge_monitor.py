"""Testes unitarios para monitoramento e execucao de micro-hedging de ticks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.services.micro_hedge_monitor import (
    evaluate_hedges_once,
    execute_micro_hedge_order,
    is_unfavorable_drawdown,
    load_micro_hedge_config,
    register_contract_for_hedge,
    start_micro_hedge_monitor_worker,
)
from src.domain.models.trade import Contract, TradeDirection, TradeStatus


def test_load_micro_hedge_config():
    """Verifica carregamento de configuracao de micro-hedging."""
    assert load_micro_hedge_config(None).enabled is False
    assert load_micro_hedge_config({}).enabled is False

    cfg_dict = {
        "orchestrator": {
            "execution": {
                "micro_hedging": {
                    "enabled": True,
                    "eval_window_seconds_min": 100.0,
                    "eval_window_seconds_max": 200.0,
                    "target_midpoint_seconds": 150.0,
                    "delta_unfavorable_ticks": 6.0,
                    "hedge_duration_seconds": 120,
                    "hedge_stake_ratio": 0.60,
                    "tick_size": 0.05,
                }
            }
        }
    }
    cfg = load_micro_hedge_config(cfg_dict)
    assert cfg.enabled is True
    assert cfg.eval_window_seconds_min == 100.0
    assert cfg.eval_window_seconds_max == 200.0
    assert cfg.delta_unfavorable_ticks == 6.0
    assert cfg.hedge_duration_seconds == 120
    assert cfg.hedge_stake_ratio == 0.60
    assert cfg.tick_size == 0.05


def test_register_contract_for_hedge():
    """Verifica registro do contrato sob observacao de hedging."""
    register_contract_for_hedge(None, 101, symbol="1HZ75V", direction=TradeDirection.CALL, entry_spot=100.0, stake=2.0)

    orch = SimpleNamespace()
    register_contract_for_hedge(
        orch,
        101,
        symbol="1HZ75V",
        direction=TradeDirection.CALL,
        entry_spot=100.0,
        stake=2.0,
        duration=300,
        entry_time=1000.0,
    )
    assert 101 in orch._active_hedge_watch
    item = orch._active_hedge_watch[101]
    assert item["symbol"] == "1HZ75V"
    assert item["direction"] == TradeDirection.CALL
    assert item["entry_spot"] == 100.0
    assert item["entry_time"] == 1000.0
    assert item["hedged"] is False


def test_is_unfavorable_drawdown():
    """Verifica identificacao de drawdown adverso para CALL e PUT."""
    assert is_unfavorable_drawdown(TradeDirection.CALL, -5.5, 5.0) is True
    assert is_unfavorable_drawdown(TradeDirection.CALL, -4.5, 5.0) is False
    assert is_unfavorable_drawdown(TradeDirection.MULTUP, -5.0, 5.0) is True

    assert is_unfavorable_drawdown(TradeDirection.PUT, 5.5, 5.0) is True
    assert is_unfavorable_drawdown(TradeDirection.PUT, 4.5, 5.0) is False
    assert is_unfavorable_drawdown(TradeDirection.MULTDOWN, 5.0, 5.0) is True

    assert is_unfavorable_drawdown(TradeDirection.CALL, 5.0, 5.0) is False


@pytest.mark.asyncio
async def test_execute_micro_hedge_order_success():
    """Verifica disparo e registro bem-sucedido de micro-hedge."""
    assert await execute_micro_hedge_order(None, "1HZ75V", TradeDirection.PUT, 1.0, 150, 101, -6.0) is None

    fake_contract = Contract(
        contract_id=202,
        proposal_id="p202",
        status=TradeStatus.OPEN,
        buy_price=1.0,
        payout=1.85,
        symbol="1HZ75V",
        direction=TradeDirection.PUT,
        stake=1.0,
        expiry_time=1200,
    )
    th = SimpleNamespace(buy_with_parameters=AsyncMock(return_value=fake_contract))
    rm = SimpleNamespace(active_contract_ids=[], contract_to_symbol={}, contract_stakes={})
    orch = SimpleNamespace(
        trade_handler=th,
        risk_manager=rm,
        state=SimpleNamespace(active_contracts={}),
    )

    contract = await execute_micro_hedge_order(orch, "1HZ75V", TradeDirection.PUT, 1.0, 150, 101, -6.0)
    assert contract is not None
    assert contract.contract_id == 202
    assert 202 in orch.state.active_contracts
    assert 202 in rm.active_contract_ids
    assert rm.contract_to_symbol[202] == "1HZ75V"
    assert rm.contract_stakes[202] == 1.0


@pytest.mark.asyncio
async def test_execute_micro_hedge_order_failure():
    """Verifica tratamento fail-safe quando a compra da Deriv falha."""
    th = SimpleNamespace(buy_with_parameters=AsyncMock(side_effect=RuntimeError("Deriv indisponivel")))
    orch = SimpleNamespace(trade_handler=th)
    res = await execute_micro_hedge_order(orch, "1HZ75V", TradeDirection.PUT, 1.0, 150, 101, -6.0)
    assert res is None


@pytest.mark.asyncio
async def test_evaluate_hedges_once():
    """Verifica varredura completa de contratos e acionamento de micro-hedge."""
    assert evaluate_hedges_once(None) == []

    cfg = {
        "orchestrator": {
            "execution": {
                "micro_hedging": {
                    "enabled": True,
                    "eval_window_seconds_min": 120.0,
                    "eval_window_seconds_max": 180.0,
                    "delta_unfavorable_ticks": 5.0,
                    "tick_size": 0.01,
                }
            }
        }
    }
    orch = SimpleNamespace(config=cfg, _active_hedge_watch={}, state=SimpleNamespace(active_contracts={}))
    assert evaluate_hedges_once(orch) == []

    orch._active_hedge_watch[100] = {"entry_time": 1000.0, "hedged": False}
    assert evaluate_hedges_once(orch, now=1150.0) == []
    assert 100 not in orch._active_hedge_watch

    dummy_contract = Contract(
        contract_id=101,
        proposal_id="p101",
        status=TradeStatus.OPEN,
        buy_price=2.0,
        payout=3.70,
        symbol="1HZ75V",
        direction=TradeDirection.CALL,
        stake=2.0,
        expiry_time=1300,
    )
    orch.state.active_contracts[101] = dummy_contract
    orch._active_hedge_watch[101] = {
        "symbol": "1HZ75V",
        "direction": TradeDirection.CALL,
        "entry_spot": 100.0,
        "stake": 2.0,
        "entry_time": 1000.0,
        "hedged": True,
    }
    assert evaluate_hedges_once(orch, now=1150.0) == []

    orch._active_hedge_watch[101]["hedged"] = False
    assert evaluate_hedges_once(orch, now=1050.0) == []

    assert evaluate_hedges_once(orch, now=1250.0) == []
    assert orch._active_hedge_watch[101]["hedged"] is True

    orch._active_hedge_watch[101]["hedged"] = False
    orch.stream = SimpleNamespace(tick_buffer=SimpleNamespace(latest_price=lambda s: None))
    assert evaluate_hedges_once(orch, now=1150.0) == []

    orch.stream.tick_buffer.latest_price = lambda s: 100.02
    assert evaluate_hedges_once(orch, now=1150.0) == []

    orch.stream.tick_buffer.latest_price = lambda s: 99.90
    triggered = evaluate_hedges_once(orch, now=1150.0)
    assert triggered == [101]
    assert orch._active_hedge_watch[101]["hedged"] is True

    orch._active_hedge_watch[101]["hedged"] = False
    from unittest.mock import patch

    with patch("src.application.services.micro_hedge_monitor.spawn_background", side_effect=RuntimeError("no loop")):
        triggered_err = evaluate_hedges_once(orch, now=1150.0)
        assert triggered_err == [101]


def test_evaluate_hedges_once_disabled():
    """Verifica saida rapida quando hedging esta desabilitado."""
    orch = SimpleNamespace(config={"orchestrator": {"execution": {"micro_hedging": {"enabled": False}}}})
    assert evaluate_hedges_once(orch) == []


@pytest.mark.asyncio
async def test_start_micro_hedge_monitor_worker():
    """Verifica execucao do worker em loop assincrono e captura de falhas."""
    from unittest.mock import patch

    class MockOrch:
        def __init__(self):
            self.running = True
            self.config = {}
            self.calls = 0

        def tick(self, *args, **kwargs):
            _ = (args, kwargs)
            self.calls += 1
            self.running = False
            return []

    orch = MockOrch()
    with patch("src.application.services.micro_hedge_monitor.evaluate_hedges_once", side_effect=orch.tick):
        await start_micro_hedge_monitor_worker(orch, poll_interval=0.001)
    assert orch.calls == 1

    orch_err = MockOrch()

    def _raise(*args, **kwargs):
        _ = (args, kwargs)
        orch_err.tick()
        raise ValueError("boom")

    with patch("src.application.services.micro_hedge_monitor.evaluate_hedges_once", side_effect=_raise):
        await start_micro_hedge_monitor_worker(orch_err, poll_interval=0.001)
    assert orch_err.calls == 1
