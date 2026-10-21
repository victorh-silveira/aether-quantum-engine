"""Testes do quote guard e captura de rejeicao sem EV em execution_orders."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_orders import (
    _attach_quote_guard_params,
    _emit_execution_ticket,
    place_order,
)
from src.domain.models.trade import TradeDirection


def test_attach_quote_guard_params_call():
    params = {}
    metrics = {"calibrated_prob": 0.60, "pending_loss_total": 0.0}
    exec_cfg = {
        "require_quote_edge": True,
        "quote_probability_haircut": 0.02,
        "min_edge_execute": 0.01,
        "recovery_neg_edge_floor": -0.08,
    }
    _attach_quote_guard_params(params, metrics, TradeDirection.CALL, exec_cfg)
    assert params["_quote_guard_side_probability"] == pytest.approx(0.58)
    assert params["_quote_guard_min_edge"] == pytest.approx(0.01)


def test_attach_quote_guard_params_put_recovery():
    params = {}
    metrics = {"calibrated_prob": 0.40, "pending_loss_total": 100.0}
    exec_cfg = {
        "require_quote_edge": True,
        "quote_probability_haircut": 0.02,
        "min_edge_execute": 0.01,
        "recovery_neg_edge_floor": -0.08,
    }
    _attach_quote_guard_params(params, metrics, TradeDirection.PUT, exec_cfg)
    assert params["_quote_guard_side_probability"] == pytest.approx(0.58)
    assert params["_quote_guard_min_edge"] == pytest.approx(0.0)


def test_demo_pilot_put_keeps_probability_and_never_enables_real_bypass():
    params = {}
    metrics = {"calibrated_prob": 0.472, "indicator_demo_pilot": True, "indicator_candle_epoch": 300}
    _attach_quote_guard_params(
        params,
        metrics,
        TradeDirection.PUT,
        {"require_quote_edge": True, "quote_probability_haircut": 0.02},
        trading_mode="demo",
    )
    assert params["_quote_guard_side_probability"] == pytest.approx(0.508)
    assert params["_quote_guard_min_edge"] == -1.0
    assert params["_quote_guard_safety_margin"] == 0.0
    assert params["min_payout_rate"] == 0.0
    real_params = {}
    _attach_quote_guard_params(
        real_params,
        metrics,
        TradeDirection.PUT,
        {"require_quote_edge": True, "quote_probability_haircut": 0.02},
        trading_mode="real",
    )
    assert real_params["_quote_guard_min_edge"] == 0.0
    assert real_params["_quote_guard_side_probability"] == pytest.approx(0.508)
    assert "_indicator_direct_buy_fallback" not in real_params


def test_pullback_demo_pilot_has_no_invented_quote_probability():
    params = {}
    metrics = {
        "calibrated_prob": 0.5,
        "indicator_demo_pilot": True,
        "indicator_model_version": "pullback-1-test",
        "indicator_candle_epoch": 300,
    }
    _attach_quote_guard_params(params, metrics, TradeDirection.CALL, {"require_quote_edge": True}, trading_mode="demo")
    assert params["_indicator_demo_pilot"] is True
    assert "_quote_guard_side_probability" not in params
    assert "_quote_guard_min_edge" not in params


def test_indicator_recovery_requires_positive_quoted_ev():
    params = {}
    metrics = {
        "calibrated_prob": 0.472,
        "indicator_controls_execution": True,
        "indicator_candle_epoch": 300,
        "pending_loss_total": 120.0,
    }
    _attach_quote_guard_params(
        params,
        metrics,
        TradeDirection.PUT,
        {
            "require_quote_edge": True,
            "recovery_neg_edge_floor": -0.08,
            "quote_safety_margin": 0.01,
        },
    )
    assert params["_quote_guard_side_probability"] == pytest.approx(0.528)
    assert params["_quote_guard_min_edge"] == 0.0
    assert params["_quote_guard_safety_margin"] == pytest.approx(0.01)
    assert "_indicator_direct_buy_fallback" not in params


def test_indicator_execution_ticket_identifies_fixed_stake():
    executor = MagicMock()
    executor.orch.config = {"orchestrator": {"execution": {}}}
    contract = SimpleNamespace(contract_id=7, payout=17.81, buy_price=10.0)
    logger = MagicMock()
    with (
        patch("src.application.services.orchestrator.execution_orders.logging.getLogger", return_value=logger),
        patch("src.application.services.orchestrator.execution_orders.store_contract_audit"),
    ):
        _emit_execution_ticket(
            executor,
            cycle_id=1,
            symbol="R_25",
            direction=TradeDirection.PUT,
            stake=10.0,
            contract=contract,
            metrics={
                "indicator_stake_mode": "INDICATOR_FIXED_1PCT",
                "trade_score": 0.528,
                "calibrated_prob": 0.472,
            },
        )
    assert any("[INDICATOR_STAKE]" in call.args[0] for call in logger.info.call_args_list)


def test_pullback_demo_ticket_does_not_claim_probability_or_ev():
    executor = MagicMock()
    executor.orch.config = {"orchestrator": {"execution": {}}}
    contract = SimpleNamespace(contract_id=19, payout=1.78, buy_price=1.0)
    logger = MagicMock()
    metrics = {
        "indicator_stake_mode": "INDICATOR_DEMO_PILOT",
        "indicator_model_version": "pullback-1-v1",
        "indicator_demo_pilot": True,
        "calibrated_prob": 0.5,
    }
    with (
        patch("src.application.services.orchestrator.execution_orders.logging.getLogger", return_value=logger),
        patch("src.application.services.orchestrator.execution_orders.store_contract_audit"),
    ):
        _emit_execution_ticket(
            executor,
            cycle_id=1,
            symbol="1HZ25V",
            direction=TradeDirection.PUT,
            stake=1.0,
            contract=contract,
            metrics=metrics,
        )
    assert metrics["quote_edge"] is None
    assert not logger.warning.called
    assert any("p_side=%s" in call.args[0] and "n/a" in call.args for call in logger.info.call_args_list)


def test_attach_quote_guard_params_disabled_or_missing():
    params = {}
    _attach_quote_guard_params(params, None, TradeDirection.CALL, {"require_quote_edge": True})
    assert params == {}

    _attach_quote_guard_params(params, {"calibrated_prob": None}, TradeDirection.CALL, {"require_quote_edge": True})
    assert params == {}

    _attach_quote_guard_params(params, {"calibrated_prob": 0.60}, TradeDirection.CALL, {"require_quote_edge": False})
    assert params == {}


@pytest.mark.asyncio
async def test_place_order_catches_quote_guard_rejection_safely():
    executor = MagicMock()
    executor.orch._active_cycle_id = 1
    executor.orch.config = {
        "orchestrator": {"execution": {"require_quote_edge": True}},
        "risk_management": {"params": {"duration": 5, "duration_unit": "m", "stake_min": 1.0}},
    }
    executor.orch.trade_handler.buy_with_parameters = AsyncMock(
        side_effect=RuntimeError("Cotacao final Rise/Fall perdeu vantagem; compra bloqueada")
    )

    metrics = {
        "calibrated_prob": 0.504,
        "indicator_controls_execution": True,
        "indicator_model_version": "v1",
        "indicator_period_seconds": 900,
        "indicator_candle_epoch": 900,
        "_register_before_tracking": True,
    }
    with patch("src.application.services.orchestrator.execution_orders.logging.getLogger") as logger:
        res = await place_order(executor, "1HZ75V", TradeDirection.PUT, 50.0, metrics=metrics)
    assert res is None
    assert "guard de EV" in logger.return_value.warning.call_args.args[0]
    parameters = executor.orch.trade_handler.buy_with_parameters.await_args.kwargs["params"]
    assert parameters["_indicator_model_version"] == "v1"
    assert parameters["_indicator_period_seconds"] == 900


@pytest.mark.asyncio
async def test_pullback_order_enforces_five_minute_stake_cap():
    executor = MagicMock()
    executor.orch._active_cycle_id = 1
    executor.orch.config = {"trading": {"mode": "demo"}, "risk_management": {"params": {"stake_min": 1}}}
    metrics = {
        "indicator_controls_execution": True,
        "indicator_model_version": "pullback-1-v2",
        "indicator_period_seconds": 300,
        "_pullback_stake_cap": 50,
        "indicator_quote_rate": 0.8,
        "_pullback_recovery_amount": 20,
    }
    with pytest.raises(RuntimeError, match="excede teto"):
        await place_order(executor, "R_25", TradeDirection.PUT, 51, metrics=metrics)
    executor.orch.trade_handler.buy_with_parameters.assert_not_called()
    executor.orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=RuntimeError("proposta indisponivel"))
    with pytest.raises(RuntimeError, match="proposta indisponivel"):
        await place_order(executor, "R_25", TradeDirection.PUT, 50, metrics=metrics)
    params = executor.orch.trade_handler.buy_with_parameters.await_args.kwargs["params"]
    assert params["duration"] == 5
    assert params["_pullback_quote_rate"] == 0.8
    assert params["_pullback_recovery_amount"] == 20


@pytest.mark.asyncio
async def test_pullback_order_attaches_prequoted_proposal():
    executor = MagicMock()
    executor.orch._active_cycle_id = 1
    executor.orch.config = {"trading": {"mode": "demo"}, "risk_management": {"params": {"stake_min": 1}}}
    metrics = {
        "indicator_controls_execution": True,
        "indicator_model_version": "pullback-1-v2",
        "indicator_period_seconds": 300,
        "_pullback_stake_cap": 50,
        "_prequoted_proposal_id": "prop-12345",
        "_prequoted_ask_price": 10.0,
        "_prequoted_payout": 19.5,
        "_prequoted_rate": 0.95,
        "_prequoted_timestamp": 1700000000.0,
    }
    executor.orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=RuntimeError("stop_flow"))
    with pytest.raises(RuntimeError, match="stop_flow"):
        await place_order(executor, "R_25", TradeDirection.CALL, 10.0, metrics=metrics)
    params = executor.orch.trade_handler.buy_with_parameters.await_args.kwargs["params"]
    assert params["_prequoted_proposal_id"] == "prop-12345"
    assert params["_prequoted_ask_price"] == 10.0
    assert params["_prequoted_payout"] == 19.5
    assert params["_prequoted_rate"] == 0.95
    assert params["_prequoted_timestamp"] == 1700000000.0


def test_attach_quote_guard_params_flips_and_error():
    params = {}
    metrics_flip = {"calibrated_prob": 0.501, "loss_clf_flip": True, "loss_clf_p_eff": 0.72}
    _attach_quote_guard_params(params, metrics_flip, TradeDirection.CALL, {"require_quote_edge": True})
    assert params["_quote_guard_side_probability"] == pytest.approx(0.72)

    params2 = {}
    metrics_atl = {"calibrated_prob": 0.501, "anti_trend_lock_flip": True, "conviction": 0.65}
    _attach_quote_guard_params(params2, metrics_atl, TradeDirection.PUT, {"require_quote_edge": True})
    assert params2["_quote_guard_side_probability"] == pytest.approx(0.499)

    params3 = {}
    metrics_err = {"calibrated_prob": "invalid_number"}
    _attach_quote_guard_params(params3, metrics_err, TradeDirection.CALL, {"require_quote_edge": True})
    assert params3 == {}


def test_attach_quote_guard_uses_zero_effective_flip_probability():
    params = {}
    metrics = {"calibrated_prob": 0.7, "loss_clf_flip": True, "loss_clf_p_eff": 0.0, "loss_clf_p_loss": 0.9}
    _attach_quote_guard_params(params, metrics, TradeDirection.PUT, {"require_quote_edge": True})
    assert params["_quote_guard_side_probability"] == 0.0


def test_attach_quote_guard_rejects_invalid_haircut():
    params = {}
    _attach_quote_guard_params(
        params,
        {"calibrated_prob": 0.6},
        TradeDirection.CALL,
        {"require_quote_edge": True, "quote_probability_haircut": "bad"},
    )
    assert params == {}
