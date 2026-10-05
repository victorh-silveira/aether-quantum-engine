"""Testes do quote guard e captura de rejeicao sem EV em execution_orders."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.execution_orders import (
    _attach_quote_guard_params,
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
    assert params["_quote_guard_min_edge"] == pytest.approx(-0.08)


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

    metrics = {"calibrated_prob": 0.504}
    res = await place_order(executor, "1HZ75V", TradeDirection.PUT, 50.0, metrics=metrics)
    assert res is None


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
