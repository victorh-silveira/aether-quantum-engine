from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.application.services.execution_price_zone_gate import (
    ZONE_BUY,
    ZONE_NONE,
    ZONE_SELL,
    _reject_reason,
    align_direction_to_price_zone,
    apply_price_zone_gate,
    resolve_price_zone_config,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.test_execution_price_zone_gate import _zone_cfg


def test_apply_price_zone_gate_aligns_when_and_flags_off():
    cfg = _zone_cfg(require_trend_agreement=False, require_tcn_agreement=False)
    metrics = {"bb_pct_b": 0.15, "keltner": 0.20, "trend_direction": "PUT", "dl_direction": "PUT"}
    assert apply_price_zone_gate(metrics, TradeDirection.PUT, cfg) is None
    assert metrics["price_zone"] == ZONE_BUY
    assert metrics["price_zone_direction"] == "CALL"
    assert metrics["price_zone_aligned"] is True
    assert align_direction_to_price_zone(TradeDirection.PUT, metrics) == TradeDirection.CALL


def test_price_zone_coverage_edges():
    with (
        patch(
            "src.application.services.execution_price_zone_gate.merge_settings_block",
            return_value={"bb_weight": 0.0, "keltner_weight": 0.0},
        ),
        pytest.raises(ValueError, match="invalidos"),
    ):
        resolve_price_zone_config(
            {
                "price_zone": {
                    "enabled": True,
                    "buy_max": 0.3,
                    "sell_min": 0.7,
                    "bb_weight": 0.0,
                    "keltner_weight": 0.0,
                    "require_trend_agreement": False,
                    "require_tcn_agreement": False,
                }
            }
        )
    conf = resolve_price_zone_config(_zone_cfg())
    assert _reject_reason(ZONE_NONE, TradeDirection.CALL, {}, conf, None) == "price_zone_none"
    assert (
        _reject_reason(
            ZONE_BUY,
            TradeDirection.PUT,
            {"trend_direction": "CALL", "dl_direction": "CALL"},
            conf,
            TradeDirection.CALL,
        )
        == "price_zone_buy_requires_call"
    )
    assert (
        _reject_reason(
            ZONE_SELL,
            TradeDirection.CALL,
            {"trend_direction": "PUT", "dl_direction": "PUT"},
            conf,
            TradeDirection.PUT,
        )
        == "price_zone_sell_requires_put"
    )
    soft = _zone_cfg(require_trend_agreement=False, require_tcn_agreement=False)
    assert apply_price_zone_gate({}, TradeDirection.CALL, soft) is None
    metrics = {"indicators": {"bb_pct_b": 0.1, "keltner_pct_b": 0.1}}
    assert apply_price_zone_gate(metrics, TradeDirection.CALL, soft) is None
    assert metrics["price_zone"] == ZONE_BUY
    assert align_direction_to_price_zone(TradeDirection.CALL, {"price_zone_direction": "PUT"}) == TradeDirection.PUT


def test_initial_direction_checks_aligns_put_in_buy_zone_when_flags_off():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.30,
            "raw_prob": 0.30,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.15,
            "keltner": 0.20,
            "trend_direction": "PUT",
            "call_votes": 1,
            "put_votes": 4,
        },
    }
    result = initial_direction_checks(entry, _zone_cfg(require_trend_agreement=False, require_tcn_agreement=False))
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1]["price_zone"] == ZONE_BUY
    assert result[1]["price_zone_aligned"] is True


def test_finalize_rejects_price_zone_none():
    entry = {"metrics": {}}
    metrics = {
        "bb_pct_b": 0.50,
        "keltner": 0.50,
        "trend_direction": "CALL",
        "dl_direction": "CALL",
        "calibrated_prob": 0.70,
        "predicted_payoff_edge": 0.1,
        "trade_score": 0.7,
    }
    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.7),
        ),
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            return_value=TradeDirection.CALL,
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.is_execution_signal_vetoed",
            return_value=False,
        ),
    ):
        out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.7,
            0.1,
            meta_applied=True,
            score=0.7,
            symbol="R_10",
            orch=SimpleNamespace(config={}),
            force=False,
            exec_cfg=_zone_cfg(require_trend_agreement=False, require_tcn_agreement=False),
        )
    assert out is None
    assert entry["metrics"]["gate_reason"] == "price_zone_none"


def test_finalize_aligns_side_eq_flip_to_zone():
    entry = {"metrics": {}}
    metrics = {
        "bb_pct_b": 0.20,
        "keltner": 0.20,
        "trend_direction": "CALL",
        "dl_direction": "CALL",
        "calibrated_prob": 0.70,
        "predicted_payoff_edge": 0.1,
        "trade_score": 0.7,
    }
    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.7),
        ),
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            return_value=TradeDirection.PUT,
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.is_execution_signal_vetoed",
            return_value=False,
        ),
    ):
        out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.7,
            0.1,
            meta_applied=True,
            score=0.7,
            symbol="R_10",
            orch=SimpleNamespace(config={}),
            force=False,
            exec_cfg=_zone_cfg(require_trend_agreement=False, require_tcn_agreement=False),
        )
    assert out is not None
    assert out[0] == TradeDirection.CALL
    assert entry["metrics"]["price_zone"] == ZONE_BUY


def test_finalize_keeps_side_eq_flip_against_price_zone():
    entry = {"metrics": {}}
    metrics = {
        "bb_pct_b": 0.85,
        "keltner": 0.80,
        "trend_direction": "PUT",
        "dl_direction": "PUT",
        "calibrated_prob": 0.35,
        "predicted_payoff_edge": 0.1,
        "trade_score": 0.65,
    }

    def _flip_to_call(_orch, _symbol, _proposed, metrics_arg, **_kwargs):
        metrics_arg["side_eq_flipped"] = True
        metrics_arg["side_eq_flip_from"] = "PUT"
        return TradeDirection.CALL

    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.PUT, 0.65),
        ),
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            side_effect=_flip_to_call,
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.is_execution_signal_vetoed",
            return_value=False,
        ),
    ):
        out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.PUT,
            0.35,
            0.1,
            meta_applied=True,
            score=0.65,
            symbol="R_10",
            orch=SimpleNamespace(config={}),
            force=False,
            exec_cfg=_zone_cfg(require_trend_agreement=False, require_tcn_agreement=False),
        )
    assert out is not None
    assert out[0] == TradeDirection.CALL
    assert entry["metrics"]["side_eq_flipped"] is True
    assert entry["metrics"]["price_zone_side_eq_override"] is True
    assert entry["metrics"]["price_zone"] == ZONE_SELL
