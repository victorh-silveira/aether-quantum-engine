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
    _trend_supports,
    _zone_side_ok,
    apply_price_zone_gate,
    direction_allowed_for_zone,
    resolve_price_zone,
    resolve_price_zone_config,
    zone_score,
)
from src.domain.models.trade import TradeDirection


def _zone_cfg(**overrides):
    base = {
        "enabled": True,
        "buy_max": 0.35,
        "sell_min": 0.65,
        "bb_weight": 0.6,
        "keltner_weight": 0.4,
        "require_trend_agreement": True,
        "require_tcn_agreement": True,
    }
    base.update(overrides)
    return {"price_zone": base}


def test_resolve_price_zone_config_clamps_and_normalizes_weights():
    cfg = resolve_price_zone_config(
        {"price_zone": {"enabled": True, "buy_max": 0.4, "sell_min": 0.3, "bb_weight": 0.0, "keltner_weight": 0.0}}
    )
    assert cfg["sell_min"] == pytest.approx(0.4)
    assert cfg["bb_weight"] + cfg["keltner_weight"] == pytest.approx(1.0)
    assert resolve_price_zone_config(None)["enabled"] is True


def test_resolve_price_zone_buy_sell_none():
    cfg = resolve_price_zone_config(_zone_cfg())
    assert resolve_price_zone({"bb_pct_b": 0.20, "keltner": 0.25}, cfg) == ZONE_BUY
    assert resolve_price_zone({"bb_pct_b": 0.80, "keltner": 0.75}, cfg) == ZONE_SELL
    assert resolve_price_zone({"bb_pct_b": 0.50, "keltner": 0.50}, cfg) == ZONE_NONE
    assert (
        resolve_price_zone({"bb_pct_b": 0.20}, resolve_price_zone_config({"price_zone": {"enabled": False}}))
        == ZONE_NONE
    )


def test_zone_score_reads_indicators_and_ignores_bad_values():
    cfg = resolve_price_zone_config(_zone_cfg(bb_weight=0.5, keltner_weight=0.5))
    assert zone_score({"bb_pct_b": 0.0, "keltner": 1.0}, cfg) == pytest.approx(0.5)
    assert zone_score({"indicators": {"bb_pct_b": 0.2, "keltner_pct_b": "x"}}, cfg) == pytest.approx(0.35)


def test_direction_allowed_buy_requires_call_and_trend():
    cfg = resolve_price_zone_config(_zone_cfg())
    metrics = {
        "bb_pct_b": 0.20,
        "keltner": 0.20,
        "trend_direction": "CALL",
        "dl_direction": "CALL",
        "call_votes": 4,
        "put_votes": 1,
    }
    assert direction_allowed_for_zone(ZONE_BUY, TradeDirection.CALL, metrics, cfg) is True
    assert direction_allowed_for_zone(ZONE_BUY, TradeDirection.PUT, metrics, cfg) is False
    assert direction_allowed_for_zone(ZONE_NONE, TradeDirection.CALL, metrics, cfg) is False
    assert (
        direction_allowed_for_zone(
            ZONE_BUY, TradeDirection.CALL, metrics, resolve_price_zone_config({"price_zone": {"enabled": False}})
        )
        is True
    )


def test_direction_allowed_trend_from_votes_and_tcn_conflict():
    cfg = resolve_price_zone_config(_zone_cfg(require_trend_agreement=True, require_tcn_agreement=True))
    metrics = {"call_votes": 5, "put_votes": 1, "dl_direction": "PUT"}
    assert (
        direction_allowed_for_zone(ZONE_BUY, TradeDirection.CALL, metrics, cfg, tcn_direction=TradeDirection.CALL)
        is False
    )
    metrics_ok = {"call_votes": "bad", "put_votes": 1, "trend_direction": "", "dl_direction": "CALL"}
    assert direction_allowed_for_zone(ZONE_BUY, TradeDirection.CALL, metrics_ok, cfg) is False
    assert _trend_supports(TradeDirection.CALL, {"call_votes": 2, "put_votes": 2}) is False
    assert _zone_side_ok("OTHER", TradeDirection.CALL) is False
    cfg_no_tcn = resolve_price_zone_config(_zone_cfg(require_tcn_agreement=False))
    assert (
        direction_allowed_for_zone(
            ZONE_BUY, TradeDirection.CALL, {"trend_direction": "CALL"}, cfg_no_tcn, tcn_direction=None
        )
        is True
    )
    assert (
        _reject_reason(
            ZONE_BUY,
            TradeDirection.CALL,
            {"trend_direction": "CALL", "dl_direction": "CALL"},
            cfg,
            TradeDirection.CALL,
        )
        == "price_zone_reject"
    )


def test_apply_price_zone_gate_reasons():
    assert (
        apply_price_zone_gate({"bb_pct_b": 0.5, "keltner": 0.5}, TradeDirection.CALL, _zone_cfg()) == "price_zone_none"
    )
    assert (
        apply_price_zone_gate(
            {"bb_pct_b": 0.8, "keltner": 0.8, "trend_direction": "PUT", "dl_direction": "PUT"},
            TradeDirection.CALL,
            _zone_cfg(),
        )
        == "price_zone_sell_requires_put"
    )
    assert (
        apply_price_zone_gate(
            {"bb_pct_b": 0.2, "keltner": 0.2, "trend_direction": "PUT", "dl_direction": "CALL"},
            TradeDirection.CALL,
            _zone_cfg(),
        )
        == "price_zone_trend_conflict"
    )
    assert (
        apply_price_zone_gate(
            {"bb_pct_b": 0.2, "keltner": 0.2, "trend_direction": "CALL"},
            TradeDirection.CALL,
            _zone_cfg(),
            tcn_direction=TradeDirection.PUT,
        )
        == "price_zone_tcn_conflict"
    )
    assert apply_price_zone_gate({"bb_pct_b": 0.2}, TradeDirection.CALL, {"price_zone": {"enabled": False}}) is None


def test_initial_direction_checks_skips_outside_zone():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.50,
            "keltner": 0.50,
            "trend_direction": "CALL",
            "call_votes": 3,
            "put_votes": 1,
        },
    }
    assert initial_direction_checks(entry, _zone_cfg()) is None
    assert entry["metrics"]["gate_reason"] == "price_zone_none"


def test_initial_direction_checks_accepts_buy_zone_call():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.20,
            "keltner": 0.25,
            "trend_direction": "CALL",
            "call_votes": 4,
            "put_votes": 1,
        },
    }
    result = initial_direction_checks(entry, _zone_cfg())
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1]["price_zone"] == ZONE_BUY


def test_initial_direction_checks_rejects_put_in_buy_zone():
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
    assert initial_direction_checks(entry, _zone_cfg()) is None


def test_finalize_rejects_side_eq_flip_against_zone():
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
            exec_cfg=_zone_cfg(),
        )
    assert out is None
    assert "price_zone" in str(entry["metrics"].get("gate_reason") or "")
