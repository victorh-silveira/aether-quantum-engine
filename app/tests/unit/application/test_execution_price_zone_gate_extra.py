from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _zone_cfg(**overrides):
    base = {
        "enabled": True,
        "buy_max": 0.35,
        "sell_min": 0.65,
        "bb_weight": 0.6,
        "keltner_weight": 0.4,
        "neutral_mode": "reject",
        "require_trend_agreement": True,
        "require_tcn_agreement": True,
    }
    base.update(overrides)
    return {"price_zone": base}


def test_initial_direction_checks_waives_price_zone_under_starvation():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.30,
            "raw_prob": 0.30,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.15,
            "keltner": 0.20,
            "trend_direction": "CALL",
            "call_votes": 4,
            "put_votes": 1,
        },
    }
    result = initial_direction_checks(
        entry,
        _zone_cfg(require_trend_agreement=False),
        skipped_cycles_counter=6,
    )
    assert result is not None
    assert result[0] == TradeDirection.PUT
    assert result[1].get("price_zone_starvation_waiver") is True
    assert result[1].get("price_zone_waived_reason") == "price_zone_tcn_conflict"


def test_initial_direction_checks_waives_price_zone_on_high_conviction():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.20,
            "raw_prob": 0.20,
            "direction_margin": 0.30,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.15,
            "keltner": 0.20,
            "trend_direction": "CALL",
            "call_votes": 4,
            "put_votes": 1,
        },
    }
    result = initial_direction_checks(
        entry,
        _zone_cfg(require_trend_agreement=False),
        skipped_cycles_counter=0,
    )
    assert result is not None
    assert result[0] == TradeDirection.PUT
    assert result[1].get("price_zone_conviction_waiver") is True


def test_resolve_execution_direction_waives_price_zone_in_finalize_under_starvation():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.30,
            "raw_prob": 0.30,
            "deploy_ok": True,
            "execute": True,
            "bb_pct_b": 0.15,
            "keltner": 0.20,
            "trend_direction": "CALL",
            "call_votes": 4,
            "put_votes": 1,
            "predicted_payoff_edge": 0.40,
            "meta_classifier_applied": True,
            "val_accuracy": 0.70,
            "indicators": {"adx": 0.30, "vol_ratio": 1.0, "bb_pct_b": 0.15, "keltner": 0.20},
        },
    }
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": False}},
            "risk_management": {"min_validation_accuracy_gate": 0.0},
        },
        _quality_skipped_cycles_counter=15,
    )
    with patch(
        "src.application.services.execution_direction_resolver.resolve_meta_payoff_edge",
        return_value=(0.40, True),
    ):
        result = resolve_execution_direction(
            entry,
            exec_cfg=_zone_cfg(require_trend_agreement=False),
            symbol="R_10",
            orch=orch,
            skipped_cycles_counter=15,
        )
    assert result is not None
    assert result[0] == TradeDirection.PUT
    assert entry["metrics"].get("price_zone_starvation_waiver") is True
    assert entry["metrics"].get("gate_reason") != "price_zone_tcn_conflict"
