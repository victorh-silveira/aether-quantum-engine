import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import (
    _trade_score,
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
)
from src.application.services.meta_payoff_veto_gate import (
    META_PAYOFF_NEGATIVE_ZSCORE_VETO,
    apply_meta_payoff_negative_zscore_veto,
)
from src.domain.models.trade import TradeDirection


def _entry(direction=TradeDirection.CALL, raw_prob=0.62, **metrics):
    base = {
        "execute": True,
        "deploy_ok": True,
        "raw_prob": raw_prob,
        "trade_score": max(raw_prob, 1.0 - raw_prob),
        "val_accuracy": 0.60,
        "edge": abs(raw_prob - 0.5),
        "trend_direction": "CALL",
        "indicators": {
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.52,
            "keltner": 0.55,
            "cmo": 0.05,
        },
    }
    base.update(metrics)
    return {"direction": direction, "metrics": base}


def test_mandatory_pool_eligible_requires_inferable_direction():
    assert mandatory_pool_eligible(_entry()) is True
    assert mandatory_pool_eligible({"direction": None, "metrics": {"deploy_ok": True}}) is False


def test_mandatory_pool_eligible_ignores_signal_veto():
    entry = _entry()
    apply_meta_payoff_negative_zscore_veto(entry["metrics"])
    assert entry["metrics"]["gate_reason"] == META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert mandatory_pool_eligible(entry) is True


def test_trade_score_falls_back_when_veto_nulled_scores():
    metrics = {"trade_score": None, "conviction": None, "raw_prob": 0.71}
    assert _trade_score(metrics) == pytest.approx(0.71, abs=1e-6)


def test_market_decision_score_prefers_higher_raw_side():
    low = market_decision_score(_entry(raw_prob=0.52)["metrics"])
    high = market_decision_score(_entry(raw_prob=0.80)["metrics"])
    assert high > low


def test_build_market_execution_candidate_uses_resolver():
    built = build_market_execution_candidate(
        "R_10",
        _entry(direction=TradeDirection.CALL, raw_prob=0.62),
        exec_cfg={"price_zone": {"enabled": False}},
    )
    assert built is not None
    symbol, direction, metrics = built
    assert symbol == "R_10"
    assert direction == TradeDirection.CALL
    assert "exec_direction" in metrics


def test_market_decision_score_recovery_indicator_adjustments():
    metrics = {
        "raw_prob": 0.62,
        "val_accuracy": 0.60,
        "edge": 0.12,
        "execute": True,
        "deploy_ok": True,
        "indicators": {"adx": 0.15, "vol_ratio": 0.90, "hurst": 0.40},
    }
    low_adx = market_decision_score(metrics, recovery_active=True, symbol="R_10")
    high_adx = market_decision_score(
        {
            **metrics,
            "indicators": {"adx": 0.30, "vol_ratio": 1.10, "hurst": 0.60},
        },
        recovery_active=True,
        symbol="R_10",
    )
    assert high_adx > low_adx


def test_market_decision_score_penalizes_low_margin():
    aligned = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.10)["metrics"])
    low_margin = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.03)["metrics"])
    assert aligned > low_margin


def test_market_decision_score_penalizes_squeeze_in_recovery():
    base = {
        "raw_prob": 0.70,
        "val_accuracy": 0.60,
        "edge": 0.20,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.20,
    }
    clean = market_decision_score(base, recovery_active=True, symbol="R_10")
    squeezed = market_decision_score(
        {**base, "meta_squeeze_downgrade": True},
        recovery_active=True,
        symbol="R_10",
    )
    assert clean > squeezed


def test_resolve_keeps_dl_side_on_low_val_accuracy():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.58,
        val_accuracy=0.45,
        trend_direction="PUT",
        call_votes=4,
        put_votes=2,
        indicators={
            "hurst": 0.52,
            "adx": 0.18,
            "vol_ratio": 0.90,
            "rsi": 0.50,
            "keltner": 0.55,
            "cmo": 0.05,
        },
    )
    result = resolve_execution_direction(entry, symbol="R_10", exec_cfg={"price_zone": {"enabled": False}})
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL


def test_market_decision_score_penalizes_mid_brier_and_ece():
    base = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.12)["metrics"])
    mid_brier = market_decision_score(
        _entry(raw_prob=0.80, direction_margin=0.12, deploy_settlement_brier=0.23)["metrics"]
    )
    high_brier = market_decision_score(
        _entry(raw_prob=0.80, direction_margin=0.12, deploy_settlement_brier=0.28)["metrics"]
    )
    high_ece = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.12, val_ece=0.12)["metrics"])
    assert mid_brier < base
    assert high_brier < mid_brier
    assert high_ece < base
