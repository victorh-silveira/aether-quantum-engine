"""Classificador micro explosion/retraction/chop (telemetria SCALE)."""

from src.application.services.execution_scale_micro import (
    _tick_confirms_side,
    classify_micro_regime,
    micro_regime_token,
)


def test_micro_regime_token():
    assert micro_regime_token("explosion") == "explos"
    assert micro_regime_token("retraction") == "retract"
    assert micro_regime_token("chop") == "chop"
    assert micro_regime_token(None) == "chop"


def test_classify_explosion_aligned():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    classify_micro_regime(metrics, "CALL", cfg={})
    assert metrics["scale_micro_regime"] == "explosion"
    assert metrics["scale_retraction_vs_tcn"] is False


def test_classify_c2_like_mili_oppose_is_chop():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
    }
    classify_micro_regime(metrics, "CALL", cfg={})
    assert metrics["scale_micro_regime"] == "chop"
    assert metrics["scale_mili_oppose_tcn"] is True


def test_classify_c4_like_retraction_vs_tcn():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    cfg = {"retraction_require_mili": True, "retraction_use_tick_accel": True}
    classify_micro_regime(metrics, "PUT", cfg=cfg)
    assert metrics["scale_micro_regime"] == "retraction"
    assert metrics["scale_micro_side"] == "CALL"
    assert metrics["scale_retraction_vs_tcn"] is True


def test_retraction_without_mili_uses_tick_when_allowed():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": None,
        "flow_features": {"price_velocity": -0.5},
    }
    cfg = {"retraction_require_mili": False, "retraction_use_tick_accel": True}
    classify_micro_regime(metrics, "CALL", cfg=cfg)
    assert metrics["scale_micro_regime"] == "retraction"
    assert metrics["scale_retraction_vs_tcn"] is True


def test_tick_confirms_handles_bad_flow_values():
    metrics = {"flow_features": {"price_velocity": object(), "micro_tick_acceleration": object()}}
    assert _tick_confirms_side(metrics, "PUT") is False
