"""Testes de adapt SCALE por explosao e mili+tape (padroes live)."""

from unittest.mock import patch

from src.application.services.execution_scale_adapt import apply_scale_direction_adapt
from src.domain.models.trade import TradeDirection


def test_adapt_c4_like_mili_tape_vs_tcn_when_mini_pair_split():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "mili_tape_vs_tcn"
    assert metrics["scale_micro_regime"] == "chop"


def test_adapt_explosion_vs_tcn():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "explosion_vs_tcn"
    assert metrics["scale_micro_regime"] == "explosion"


def test_adapt_explosion_disabled_falls_to_mili_tape():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_on_retraction": True,
            "adapt_on_explosion": False,
            "adapt_on_mili_tape": True,
            "adapt_require_bar_pair_agree": True,
            "adapt_require_raw_extreme": True,
            "adapt_allow_strong_tape": True,
            "retraction_require_mili": True,
            "retraction_use_tick_accel": True,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "mili_tape_vs_tcn"


def test_adapt_explosion_invalid_live_side_noop():
    from src.application.services.execution_scale_adapt_regimes import adapt_on_explosion

    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }

    def _fake(m, _tcn, cfg=None):
        m["scale_micro_regime"] = "explosion"
        m["scale_micro_side"] = None
        return m

    with patch(
        "src.application.services.execution_scale_adapt_regimes.classify_micro_regime",
        side_effect=_fake,
    ):
        assert adapt_on_explosion(metrics, TradeDirection.CALL, {"adapt_on_explosion": True}) is None

    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_on_retraction": True,
            "adapt_on_explosion": True,
            "adapt_on_mili_tape": False,
            "adapt_require_bar_pair_agree": True,
            "adapt_require_raw_extreme": True,
            "adapt_allow_strong_tape": True,
            "retraction_require_mili": True,
            "retraction_use_tick_accel": True,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapt_reason"] == "need_bar_pair"
