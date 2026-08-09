"""Testes adicionais de adaptacao SCALE direcao, tape e sizing."""

from unittest.mock import patch

from src.application.services.execution_scale_adapt import apply_scale_direction_adapt
from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.domain.models.trade import TradeDirection


def _pair_call(**extra):
    base = {
        "scale_tape_consensus": "CALL",
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_tape_strong": True,
        "scale_mili_dir": "CALL",
    }
    base.update(extra)
    return base


def test_adapt_when_raw_extreme_flag_off():
    metrics = _pair_call(calibration_mode="calibrated", scale_tape_strong=False)
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_require_raw_extreme": False,
            "adapt_require_bar_pair_agree": True,
            "adapt_allow_strong_tape": False,
            "adapt_on_majority_votes": False,
            "adapt_skip_chop": False,
            "adapt_require_cal_agree": False,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapt_reason"] == "tape_vs_tcn"


def test_compute_tape_strong_via_micro_pair():
    from src.application.services.execution_scale_tape import compute_tape_strong

    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": None,
        "scale_micro_prev_bar_dir": "CALL",
        "scale_micro_bar_dir": "CALL",
    }
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=True) is True
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=False) is True
    weak = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "scale_micro_prev_bar_dir": None,
        "scale_micro_bar_dir": None,
    }
    assert compute_tape_strong(weak, "CALL", mini_pair_sufficient=False) is False
    assert compute_tape_strong({"scale_mini_prev_bar_dir": "PUT", "scale_mini_bar_dir": "CALL"}, "CALL") is False


def test_c7_style_mini_pair_strong_without_mili():
    from src.application.services.execution_scale_tape import compute_tape_strong, mini_pair_opposes_tcn

    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
    }
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=True) is True
    assert mini_pair_opposes_tcn(metrics, "PUT") is True


def test_adapt_aligned_consensus():
    metrics = _pair_call(
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_mili_dir="PUT",
    )
    metrics["calibration_mode"] = "raw_extreme"
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "aligned"


def test_adapt_disabled_flags():
    metrics = _pair_call(calibration_mode="raw_extreme")
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": False,
            "adapt_direction_enabled": True,
            "adapt_require_raw_extreme": True,
            "adapt_require_bar_pair_agree": True,
            "adapt_allow_strong_tape": True,
        },
    ):
        assert apply_scale_direction_adapt(metrics, TradeDirection.PUT) == TradeDirection.PUT
    metrics2 = _pair_call(calibration_mode="raw_extreme")
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": False,
            "adapt_require_raw_extreme": True,
            "adapt_require_bar_pair_agree": True,
            "adapt_allow_strong_tape": True,
        },
    ):
        assert apply_scale_direction_adapt(metrics2, TradeDirection.PUT) == TradeDirection.PUT
        assert metrics2["scale_adapt_reason"] == "adapt_off"


def test_sizing_on_adapted_sets_force_explore_and_cap():
    metrics = {"kelly_fraction_scale": 1.0, "scale_adapted": True, "scale_discordance": False}
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.CALL, metrics)
    assert metrics["scale_force_explore"] is True
    assert metrics["kelly_fraction_scale"] < 1.0
    assert float(metrics["scale_max_stake_pct"]) > 0.0
