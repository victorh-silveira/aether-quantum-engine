"""Testes de adaptacao de direcao e teto de stake SCALE."""

from unittest.mock import patch

from src.application.services.execution_scale_adapt import apply_scale_direction_adapt
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


def test_adapt_direction_tape_vs_tcn_raw_extreme():
    metrics = _pair_call(calibration_mode="raw_extreme")
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["tcn_direction"] == "PUT"
    assert metrics["scale_adapt_reason"] == "majority_votes"


def test_adapt_strong_cal_allows_majority_vs_tcn():
    metrics = _pair_call(calibration_mode="calibrated", direction_margin=0.07, scale_tape_strong=True)
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "majority_votes"


def test_adapt_pending_strong_cal_tape_without_majority():
    metrics = _pair_call(
        calibration_mode="calibrated",
        direction_margin=0.12,
        scale_tape_strong=True,
        pending_loss_total=25.0,
    )
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_require_raw_extreme": False,
            "adapt_require_bar_pair_agree": True,
            "adapt_allow_strong_tape": True,
            "adapt_on_retraction": True,
            "adapt_on_explosion": True,
            "adapt_on_mili_tape": True,
            "adapt_on_majority_votes": False,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is True


def test_adapt_raw_extreme_flips_with_strong_cal_margin():
    metrics = _pair_call(calibration_mode="raw_extreme", direction_margin=0.07)
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "majority_votes"


def test_kelly_side_sync_floors_conviction_after_adapt():
    from src.application.services.execution_scale_adapt import apply_scale_kelly_side_sync

    metrics = {
        "scale_adapted": True,
        "tcn_direction": "PUT",
        "calibrated_prob": 0.48,
        "trade_score": 0.48,
        "conviction": 0.48,
    }
    apply_scale_kelly_side_sync(metrics, TradeDirection.CALL)
    assert metrics["scale_kelly_side_synced"] is True
    assert float(metrics["conviction"]) >= 0.55
    assert float(metrics["trade_score"]) >= 0.55


def test_kelly_side_sync_floors_when_not_adapted():
    from src.application.services.execution_scale_adapt import apply_scale_kelly_side_sync

    metrics = {"scale_adapted": False, "calibrated_prob": 0.48, "conviction": 0.48}
    apply_scale_kelly_side_sync(metrics, TradeDirection.CALL)
    assert metrics["scale_kelly_side_synced"] is False
    assert float(metrics["conviction"]) >= 0.55


def test_kelly_side_sync_bad_cal_and_same_side():
    from src.application.services.execution_scale_adapt import apply_scale_kelly_side_sync

    metrics = {"scale_adapted": True, "tcn_direction": "PUT", "calibrated_prob": None, "conviction": 0.4}
    apply_scale_kelly_side_sync(metrics, TradeDirection.CALL)
    assert metrics["scale_kelly_side_synced"] is False
    assert float(metrics["conviction"]) >= 0.55
    metrics_bad = {"scale_adapted": True, "tcn_direction": "PUT", "calibrated_prob": "x", "conviction": 0.4}
    apply_scale_kelly_side_sync(metrics_bad, TradeDirection.CALL)
    assert metrics_bad["scale_kelly_side_synced"] is False
    assert float(metrics_bad["conviction"]) >= 0.55
    metrics2 = {"scale_adapted": True, "tcn_direction": "CALL", "calibrated_prob": 0.7, "conviction": 0.7}
    apply_scale_kelly_side_sync(metrics2, TradeDirection.CALL)
    assert metrics2["scale_kelly_side_synced"] is False
    assert float(metrics2["conviction"]) >= 0.7


def test_tape_strong_mili_reinforce_and_oppose_edges():
    from src.application.services.execution_scale_tape import compute_tape_strong, mini_pair_opposes_tcn

    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_micro_prev_bar_dir": None,
        "scale_micro_bar_dir": None,
    }
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=False) is True
    assert mini_pair_opposes_tcn(metrics, None) is False
    assert mini_pair_opposes_tcn({"scale_mini_prev_bar_dir": None, "scale_mini_bar_dir": "CALL"}, "PUT") is False
    assert mini_pair_opposes_tcn({"scale_mini_prev_bar_dir": "CALL", "scale_mini_bar_dir": "PUT"}, "PUT") is False


def test_adapt_blocks_split_mini_pair_like_c1():
    metrics = {
        "scale_tape_consensus": "CALL",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_tape_strong": False,
        "calibration_mode": "raw_extreme",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "chop_hold"


def test_adapt_requires_raw_extreme_when_tape_not_strong():
    metrics = _pair_call(
        calibration_mode="calibrated",
        scale_tape_strong=False,
        scale_mili_dir="PUT",
    )
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "chop_hold"


def test_adapt_skip_chop_holds_tcn():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "calibration_mode": "raw_extreme",
        "calibrated_prob": 0.45,
        "rsi": 0.35,
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.CALL
    assert metrics["scale_micro_regime"] == "chop"
    assert metrics["scale_adapt_reason"] == "chop_hold"


def test_adapt_cal_disagree_blocks_anti_cal_flip():
    metrics = _pair_call(
        calibration_mode="raw_extreme",
        calibrated_prob=0.56,
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_mili_dir="PUT",
        rsi=0.35,
    )
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "cal_disagree"


def test_adapt_no_consensus_keeps_tcn():
    metrics = {"scale_tape_consensus": None, "calibration_mode": "raw_extreme"}
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "chop_hold"


def test_adapt_invalid_calibrated_prob_and_need_raw_extreme():
    metrics = _pair_call(
        calibration_mode="calibrated",
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_tape_strong=False,
        scale_mili_dir="PUT",
        calibrated_prob=0.44,
    )
    cfg = {
        "enabled": True,
        "adapt_direction_enabled": True,
        "adapt_require_raw_extreme": True,
        "adapt_require_bar_pair_agree": True,
        "adapt_allow_strong_tape": False,
        "adapt_on_majority_votes": False,
        "adapt_skip_chop": True,
        "adapt_require_cal_agree": True,
    }
    with (
        patch("src.application.services.execution_scale_adapt.parse_scale_vision_config", return_value=cfg),
        patch("src.application.services.execution_scale_adapt.try_regime_adapts", return_value=None),
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapt_reason"] == "need_raw_extreme"

    metrics_bad_cal = _pair_call(
        calibration_mode="raw_extreme",
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_tape_strong=True,
        scale_mili_dir="PUT",
        calibrated_prob="bad",
    )
    with patch("src.application.services.execution_scale_adapt.parse_scale_vision_config", return_value=cfg):
        bad_out = apply_scale_direction_adapt(metrics_bad_cal, TradeDirection.CALL)
    assert bad_out == TradeDirection.PUT

    put_cal = _pair_call(
        calibration_mode="raw_extreme",
        calibrated_prob=0.44,
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_mili_dir="PUT",
    )
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={**cfg, "adapt_on_majority_votes": False, "adapt_skip_chop": False},
    ):
        put_side = apply_scale_direction_adapt(put_cal, TradeDirection.CALL)
    assert put_side == TradeDirection.PUT

    none_cal = _pair_call(calibration_mode="raw_extreme", calibrated_prob=None, scale_tape_consensus="PUT")
    with (
        patch("src.application.services.execution_scale_adapt.parse_scale_vision_config", return_value=cfg),
        patch(
            "src.application.services.execution_scale_adapt.adapt_on_majority_votes",
            return_value=TradeDirection.CALL,
        ),
    ):
        held = apply_scale_direction_adapt(none_cal, TradeDirection.CALL)
    assert held == TradeDirection.CALL

    metrics_put = _pair_call(
        calibration_mode="raw_extreme",
        calibrated_prob=0.44,
        scale_tape_consensus="PUT",
        scale_mini_prev_bar_dir="PUT",
        scale_mini_bar_dir="PUT",
        scale_mili_dir="PUT",
    )
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            **cfg,
            "adapt_on_majority_votes": True,
            "adapt_majority_min_lead": 1,
            "adapt_majority_min_votes": 2,
            "adapt_majority_include_rsi": False,
            "adapt_require_cal_agree": False,
            "adapt_skip_chop": False,
        },
    ):
        same = apply_scale_direction_adapt(metrics_put, TradeDirection.PUT)
    assert same == TradeDirection.PUT
