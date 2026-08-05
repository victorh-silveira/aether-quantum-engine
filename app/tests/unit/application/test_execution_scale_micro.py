"""Testes do classificador micro explosion/retraction/chop."""

from src.application.services.execution_scale_adapt import apply_scale_direction_adapt
from src.application.services.execution_scale_micro import classify_micro_regime, micro_regime_token
from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.domain.models.trade import TradeDirection


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


def test_adapt_on_retraction_vs_tcn_call():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "raw_extreme",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "retraction"


def test_adapt_retraction_when_need_bar_pair_blocks_tape():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "retraction"


def test_sizing_dampens_c2_like_chop_mili_oppose():
    metrics = {
        "kelly_fraction_scale": 1.0,
        "scale_adapted": False,
        "scale_discordance": False,
        "scale_micro_regime": "chop",
        "scale_mili_oppose_tcn": True,
    }
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.CALL, metrics)
    assert metrics["kelly_fraction_scale"] < 1.0
    assert metrics["scale_force_explore"] is True


def test_sizing_dampens_retraction_regime():
    metrics = {
        "kelly_fraction_scale": 1.0,
        "scale_adapted": False,
        "scale_discordance": False,
        "scale_micro_regime": "retraction",
        "scale_mili_oppose_tcn": False,
    }
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.CALL, metrics)
    assert metrics["scale_sizing_reason"] == "retraction"
    assert float(metrics["scale_max_stake_pct"]) > 0.0


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
    from src.application.services.execution_scale_micro import _tick_confirms_side

    metrics = {"flow_features": {"price_velocity": object(), "micro_tick_acceleration": object()}}
    assert _tick_confirms_side(metrics, "PUT") is False


def test_adapt_on_retraction_disabled_and_no_consensus():
    from unittest.mock import patch

    metrics = {
        "scale_tape_consensus": None,
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_on_retraction": False,
            "adapt_on_explosion": False,
            "adapt_on_mili_tape": False,
            "adapt_require_bar_pair_agree": True,
            "adapt_require_raw_extreme": True,
            "adapt_allow_strong_tape": True,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapt_reason"] == "no_consensus"


def test_adapt_on_retraction_same_side_noop():
    from unittest.mock import patch

    from src.application.services.execution_scale_adapt import _adapt_on_retraction

    metrics = {}

    def _fake(m, _tcn, cfg=None):
        m["scale_retraction_vs_tcn"] = True
        m["scale_micro_side"] = "CALL"
        return m

    with patch("src.application.services.execution_scale_adapt_regimes.classify_micro_regime", side_effect=_fake):
        assert _adapt_on_retraction(metrics, TradeDirection.CALL, {"adapt_on_retraction": True}) is None


def test_adapt_on_retraction_invalid_live_side():
    from unittest.mock import patch

    from src.application.services.execution_scale_adapt import _adapt_on_retraction

    metrics = {}

    def _fake(m, _tcn, cfg=None):
        m["scale_retraction_vs_tcn"] = True
        m["scale_micro_side"] = None
        return m

    with patch("src.application.services.execution_scale_adapt_regimes.classify_micro_regime", side_effect=_fake):
        assert _adapt_on_retraction(metrics, TradeDirection.CALL, {"adapt_on_retraction": True}) is None


def test_adapt_retraction_when_no_tape_consensus():
    metrics = {
        "scale_tape_consensus": None,
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "retraction"


def test_adapt_retraction_after_need_raw_extreme_gate():
    from unittest.mock import patch

    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    with patch(
        "src.application.services.execution_scale_adapt._adapt_on_retraction",
        return_value=TradeDirection.PUT,
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT


def test_adapt_tape_without_raw_extreme_flag():
    from unittest.mock import patch

    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_strong": False,
        "calibration_mode": "calibrated",
    }
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={
            "enabled": True,
            "adapt_direction_enabled": True,
            "adapt_on_retraction": False,
            "adapt_on_explosion": False,
            "adapt_on_mili_tape": False,
            "adapt_require_bar_pair_agree": True,
            "adapt_require_raw_extreme": False,
            "adapt_allow_strong_tape": True,
            "retraction_require_mili": True,
        },
    ):
        out = apply_scale_direction_adapt(metrics, TradeDirection.CALL)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "tape_vs_tcn"
