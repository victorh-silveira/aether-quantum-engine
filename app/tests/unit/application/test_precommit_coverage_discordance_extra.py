from src.application.services.execution_direction_discordance import (
    _multi_bar_ema_trend_alignment,
    align_direction_to_rsi_trend,
    apply_technical_agreement,
)
from src.domain.models.trade import TradeDirection


def test_apply_technical_agreement_edge_waives_discordance():
    metrics = {
        "call_votes": 0,
        "put_votes": 4,
        "trend_direction": "PUT",
        "direction_margin": 0.04,
        "calibrated_prob": 0.54,
        "predicted_payoff_edge": 0.20,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.54,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
        },
        skipped_cycles_counter=0,
    )
    assert veto is False
    assert metrics.get("indicator_discordance_edge_waiver") is True
    bad = {
        "call_votes": 0,
        "put_votes": 4,
        "trend_direction": "PUT",
        "direction_margin": 0.15,
        "calibrated_prob": 0.65,
        "predicted_payoff_edge": "bad",
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto_bad = apply_technical_agreement(
        bad,
        TradeDirection.CALL,
        0.65,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
            "discordance": {"waiver_margin": 0.40},
        },
        skipped_cycles_counter=0,
    )
    assert veto_bad is True


def test_multi_bar_ema_trend_alignment():
    assert _multi_bar_ema_trend_alignment({"macro_indicators": {"ema_9": 1.1, "ema_21": 1.0}}) == "CALL"
    assert _multi_bar_ema_trend_alignment({"macro_indicators": {"ema_9": 1.0, "ema_21": 1.1}}) == "PUT"
    assert _multi_bar_ema_trend_alignment({}) is None
    assert _multi_bar_ema_trend_alignment({"macro_indicators": {"ema_9": 1.0}}) is None


def test_align_direction_rsi_trend_hurst_above_50_extremes():
    assert (
        align_direction_to_rsi_trend(
            TradeDirection.CALL,
            {"macro_indicators": {"rsi": 0.85, "hurst": 0.60, "adx": 0.25}},
        )
        == TradeDirection.PUT
    )
    assert (
        align_direction_to_rsi_trend(
            TradeDirection.PUT,
            {"macro_indicators": {"rsi": 0.10, "hurst": 0.60, "adx": 0.25}},
        )
        == TradeDirection.CALL
    )


def test_align_direction_rsi_trend_ema_multi_bar():
    assert (
        align_direction_to_rsi_trend(
            TradeDirection.CALL,
            {"macro_indicators": {"rsi": 0.70, "hurst": 0.60, "ema_9": 1.1, "ema_21": 1.0}},
        )
        == TradeDirection.CALL
    )
    assert (
        align_direction_to_rsi_trend(
            TradeDirection.PUT,
            {"macro_indicators": {"rsi": 0.30, "hurst": 0.60, "ema_9": 1.0, "ema_21": 1.1}},
        )
        == TradeDirection.PUT
    )
