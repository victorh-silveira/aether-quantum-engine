from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_discordance import (
    _macro_indicator_float,
    _rsi_di_oppose_direction,
    align_direction_to_rsi_trend,
    apply_technical_agreement,
)
from src.application.services.execution_price_zone_gate import align_or_keep_meta_side
from src.domain.models.trade import TradeDirection


def test_macro_indicator_float_and_rsi_di():
    assert _macro_indicator_float({}, "rsi") is None
    assert _macro_indicator_float({"indicators": {"rsi": "bad"}}, "rsi") is None
    assert _macro_indicator_float({"macro_indicators": {"rsi": 0.7}}, "rsi") == 0.7
    assert _rsi_di_oppose_direction({}, TradeDirection.CALL) is False
    assert (
        _rsi_di_oppose_direction(
            {"macro_indicators": {"rsi": 0.51, "di_diff": 0.1}},
            TradeDirection.CALL,
        )
        is False
    )
    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.50}}, TradeDirection.CALL) is False
    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.55}}, TradeDirection.PUT) is False
    assert (
        _rsi_di_oppose_direction(
            {"macro_indicators": {"rsi": 0.60, "di_diff": 0.02}},
            TradeDirection.PUT,
        )
        is False
    )
    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.58}}, TradeDirection.PUT) is False
    assert (
        _rsi_di_oppose_direction(
            {"macro_indicators": {"rsi": 0.2, "di_diff": -0.1}},
            TradeDirection.CALL,
        )
        is True
    )


def test_apply_technical_agreement_margin_waives_discordance():
    metrics = {
        "call_votes": 1,
        "put_votes": 3,
        "trend_direction": "PUT",
        "direction_margin": 0.11,
        "calibrated_prob": 0.61,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.61,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
        },
        skipped_cycles_counter=0,
    )
    assert veto is False
    assert metrics.get("indicator_discordance_margin_waiver") is True
    assert metrics.get("gate_reason") is None


def test_apply_technical_agreement_trend_needs_vote_opposition():
    metrics = {
        "call_votes": 3,
        "put_votes": 1,
        "trend_direction": "PUT",
        "macro_indicators": {"rsi": 0.50},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.62,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
        },
    )
    assert veto is False
    assert metrics.get("indicator_trend_discordance") is not True


def test_apply_technical_agreement_vetoes():
    metrics = {
        "call_votes": 1,
        "put_votes": 3,
        "trend_direction": "PUT",
        "direction_margin": 0.15,
        "calibrated_prob": 0.65,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.7,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": False,
            "dynamic_threshold": {"require_indicator_consensus": True},
            "discordance": {"waiver_margin": 0.40},
        },
    )
    assert veto is True
    assert metrics.get("indicator_side_discordance") is True
    assert metrics.get("indicator_trend_discordance") is True


def test_apply_technical_agreement_starvation_waives_discordance():
    metrics = {
        "call_votes": 1,
        "put_votes": 3,
        "trend_direction": "PUT",
        "direction_margin": 0.15,
        "calibrated_prob": 0.65,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.7,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
            "discordance": {"waiver_margin": 0.40},
        },
        skipped_cycles_counter=6,
    )
    assert veto is False
    assert metrics.get("indicator_discordance_starvation_waiver") is True
    assert metrics.get("gate_reason") is None


def test_apply_technical_agreement_high_conviction_waives_discordance():
    metrics = {
        "call_votes": 0,
        "put_votes": 4,
        "trend_direction": "PUT",
        "direction_margin": 0.30,
        "calibrated_prob": 0.80,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.80,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
        },
        skipped_cycles_counter=0,
    )
    assert veto is False
    assert metrics.get("indicator_discordance_conviction_waiver") is True


def test_align_direction_to_rsi_trend():
    assert align_direction_to_rsi_trend(TradeDirection.PUT, {}) == TradeDirection.PUT

    m_candle = {"macro_indicators": {"open": 100.0, "close": 105.0}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_candle) == TradeDirection.CALL
    assert m_candle.get("candle_color_direction") == "CALL"
    from src.application.services.execution_direction_checks import _is_neutral_clamp

    assert _is_neutral_clamp(m_candle) is False

    m_neutral = {"macro_indicators": {"rsi": 0.50}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_neutral) == TradeDirection.PUT

    m_bull = {"macro_indicators": {"rsi": 0.70, "hurst": 0.55}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_bull) == TradeDirection.CALL
    assert m_bull.get("rsi_trend_flipped") is True

    m_bear = {"macro_indicators": {"rsi": 0.40, "hurst": 0.55}}
    assert align_direction_to_rsi_trend(TradeDirection.CALL, m_bear) == TradeDirection.PUT
    assert m_bear.get("rsi_trend_flipped") is True

    m_bull_same = {"macro_indicators": {"rsi": 0.70, "hurst": 0.55}}
    assert align_direction_to_rsi_trend(TradeDirection.CALL, m_bull_same) == TradeDirection.CALL

    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.70}}, TradeDirection.PUT) is True

    m_overbought = {"macro_indicators": {"rsi": 0.80, "adx": 0.25, "hurst": 0.55}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_overbought) == TradeDirection.CALL
    assert m_overbought.get("senior_trader_regime") == "strong_trend_impulse"

    # Hurst < 0.50 + RSI <= 0.32 = exaustao de fundo -> CALL (reversao a media)
    m_revert = {"macro_indicators": {"rsi": 0.30, "adx": 0.15, "hurst": 0.40}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_revert) == TradeDirection.CALL
    assert m_revert.get("micro_regime_mean_reversion") is True
    assert m_revert.get("senior_trader_regime") == "mean_reversion_exhaustion_bottom"

    # Hurst < 0.50 + RSI >= 0.68 = exaustao de topo -> PUT (reversao a media)
    m_exhaustion_top = {"macro_indicators": {"rsi": 0.75, "adx": 0.15, "hurst": 0.40}}
    assert align_direction_to_rsi_trend(TradeDirection.CALL, m_exhaustion_top) == TradeDirection.PUT
    assert m_exhaustion_top.get("senior_trader_regime") == "mean_reversion_exhaustion_top"

    # Hurst < 0.50 + RSI neutro (range) = range_momentum_alignment -> segue RSI
    m_range = {"macro_indicators": {"rsi": 0.60, "hurst": 0.40}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_range) == TradeDirection.CALL
    assert m_range.get("senior_trader_regime") == "range_momentum_alignment"

    # Sem hurst informado (default 0.40 < 0.50) + RSI < 0.32 = exaustao de fundo
    m_oversold = {"macro_indicators": {"rsi": 0.20}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_oversold) == TradeDirection.CALL
    assert m_oversold.get("senior_trader_regime") == "mean_reversion_exhaustion_bottom"

    m_align_low_margin = {
        "rsi_trend_align_enabled": True,
        "direction_margin": 0.010,
        "macro_indicators": {"rsi": 0.65},
    }
    assert (
        align_or_keep_meta_side(
            TradeDirection.PUT,
            m_align_low_margin,
            dl_dir=TradeDirection.PUT,
            predicted_edge=0.05,
            meta_applied=True,
        )
        == TradeDirection.CALL
    )


def test_initial_direction_checks_discordance_reject():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.65,
            "direction_margin": 0.15,
            "deploy_ok": True,
            "execute": True,
            "call_votes": 0,
            "put_votes": 4,
            "trend_direction": "PUT",
            "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
        },
    }
    assert (
        initial_direction_checks(
            entry,
            {
                "discordance_veto_enabled": True,
                "require_indicator_consensus": True,
                "discordance": {"waiver_margin": 0.40},
            },
        )
        is None
    )
    assert entry["metrics"].get("gate_reason") == "indicator_discordance"
    assert entry["metrics"].get("indicator_discordance_kind") == "votes+side+trend"


def test_apply_technical_agreement_weak_tcn_defers_discordance():
    metrics = {
        "call_votes": 0,
        "put_votes": 4,
        "trend_direction": "PUT",
        "direction_margin": 0.02,
        "calibrated_prob": 0.52,
        "macro_indicators": {"rsi": 0.2, "di_diff": -0.2},
    }
    _, veto = apply_technical_agreement(
        metrics,
        TradeDirection.CALL,
        0.52,
        {
            "discordance_veto_enabled": True,
            "require_indicator_consensus": True,
            "dynamic_threshold": {"require_indicator_consensus": True},
        },
        skipped_cycles_counter=0,
    )
    assert veto is False
    assert metrics.get("indicator_discordance_weak_tcn_defer") is True
