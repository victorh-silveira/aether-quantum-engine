from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_discordance import (
    _macro_indicator_float,
    _rsi_di_oppose_direction,
    align_direction_to_rsi_trend,
    apply_technical_agreement,
)
from src.application.services.execution_price_zone_gate import align_or_keep_meta_side
from src.application.services.side_equilibrium_helpers import (
    alternate_side_is_preferable,
    flip_conflicts_price_zone,
    primary_side_is_toxic,
)
from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_PASS,
    SideEquilibriumDecision,
)
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
    assert (
        _rsi_di_oppose_direction(
            {"macro_indicators": {"rsi": 0.2, "di_diff": -0.1}},
            TradeDirection.CALL,
        )
        is True
    )


def test_apply_technical_agreement_vetoes():
    metrics = {
        "call_votes": 1,
        "put_votes": 3,
        "trend_direction": "PUT",
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
        },
    )
    assert veto is True
    assert metrics.get("indicator_side_discordance") is True
    assert metrics.get("indicator_trend_discordance") is True


def test_align_direction_to_rsi_trend():
    assert align_direction_to_rsi_trend(TradeDirection.PUT, {}) == TradeDirection.PUT

    m_neutral = {"macro_indicators": {"rsi": 0.51}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_neutral) == TradeDirection.PUT

    m_bull = {"macro_indicators": {"rsi": 0.70}}
    assert align_direction_to_rsi_trend(TradeDirection.PUT, m_bull) == TradeDirection.CALL
    assert m_bull.get("rsi_trend_flipped") is True

    m_bear = {"macro_indicators": {"rsi": 0.30}}
    assert align_direction_to_rsi_trend(TradeDirection.CALL, m_bear) == TradeDirection.PUT
    assert m_bear.get("rsi_trend_flipped") is True

    m_bull_same = {"macro_indicators": {"rsi": 0.70}}
    assert align_direction_to_rsi_trend(TradeDirection.CALL, m_bull_same) == TradeDirection.CALL

    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.70}}, TradeDirection.PUT) is True

    m_align = {"rsi_trend_align_enabled": True, "macro_indicators": {"rsi": 0.70}}
    assert (
        align_or_keep_meta_side(
            TradeDirection.PUT,
            m_align,
            dl_dir=TradeDirection.PUT,
            predicted_edge=-0.05,
            meta_applied=True,
        )
        == TradeDirection.CALL
    )


def test_initial_direction_checks_discordance_reject():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
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
            {"discordance_veto_enabled": True, "require_indicator_consensus": True},
        )
        is None
    )
    assert entry["metrics"].get("gate_reason") == "indicator_discordance"


def test_side_eq_helpers():
    pass_dec = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.55)
    assert primary_side_is_toxic(pass_dec) is False
    toxic = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="x", side_wr=None)
    assert primary_side_is_toxic(toxic) is True
    assert flip_conflicts_price_zone(TradeDirection.CALL, {"price_zone_direction": "PUT"}) is True
    alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.6, call_n=3, call_wins=2)
    pri = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="x", side_wr=None)
    assert alternate_side_is_preferable(pri, alt, opposite=TradeDirection.CALL) is True
    alt_none = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=None, call_n=3)
    assert alternate_side_is_preferable(pri, alt_none, opposite=TradeDirection.CALL) is False
    thin_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.6, call_n=2, call_wins=1)
    assert alternate_side_is_preferable(pri, thin_alt, opposite=TradeDirection.CALL) is False
    weak_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.50, call_n=3, call_wins=1)
    toxic_pri = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="x", side_wr=0.0)
    assert alternate_side_is_preferable(toxic_pri, weak_alt, opposite=TradeDirection.CALL) is False
    strong_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.55, call_n=3, call_wins=2)
    assert alternate_side_is_preferable(toxic_pri, strong_alt, opposite=TradeDirection.CALL) is True
    better_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.70, call_n=3, call_wins=2)
    assert alternate_side_is_preferable(toxic_pri, better_alt, opposite=TradeDirection.CALL) is True
