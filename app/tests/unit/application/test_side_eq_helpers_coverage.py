from unittest.mock import MagicMock, patch

from src.application.services.side_equilibrium_helpers import (
    alternate_side_is_preferable,
    flip_conflicts_price_zone,
    primary_side_is_toxic,
)
from src.domain.analytics.side_equilibrium import ACTION_HARD_SKIP, ACTION_PASS, SideEquilibriumDecision
from src.domain.models.trade import TradeDirection


def test_side_eq_helpers():
    pass_dec = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.55)
    assert primary_side_is_toxic(pass_dec) is False
    toxic = SideEquilibriumDecision(
        action=ACTION_HARD_SKIP, reason="x", side_wr=None, put_n=8, put_wins=0, call_n=0, call_wins=0
    )
    assert primary_side_is_toxic(toxic) is True
    assert flip_conflicts_price_zone(TradeDirection.CALL, {"price_zone_direction": "PUT"}) is True
    alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.6, call_n=8, call_wins=5)
    pri = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="x", side_wr=None, put_n=8, put_wins=0)
    assert alternate_side_is_preferable(pri, alt, opposite=TradeDirection.CALL) is True
    alt_none = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=None, call_n=8)
    assert alternate_side_is_preferable(pri, alt_none, opposite=TradeDirection.CALL) is False
    thin_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.6, call_n=2, call_wins=1)
    assert alternate_side_is_preferable(pri, thin_alt, opposite=TradeDirection.CALL) is False
    weak_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.50, call_n=8, call_wins=4)
    toxic_pri = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="x", side_wr=0.0, put_n=8, put_wins=0)
    assert alternate_side_is_preferable(toxic_pri, weak_alt, opposite=TradeDirection.CALL) is False
    strong_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.55, call_n=8, call_wins=5)
    assert alternate_side_is_preferable(toxic_pri, strong_alt, opposite=TradeDirection.CALL) is True
    better_alt = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.70, call_n=8, call_wins=6)
    assert alternate_side_is_preferable(toxic_pri, better_alt, opposite=TradeDirection.CALL) is True


def test_side_eq_gate_rsi_trend_conflict():

    from src.application.services.side_equilibrium_gate import resolve_direction_with_side_equilibrium

    orch = MagicMock()

    metrics = {
        "macro_indicators": {"rsi": 0.65},
        "price_zone_direction": "CALL",
    }
    with patch("src.application.services.side_equilibrium_gate.evaluate_proposed_side_equilibrium") as mock_eval:
        hard_skip = SideEquilibriumDecision(
            action=ACTION_HARD_SKIP,
            reason="side_imbalance_small_n",
            side_wr=0.45,
            call_n=8,
            call_wins=4,
        )
        pass_dec = SideEquilibriumDecision(action=ACTION_PASS, reason="ok", side_wr=0.70, put_n=8, put_wins=6)
        mock_eval.side_effect = [hard_skip, pass_dec]
        res = resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.CALL, metrics, recovery_active=False)
        assert res is None
        assert metrics.get("gate_reason") == "side_imbalance_rsi_trend_conflict"

        metrics2 = {"macro_indicators": {"rsi": 0.65}}
        mock_eval.side_effect = [hard_skip, pass_dec]
        res2 = resolve_direction_with_side_equilibrium(
            orch, "R_10", TradeDirection.CALL, metrics2, recovery_active=True
        )
        assert res2 == TradeDirection.CALL
        assert metrics2.get("side_eq_recovery_rsi_keep") is True
