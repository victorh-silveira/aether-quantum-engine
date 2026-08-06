"""Testes de sizing SIDE_EQ soft (sem veto de direcao)."""

from collections import deque

from src.application.services.execution_side_eq_sizing import (
    _pick_sizing_decision,
    _soft_from_decision,
    apply_side_eq_kelly_sizing,
)
from src.domain.analytics.side_equilibrium import ACTION_HARD_SKIP, ACTION_PASS, ACTION_SOFT, SideEquilibriumDecision
from src.domain.models.trade import TradeDirection


def _orch_with_hist(rows: list[tuple[str, bool]], *, enabled: bool = True):
    return type(
        "Orch",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "side_equilibrium": {
                            "enabled": enabled,
                            "small_window": 24,
                            "large_window": 100,
                            "n_min_small": 8,
                            "n_min_large": 40,
                            "wr_floor_small": 0.4,
                            "wr_floor_large": 0.45,
                            "freq_bias_max_small": 0.7,
                            "freq_bias_max_large": 0.65,
                            "kelly_mult_soft": 0.55,
                            "margin_boost_soft": 0.03,
                            "break_even_wr": 0.55,
                        }
                    }
                }
            },
            "_side_equilibrium_hist": {"OTC_SPC": deque(rows, maxlen=120)},
        },
    )()


def test_side_eq_sizing_disabled():
    orch = _orch_with_hist([("PUT", False)] * 20, enabled=False)
    metrics = {"kelly_fraction_scale": 1.0}
    apply_side_eq_kelly_sizing(orch, "OTC_SPC", TradeDirection.PUT, metrics)
    assert metrics["side_eq_reason"] == "disabled"
    assert metrics["kelly_fraction_scale"] == 1.0
    assert metrics["side_eq_blocked"] is False


def test_side_eq_sizing_soft_reduces_kelly_on_toxic_put():
    rows = [("PUT", False)] * 30 + [("CALL", True)] * 10
    orch = _orch_with_hist(rows, enabled=True)
    metrics = {"kelly_fraction_scale": 1.0}
    apply_side_eq_kelly_sizing(orch, "OTC_SPC", TradeDirection.PUT, metrics)
    assert metrics["side_eq_blocked"] is False
    assert metrics["side_eq_action"] == "soft_penalty"
    assert metrics["kelly_fraction_scale"] < 1.0


def test_side_eq_sizing_never_hard_skip_in_metrics():
    rows = [("PUT", False)] * 12
    orch = _orch_with_hist(rows, enabled=True)
    metrics = {"kelly_fraction_scale": 1.0}
    apply_side_eq_kelly_sizing(orch, "OTC_SPC", TradeDirection.PUT, metrics)
    assert metrics["side_eq_action"] != "hard_skip"
    assert metrics.get("side_eq_blocked") is False


def test_side_eq_sizing_no_orch_and_pass_path():
    metrics = {"kelly_fraction_scale": 1.0}
    apply_side_eq_kelly_sizing(None, "OTC_SPC", TradeDirection.CALL, metrics)
    assert metrics["side_eq_reason"] == "no_orch"
    orch = _orch_with_hist([("CALL", True)] * 20 + [("PUT", True)] * 20, enabled=True)
    metrics2 = {"kelly_fraction_scale": 1.0}
    apply_side_eq_kelly_sizing(orch, "OTC_SPC", TradeDirection.CALL, metrics2)
    assert metrics2["side_eq_kelly_mult"] == 1.0
    assert metrics2["side_eq_action"] == "pass"


def test_pick_sizing_prefers_small_soft_and_disabled_large():
    hard = SideEquilibriumDecision(action=ACTION_HARD_SKIP, reason="side_imbalance_small_n")
    mapped = _soft_from_decision(hard, 0.55)
    assert mapped.action == ACTION_SOFT
    assert mapped.kelly_mult == 0.55
    pass_ok = SideEquilibriumDecision(action=ACTION_PASS, reason="ok")
    picked = _pick_sizing_decision(small=mapped, large=pass_ok, kelly_mult_soft=0.55)
    assert picked.action == ACTION_SOFT
    disabled = SideEquilibriumDecision(action=ACTION_PASS, reason="disabled")
    assert _pick_sizing_decision(small=pass_ok, large=disabled, kelly_mult_soft=0.55).reason == "ok"
