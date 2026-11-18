"""Testes de imutabilidade de SignalEvaluation e StakeDecision no dominio de risco."""

import copy

import pytest

from src.domain.models.signal_evaluation import SignalEvaluation, StakeDecision
from src.domain.risk.risk_manager import RiskManager


def _base_config() -> dict:
    return {
        "kelly": {
            "kelly_fraction": 0.25,
            "max_stake_pct": 0.05,
            "super_concordance_enabled": True,
            "super_concordance_prob_min": 0.70,
            "super_concordance_booster": 1.5,
        },
        "risk_management": {"params": {"stake_min": 1.0, "stop_win_pct": 0.20}},
        "orchestrator": {"execution": {"mandatory_trade_each_cycle": False}},
    }


def test_calculate_stake_decision_nao_muta_dl_metrics_dict():
    rm = RiskManager(_base_config())
    original = {
        "execute": True,
        "signal_status": "SOFT_VETO",
        "meta_veto_mode": "soft",
        "senior_trader_conviction": 0.72,
        "trade_score": 0.75,
        "conviction": 0.75,
        "calibrated_prob": 0.75,
        "call_votes": 3,
        "put_votes": 0,
        "live_n": 10,
        "kelly_fraction_scale": 1.0,
        "deploy_provisional": True,
        "provisional_max_stake_pct": 0.01,
    }
    snapshot = copy.deepcopy(original)

    decision = rm.calculate_stake_decision(
        1000.0,
        "OTC_SSMI",
        conviction=0.75,
        silent=True,
        dl_metrics=original,
        order_direction="CALL",
    )

    assert isinstance(decision, StakeDecision)
    assert original == snapshot
    assert "f_star" not in original
    assert "explore_stake_scale" not in original
    assert "provisional_stake_cap_applied" not in original
    updates = decision.to_telemetry_updates()
    assert updates["stake_regime"] == "EXPLORE"
    assert "f_star" in updates
    assert updates["provisional_stake_cap_applied"] is True


def test_signal_evaluation_is_frozen_and_readonly():
    ev = SignalEvaluation.from_metrics({"execute": True, "live_n": 5, "custom_key": 42})
    assert SignalEvaluation.from_metrics(None) == SignalEvaluation()
    assert SignalEvaluation.from_metrics(ev) is ev
    assert ev.live_n == 5
    assert ev.get("custom_key") == 42
    assert ev["custom_key"] == 42
    assert "custom_key" in ev
    assert ev.to_dict()["custom_key"] == 42
    with pytest.raises(TypeError):
        ev.raw["custom_key"] = 99  # type: ignore[index]

    bad_ev = SignalEvaluation.from_metrics({"live_wr": "invalid", "kelly_fraction_scale": "bad"})
    assert bad_ev.live_wr is None
    assert bad_ev.kelly_fraction_scale == 1.0

    dec = StakeDecision(
        stake=10.0,
        f_star=0.02,
        explore_stake_scale=0.5,
        super_concordance_booster_active=True,
        super_concordance_fraction_multiplier=1.5,
        kelly_fraction_effective=0.03,
        provisional_stake_cap_applied=True,
    )
    tele = dec.to_telemetry_updates()
    assert tele["super_concordance_booster_active"] is True
    assert tele["super_concordance_fraction_multiplier"] == 1.5
    assert tele["kelly_fraction_effective"] == 0.03
    assert tele["explore_stake_scale"] == 0.5
    assert tele["provisional_stake_cap_applied"] is True
