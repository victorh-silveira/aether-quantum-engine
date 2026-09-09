"""Testes do loader de invariantes da doutrina."""

from __future__ import annotations

import copy

import pytest

from src.application.services.doctrine_invariants import (
    assert_production_doctrine,
    load_doctrine_invariants,
    reset_doctrine_invariants_cache,
)
from src.domain.config_knobs import load_settings_json


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_load_doctrine_invariants_from_ssot():
    inv = load_doctrine_invariants()
    assert inv["force_trade_every_cycle"] is False
    assert inv["mandatory_trade_each_cycle"] is False
    assert inv["online_training"] is False
    assert inv["loss_clf_veto_mode"] == "hard"
    assert inv["loss_clf_hard_p_loss_floor"] == pytest.approx(0.55)
    assert inv["loss_clf_flip_trust_n"] == 32
    assert inv["loss_clf_flip_young_shrink"] == pytest.approx(0.35)
    assert inv["loss_clf_flip_young_p_eff_floor"] == pytest.approx(0.55)
    assert inv["loss_clf_enabled"] is True
    assert inv["watchdog_stale_tick_seconds"] == 300
    assert inv["settlement_tolerance_window_seconds"] == 600
    assert inv["post_settlement_is_trading_wait_seconds"] == 90
    assert inv["cover_enabled"] is False
    assert inv["neutral_bankroll_pct"] == pytest.approx(0.01)
    assert inv["min_stake_pct"] == pytest.approx(0.01)
    assert inv["max_safe_stake_pct_linear3"] == pytest.approx(0.025)
    assert inv["large_account_stop_win_pct"] == pytest.approx(4.31)
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] == pytest.approx(0.40)


def test_assert_production_doctrine_rejects_online_training():
    settings = copy.deepcopy(load_settings_json())
    settings["deep_learning"]["online_training"] = True
    with pytest.raises(ValueError, match="online_training"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_soft_veto_mode():
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"]["veto_mode"] = "soft"
    with pytest.raises(ValueError, match="veto_mode"):
        assert_production_doctrine(settings)


def test_load_doctrine_invariants_missing_execution():
    with pytest.raises(ValueError, match="orchestrator"):
        load_doctrine_invariants({})


def test_assert_production_doctrine_rejects_force_trade():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["force_trade_every_cycle"] = True
    with pytest.raises(ValueError, match="force_trade"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_low_acc():
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["min_validation_accuracy_gate"] = 0.40
    with pytest.raises(ValueError, match="min_validation_accuracy_gate"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_signal_skip_block():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["signal_skip"] = {"enabled": True}
    with pytest.raises(ValueError, match="signal_skip"):
        assert_production_doctrine(settings)
