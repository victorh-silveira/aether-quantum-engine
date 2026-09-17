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
    assert inv["loss_clf_hard_p_loss_floor"] == pytest.approx(0.58)
    assert inv["loss_clf_flip_trust_n"] == 32
    assert inv["loss_clf_flip_young_shrink"] == pytest.approx(0.35)
    assert inv["loss_clf_flip_young_p_eff_floor"] == pytest.approx(0.58)
    assert inv["loss_clf_flip_min_n_train"] == 4
    assert inv["loss_clf_bootstrap_exit_n"] == 4
    assert inv["loss_clf_ready_n"] == 32
    assert inv["loss_clf_retrain_min_n"] == 12
    assert inv["loss_clf_retrain_on_loss_min_n"] == 4
    assert inv["loss_clf_min_win_for_loss_retrain"] == 4
    assert inv["loss_clf_enabled"] is True
    assert inv["watchdog_stale_tick_seconds"] == 300
    assert inv["settlement_tolerance_window_seconds"] == 600
    assert inv["post_settlement_is_trading_wait_seconds"] == 90
    assert inv["cover_enabled"] is True
    assert inv["cover_multiple"] == pytest.approx(1.0)
    assert inv["neutral_bankroll_pct"] == pytest.approx(0.01)
    assert inv["min_stake_pct"] == pytest.approx(0.01)
    assert inv["max_safe_stake_pct_linear3"] == pytest.approx(0.025)
    assert inv["large_account_stop_win_pct"] == pytest.approx(4.31)
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] == pytest.approx(0.40)
    assert inv["invert_exec_side"] is False
    assert inv["skip_exec_vs_candle"] is False
    assert inv["skip_below_soft_min_acc"] is False
    assert inv["skip_scale_candle_discord"] is False
    assert inv["skip_neg_edge"] is True
    assert inv["skip_doji"] is False
    assert inv["amort_cycles_min"] == 1
    assert inv["amort_cycles_max"] == 1
    assert inv["max_safe_stake_pct"] == pytest.approx(0.035)


def test_assert_production_doctrine_rejects_invert_exec_side_true():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["invert_exec_side"] = True
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="invert_exec_side"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_skip_exec_vs_candle_on():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["skip_exec_vs_candle"] = True
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="skip_exec_vs_candle"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_skip_below_soft_min_acc_on():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["skip_below_soft_min_acc"] = True
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="skip_below_soft_min_acc"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_skip_scale_candle_discord_on():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["skip_scale_candle_discord"] = True
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="skip_scale_candle_discord"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_skip_neg_edge_off():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["skip_neg_edge"] = False
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="skip_neg_edge"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_skip_doji_on():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["skip_doji"] = True
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="skip_doji"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_amort_not_one():
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["amort_cycles_min"] = 2
    settings["risk_management"]["soft_recovery"]["amort_cycles_max"] = 2
    reset_doctrine_invariants_cache()
    with pytest.raises(ValueError, match="amort_cycles"):
        assert_production_doctrine(settings)


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


def test_assert_production_doctrine_rejects_adapt_retract_on():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["adapt_retract_enabled"] = True
    with pytest.raises(ValueError, match="adapt_retract_enabled"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_adapt_tape_require_strong_off():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["adapt_tape_require_strong"] = False
    with pytest.raises(ValueError, match="adapt_tape_require_strong"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_adapt_explos_max_tcn_edge():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["adapt_explos_max_tcn_edge"] = 0.10
    with pytest.raises(ValueError, match="adapt_explos_max_tcn_edge"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_flip_min_n_train():
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"]["flip_min_n_train"] = 2
    with pytest.raises(ValueError, match="flip_min_n_train"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_cal_soft_knobs():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["cal_soft_edge_skip_enabled"] = True
    with pytest.raises(ValueError, match="cal_soft_edge"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_scale_missing():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"] = None
    with pytest.raises(ValueError, match="scale_vision ausente"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_adapt_soft_margin_knob():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["adapt_soft_margin"] = 0.05
    with pytest.raises(ValueError, match="adapt_soft_margin"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_bootstrap_exit_n():
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"]["bootstrap_exit_n"] = 12
    with pytest.raises(ValueError, match="bootstrap_exit_n"):
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


def test_load_doctrine_invariants_requires_loss_streak_knobs():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["post_loss_cooldown"]
    with pytest.raises(ValueError, match="post_loss_cooldown"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["deep_learning"]["session_pause_cycles"]
    with pytest.raises(ValueError, match="session_pause_cycles"):
        load_doctrine_invariants(settings)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("ready_n", 8, "ready_n"),
        ("retrain_min_n", 1, "retrain_min_n"),
        ("retrain_on_loss_min_n", 1, "retrain_on_loss_min_n"),
        ("min_win_for_loss_retrain", 1, "min_win_for_loss_retrain"),
    ],
)
def test_assert_production_doctrine_rejects_loss_sample_floors(key, value, match):
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"][key] = value
    with pytest.raises(ValueError, match=match):
        assert_production_doctrine(settings)
