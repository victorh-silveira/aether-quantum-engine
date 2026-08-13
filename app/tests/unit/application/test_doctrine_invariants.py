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
    assert inv["invert_exec_side"] is False
    assert inv["online_training"] is False
    assert inv["flip_require_auto_learn"] is True
    assert inv["flip_seed_waive_edge_min"] == pytest.approx(-0.08)
    assert inv["fusion_block_when_tcn_pos_edge"] is True
    assert inv["fusion_block_when_tcn_candle_agree"] is True
    assert inv["fusion_loss_requires_auto_learn"] is True
    assert inv["fusion_loss_seed_weight_mult"] == pytest.approx(0.0)
    assert inv["neg_edge_deep_edge_floor"] == pytest.approx(-0.12)
    assert inv["watchdog_stale_tick_seconds"] == 300
    assert inv["settlement_tolerance_window_seconds"] == 90
    assert inv["post_settlement_is_trading_wait_seconds"] == 90
    assert inv["amort_cycles_min"] == 1
    assert inv["amort_cycles_max"] == 1
    assert inv["cover_multiple"] == pytest.approx(1.5)
    assert inv["max_safe_stake_pct_linear3"] == pytest.approx(0.025)
    assert inv["large_account_stop_win_pct"] == pytest.approx(3.0)
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] > 0.0


def test_assert_production_doctrine_rejects_online_training():
    settings = copy.deepcopy(load_settings_json())
    settings["deep_learning"]["online_training"] = True
    with pytest.raises(ValueError, match="online_training"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_flip_seed_waive():
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"]["flip_require_auto_learn"] = False
    with pytest.raises(ValueError, match="flip_require_auto_learn"):
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


def test_load_doctrine_invariants_cache_hit():
    first = load_doctrine_invariants()
    second = load_doctrine_invariants()
    assert first == second


def test_load_doctrine_missing_soft_recovery():
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["soft_recovery"]
    with pytest.raises(ValueError, match="soft_recovery"):
        load_doctrine_invariants(settings)


def test_assert_production_doctrine_rejects_bad_explore_floor():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["sample_size_policy"]["explore_stake_scale_floor"] = 0.0
    with pytest.raises(ValueError, match="explore_stake_scale_floor"):
        assert_production_doctrine(settings)


def test_assert_production_signal_skip_margin_bounds():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["signal_skip"]["min_direction_margin"] = 0.01
    with pytest.raises(ValueError, match="min_direction_margin"):
        assert_production_doctrine(settings)
    settings["orchestrator"]["execution"]["signal_skip"]["min_direction_margin"] = 0.06
    with pytest.raises(ValueError, match="min_direction_margin"):
        assert_production_doctrine(settings)


def test_load_doctrine_signal_skip_from_ssot():
    inv = load_doctrine_invariants()
    assert inv["signal_skip_enabled"] is True
    assert 0.015 <= float(inv["signal_skip_min_direction_margin"]) <= 0.05


def test_assert_production_doctrine_rejects_bad_caps():
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["max_safe_stake_pct"] = 0.0
    with pytest.raises(ValueError, match="max_safe_stake"):
        assert_production_doctrine(settings)


def test_load_doctrine_missing_execution_block():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]
    with pytest.raises(ValueError, match="execution"):
        load_doctrine_invariants(settings)


def test_load_doctrine_missing_acc_gate():
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["min_validation_accuracy_gate"]
    with pytest.raises(ValueError, match="min_validation_accuracy_gate"):
        load_doctrine_invariants(settings)


def test_load_doctrine_sample_policy_not_dict():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["sample_size_policy"] = []
    with pytest.raises(ValueError, match="sample_size_policy"):
        load_doctrine_invariants(settings)


def test_load_doctrine_missing_infra_loss():
    settings = copy.deepcopy(load_settings_json())
    del settings["infra"]
    with pytest.raises(ValueError, match="infra"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["infra"]["loss_classifier"]
    with pytest.raises(ValueError, match="loss_classifier"):
        load_doctrine_invariants(settings)


def test_load_doctrine_missing_fusion_and_neg_edge():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["scale_vision"]["fusion_block_when_tcn_pos_edge"]
    with pytest.raises(ValueError, match="fusion_block"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["signal_skip"]["neg_edge_deep_edge_floor"]
    with pytest.raises(ValueError, match="neg_edge_deep"):
        load_doctrine_invariants(settings)


def test_load_doctrine_missing_stop_win_and_online():
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["large_account_stop_win_pct"]
    with pytest.raises(ValueError, match="large_account_stop_win_pct"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["deep_learning"]["online_training"]
    with pytest.raises(ValueError, match="online_training"):
        load_doctrine_invariants(settings)


def test_load_doctrine_missing_recovery_timing_keys():
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["soft_recovery"]["amort_cycles_min"]
    with pytest.raises(ValueError, match="amort_cycles"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["watchdog_stale_tick_seconds"]
    with pytest.raises(ValueError, match="watchdog_stale_tick"):
        load_doctrine_invariants(settings)


def test_assert_production_rejects_ssot_knobs():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["mandatory_trade_each_cycle"] = True
    with pytest.raises(ValueError, match="mandatory_trade"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["invert_exec_side"] = True
    with pytest.raises(ValueError, match="invert_exec_side"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["infra"]["loss_classifier"]["flip_seed_waive_edge_min"] = -1.0
    with pytest.raises(ValueError, match="flip_seed_waive"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["fusion_block_when_tcn_pos_edge"] = False
    with pytest.raises(ValueError, match="fusion_block"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["fusion_block_when_tcn_candle_agree"] = False
    with pytest.raises(ValueError, match="fusion_block_when_tcn_candle_agree"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["fusion_loss_requires_auto_learn"] = False
    with pytest.raises(ValueError, match="fusion_loss_requires_auto_learn"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["scale_vision"]["fusion_loss_seed_weight_mult"] = 0.5
    with pytest.raises(ValueError, match="fusion_loss_seed_weight_mult"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["signal_skip"]["neg_edge_deep_edge_floor"] = -1.0
    with pytest.raises(ValueError, match="neg_edge_deep"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["watchdog_stale_tick_seconds"] = 30
    with pytest.raises(ValueError, match="watchdog_stale"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["settlement_tolerance_window_seconds"] = 300
    with pytest.raises(ValueError, match="settlement_tolerance"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 35
    with pytest.raises(ValueError, match="post_settlement"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["amort_cycles_max"] = 9
    with pytest.raises(ValueError, match="amort_cycles"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["cover_multiple"] = 2.0
    with pytest.raises(ValueError, match="cover_multiple"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["max_safe_stake_pct_linear3"] = 0.05
    with pytest.raises(ValueError, match="linear3"):
        assert_production_doctrine(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["large_account_stop_win_pct"] = 2.6
    with pytest.raises(ValueError, match="large_account_stop_win"):
        assert_production_doctrine(settings)


def test_assert_production_ok_from_ssot():
    inv = assert_production_doctrine()
    assert inv["online_training"] is False
