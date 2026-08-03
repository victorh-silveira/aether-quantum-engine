"""Testes do loader de invariantes da doutrina."""

from __future__ import annotations

import copy

import pytest

from src.application.services.doctrine_invariants import (
    assert_production_doctrine,
    load_doctrine_invariants,
    reset_doctrine_invariants_cache,
    resolve_hard_cal_margin_floor,
)
from src.application.services.execution_quality_reject import reject_on_quality_gate
from src.domain.config_knobs import load_settings_json


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_load_doctrine_invariants_from_ssot():
    inv = load_doctrine_invariants()
    assert inv["force_trade_every_cycle"] is False
    assert inv["hard_cal_margin_floor"] == pytest.approx(0.05)
    assert inv["min_payoff_edge"] >= 0.0
    assert inv["min_validation_accuracy_gate"] >= 0.53


def test_load_doctrine_invariants_missing_execution():
    with pytest.raises(ValueError, match="orchestrator"):
        load_doctrine_invariants({})


def test_load_doctrine_invariants_missing_hard_floor():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["hard_cal_margin_floor"]
    with pytest.raises(ValueError, match="hard_cal_margin_floor"):
        load_doctrine_invariants(settings)


def test_resolve_hard_cal_margin_floor_override_zero():
    assert resolve_hard_cal_margin_floor({"hard_cal_margin_floor": 0.0}) == pytest.approx(0.0)


def test_resolve_hard_cal_margin_floor_from_ssot():
    assert resolve_hard_cal_margin_floor({}) == pytest.approx(0.05)
    assert resolve_hard_cal_margin_floor(None) == pytest.approx(0.05)


def test_reject_uses_ssot_floor_when_exec_omits_key():
    metrics = {"calibrated_prob": 0.52, "raw_prob": 0.52, "val_accuracy": 0.70}
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"quality_gate": {"regular": {"min_direction_margin": 0.0}}},
    )
    assert rejected is True
    assert metrics["gate_reason"] == "cal_margin_floor"


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


def test_assert_production_doctrine_rejects_negative_edge():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["quality_gate"]["min_payoff_edge"] = -0.1
    with pytest.raises(ValueError, match="min_payoff_edge"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_low_hard_cal():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["hard_cal_margin_floor"] = 0.01
    with pytest.raises(ValueError, match="hard_cal_margin_floor"):
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


def test_assert_production_doctrine_rejects_bad_caps():
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"]["max_safe_stake_pct"] = 0.0
    with pytest.raises(ValueError, match="max_safe_stake"):
        assert_production_doctrine(settings)


def test_assert_production_doctrine_rejects_regular_negative_edge():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["quality_gate"]["regular"]["min_payoff_edge"] = -0.2
    with pytest.raises(ValueError, match="regular.min_payoff_edge"):
        assert_production_doctrine(settings)


def test_load_doctrine_missing_quality_gate():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["quality_gate"]
    with pytest.raises(ValueError, match="quality_gate"):
        load_doctrine_invariants(settings)


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


def test_load_doctrine_quality_gate_not_dict():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["quality_gate"] = []
    with pytest.raises(ValueError, match="quality_gate"):
        load_doctrine_invariants(settings)


def test_load_doctrine_regular_edge_missing():
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["quality_gate"]["regular"]["min_payoff_edge"]
    with pytest.raises(ValueError, match="regular.min_payoff_edge"):
        load_doctrine_invariants(settings)


def test_load_doctrine_sample_policy_not_dict():
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["sample_size_policy"] = []
    with pytest.raises(ValueError, match="sample_size_policy"):
        load_doctrine_invariants(settings)
