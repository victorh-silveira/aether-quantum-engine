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
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] > 0.0


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
