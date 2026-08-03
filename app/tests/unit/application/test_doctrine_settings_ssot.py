"""Congela config/settings.json contra os pisos da doutrina AGENTS."""

from __future__ import annotations

import pytest

from src.application.services.doctrine_invariants import (
    assert_production_doctrine,
    reset_doctrine_invariants_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_production_settings_pass_doctrine_invariants():
    inv = assert_production_doctrine()
    assert inv["force_trade_every_cycle"] is False
    assert inv["hard_cal_margin_floor"] >= 0.05
    assert inv["min_payoff_edge"] >= 0.0
    assert inv["regular_min_payoff_edge"] >= 0.0
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] > 0.0
    assert inv["max_safe_stake_cap"] > 0.0
    assert inv["max_safe_stake_pct"] > 0.0
