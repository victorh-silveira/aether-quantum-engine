"""Congela Single-Strike 4.31% e anti-loss RSI no SSOT."""

from __future__ import annotations

import pytest

from src.application.services.doctrine_invariants import reset_doctrine_invariants_cache
from src.domain.config_knobs import load_settings_json


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_production_single_strike_and_anti_loss_rsi_ssot():
    settings = load_settings_json()
    params = settings["risk_management"]["params"]
    kelly = settings["risk_management"]["kelly"]
    skip = settings["orchestrator"]["execution"]["signal_skip"]
    assert float(kelly["stop_win_kelly_min_fraction"]) == pytest.approx(1.0)
    assert float(kelly["stop_win_kelly_max_fraction"]) == pytest.approx(1.0)
    assert float(kelly["stop_win_kelly_min_conviction"]) == pytest.approx(0.52)
    assert float(params["compounding_rate_daily"]) == pytest.approx(0.0431)
    assert float(skip["anti_loss_rsi_min"]) == pytest.approx(0.30)
    assert float(skip["anti_loss_rsi_max"]) == pytest.approx(0.70)
    single_strike = float(params["compounding_rate_daily"]) / float(params["payout_estimate"])
    assert single_strike == pytest.approx(0.0431 / 0.85)
    assert single_strike == pytest.approx(0.0507058823)
    assert float(kelly["max_stake_pct"]) == pytest.approx(0.05)
