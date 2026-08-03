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


def test_production_deploy_gate_armed():
    from src.domain.config_knobs import load_settings_json

    settings = load_settings_json()
    dl = settings["deep_learning"]
    gate = dl["deploy_gate"]
    assert gate["enabled"] is True
    assert gate["force_ok"] is False
    assert float(gate["soft_min_val_accuracy"]) >= 0.53
    assert int(dl.get("min_epochs", 0)) >= 40
    assert settings["orchestrator"]["execution"]["bypass_deploy_gate"] is False


def test_production_logging_ssot():
    from src.domain.config_knobs import load_settings_json
    from src.presentation.terminal.logging_config import resolve_logging_config

    settings = load_settings_json()
    block = settings["logging"]
    assert block["level"] == "INFO"
    assert block["log_file"]
    assert "settle_enqueue" in block["quiet_channels"]
    cfg = resolve_logging_config(settings)
    assert cfg["level"] == 20
    assert "settle_enqueue" in cfg["quiet_channels"]
