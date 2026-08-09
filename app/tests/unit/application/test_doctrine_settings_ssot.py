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
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] > 0.0
    assert inv["max_safe_stake_cap"] > 0.0
    assert inv["max_safe_stake_pct"] > 0.0
    assert inv["signal_skip_enabled"] is True
    assert 0.015 <= float(inv["signal_skip_min_direction_margin"]) <= 0.05


def test_production_deploy_gate_armed():
    from src.domain.config_knobs import load_settings_json

    settings = load_settings_json()
    dl = settings["deep_learning"]
    gate = dl["deploy_gate"]
    assert gate["enabled"] is True
    assert gate["force_ok"] is False
    assert float(gate["soft_min_val_accuracy"]) >= 0.53
    assert float(gate.get("soft_max_brier", 0.0)) >= 0.26
    assert int(dl.get("min_epochs", 0)) >= 40
    assert int(dl.get("early_stopping_patience", 0)) >= 40
    assert str(dl.get("label_mode")) == "ma_trend"
    assert settings["orchestrator"]["execution"]["bypass_deploy_gate"] is False
    assert "quality_gate" not in settings["orchestrator"]["execution"]
    assert "indicator_gating" not in dl
    skip = settings["orchestrator"]["execution"]["signal_skip"]
    assert skip["enabled"] is True
    assert 0.015 <= float(skip["min_direction_margin"]) <= 0.05


def test_production_logging_ssot():
    from src.domain.config_knobs import load_settings_json
    from src.presentation.terminal.logging_config import resolve_logging_config

    settings = load_settings_json()
    block = settings["logging"]
    assert block["level"] == "INFO"
    assert block["log_file"]
    assert "settle_enqueue" in block["quiet_channels"]
    assert "settle_tolerance" in block["quiet_channels"]
    cfg = resolve_logging_config(settings)
    assert cfg["level"] == 20
    assert "settle_enqueue" in cfg["quiet_channels"]
    assert "settle_tolerance" in cfg["quiet_channels"]


def test_production_loss_classifier_soft_veto_ssot():
    from src.domain.config_knobs import load_settings_json
    from src.infrastructure.inference.loss_classifier_client import resolve_loss_classifier_config

    settings = load_settings_json()
    block = settings["infra"]["loss_classifier"]
    assert str(block["veto_mode"]).strip().lower() == "soft"
    assert float(block["veto_p_loss_floor"]) == pytest.approx(0.65)
    assert float(block["hard_p_loss_floor"]) == pytest.approx(0.90)
    assert bool(block["hard_blocks_pending_waive"]) is True
    assert float(block["soft_kelly_mult"]) == pytest.approx(0.55)
    assert float(block["soft_kelly_mult_high"]) == pytest.approx(0.40)
    assert float(block["soft_p_loss_high"]) == pytest.approx(0.85)
    assert float(block["soft_max_stake_pct_high"]) == pytest.approx(0.0025)
    assert float(block["timeout_seconds"]) == pytest.approx(8.0)
    assert int(block["retrain_on_loss_min_n"]) == 2
    assert bool(block["flip_require_auto_learn"]) is True
    assert bool(block["flip_allow_seed_on_scale_discord"]) is True
    assert bool(block["flip_allow_seed_on_cal_discord"]) is True
    resolved = resolve_loss_classifier_config(None)
    assert resolved["veto_mode"] == "soft"
    assert resolved["veto_p_loss_floor"] == pytest.approx(0.65)
    assert resolved["hard_p_loss_floor"] == pytest.approx(0.90)
    assert resolved["hard_blocks_pending_waive"] is True
    assert resolved["soft_kelly_mult"] == pytest.approx(0.55)
    assert resolved["soft_kelly_mult_high"] == pytest.approx(0.40)
    assert resolved["soft_p_loss_high"] == pytest.approx(0.85)
    assert resolved["soft_max_stake_pct_high"] == pytest.approx(0.0025)
    assert resolved["retrain_on_loss_min_n"] == 2
    assert resolved["flip_require_auto_learn"] is True
    assert resolved["flip_allow_seed_on_scale_discord"] is True
    assert resolved["flip_allow_seed_on_cal_discord"] is True
    soft_rec = settings["risk_management"]["soft_recovery"]
    assert int(soft_rec["amort_cycles_min"]) == 1
    assert int(soft_rec["amort_cycles_max"]) == 1
    skip = settings["orchestrator"]["execution"]["signal_skip"]
    assert "direction_loss_lock_min" not in skip
    assert "direction_loss_toxic_escape" not in skip
    assert "direction_loss_flip_kelly_mult" not in skip
    assert "direction_loss_both_soft_kelly_mult" not in skip
    assert "direction_loss_lock_ttl_seconds" not in skip
    assert float(skip["cal_margin_soft_kelly_mult"]) == pytest.approx(0.55)
    assert bool(skip["chop_pause_enabled"]) is True
    assert float(skip["chop_adx_max"]) == pytest.approx(0.22)
    assert float(skip["chop_hurst_min"]) == pytest.approx(0.47)
    assert float(skip["chop_hurst_max"]) == pytest.approx(0.53)
    assert float(skip["chop_soft_kelly_mult"]) == pytest.approx(0.55)
    assert float(skip["neg_edge_soft_kelly_mult"]) == pytest.approx(0.55)
    assert "calib_gray_margin_floor" not in skip
    assert "calib_gray_soft_kelly_mult" not in skip
    assert "calib_gray_max_stake_pct" not in skip
    assert float(skip["waive_mini_pair_min_margin"]) == pytest.approx(0.0)
    scale = settings["orchestrator"]["execution"]["scale_vision"]
    assert "adapt_min_cal_margin" not in scale
    assert "adapt_max_cal_margin" not in scale
    assert scale["adapt_on_majority_votes"] is True
    assert scale["adapt_mili_tape_skip_chop"] is True
    assert scale["adapt_skip_chop"] is True
    assert scale["adapt_require_cal_agree"] is True
    assert int(scale["adapt_majority_min_votes"]) == 3
    assert int(scale["adapt_majority_min_lead"]) == 2
    assert scale["adapt_majority_include_rsi"] is True
    assert scale["adapt_majority_include_micro_bar"] is False
    data = settings["data_handler"]
    assert int(data["micro_granularity"]) == 120
    assert int(data["mini_granularity"]) == 120
    assert int(data["granularity"]) == 3600
    assert int(data["fetch_count"]) == 2000
    assert int(data["micro_fetch_count"]) == 2000
    assert int(data["mini_fetch_count"]) == 256
    orch = settings["orchestrator"]
    assert int(orch["cycle_interval_seconds"]) == 60
    assert int(orch["signature_boundary_seconds"]) == 60
    assert int(orch["exec_empty_retry_seconds"]) == 60
    assert int(orch["settlement_tolerance_window_seconds"]) == 90
    assert int(orch["watchdog_stale_tick_seconds"]) == 300
    assert int(orch["post_settlement_is_trading_wait_seconds"]) == 90
    assert int(settings["risk_management"]["kelly"]["cycle_stake_baseline_seconds"]) == 120
    params = settings["risk_management"]["params"]
    assert int(params["duration"]) == 2
    assert str(params["duration_unit"]).lower() == "m"
    assert float(params["payout_estimate"]) == pytest.approx(0.72)
    kelly = settings["risk_management"]["kelly"]
    assert float(kelly["default_payout"]) == pytest.approx(0.72)
    assert float(kelly["payout_fallback"]) == pytest.approx(0.72)
    assert bool(kelly["stop_win_kelly_enabled"]) is True
    assert int(kelly["stop_win_kelly_cycles_target"]) == 4
    assert int(kelly["stop_win_kelly_live_n_min"]) == 0
    assert float(kelly["max_stake_pct"]) == pytest.approx(0.05)
    assert float(kelly["max_bankroll_stake_fraction"]) == pytest.approx(0.05)
    assert float(kelly["min_stake_pct"]) == pytest.approx(0.0025)
    assert float(kelly["neutral_bankroll_pct"]) == pytest.approx(0.0025)
    assert float(kelly["stop_win_max_stake_pct"]) == pytest.approx(0.05)
    assert float(kelly["fraction"]) == pytest.approx(0.08)
    soft_rec_caps = settings["risk_management"]["soft_recovery"]
    assert float(soft_rec_caps["max_safe_stake_pct"]) == pytest.approx(0.05)
    assert float(soft_rec_caps["max_safe_stake_pct_linear2"]) == pytest.approx(0.05)
    assert float(soft_rec_caps["max_safe_stake_pct_linear3"]) == pytest.approx(0.05)
    assert float(soft_rec_caps["linear_bankroll_pct"]) == pytest.approx(0.0025)
    assert float(soft_rec_caps["cover_multiple"]) == pytest.approx(2.0)
    ssp = settings["orchestrator"]["execution"]["sample_size_policy"]
    assert int(ssp["evidence_n_min"]) == 12
    assert int(ssp["toxic_side_n_min"]) == 4
    assert int(ssp["large_n_min"]) == 32
    side_eq = settings["orchestrator"]["execution"]["side_equilibrium"]
    assert int(side_eq["n_min_small"]) == 4
    assert int(side_eq["large_window"]) == 64
    assert settings["gating"]["price_zone_gate_enabled"] is False
    dl = settings["deep_learning"]
    assert int(dl["training_history_bars"]) == 2000
    assert float(dl["train_history_shortfall_ratio"]) == pytest.approx(0.95)
    assert int(dl["bootstrap_max_wait_rounds"]) == 16
