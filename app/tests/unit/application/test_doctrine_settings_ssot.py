"""Congela config/settings.json contra os pisos da doutrina AGENTS."""

from __future__ import annotations

import pytest

from src.application.services.doctrine_invariants import assert_production_doctrine, reset_doctrine_invariants_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_production_settings_pass_doctrine_invariants():
    inv = assert_production_doctrine()
    assert inv["force_trade_every_cycle"] is False
    assert inv["min_validation_accuracy_gate"] >= 0.53
    assert inv["explore_stake_scale_floor"] == pytest.approx(0.40)
    assert inv["max_safe_stake_cap"] > 0.0
    assert inv["max_safe_stake_pct"] > 0.0
    assert inv["loss_clf_veto_mode"] == "hard"
    assert float(inv["loss_clf_hard_p_loss_floor"]) == pytest.approx(0.58)
    assert int(inv["loss_clf_flip_trust_n"]) == 32
    assert float(inv["loss_clf_flip_young_shrink"]) == pytest.approx(0.35)
    assert float(inv["loss_clf_flip_young_p_eff_floor"]) == pytest.approx(0.55)
    assert int(inv["loss_clf_ready_n"]) == 32
    assert int(inv["loss_clf_retrain_min_n"]) == 12
    assert int(inv["loss_clf_retrain_on_loss_min_n"]) == 4
    assert int(inv["loss_clf_min_win_for_loss_retrain"]) == 4


def test_production_deploy_gate_armed():
    from src.domain.config_knobs import load_settings_json

    settings = load_settings_json()
    dl = settings["deep_learning"]
    gate = dl["deploy_gate"]
    assert gate["enabled"] is True
    assert gate["force_ok"] is False
    assert float(gate["soft_min_val_accuracy"]) >= 0.53
    assert float(gate["max_label_call_frac_bias"]) == pytest.approx(0.20)
    assert bool(dl.get("allow_undeployed_inference")) is False
    assert int(dl.get("training_history_bars", 0)) == 2000
    assert int(dl.get("lookback", 0)) == 30
    assert int(dl.get("label_horizon_bars", 0)) == 1
    assert int(settings["risk_management"]["params"]["duration"]) == 5
    assert str(settings["risk_management"]["params"]["duration_unit"]) == "m"
    assert settings["orchestrator"]["execution"]["bypass_deploy_gate"] is False
    assert "quality_gate" not in settings["orchestrator"]["execution"]
    assert "signal_skip" not in settings["orchestrator"]["execution"]
    assert "invert_exec_side" not in settings["orchestrator"]["execution"]


def test_production_logging_ssot():
    from src.domain.config_knobs import load_settings_json
    from src.presentation.terminal.logging_config import resolve_logging_config

    settings = load_settings_json()
    block = settings["logging"]
    assert block["level"] == "INFO"
    assert block["log_file"]
    cfg = resolve_logging_config(settings)
    assert cfg["level"] == 20


def test_production_loss_classifier_flip_floor_ssot():
    from src.domain.config_knobs import load_settings_json
    from src.infrastructure.inference.loss_classifier_client import resolve_loss_classifier_config

    settings = load_settings_json()
    block = settings["infra"]["loss_classifier"]
    assert str(block["veto_mode"]).strip().lower() == "hard"
    assert float(block["hard_p_loss_floor"]) == pytest.approx(0.58)
    assert int(block["flip_trust_n"]) == 32
    assert float(block["flip_young_shrink"]) == pytest.approx(0.35)
    assert float(block["flip_young_p_eff_floor"]) == pytest.approx(0.55)
    assert "flip_require_auto_learn" not in block
    assert "soft_kelly_mult" not in block
    assert "veto_p_loss_floor" not in block
    resolved = resolve_loss_classifier_config(None)
    assert resolved["veto_mode"] == "hard"
    assert resolved["hard_p_loss_floor"] == pytest.approx(0.58)
    assert resolved["flip_trust_n"] == 32
    assert resolved["flip_young_shrink"] == pytest.approx(0.35)
    assert resolved["flip_young_p_eff_floor"] == pytest.approx(0.55)
    assert int(block["ready_n"]) == 32
    assert int(block["retrain_min_n"]) == 12
    assert int(block["min_win_for_loss_retrain"]) == 4
    assert bool(settings["orchestrator"]["execution"]["allow_undeployed"]) is False
    soft_rec = settings["risk_management"]["soft_recovery"]
    assert bool(soft_rec["cover_enabled"]) is False
    assert float(soft_rec["max_safe_stake_pct_linear3"]) == pytest.approx(0.025)
    scale = settings["orchestrator"]["execution"]["scale_vision"]
    assert scale["enabled"] is True
    assert int(scale["ops_window_bars"]) == 3
    assert "fusion_enabled" not in scale
    assert "adapt_direction_enabled" not in scale
    assert bool(settings["orchestrator"]["execution"]["mandatory_trade_each_cycle"]) is False
    assert float(settings["risk_management"]["large_account_stop_win_pct"]) == pytest.approx(4.31)
    data = settings["data_handler"]
    assert int(data["micro_granularity"]) == 300
    assert int(data["micro_history_bars"]) == 2000
    assert int(data["micro_fetch_count"]) == 2000
    assert int(data["granularity"]) == 86400
    dl = settings["deep_learning"]
    assert bool(dl["online_training"]) is False
    assert float(dl["confidence_call_threshold"]) == pytest.approx(0.55)
    assert float(dl["confidence_put_threshold"]) == pytest.approx(0.45)
    cal = dl["calibration"]
    assert cal["calibration_neutral_drift"] == [0.45, 0.55]
    assert float(cal["neutral_half_width"]) == pytest.approx(0.05)
    assert float(cal["neutral_calibration_half_width"]) == pytest.approx(0.05)
    assert float(cal["min_calibration_margin_floor"]) == pytest.approx(0.05)
    assert float(cal["temperature_min"]) == pytest.approx(0.75)
    assert float(cal["min_calibration_sharpness"]) == pytest.approx(0.03)
    assert float(cal["min_oos_sharpness"]) == pytest.approx(0.03)
    assert float(cal["max_calibrated_raw_gap"]) == pytest.approx(0.05)
    assert float(dl["label_smoothing"]) == pytest.approx(0.02)
    assert float(dl["tcn"]["dropout"]) == pytest.approx(0.20)
    orch = settings["orchestrator"]
    assert int(orch["cycle_interval_seconds"]) == 300
    assert bool(orch["require_signature_boundary"]) is True
    assert settings.get("anchor") == "1HZ75V"
    kelly = settings["risk_management"]["kelly"]
    assert float(kelly["default_payout"]) == pytest.approx(0.85)
    assert float(kelly["max_stake_pct"]) == pytest.approx(0.05)
    assert float(kelly["min_stake_pct"]) == pytest.approx(0.01)
    assert int(kelly["stop_win_kelly_live_n_min"]) == 12
    assert float(kelly["stop_win_kelly_min_conviction"]) == pytest.approx(0.58)
    assert float(kelly["kelly_p_floor"]) == pytest.approx(0.55)
    ssp = settings["orchestrator"]["execution"]["sample_size_policy"]
    assert float(ssp["explore_stake_scale_floor"]) == pytest.approx(0.40)
    assert int(ssp["evidence_n_min"]) == 12
    cool = settings["orchestrator"]["execution"]["post_loss_cooldown"]
    assert int(cool["lin_min"]) == 1
    assert float(cool["delay_seconds_lin1"]) == pytest.approx(300.0)
    assert float(cool["delay_seconds_lin2"]) == pytest.approx(300.0)
    assert float(cool["delay_seconds_lin3"]) == pytest.approx(600.0)
    assert float(cool["delay_seconds_lin4"]) == pytest.approx(900.0)
    from src.application.services.deep_learning.dl_outcomes import resolve_session_pause_config
    from src.application.services.deep_learning.dl_params import parse_dl_params
    from src.application.services.execution_runtime_config import resolve_post_loss_cooldown_config

    ladder = resolve_post_loss_cooldown_config(None)
    assert int(ladder["lin_min"]) == 1
    assert float(ladder["delay_seconds_lin3"]) == pytest.approx(600.0)
    pause = resolve_session_pause_config(None)
    assert int(pause["session_max_losses_in_window"]) == 3
    assert int(pause["session_window_trades"]) == 5
    assert int(pause["session_pause_cycles"]) == 2
    assert int(dl["session_max_losses_in_window"]) == 3
    assert int(dl["session_window_trades"]) == 5
    assert int(dl["session_pause_cycles"]) == 2
    params = parse_dl_params(dl, data, settings["risk_management"]["params"])
    assert int(params["inference_history_bars"]) >= 288 + 30 + 16
