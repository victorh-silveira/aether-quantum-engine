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
    assert inv["signal_skip_enabled"] is True
    assert float(inv["signal_skip_min_direction_margin"]) == pytest.approx(0.005)


def test_production_deploy_gate_armed():
    from src.domain.config_knobs import load_settings_json

    settings = load_settings_json()
    dl = settings["deep_learning"]
    gate = dl["deploy_gate"]
    assert gate["enabled"] is True
    assert gate["force_ok"] is False
    assert float(gate["soft_min_val_accuracy"]) >= 0.53
    assert float(gate.get("soft_max_brier", 0.0)) == pytest.approx(0.28)
    assert float(gate.get("max_brier", 0.0)) == pytest.approx(0.28)
    assert int(dl.get("min_epochs", 0)) >= 15
    assert int(dl.get("early_stopping_patience", 0)) >= 12
    assert float(dl.get("weight_decay", 0.0)) == pytest.approx(0.005)
    assert float(dl.get("tcn", {}).get("dropout", 0.0)) == pytest.approx(0.35)
    assert float(dl.get("learning_rate", 0.0)) == pytest.approx(0.001)
    assert int(dl.get("train_deploy_retries", 0)) >= 1
    assert int(dl.get("sample_weighting", {}).get("recency_half_life_n", 0)) == 365
    assert str(dl.get("label_mode")) == "supertrend_atr"
    assert int(dl.get("lookback", 0)) == 20
    assert int(dl.get("label_horizon_bars", 0)) == 1
    assert int(settings["risk_management"]["params"]["duration"]) == 15
    assert str(settings["risk_management"]["params"]["duration_unit"]) == "m"
    assert int(dl["horizon_sweep"]["ops_contract_duration_minutes"]) == 15
    assert float(dl.get("min_edge_execute", 0.0)) == pytest.approx(0.01)
    assert settings["orchestrator"]["execution"]["bypass_deploy_gate"] is False
    assert "quality_gate" not in settings["orchestrator"]["execution"]
    assert "indicator_gating" not in dl
    skip = settings["orchestrator"]["execution"]["signal_skip"]
    assert skip["enabled"] is True
    assert float(skip["min_direction_margin"]) == pytest.approx(0.005)


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
    assert float(block["soft_max_stake_pct_high"]) == pytest.approx(0.01)
    assert float(block["timeout_seconds"]) == pytest.approx(8.0)
    assert int(block["retrain_min_n"]) == 1
    assert int(block["retrain_on_loss_min_n"]) == 1
    assert bool(block["flip_require_auto_learn"]) is True
    assert bool(block["flip_allow_seed_on_scale_discord"]) is True
    assert bool(block["flip_allow_seed_on_cal_discord"]) is True
    assert float(block["flip_cal_discord_margin"]) == pytest.approx(0.03)
    assert bool(block["flip_require_pos_edge"]) is False
    assert float(block["flip_min_edge_execute"]) == pytest.approx(0.04)
    resolved = resolve_loss_classifier_config(None)
    assert resolved["veto_mode"] == "soft"
    assert resolved["veto_p_loss_floor"] == pytest.approx(0.65)
    assert resolved["hard_p_loss_floor"] == pytest.approx(0.90)
    assert resolved["hard_blocks_pending_waive"] is True
    assert resolved["soft_kelly_mult"] == pytest.approx(0.55)
    assert resolved["soft_kelly_mult_high"] == pytest.approx(0.40)
    assert resolved["soft_p_loss_high"] == pytest.approx(0.85)
    assert resolved["soft_max_stake_pct_high"] == pytest.approx(0.01)
    assert resolved["retrain_min_n"] == 1
    assert resolved["retrain_on_loss_min_n"] == 1
    assert resolved["flip_require_auto_learn"] is True
    assert resolved["flip_allow_seed_on_scale_discord"] is True
    assert resolved["flip_allow_seed_on_cal_discord"] is True
    assert resolved["flip_cal_discord_margin"] == pytest.approx(0.03)
    assert resolved["flip_require_pos_edge"] is False
    assert resolved["flip_min_edge_execute"] == pytest.approx(0.04)
    assert resolved["flip_waive_on_closed_candle"] is False
    assert resolved["flip_candle_p_loss_floor"] == pytest.approx(0.85)
    assert resolved["flip_waive_scale_above_p_loss"] == pytest.approx(0.95)
    assert resolved["flip_block_when_tcn_pos_edge"] is True and resolved["flip_waive_tcn_pos_edge_on_discord"] is True
    assert resolved["flip_waive_edge_min"] == pytest.approx(-1.0)
    assert resolved["flip_seed_block_against_closed_candle"] is True
    assert resolved["flip_seed_waive_edge_min"] == pytest.approx(-0.08)
    soft_rec = settings["risk_management"]["soft_recovery"]
    assert int(soft_rec["amort_cycles_min"]) == 1
    assert int(soft_rec["amort_cycles_max"]) == 1
    assert float(soft_rec["cover_multiple"]) == pytest.approx(1.5)
    assert float(soft_rec["max_safe_stake_pct_linear2"]) == pytest.approx(0.04)
    assert float(soft_rec["max_safe_stake_pct_linear3"]) == pytest.approx(0.025)
    assert int(soft_rec["fixed_step_linear_max"]) == 4
    assert float(soft_rec["live_evidence_force_explore_wr_max"]) == pytest.approx(0.62)
    skip = settings["orchestrator"]["execution"]["signal_skip"]
    assert "direction_loss_lock_min" not in skip
    assert "direction_loss_toxic_escape" not in skip
    assert "direction_loss_flip_kelly_mult" not in skip
    assert "direction_loss_both_soft_kelly_mult" not in skip
    assert "direction_loss_lock_ttl_seconds" not in skip
    assert float(skip["cal_margin_soft_kelly_mult"]) == pytest.approx(0.75)
    assert bool(skip["chop_pause_enabled"]) is False
    assert float(skip["chop_adx_max"]) == pytest.approx(0.10)
    assert float(skip["chop_hurst_min"]) == pytest.approx(0.45)
    assert float(skip["chop_hurst_max"]) == pytest.approx(0.55)
    assert float(skip["chop_soft_kelly_mult"]) == pytest.approx(0.75)
    assert float(skip["neg_edge_soft_kelly_mult"]) == pytest.approx(0.55)
    assert bool(skip["neg_edge_hard_skip"]) is False
    assert bool(skip["neg_edge_soft_when_closed_candle_agree"]) is True
    assert float(skip["neg_edge_soft_min_edge"]) == pytest.approx(-1.0)
    assert float(skip["neg_edge_bootstrap_soft_kelly_mult"]) == pytest.approx(0.25)
    assert float(skip["neg_edge_deep_edge_floor"]) == pytest.approx(-0.12)
    assert bool(skip["anti_loss_seed_discord_enabled"]) is True
    assert float(skip["anti_loss_p_loss_floor"]) == pytest.approx(0.85)
    assert bool(skip["anti_loss_require_seed"]) is True
    assert bool(skip["anti_loss_hard_skip"]) is False
    assert float(skip["anti_loss_soft_kelly_mult"]) == pytest.approx(1.0)
    assert bool(skip["anti_loss_require_tcn_pos_edge"]) is True
    assert float(skip["anti_loss_min_candle_body"]) == pytest.approx(0.02)
    assert bool(skip["anti_loss_live_weak_candle_enabled"]) is False
    assert bool(skip["anti_loss_live_confirm_enabled"]) is False
    assert float(skip["anti_loss_live_confirm_min_body"]) == pytest.approx(0.05)
    assert bool(skip["anti_loss_live_exec_candle_enabled"]) is False
    assert "anti_loss_hard_skip_explore" not in skip and "anti_loss_recover_soft_kelly_mult" not in skip
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
    assert scale["adapt_majority_include_micro_bar"] is True
    assert scale["fusion_enabled"] is True
    assert scale["fusion_replace_adapt_flip"] is True
    assert float(scale["fusion_w_macro"]) == pytest.approx(0.45)
    assert float(scale["fusion_w_micro_bar"]) == pytest.approx(0.10)
    assert float(scale["fusion_w_mini"]) == pytest.approx(0.05)
    assert float(scale["fusion_w_mili"]) == pytest.approx(0.02)
    assert float(scale["fusion_w_tape"]) == pytest.approx(0.45)
    assert float(scale["fusion_meta_ev_weight"]) == pytest.approx(0.10)
    assert float(scale["fusion_loss_weight"]) == pytest.approx(0.45)
    assert float(scale["fusion_tcn_shrink_near_half"]) == pytest.approx(0.25)
    assert scale["fusion_block_when_tcn_pos_edge"] is True
    assert scale["fusion_block_when_tcn_candle_agree"] is False and int(scale["ops_window_bars"]) == 3
    assert scale["fusion_loss_requires_auto_learn"] is True
    assert float(scale["fusion_loss_seed_weight_mult"]) == pytest.approx(0.0)
    assert float(scale["fusion_min_edge_execute"]) == pytest.approx(0.035)
    assert float(scale["fusion_weak_ev_soft_kelly_mult"]) == pytest.approx(0.50)
    assert float(scale["fusion_weak_ev_seed_soft_kelly_mult"]) == pytest.approx(0.25)
    assert bool(settings["orchestrator"]["execution"]["invert_exec_side"]) is False
    assert bool(settings["orchestrator"]["execution"]["mandatory_trade_each_cycle"]) is False
    assert float(settings["risk_management"]["large_account_stop_win_pct"]) == pytest.approx(4.31)
    from src.application.services.execution_direction_fusion import parse_direction_fusion_config

    fusion = parse_direction_fusion_config({})
    assert fusion["fusion_enabled"] is True
    assert fusion["fusion_replace_adapt_flip"] is True
    assert float(fusion["fusion_weak_ev_soft_kelly_mult"]) == pytest.approx(0.50)
    assert float(fusion["fusion_weak_ev_seed_soft_kelly_mult"]) == pytest.approx(0.25)
    assert fusion["fusion_block_when_tcn_pos_edge"] is True
    assert fusion["fusion_block_when_tcn_candle_agree"] is False
    assert fusion["fusion_loss_requires_auto_learn"] is True
    assert float(fusion["fusion_loss_seed_weight_mult"]) == pytest.approx(0.0)
    assert float(fusion["fusion_loss_weight"]) == pytest.approx(0.45)
    assert float(fusion["fusion_tcn_shrink_near_half"]) == pytest.approx(0.25)
    data = settings["data_handler"]
    assert int(data["micro_granularity"]) == 900
    assert int(data["mini_granularity"]) == 900
    assert int(data["granularity"]) == 86400
    dl = settings["deep_learning"]
    assert bool(dl["online_training"]) is False
    assert int(dl["rolling_retrain_bars"]) == 48
    assert int(dl["retrain_min_bars"]) == 12
    meta = settings["infra"]["meta_classifier"]
    assert bool(meta["online_learn"]) is True
    assert int(meta["retrain_min_n"]) == 2
    assert int(meta["max_buffer"]) == 2000
    assert float(meta["timeout_seconds"]) == pytest.approx(8.0)
    assert int(data["fetch_count"]) == 100
    assert int(data["micro_fetch_count"]) == 100
    assert int(data["mini_fetch_count"]) == 100
    orch = settings["orchestrator"]
    assert int(orch["cycle_interval_seconds"]) == 120
    assert int(orch["signature_boundary_seconds"]) == 900
    assert int(orch["exec_empty_retry_seconds"]) == 120
    assert int(orch["settlement_tolerance_window_seconds"]) == 600
    assert int(orch["watchdog_stale_tick_seconds"]) == 300
    assert int(orch["post_settlement_is_trading_wait_seconds"]) == 90
    assert int(settings["risk_management"]["kelly"]["cycle_stake_baseline_seconds"]) == 900
    assert settings.get("anchor") == "OTC_SPC"
    assert list(settings.get("symbols") or []) == ["OTC_SPC"]
    assert list(dl.get("train_symbols") or []) == ["OTC_SPC"]
    assert int(settings["risk_management"]["params"]["duration"]) == 15
    assert str(settings["risk_management"]["params"]["duration_unit"]) == "m"
    params = settings["risk_management"]["params"]
    assert float(params["payout_estimate"]) == pytest.approx(0.85)
    kelly = settings["risk_management"]["kelly"]
    assert float(kelly["default_payout"]) == pytest.approx(0.85)
    assert float(kelly["payout_fallback"]) == pytest.approx(0.85)
    assert bool(kelly["stop_win_kelly_enabled"]) is True
    assert int(kelly["stop_win_kelly_cycles_target"]) == 1
    assert int(kelly["stop_win_kelly_live_n_min"]) == 0
    assert float(kelly["max_stake_pct"]) == pytest.approx(0.05)
    assert float(kelly["max_bankroll_stake_fraction"]) == pytest.approx(0.05)
    assert float(kelly["min_stake_pct"]) == pytest.approx(0.0025)
    assert float(kelly["neutral_bankroll_pct"]) == pytest.approx(0.0025)
    assert float(kelly["target_damping_floor"]) == pytest.approx(0.50)
    assert float(kelly["target_damping_span"]) == pytest.approx(0.50)
    assert float(kelly["stop_win_max_stake_pct"]) == pytest.approx(0.05)
    assert float(kelly["fraction"]) == pytest.approx(0.08)
    assert float(kelly["kelly_p_floor"]) == pytest.approx(0.55)
    ssp = settings["orchestrator"]["execution"]["sample_size_policy"]
    assert float(ssp["explore_stake_scale_floor"]) == pytest.approx(0.40)
    soft_rec_caps = settings["risk_management"]["soft_recovery"]
    assert float(soft_rec_caps["max_safe_stake_pct"]) == pytest.approx(0.05)
    assert float(soft_rec_caps["max_safe_stake_pct_linear2"]) == pytest.approx(0.04)
    assert float(soft_rec_caps["max_safe_stake_pct_linear3"]) == pytest.approx(0.025)
    assert float(soft_rec_caps["linear_bankroll_pct"]) == pytest.approx(0.0025)
    assert float(soft_rec_caps["cover_multiple"]) == pytest.approx(1.5)
    ssp = settings["orchestrator"]["execution"]["sample_size_policy"]
    assert int(ssp["evidence_n_min"]) == 12
    assert int(ssp["toxic_side_n_min"]) == 4
    assert int(ssp["large_n_min"]) == 32
    side_eq = settings["orchestrator"]["execution"]["side_equilibrium"]
    assert int(side_eq["n_min_small"]) == 4
    assert int(side_eq["large_window"]) == 64
    assert "gating" not in settings
    assert "risk" not in settings
    assert "ws_connect_max_attempts" not in settings.get("api_config", {})
    assert "history_fetch_chunk" not in settings.get("data_handler", {})
    dl = settings["deep_learning"]
    assert int(dl["training_history_bars"]) == 100
    assert float(dl["train_history_shortfall_ratio"]) == pytest.approx(0.95)
    assert int(dl["bootstrap_max_wait_rounds"]) == 16
    cal = dl["calibration"]
    assert float(cal["temperature_min"]) == pytest.approx(1.0)
    assert float(cal["max_calibrated_raw_gap"]) == pytest.approx(0.08)
    assert "tf_sweep" not in dl
    h_sweep = dl["horizon_sweep"]
    assert bool(h_sweep["enabled"]) is False
    assert bool(h_sweep["run_in_launch_train"]) is False
    assert bool(h_sweep["auto_promote"]) is False
    assert int(h_sweep["ops_contract_duration_minutes"]) == 15
    assert bool(h_sweep["quiet_train_logs"]) is True
    assert int(h_sweep["train_deploy_retries"]) == 1
    assert bool(h_sweep["disable_infra_during_sweep"]) is True
    assert int(h_sweep["min_settle_n"]) == 16
    assert int(h_sweep["min_history_bars"]) == 800
    assert "launch_only" not in h_sweep
    assert float(h_sweep["min_edge_vs_breakeven"]) == pytest.approx(0.03)
    assert list(h_sweep["n_bars"]) == [1, 2, 3, 4]
    assert list(h_sweep["duration_minutes"]) == [15, 30, 45, 60]
    assert list(h_sweep["symbols"]) == ["OTC_SPC"]
