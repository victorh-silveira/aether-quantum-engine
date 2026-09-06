from src.application.services.execution_micro_protect import (
    apply_chop_loss_risk_hard_skip,
    apply_micro_discord_hard_skip,
    apply_micro_protect_gates,
    apply_soft_confirm_weak_hard_skip,
    parse_micro_protect_config,
    score_soft_confirmations,
)


def test_parse_micro_protect_ssot():
    cfg = parse_micro_protect_config()
    assert cfg["micro_discord_hard_skip"] is True
    assert cfg["micro_discord_follow_candle"] is True
    assert float(cfg["micro_discord_follow_kelly_mult"]) == 0.55
    assert cfg["chop_loss_risk_hard_skip"] is True
    assert float(cfg["chop_loss_risk_p_loss_floor"]) == 0.90
    assert cfg["soft_confirm_weak_hard_skip"] is True
    assert int(cfg["soft_exec_min_confirmations"]) == 2


def test_parse_micro_protect_rejects_bad_body():
    try:
        parse_micro_protect_config({"micro_discord_min_body": -1.0})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "micro_discord_min_body" in str(exc)


def test_parse_micro_protect_rejects_bad_floor_and_min_conf():
    try:
        parse_micro_protect_config({"chop_loss_risk_p_loss_floor": 1.5})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "chop_loss_risk_p_loss_floor" in str(exc)
    try:
        parse_micro_protect_config({"soft_exec_min_confirmations": 0})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "soft_exec_min_confirmations" in str(exc)
    try:
        parse_micro_protect_config({"micro_discord_follow_kelly_mult": 0.0})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "micro_discord_follow_kelly_mult" in str(exc)


def test_micro_discord_hard_skip_preserves_call():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "resolved_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 6.0,
        "scale_tape_consensus": "PUT",
        "scale_mini_bar_dir": "PUT",
        "calibrated_prob": 0.72,
    }
    orch = type("O", (), {})()
    assert apply_micro_discord_hard_skip(metrics, orch=orch) is True
    assert metrics["gate_reason"] == "micro_discord"
    assert metrics["gate_verdict"] == "HARD_SKIP"
    assert metrics["execution_candidate_ready"] is False
    assert metrics["exec_direction"] == "CALL"
    assert metrics["micro_discord_confirmed"] is True
    assert metrics.get("micro_discord_followed") is not True


def test_micro_discord_follow_candle_soft_when_edge_ok():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "resolved_direction": "PUT",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 4.0,
        "calibrated_prob": 0.56,
        "kelly_fraction_scale": 1.0,
        "min_edge_explore": 0.015,
    }
    orch = type("O", (), {})()
    assert apply_micro_discord_hard_skip(metrics, orch=orch) is False
    assert metrics["exec_direction"] == "CALL"
    assert metrics["resolved_direction"] == "CALL"
    assert metrics["micro_discord_followed"] is True
    assert metrics["micro_discord_follow_from"] == "PUT"
    assert metrics["micro_discord_follow_soft"] is True
    assert metrics["gate_verdict"] == "SOFT_SIZE"
    assert metrics["execution_candidate_ready"] is True
    assert float(metrics["micro_discord_follow_candle_edge"]) >= 0.015


def test_micro_discord_follow_disabled_stays_hard():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 4.0,
        "calibrated_prob": 0.56,
    }
    cfg = parse_micro_protect_config({"micro_discord_follow_candle": False})
    assert apply_micro_discord_hard_skip(metrics, cfg=cfg) is True
    assert metrics["gate_reason"] == "micro_discord"
    assert metrics["exec_direction"] == "PUT"


def test_micro_discord_force_and_not_ready_and_skip_status():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 6.0,
        "scale_tape_consensus": "PUT",
    }
    assert apply_micro_discord_hard_skip(metrics, force=True) is False
    metrics["execution_candidate_ready"] = False
    assert apply_micro_discord_hard_skip(metrics) is False
    metrics["execution_candidate_ready"] = True
    metrics["signal_status"] = "SKIP:NEG_EDGE"
    assert apply_micro_discord_hard_skip(metrics) is False


def test_micro_discord_disabled():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 6.0,
        "scale_tape_consensus": "PUT",
    }
    cfg = parse_micro_protect_config({"micro_discord_hard_skip": False})
    assert apply_micro_discord_hard_skip(metrics, cfg=cfg) is False


def test_micro_discord_body_fallback_ops_window():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": "bad",
        "ops_window_candle_body": 3.0,
        "ops_window_candle_dir": "CALL",
        "calibrated_prob": 0.28,
    }
    assert apply_micro_discord_hard_skip(metrics) is True
    assert metrics["gate_reason"] == "micro_discord"


def test_micro_discord_body_too_small_or_zero():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 0.0,
        "ops_window_candle_body": None,
        "scale_tape_consensus": "PUT",
    }
    assert apply_micro_discord_hard_skip(metrics) is False
    metrics["closed_micro_candle_body"] = 0.01
    assert apply_micro_discord_hard_skip(metrics) is False


def test_micro_discord_hard_without_tape_confirm():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 0.15,
        "scale_tape_consensus": "PUT",
        "scale_mini_bar_dir": "PUT",
        "calibrated_prob": 0.545,
    }
    assert apply_micro_discord_hard_skip(metrics) is True
    assert metrics["gate_reason"] == "micro_discord"
    assert metrics["micro_discord_confirmed"] is False
    assert metrics.get("micro_discord_follow_blocked") == "edge_subfloor"


def test_micro_discord_noop_when_candle_agrees():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 6.0,
        "scale_tape_consensus": "PUT",
    }
    assert apply_micro_discord_hard_skip(metrics) is False


def test_chop_loss_risk_hard_skip():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.96,
        "loss_clf_flip_blocked": "tcn_pos_edge",
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is True
    assert metrics["gate_reason"] == "chop_loss_risk"
    assert metrics["chop_loss_risk_candle"] == "PUT"


def test_chop_loss_risk_any_regime_soft():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "scale_micro_regime": "retraction",
        "loss_clf_p_loss": 0.91,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is True
    assert metrics["gate_reason"] == "chop_loss_risk"


def test_chop_loss_risk_soft_path_and_guards():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.91,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics, force=True) is False
    assert apply_chop_loss_risk_hard_skip(metrics) is True
    metrics = {
        "execution_candidate_ready": False,
        "signal_status": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.99,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "SKIP:X",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.99,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    cfg = parse_micro_protect_config({"chop_loss_risk_hard_skip": False})
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.99,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics, cfg=cfg) is False


def test_chop_loss_risk_waives_when_candle_agrees_or_missing():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "CALL",
        "loss_clf_p_loss": 0.96,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "loss_clf_p_loss": 0.96,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.96,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False


def test_chop_loss_risk_bad_p_loss_and_below_floor_and_no_soft():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": object(),
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.50,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.89,
        "loss_clf_soft": True,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.99,
    }
    assert apply_chop_loss_risk_hard_skip(metrics) is False


def test_score_soft_confirmations():
    score, peers = score_soft_confirmations(
        {
            "closed_micro_candle_dir": "CALL",
            "scale_tape_consensus": "CALL",
            "scale_mini_bar_dir": "PUT",
            "scale_mili_dir": "PUT",
        },
        "CALL",
    )
    assert score == 2
    assert peers == ["candle", "tape"]


def test_soft_confirm_weak_hard_skip():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "loss_clf_soft": True,
        "loss_clf_p_loss": 0.70,
        "closed_micro_candle_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    orch = type("O", (), {})()
    assert apply_soft_confirm_weak_hard_skip(metrics, orch=orch) is True
    assert metrics["gate_reason"] == "soft_confirm_weak"
    assert metrics["soft_confirm_score"] == 1
    assert metrics["exec_direction"] == "CALL"


def test_soft_confirm_weak_passes_with_enough_peers():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "loss_clf_soft": True,
        "loss_clf_p_loss": 0.70,
        "closed_micro_candle_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    assert apply_soft_confirm_weak_hard_skip(metrics) is False
    assert metrics["soft_confirm_score"] == 2


def test_soft_confirm_weak_guards():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "loss_clf_p_loss": 0.25,
        "closed_micro_candle_dir": "CALL",
    }
    assert apply_soft_confirm_weak_hard_skip(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "loss_clf_soft": True,
        "closed_micro_candle_dir": "CALL",
    }
    assert apply_soft_confirm_weak_hard_skip(metrics) is False
    cfg = parse_micro_protect_config({"soft_confirm_weak_hard_skip": False})
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "loss_clf_soft": True,
        "closed_micro_candle_dir": "CALL",
    }
    assert apply_soft_confirm_weak_hard_skip(metrics, cfg=cfg) is False
    assert apply_soft_confirm_weak_hard_skip(metrics, force=True) is False
    metrics["execution_candidate_ready"] = False
    assert apply_soft_confirm_weak_hard_skip(metrics) is False
    metrics["execution_candidate_ready"] = True
    metrics["signal_status"] = "SKIP:X"
    assert apply_soft_confirm_weak_hard_skip(metrics) is False


def test_soft_confirm_uses_mini_dir_fallback():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "loss_clf_flip_blocked": "tcn_pos_edge",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    assert apply_soft_confirm_weak_hard_skip(metrics) is True
    assert metrics["gate_reason"] == "soft_confirm_weak"
    assert metrics["soft_confirm_score"] == 0


def test_apply_micro_protect_prefers_discord():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 5.0,
        "loss_clf_p_loss": 0.99,
        "loss_clf_soft": True,
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "calibrated_prob": 0.72,
    }
    assert apply_micro_protect_gates(metrics) is True
    assert metrics["gate_reason"] == "micro_discord"


def test_apply_micro_protect_falls_through_to_chop():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 0.05,
        "scale_tape_consensus": "CALL",
        "loss_clf_p_loss": 0.91,
        "loss_clf_soft": True,
    }
    assert apply_micro_protect_gates(metrics) is True
    assert metrics["gate_reason"] == "chop_loss_risk"


def test_apply_micro_protect_chop_waives_when_candle_agrees():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 5.0,
        "scale_tape_consensus": "CALL",
        "scale_mini_bar_dir": "CALL",
        "loss_clf_p_loss": 0.91,
        "loss_clf_soft": True,
    }
    assert apply_micro_protect_gates(metrics) is False


def test_apply_micro_protect_falls_through_to_confirm_weak():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 5.0,
        "loss_clf_p_loss": 0.70,
        "loss_clf_soft": True,
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    assert apply_micro_protect_gates(metrics) is True
    assert metrics["gate_reason"] == "soft_confirm_weak"
