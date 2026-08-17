from src.application.services.market_audit_log import format_gates_audit_line


def test_format_gates_audit_line():
    metrics = {
        "loss_clf_soft": True,
        "loss_clf_p_loss": 0.86,
        "loss_clf_soft_kelly_mult": 0.4,
        "loss_clf_auto_learn": False,
        "loss_clf_n_train": 64,
        "regime_chop_soft": True,
        "regime_chop_adx": 0.16,
        "regime_chop_hurst": 0.48,
        "regime_chop_via_scale": True,
        "cal_side_edge": -0.09,
        "cal_side_edge_floor": 0.04,
        "signal_skip_waived": "neg_edge_soft",
        "exec_direction": "CALL",
    }
    line = format_gates_audit_line(metrics)
    assert line.startswith("[GATES] || LOSS_CLF: SOFT")
    assert "CHOP adx=" in line
    assert "\n[GATES] || NEG_EDGE soft side=CALL" in line
    assert "FUSION:" in line
    hard_neg = format_gates_audit_line(
        {
            "gate_reason": "neg_edge",
            "signal_status": "SKIP:NEG_EDGE",
            "cal_side_edge": -0.083,
            "cal_side_edge_floor": 0.04,
            "calibrated_prob": 0.533,
            "raw_prob": 0.99,
            "exec_direction": "CALL",
            "loss_clf_p_loss": -1.0,
        }
    )
    assert "NEG_EDGE hard side=CALL" in hard_neg
    assert "edge=-0.0830" in hard_neg
    assert "raw_edge=" in hard_neg
    assert "be=0.581" in hard_neg
    assert "skip=neg_edge" in hard_neg
    assert hard_neg.count("[GATES]") == 4
    assert "ANTI_LOSS off" in hard_neg
    anti_line = format_gates_audit_line(
        {
            "anti_loss_seed_discord": True,
            "anti_loss_why": "seed_discord",
            "anti_loss_p_loss": 0.93,
            "anti_loss_tcn": "PUT",
            "anti_loss_candle": "CALL",
            "gate_reason": "anti_loss_seed_discord",
            "signal_status": "SKIP:ANTI_LOSS_SEED_DISCORD",
            "loss_clf_p_loss": 0.93,
            "exec_direction": "PUT",
        }
    )
    assert "ANTI_LOSS skip why=seed_discord" in anti_line
    assert "side=PUT candle=CALL" in anti_line
    assert "skip=anti_loss_seed_discord" in anti_line
    weak_line = format_gates_audit_line(
        {
            "anti_loss_seed_discord": True,
            "anti_loss_why": "seed_weak_candle",
            "anti_loss_p_loss": 0.91,
            "anti_loss_tcn": "PUT",
            "anti_loss_candle": "PUT",
            "anti_loss_body": 0.038,
            "gate_reason": "anti_loss_seed_discord",
            "signal_status": "SKIP:ANTI_LOSS_SEED_DISCORD",
            "loss_clf_p_loss": 0.91,
            "exec_direction": "PUT",
        }
    )
    assert "ANTI_LOSS skip why=seed_weak_candle" in weak_line
    assert "candle=PUT body=0.038" in weak_line
    bad_body_line = format_gates_audit_line(
        {
            "anti_loss_seed_discord": True,
            "anti_loss_why": "seed_weak_candle",
            "anti_loss_p_loss": 0.91,
            "anti_loss_tcn": "PUT",
            "anti_loss_candle": "PUT",
            "anti_loss_body": "x",
            "gate_reason": "anti_loss_seed_discord",
            "signal_status": "SKIP:ANTI_LOSS_SEED_DISCORD",
            "loss_clf_p_loss": 0.91,
            "exec_direction": "PUT",
        }
    )
    assert "body=" not in bad_body_line.split("ANTI_LOSS", 1)[-1]
    anti_soft = format_gates_audit_line(
        {
            "anti_loss_seed_discord": True,
            "anti_loss_soft": True,
            "anti_loss_why": "seed_discord",
            "anti_loss_p_loss": 0.91,
            "anti_loss_tcn": "PUT",
            "anti_loss_candle": "CALL",
            "loss_clf_p_loss": 0.91,
            "exec_direction": "PUT",
            "signal_skip_waived": "anti_loss_soft",
        }
    )
    assert "ANTI_LOSS soft why=seed_discord" in anti_soft
    soft_gap = format_gates_audit_line(
        {
            "cal_side_edge": -0.09,
            "cal_side_edge_floor": 0.04,
            "signal_skip_waived": "neg_edge_soft",
            "calibrated_prob": 0.53,
            "raw_prob": 0.98,
            "exec_direction": "CALL",
            "loss_clf_p_loss": -1.0,
        }
    )
    assert "NEG_EDGE soft" in soft_gap
    assert "raw_edge=" in soft_gap
    assert "be=0.581" in soft_gap
    candle_gap = format_gates_audit_line(
        {
            "cal_side_edge": -0.02,
            "cal_side_edge_floor": 0.04,
            "signal_skip_waived": "neg_edge_soft",
            "neg_edge_candle_soft": True,
            "exec_direction": "PUT",
            "loss_clf_p_loss": -1.0,
        }
    )
    assert "NEG_EDGE candle side=PUT" in candle_gap
    p_ovr_gap = format_gates_audit_line(
        {
            "cal_side_edge": -0.08,
            "cal_side_edge_floor": 0.04,
            "signal_skip_waived": "neg_edge_soft",
            "neg_edge_p_ovr_soft": True,
            "exec_direction": "PUT",
            "loss_clf_p_loss": -1.0,
        }
    )
    assert "NEG_EDGE p_ovr side=PUT" in p_ovr_gap
    tcn_block = format_gates_audit_line(
        {
            "loss_clf_flip_blocked": "tcn_pos_edge",
            "loss_clf_p_loss": 0.95,
            "loss_clf_soft_kelly_mult": 0.4,
            "loss_clf_auto_learn": False,
        }
    )
    assert "FLIP_BLOCK:tcn_edge" in tcn_block
    flip_line = format_gates_audit_line(
        {
            "loss_clf_flip": True,
            "loss_clf_p_loss": 0.91,
            "loss_clf_auto_learn": True,
            "loss_clf_n_train": 12,
            "loss_clf_flip_ref": "PUT",
            "exec_direction": "CALL",
            "loss_clf_flip_reason": "cal_ovr",
        }
    )
    assert "FLIP PUT->CALL why=cal_ovr auto=1" in flip_line
    block_line = format_gates_audit_line(
        {
            "loss_clf_flip_blocked": "scale_consensus",
            "loss_clf_p_loss": 0.91,
            "loss_clf_soft_kelly_mult": 0.4,
            "loss_clf_auto_learn": False,
        }
    )
    assert "FLIP_BLOCK:scale" in block_line
    neg_block = format_gates_audit_line(
        {
            "loss_clf_flip_blocked": "neg_edge",
            "loss_clf_p_loss": 0.95,
            "loss_clf_soft_kelly_mult": 0.4,
            "loss_clf_auto_learn": False,
        }
    )
    assert "FLIP_BLOCK:neg_edge" in neg_block
    other_block = format_gates_audit_line(
        {
            "loss_clf_flip_blocked": "custom_reason",
            "loss_clf_p_loss": 0.91,
            "loss_clf_soft_kelly_mult": 0.4,
        }
    )
    assert "FLIP_BLOCK:custom_reason" in other_block
    ok_line = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.42,
            "loss_clf_auto_learn": True,
            "loss_clf_n_train": 8,
            "loss_clf_veto_ready": True,
            "loss_clf_model_version": "v1",
        }
    )
    assert "LOSS_CLF: OK auto=1" in ok_line
    fusion_line = format_gates_audit_line(
        {
            "fusion_applied": True,
            "fusion_ev_call": -0.05,
            "fusion_ev_put": 0.02,
            "fusion_p_eff": 0.60,
            "fusion_reason": "ev_put",
            "exec_direction": "PUT",
            "loss_clf_p_loss": -1.0,
        }
    )
    assert "FUSION: side=PUT" in fusion_line
    assert "why=ev_put" in fusion_line
    seed_cndl = format_gates_audit_line(
        {
            "loss_clf_flip_blocked": "seed_candle",
            "loss_clf_p_loss": 0.96,
            "loss_clf_soft_kelly_mult": 0.4,
            "loss_clf_auto_learn": False,
        }
    )
    assert "FLIP_BLOCK:seed_candle" in seed_cndl
    boot_deep = format_gates_audit_line(
        {
            "gate_reason": "neg_edge",
            "signal_status": "SKIP:NEG_EDGE",
            "neg_edge_bootstrap_deep": True,
            "cal_side_edge": -0.20,
            "cal_side_edge_floor": 0.04,
            "calibrated_prob": 0.40,
            "raw_prob": 0.40,
            "exec_direction": "PUT",
        }
    )
    assert "NEG_EDGE boot_deep side=PUT" in boot_deep
