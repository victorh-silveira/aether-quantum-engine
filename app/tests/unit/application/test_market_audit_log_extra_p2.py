"""CLUSTER audit line extras (parte 2)."""

from src.application.services.market_audit_log import format_cluster_audit_line


def test_format_cluster_neutral_zone_shows_edge_and_be():
    decisions = {
        "1HZ75V": {
            "direction": "PUT",
            "metrics": {
                "raw_prob": 0.46814,
                "calibrated_prob": 0.46814,
                "exec_direction": "PUT",
                "gate_reason": "neutral_zone",
                "signal_status": "SKIP:NEUTRAL_ZONE",
            },
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M5")
    assert "SKIP:NEUTRAL_ZONE" in line
    assert "Edge:" in line and "raw_edge:" in line and "be=0.541" in line
    assert "Margin:" in line
    assert "p_call: 0.46814" in line and "p_put: 0.53186" in line


def test_format_cluster_neg_edge_shows_raw_edge_and_be():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "raw_prob": 0.99,
                "calibrated_prob": 0.533,
                "exec_direction": "CALL",
                "gate_reason": "neg_edge",
                "signal_status": "SKIP:NEG_EDGE",
            },
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "raw_edge:" in line and "be=0.541" in line and "Edge: -0.014" in line
    assert "p_call: 0.53300" in line and "p_put: 0.46700" in line
    assert "NEG_EDGE" in line or "neg_edge" in line


def test_format_cluster_edge_gap_before_gate():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {"raw_prob": 0.98783, "calibrated_prob": 0.53338, "exec_direction": "CALL"},
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "Edge: -0.013" in line and "raw_edge: +0.827" in line and "be=0.541" in line
    assert "Margin: 0.033" in line and "p_call: 0.53338" in line and "p_put: 0.46662" in line


def test_format_cluster_leans_call_when_direction_missing():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {"raw_prob": 0.37115, "calibrated_prob": 0.52497},
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "R_10: CALL (" in line and "p_call: 0.52497" in line and "p_put: 0.47503" in line


def test_format_decision_origin_line_variants():
    from src.application.services.market_audit_cycle import format_decision_origin_line

    line_direct = format_decision_origin_line(
        "1HZ75V",
        "CALL",
        {"direction_origin": "TCN_DIRECT", "conviction": 0.62, "cal_side_edge": 0.12, "trend_direction": "CALL"},
    )
    assert "[DECISION] || CALL [1HZ75V] || ORIGEM: TCN_DIRECT | p=0.620 | edge=+0.120 | trend=CALL" in line_direct

    line_flip_clf = format_decision_origin_line(
        "1HZ75V",
        "PUT",
        {
            "direction_origin": "FLIP_LOSS_CLF",
            "tcn_direction": "CALL",
            "loss_clf_p_eff": 0.72,
            "trend_direction": "NEUTRAL",
        },
    )
    assert "[DECISION] || PUT [1HZ75V] || ORIGEM: FLIP loss_clf (CALL->PUT) | pe=0.720 | trend=NEUTRAL" in line_flip_clf

    line_flip_lock = format_decision_origin_line(
        "1HZ75V",
        "PUT",
        {
            "direction_origin": "FLIP_ANTI_TREND_LOCK",
            "anti_trend_lock_from": "CALL",
            "conviction": 0.55,
            "trend_direction": "PUT",
        },
    )
    assert (
        "[DECISION] || PUT [1HZ75V] || ORIGEM: FLIP anti_trend_lock (CALL->PUT) | trend=PUT | p_orig=0.550"
        in line_flip_lock
    )

    line_unknown_origin = format_decision_origin_line(
        "1HZ75V",
        "PUT",
        {
            "direction_origin": "TCN_DIRECT",
            "tcn_direction": "CALL",
            "cal_side_edge": 0.036,
        },
    )
    assert "[DECISION] || PUT [1HZ75V] || ORIGEM: TCN_DIRECT" in line_unknown_origin


def test_format_market_summary_line():
    from src.application.services.market_audit_cycle import format_market_summary_line

    metrics = {
        "indicators": {
            "rsi": 0.5432,
            "adx": 0.2850,
            "atr": 12.34,
            "bb_width": 0.0456,
        },
        "closed_micro_candle_dir": "CALL",
        "trend_direction": "CALL",
        "scale_micro_regime": "TRENDING",
    }
    line = format_market_summary_line("1HZ75V", metrics)
    assert "[MARKET] || 1HZ75V || RSI: 0.543 | ADX: 0.285 | ATR: 12.34 | BB_W: 0.0456" in line
    assert "CANDLE: CALL | TREND: CALL | REGIME: trending" in line


def test_format_settlement_audit_line_session_pnl():
    from src.application.services.market_audit_cycle import format_settlement_audit_line

    line = format_settlement_audit_line(
        1,
        "WIN",
        50.0,
        "CALL",
        "1HZ75V",
        0.10,
        session_pnl=120.50,
        target_pnl=384.94,
    )
    assert "SESSAO: +120.50" in line
    assert "ALVO: $384.94 (31.3%)" in line

    line_no_target = format_settlement_audit_line(
        1,
        "LOSS",
        -25.0,
        "PUT",
        "1HZ75V",
        -0.05,
        session_pnl=-50.0,
        target_pnl=0.0,
    )
    assert "SESSAO:  -50.00" in line_no_target
    assert "ALVO:" not in line_no_target
