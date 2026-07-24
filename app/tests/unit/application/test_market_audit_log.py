from types import SimpleNamespace

from src.application.services.market_audit_log import (
    format_cluster_audit_line,
    format_execution_ticket_line,
    format_indicators_audit_line,
    format_settlement_audit_line,
    resolve_cluster_timeframe,
    resolve_settlement_tag,
    resolve_stake_audit_context,
    resolve_stake_mode_tag,
)
from src.domain.models.trade import TradeDirection


def test_resolve_cluster_timeframe_branches():
    assert resolve_cluster_timeframe(None) == "M5"
    assert resolve_cluster_timeframe({"data_handler": "x"}) == "M5"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 900}}) == "M15"
    assert resolve_cluster_timeframe({"data_handler": {"micro_granularity": 300}}) == "M5"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 120}}) == "M2"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 30}}) == "S30"


def test_format_settlement_audit_line_default_tag_flat_keep():
    line = format_settlement_audit_line(1, "WIN", 2.0, "CALL", "R_10", 0.1)
    assert "FLAT_KEEP" in line
    assert resolve_settlement_tag(profit=1.0, linear_before=2) == "RESET_LINEAR"


def test_format_cluster_veto_and_metric_float_paths():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "exec_direction": "CALL",
                "raw_prob": "bad",
                "calibrated_prob": 0.55,
                "quality_guard_reject": True,
            },
        },
        "R_25": {
            "direction": "CALL",
            "metrics": {"exec_direction": "CALL", "signal_status": "SKIP"},
        },
        "R_50": {
            "direction": None,
            "metrics": {"execute": False, "deploy_ok": True, "dl_direction": "PUT"},
        },
        "R_100": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "raw_prob": object(),
                "calibrated_prob": 0.61,
                "predicted_payoff_edge": 0.2,
                "gate_reason": "adx_starvation",
            },
        },
    }
    line = format_cluster_audit_line(decisions, timeframe="M5")
    assert "R_10: CALL (Prob:" in line and "NEUTRO_SKIP)" in line
    assert "R_25: CALL (Prob:" in line and "NEUTRO_SKIP)" in line
    assert "R_50: PUT (Prob:" in line and "NEUTRO_SKIP)" in line
    assert "R_100: CALL (Prob:" in line and "ADX_STARVATION)" in line


def test_metric_float_skips_invalid_then_uses_default_in_cluster():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "raw_prob": object(),
                "calibrated_prob": object(),
                "predicted_payoff_edge": 0.1,
            },
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M5")
    assert "R_10: PUT (Prob: 0.500 Cal: 0.500 Margin: 0.000 Edge: +0.100)" in line


def test_format_settlement_audit_line():
    line = format_settlement_audit_line(
        3,
        "WIN",
        1.63,
        "CALL",
        "R_10",
        0.1234,
        settlement_tag="RESET_LINEAR",
    )
    assert line == "[C0003] RESOLVED || STATUS: WIN  | P&L:   +1.63 | RESET_LINEAR"


def test_format_settlement_audit_line_loss_cooldown():
    line = format_settlement_audit_line(
        3,
        "LOSS",
        -1.0,
        "PUT",
        "R_10",
        -0.05,
        settlement_tag=resolve_settlement_tag(profit=-1.0, linear_before=0),
    )
    assert line == "[C0003] RESOLVED || STATUS: LOSS | P&L:   -1.00 | COOLDOWN_L1"


def test_format_cluster_audit_line():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "raw_prob": 0.377,
                "calibrated_prob": 0.365,
                "predicted_payoff_edge": 0.95,
            },
        },
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "quality_gate_reason": "neutral_clamp"},
        },
    }
    line = format_cluster_audit_line(decisions, timeframe="M5")
    assert line.startswith("[CLUSTER] M5 || ")
    assert "R_10: PUT (Prob: 0.377 Cal: 0.365 Margin: 0.135 Edge: +0.950)" in line
    assert "R_50: CALL (Prob:" in line and "NEUTRO_SKIP)" in line


def test_format_execution_ticket_line():
    line = format_execution_ticket_line(
        6,
        direction="PUT",
        symbol="R_10",
        stake=2.06,
        mode_tag="RECOVER_DAL_L1",
        pending=1.62,
        bankroll=87.69,
        contract_id=1129497159,
        payout=1.79,
        linear=1,
        cap=4.20,
        recovery_infeasible=False,
    )
    assert line == (
        "[C0006] EXEC || PUT [R_10] || "
        "STAKE: 2.06 (RECOVER_DAL_L1) | PEND: 1.62 | LIN: 1 | CAP: 4.20 | "
        "BANCA: 87.69 || "
        "CID: 1129497159 | PAY: 1.79"
    )


def test_resolve_stake_mode_tag():
    assert resolve_stake_mode_tag("DALEMBERT", 1, stake_regime="RECOVER") == "RECOVER_DAL_L1"
    assert resolve_stake_mode_tag("KELLY", 0, stake_regime="EXPLORE") == "EXPLORE_KELLY"
    assert resolve_stake_mode_tag("EXPLORE_KELLY", 0) == "EXPLORE_KELLY"
    assert resolve_stake_mode_tag("RECOVER_DAL_L2", 2) == "RECOVER_DAL_L2"


def test_resolve_stake_mode_tag_invalid_regime_defaults_explore():
    assert resolve_stake_mode_tag("KELLY", 0, stake_regime="WEIRD") == "EXPLORE_KELLY"


def test_format_settlement_audit_line_with_finance_telemetry():
    line = format_settlement_audit_line(
        3,
        "WIN",
        1.5,
        "CALL",
        "R_10",
        0.1,
        pending=2.0,
        linear=1,
        mode_tag="RECOVER_DAL_L1",
        recovery_infeasible=True,
    )
    assert "PEND: 2.00" in line
    assert "LIN: 1" in line
    assert "MODE: RECOVER_DAL_L1" in line
    assert "RECOVERY_INFEASIBLE" in line


def test_format_indicators_audit_line():
    metrics = {
        "indicators": {
            "rsi": 0.4859,
            "adx": 0.2017,
            "hurst": 0.5671,
            "atr_norm": -0.9558,
            "bb_width": -0.2226,
            "vol_ratio": 1.0720,
        },
        "edge_zscore": 0.60,
        "val_accuracy": 0.6433,
        "direction_margin": 0.12,
        "calibration_mode": "calibrated",
        "meta_veto_mode": "none",
    }
    line = format_indicators_audit_line(6, "R_10", metrics)
    assert line.startswith("[C0006] IND || ")
    assert "RSI:" in line and "0.4859" in line
    assert "ADX:" in line and "0.2017" in line
    assert "HURST:" in line and "0.5671" in line
    assert "ATR:" in line and "-0.9558" in line
    assert "BBW:" in line and "-0.2226" in line
    assert "VOL_R:" in line and "1.0720" in line
    assert "Z:" in line and "+0.60" in line
    assert "ACC:" in line and "0.6433" in line
    assert "MARGIN:" in line and "0.120" in line
    assert "NEUTRAL: calibrated" in line
    assert "META_VETO: none" in line


def test_format_indicators_audit_line_ignores_none_and_invalid():
    metrics = {"indicators": {"rsi": None, "hurst": 0.61, "adx": "bad"}, "val_accuracy": 0.5}
    line = format_indicators_audit_line(5, "R_10", metrics)
    assert "0.6100" in line
    assert "RSI:" in line
    assert "0.0000" in line


def test_format_indicators_audit_line_marks_neutral_clamp():
    metrics = {
        "indicators": {"rsi": 0.5, "adx": 0.2, "hurst": 0.5},
        "direction_margin": 0.01,
        "gate_reason": "neutral_clamp",
        "meta_veto_mode": "soft",
    }
    line = format_indicators_audit_line(7, "R_10", metrics)
    assert "NEUTRAL: neutral_clamp" in line
    assert "META_VETO: soft" in line


def test_resolve_stake_audit_context_from_audit():
    rm = SimpleNamespace(
        _last_stake_audit={
            "mode_tag": "RECOVER_DAL_L1",
            "pending": 1.5,
            "bankroll": 90.0,
            "linear_losses": 1,
            "cap": 4.2,
            "recovery_infeasible": False,
        },
        pending_loss_total=lambda: 9.0,
        bankroll=80.0,
    )
    audit = resolve_stake_audit_context(rm)
    assert audit["mode_tag"] == "RECOVER_DAL_L1"
    assert audit["pending"] == 1.5
    assert audit["bankroll"] == 90.0
    assert audit["linear"] == 1
    assert audit["cap"] == 4.2


def test_resolve_stake_audit_context_fallback_balance():
    rm = SimpleNamespace(
        bankroll=70.0,
        initial_bankroll=70.0,
        pending_loss_total=lambda: 2.0,
        consecutive_losses_linear=0,
    )
    audit = resolve_stake_audit_context(rm, balance_fallback=88.5)
    assert audit["mode_tag"] == "EXPLORE_KELLY"
    assert audit["pending"] == 2.0
    assert audit["bankroll"] == 88.5


def test_format_cluster_audit_line_empty():
    assert format_cluster_audit_line({}, timeframe="M5") == "[CLUSTER] M5 || EMPTY"
