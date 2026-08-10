from unittest.mock import MagicMock

from src.application.services.market_audit_log import (
    emit_audit_info,
    format_cluster_audit_line,
    format_execution_ticket_line,
    format_kelly_audit_line,
    format_settlement_audit_line,
    resolve_cluster_timeframe,
    resolve_settlement_tag,
    resolve_stake_mode_tag,
)
from src.domain.models.trade import TradeDirection


def test_emit_audit_info_splits_multiline():
    logger = MagicMock()
    emit_audit_info(logger, "[IND] || A\n[IND] || B\n\n[IND] || C")
    assert logger.info.call_count == 3
    assert logger.info.call_args_list[0].args == ("%s", "[IND] || A")
    assert logger.info.call_args_list[1].args == ("%s", "[IND] || B")
    assert logger.info.call_args_list[2].args == ("%s", "[IND] || C")
    emit_audit_info(logger, "   \n  ")
    assert logger.info.call_count == 3


def test_resolve_cluster_timeframe_branches():
    assert resolve_cluster_timeframe(None) == "M1"
    assert resolve_cluster_timeframe({"data_handler": "x"}) == "M1"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 900}}) == "M15"
    assert resolve_cluster_timeframe({"data_handler": {"micro_granularity": 300}}) == "M5"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 300, "micro_granularity": 60}}) == "M1"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 600, "micro_granularity": 120}}) == "M2"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 120}}) == "M2"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 30}}) == "S30"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 86400}}) == "D1"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 172800}}) == "D2"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 3600}}) == "H1"
    assert resolve_cluster_timeframe({"data_handler": {"granularity": 7200}}) == "H2"


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
    assert "R_100: CALL (Prob:" in line and "SKIP:ADX_STARVATION)" in line


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
    assert "R_10: PUT (Prob: 0.50000 Cal: 0.50000 Margin: 0.000 Edge: +0.000" in line
    assert "raw_edge:" in line and "be=0.581" in line


def test_metric_float_conversion_error_branch():
    from src.application.services.market_audit_log_helpers import metric_float

    assert metric_float({"trade_score": "bad"}, "trade_score", default=0.0) == 0.0


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
    assert line == "[RESOLVED] || STATUS: WIN  | P&L:   +1.63 | RESET_LINEAR | PEND: n/a | LIN: n/a | MODE: n/a"


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
    assert line.startswith("[RESOLVED] || STATUS: LOSS | P&L:   -1.00 | COOLDOWN_L1")
    assert resolve_settlement_tag(profit=-1.0, linear_before=2) == "COOLDOWN_L3"


def test_format_settlement_audit_line_default_loss_tag():
    line = format_settlement_audit_line(4, "LOSS", -2.0, "PUT", "R_10", -0.2)
    assert "COOLDOWN_L1" in line


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
    assert line.startswith("[CLUSTER] || M5 || ")
    assert "R_10: PUT (Prob: 0.62300 Cal: 0.63500 Margin: 0.135 Edge: +0.092" in line
    assert "raw_edge:" in line and "be=0.581" in line
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
    assert line.startswith("[EXEC] || PUT [R_10] || STAKE: 2.06 (RECOVER_DAL_L1)")
    assert "PEND: 1.62" in line and "LIN: 1" in line and "CAP: 4.20" in line
    assert "BANCA: 87.69" in line and "CID: 1129497159" in line and "PAY: 1.79" in line
    assert "\n" not in line


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
        learn_detail="label=WIN buffer_n=1",
    )
    assert line.startswith("[RESOLVED] ||")
    assert "PEND: 2.00" in line
    assert "LIN: 1" in line
    assert "MODE: RECOVER_DAL_L1" in line
    assert "RECOVERY_INFEASIBLE" in line
    assert "LEARN: label=WIN buffer_n=1" in line


def test_format_kelly_audit_line():
    line = format_kelly_audit_line(
        {"conviction": 0.58, "live_wr": 0.5, "live_n": 4, "f_star": 0.001},
        stake=22.5,
        mode_tag="EXPLORE_KELLY",
        audit={"mode_tag": "EXPLORE_KELLY"},
    )
    assert line.startswith("[KELLY] || p=0.5800")
    assert "stake=22.50 (EXPLORE_KELLY)" in line
    custom_mode = format_kelly_audit_line(
        {"conviction": 0.58, "stake_regime": "recover"},
        stake=10.0,
        mode_tag="CUSTOM",
        audit={},
    )
    assert "mode=recover" in custom_mode
