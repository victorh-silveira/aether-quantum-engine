from unittest.mock import patch

from src.domain.risk.risk_manager import RiskManager


def test_martingale_blocked_by_dl_metrics(kelly_config):
    rm = RiskManager(kelly_config)
    rm.recovery_threshold = 0.5
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.75,
            dl_metrics={"gate_reason": "deploy", "val_brier": 0.1, "deploy_ok": True},
        )
        is False
    )
    rm.pending_loss["RDBULL"] = 50.0
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.75,
            dl_metrics={"gate_reason": "deploy", "val_brier": 0.1, "deploy_ok": True},
        )
        is True
    )
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.75,
            dl_metrics={"val_brier": 0.35, "deploy_ok": True},
            max_val_brier=0.28,
        )
        is True
    )
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.47,
            dl_metrics={"gate_reason": "raw_conviction", "val_brier": 0.1, "deploy_ok": False},
            order_direction="PUT",
        )
        is True
    )
    rm.pending_loss.clear()
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.75,
            dl_metrics={"val_brier": 0.1, "deploy_ok": False},
        )
        is False
    )


def test_martingale_blocked_repeat_loss_direction(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["RDBULL"] = 20.0
    rm.last_loss_symbol = "RDBULL"
    rm.last_loss_direction = "CALL"
    assert (
        rm._martingale_allowed(
            "RDBULL",
            0.8,
            dl_metrics={"val_brier": 0.1, "deploy_ok": True},
            order_direction="CALL",
        )
        is False
    )


def test_calculate_stake_logs_when_martingale_blocked_by_repeat_loss(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["RDBULL"] = 50.0
    rm.last_loss_symbol = "RDBULL"
    rm.last_loss_direction = "PUT"
    with patch.object(rm.logger, "info") as mock_info:
        rm.calculate_stake(
            1000.0,
            "RDBULL",
            conviction=0.6,
            cycle_id=2,
            order_direction="PUT",
            dl_metrics={"val_brier": 0.1, "deploy_ok": True},
        )
    assert any("Martingale bloqueado" in str(c) for c in mock_info.call_args_list)


def test_symbol_loss_cooldown_records_direction(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 1
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "RDBULL", direction="PUT")
    assert rm.last_loss_direction == "PUT"
