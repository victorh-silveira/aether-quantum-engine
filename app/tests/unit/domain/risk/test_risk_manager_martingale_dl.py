from src.domain.risk.risk_manager import RiskManager


def test_martingale_native_always_on_with_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 50.0
    assert rm._martingale_allowed(
        "R_50",
        0.40,
        dl_metrics={"gate_reason": "deploy", "val_brier": 0.35, "deploy_ok": False},
        order_direction="PUT",
    )
    assert rm._martingale_allowed(
        "R_75",
        0.56,
        dl_metrics={"gate_reason": "brier", "val_brier": 0.40},
        order_direction="CALL",
        last_loss_symbol="R_50",
        last_loss_direction="CALL",
    )


def test_martingale_native_progression_doubles_after_loss(kelly_config):
    kelly_config["kelly"]["martingale_multiplier"] = 2.0
    kelly_config["kelly"]["max_recovery_stake_pct"] = 0.10
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    first = rm.calculate_stake(1000.0, "R_75", conviction=0.56)
    rm.last_martingale_stake = first
    rm.consecutive_losses = 1
    rm.pending_loss["R_50"] = 10.0
    second = rm.calculate_stake(990.0, "R_75", conviction=0.56)
    assert second >= first * 1.9


def test_symbol_loss_cooldown_records_direction(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 1
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "R_50", direction="PUT")
    assert rm.last_loss_direction == "PUT"
