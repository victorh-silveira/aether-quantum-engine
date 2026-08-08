from src.domain.risk.risk_manager import RiskManager


def test_symbol_loss_cooldown_always_disabled_and_inactive(kelly_config):

    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 2
    kelly_config["kelly"]["symbol_loss_cooldown_candles"] = 2
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_10")
    assert rm.last_loss_symbol == "R_10"
    assert rm.is_symbol_on_loss_cooldown("R_10") is False
    assert rm.symbol_loss_cooldown == {}
    rm.tick_symbol_loss_cycle_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_10") is False
    rm.tick_symbol_loss_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_10") is False
