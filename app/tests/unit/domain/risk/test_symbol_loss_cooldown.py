from src.domain.risk.risk_manager import RiskManager


def test_symbol_loss_cooldown_after_negative_result(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 2
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    assert rm.last_loss_symbol == "R_50"
    assert rm.is_symbol_on_loss_cooldown("R_50") is True
    rm.tick_symbol_loss_cycle_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_50") is True
    rm.tick_symbol_loss_cycle_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_50") is False


def test_symbol_loss_cooldown_zero_disables(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 0
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    assert rm.symbol_loss_cooldown == {}


def test_symbol_loss_cooldown_disabled_without_cycles_key(kelly_config):
    kelly_config["kelly"].pop("symbol_loss_cooldown_cycles", None)
    kelly_config["kelly"]["symbol_loss_cooldown_candles"] = 0
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    assert rm.symbol_loss_cooldown == {}


def test_symbol_loss_cooldown_candle_unit_ticks_only_on_candle(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_candles"] = 2
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    rm.tick_symbol_loss_cycle_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_50") is True
    rm.tick_symbol_loss_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_50") is True
    rm.tick_symbol_loss_cooldowns()
    assert rm.is_symbol_on_loss_cooldown("R_50") is False
