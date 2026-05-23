"""Testes de cobertura para RiskManager Kelly."""

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_rolling_wins_cap():
    """Cobre o limite de amostras no histórico de vitórias."""
    rm = RiskManager({})
    for _ in range(110):
        rm.record_trade_outcome("SYM", won=True)
    wr, n = rm.get_wr_rolling_stats("SYM")
    assert n == 100


def test_risk_manager_cooldown_active():
    """Cobre o estado de cooldown ativo."""
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.last_result_tick = 100
    assert rm.is_on_cooldown(105) is True
    assert rm.is_on_cooldown(115) is False


def test_risk_manager_high_conviction_shorter_cooldown():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 12,
                "entry_cooldown_ticks_high_conviction": 6,
                "high_conviction_cooldown_threshold": 0.85,
            }
        }
    )
    rm.current_cooldown_ticks = 12
    rm.register_entry_conviction(0.9)
    rm.last_result_tick = 100
    assert rm.effective_cooldown_ticks() == 6
    assert rm.is_on_cooldown(105) is True
    assert rm.is_on_cooldown(107) is False


def test_risk_manager_effective_cooldown_when_active_zero():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 12,
                "entry_cooldown_ticks_high_conviction": 6,
                "high_conviction_cooldown_threshold": 0.85,
            }
        }
    )
    rm.current_cooldown_ticks = 0
    rm.register_entry_conviction(0.9)
    assert rm.effective_cooldown_ticks() == 6


def test_risk_manager_effective_cooldown_target_zero():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 0}})
    rm.current_cooldown_ticks = 8
    assert rm.effective_cooldown_ticks() == 8
