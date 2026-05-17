"""Testes de cobertura para RiskManager Kelly."""

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_rolling_wins_cap():
    """Cobre o limite de amostras no histórico de vitórias."""
    rm = RiskManager({})
    for _ in range(110):
        rm.record_trade_outcome("SYM", won=True)
    wr, n = rm.get_wr_rolling_stats("SYM")
    assert n == 100


def test_risk_manager_kelly_stubs():
    """Cobre os métodos stub mantidos para compatibilidade."""
    rm = RiskManager({})
    assert rm.get_pending_recoveries() == {}
    assert rm.is_in_recovery() is False


def test_risk_manager_cooldown_active():
    """Cobre o estado de cooldown ativo."""
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.last_result_tick = 100
    assert rm.is_on_cooldown(105) is True
    assert rm.is_on_cooldown(115) is False


def test_risk_manager_load_state_empty():
    """Cobre o carregamento de estado vazio."""
    rm = RiskManager({})
    rm.load_state({})
    assert rm.initial_bankroll == 0.0
