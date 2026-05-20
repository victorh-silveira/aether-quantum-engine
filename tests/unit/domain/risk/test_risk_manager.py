"""Testes unitários para o sistema RiskManager baseado em Critério de Kelly."""

import pytest

from src.domain.risk.risk_manager import RiskManager


@pytest.fixture
def kelly_config():
    """Fornece uma configuração de Kelly padrão para os testes."""
    return {
        "kelly": {
            "base_win_rate": 0.50,
            "dynamic_win_rate": True,
            "dynamic_min_samples": 5,
            "fraction": 0.1,
            "max_stake_pct": 0.05,
        },
        "params": {"payout_estimate": 0.95, "stake_min": 1.0, "entry_cooldown_ticks": 0},
    }


def test_kelly_calculation_standard(kelly_config):
    """Verifica o cálculo de Kelly com probabilidade e payout padrão."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.6)
    assert stake == pytest.approx(17.89, abs=0.1)


def test_kelly_negative_edge_returns_min_stake(kelly_config):
    """Verifica que agora forçamos a stake_min mesmo sem vantagem (No-Idle)."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.5)
    assert stake == 1.0


def test_kelly_respects_max_stake_pct(kelly_config):
    """Verifica que a stake respeita o teto de porcentagem da banca."""
    kelly_config["kelly"]["fraction"] = 1.0
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake == 50.0


def test_kelly_dynamic_win_rate(kelly_config):
    """Verifica a ponderação do win rate dinâmico no cálculo de probabilidade."""
    rm = RiskManager(kelly_config)
    for _ in range(5):
        rm.active_contract_ids = [1]
        rm.register_result(10.0, 1, "OTC_FCHI")

    p = rm.effective_win_rate("OTC_FCHI", conviction=0.6)
    assert p == pytest.approx(0.72)


def test_kelly_respects_stake_min(kelly_config):
    """Verifica que a stake mínima é respeitada se houver edge."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(10.0, "OTC_FCHI", conviction=0.6)
    assert stake == 1.0


def test_kelly_intelligent_recovery(kelly_config):
    """Verifica se a stake aumenta para recuperar perdas em trades de alta convicção."""
    kelly_config["kelly"]["recovery_conviction_threshold"] = 0.70
    rm = RiskManager(kelly_config)

    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "OTC_FCHI")
    assert rm.pending_loss["OTC_FCHI"] == 10.0

    stake_low = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.6)
    assert stake_low == pytest.approx(17.89, abs=0.1)

    stake_high = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake_high == pytest.approx(60.52, abs=0.1)

    rm.active_contract_ids = [2]
    rm.register_result(57.49, 2, "OTC_FCHI")
    assert rm.pending_loss["OTC_FCHI"] == 0.0


def test_kelly_recovery_safety_cap(kelly_config):
    """Verifica se a recuperação respeita o limite máximo de stake (Safety Cap)."""
    kelly_config["kelly"]["max_recovery_stake_pct"] = 0.10
    rm = RiskManager(kelly_config)

    rm.pending_loss["OTC_FCHI"] = 500.0

    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake == 100.0


def test_kelly_stop_win_zero_stake(kelly_config):
    """Verifica que a stake é zero se o stop win foi atingido."""
    kelly_config["large_account_stop_win_pct"] = 3.0
    kelly_config["small_account_threshold"] = 0.0
    rm = RiskManager(kelly_config)
    rm.set_initial_bankroll(1000.0)
    rm.total_session_profit = 35.0

    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake == 0.0
