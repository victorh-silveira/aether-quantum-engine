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


def test_kelly_high_conviction_raises_stake_cap(kelly_config):
    """Conviccao alta usa max_stake_pct_high_conviction quando configurado."""
    kelly_config["kelly"]["fraction"] = 1.0
    kelly_config["kelly"]["max_stake_pct"] = 0.02
    kelly_config["kelly"]["max_stake_pct_high_conviction"] = 0.04
    kelly_config["kelly"]["high_conviction_stake_threshold"] = 0.85
    rm = RiskManager(kelly_config)
    low = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.7)
    high = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.9)
    assert high > low
    assert high == 40.0


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


def test_risk_manager_consecutive_losses_reset_on_win(kelly_config):
    rm = RiskManager(kelly_config)

    assert rm.consecutive_losses == 0

    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "OTC_FCHI")
    assert rm.consecutive_losses == 1

    rm.active_contract_ids = [2]
    rm.register_result(-10.0, 2, "OTC_FCHI")
    assert rm.consecutive_losses == 2

    rm.active_contract_ids = [3]
    rm.register_result(15.0, 3, "OTC_FCHI")
    assert rm.consecutive_losses == 0
    assert rm.is_on_cooldown(99) is False


def test_risk_manager_consecutive_losses_fraction_reduction(kelly_config):
    """Verifica se a fração de Kelly é reduzida exponencialmente após perdas consecutivas."""
    rm = RiskManager(kelly_config)

    # Base stake sem perdas
    stake_base = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.6)

    # Adiciona 1 perda no símbolo OTC_FCHI
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "OTC_FCHI")

    # Fração deve reduzir pela metade para um novo símbolo sem perdas (reduction = 0.5)
    stake_loss_1 = rm.calculate_stake(1000.0, "OTC_GDAXI", conviction=0.6)
    assert stake_loss_1 < stake_base
    assert stake_loss_1 == pytest.approx(stake_base * 0.5, abs=0.5)


def test_risk_manager_consecutive_losses_recovery_conviction(kelly_config):
    """Verifica se a convicção exigida para recuperação permanece estável e funcional."""
    kelly_config["kelly"]["recovery_conviction_threshold"] = 0.70
    rm = RiskManager(kelly_config)

    # 1ª perda
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "OTC_FCHI")

    # 2ª perda
    rm.active_contract_ids = [2]
    rm.register_result(-5.0, 2, "OTC_FCHI")

    # Conviction de 0.65 (menor que o limiar 0.70) não deve acionar recuperação
    stake_no_rec = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.65)

    # Conviction de 0.70 deve acionar recuperação bypassando a redução de tamanho
    stake_rec = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.70)
    assert stake_rec > stake_no_rec


def test_risk_manager_get_state_exports(kelly_config):
    """Verifica se get_state exporta corretamente as novas métricas de perdas consecutivas."""
    rm = RiskManager(kelly_config)
    rm.consecutive_losses = 3
    rm.current_cooldown_ticks = 80
    state = rm.get_state()
    assert state["consecutive_losses"] == 3
    assert state["current_cooldown_ticks"] == 80
