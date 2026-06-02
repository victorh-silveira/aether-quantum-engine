"""Testes unitários para o sistema RiskManager baseado em Critério de Kelly."""

import datetime
from unittest.mock import patch

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


def test_kelly_caps_stake_by_max_pct(kelly_config):
    """Verifica que a stake Kelly respeita max_stake_pct da configuracao."""
    kelly_config["kelly"]["fraction"] = 1.0
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake == pytest.approx(50.0, abs=0.1)


def test_kelly_high_conviction_cap(kelly_config):
    """Verifica que conviccoes altas usam max_stake_pct_high_conviction."""
    kelly_config["kelly"]["fraction"] = 1.0
    kelly_config["kelly"]["max_stake_pct"] = 0.02
    kelly_config["kelly"]["max_stake_pct_high_conviction"] = 0.04
    kelly_config["kelly"]["high_conviction_stake_threshold"] = 0.85
    rm = RiskManager(kelly_config)
    low = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.7)
    high = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.9)
    assert high > low
    assert low == pytest.approx(20.0, abs=0.1)
    assert high == pytest.approx(40.0, abs=0.1)


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
    assert stake_low == pytest.approx(
        8.94, abs=0.1
    )  # Nova regra: reduz pela metade por perdas consecutivas se não for tentativa de recuperação

    stake_high = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake_high == pytest.approx(50.0, abs=0.1)

    rm.active_contract_ids = [2]
    rm.register_result(57.49, 2, "OTC_FCHI")
    assert rm.pending_loss["OTC_FCHI"] == 0.0


def test_kelly_recovery_respects_safety_cap(kelly_config):
    """Verifica se a recuperacao respeita max_recovery_stake_pct."""
    kelly_config["kelly"]["max_recovery_stake_pct"] = 0.10
    rm = RiskManager(kelly_config)

    rm.pending_loss["OTC_FCHI"] = 500.0

    stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.8)
    assert stake == pytest.approx(100.0, abs=0.1)


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
    kelly_config["kelly"]["recovery_conviction_threshold"] = 0.70
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


def test_single_strike_stake_boost_in_window(kelly_config):
    """Verifica se a stake de Single Strike e aplicada na janela com alta conviccao."""
    kelly_config["small_account_stop_win"] = 100.0
    kelly_config["small_account_threshold"] = 100.0
    kelly_config["kelly"]["stop_win_aggressive"] = False
    kelly_config["kelly"]["fraction"] = 0.001
    rm = RiskManager(kelly_config)
    rm.set_initial_bankroll(1000.0)
    rm.total_session_profit = 0.0

    mock_now = datetime.datetime(2026, 6, 1, 14, 0, 0, tzinfo=datetime.UTC)

    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        mock_dt.timezone = datetime.timezone
        mock_dt.UTC = datetime.UTC

        stake = rm.calculate_stake(1000.0, "OTC_FCHI", conviction=0.85)
        assert stake == pytest.approx(50.0, abs=0.1)


def test_cross_symbol_recovery(kelly_config):
    """Verifica se uma perda em um símbolo pode ser recuperada em outro símbolo com alta convicção."""
    kelly_config["kelly"]["recovery_conviction_threshold"] = 0.70
    rm = RiskManager(kelly_config)

    # Registra uma perda no símbolo A
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "OTC_FCHI")
    assert sum(rm.pending_loss.values()) == 10.0

    # Tenta calcular stake com alta convicção no símbolo B
    stake_b_high = rm.calculate_stake(1000.0, "OTC_GDAXI", conviction=0.8)
    assert stake_b_high == pytest.approx(50.0, abs=0.1)

    # Se ganhar no símbolo B, o lucro reduz a perda pendente globalmente
    rm.active_contract_ids = [2]
    rm.register_result(12.0, 2, "OTC_GDAXI")
    assert sum(rm.pending_loss.values()) == 0.0


def test_partial_loss_recovery_and_break(kelly_config):
    """Verifica a redução parcial da perda pendente e o fluxo de break no loop de lucros."""
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1, 2]
    rm.register_result(-5.0, 1, "OTC_FCHI")
    rm.register_result(-5.0, 2, "OTC_GDAXI")
    assert sum(rm.pending_loss.values()) == 10.0

    # Registrar ganho de 3.0 (menor que a primeira perda de 5.0)
    rm.active_contract_ids = [3]
    rm.register_result(3.0, 3, "OTC_GDAXI")

    # Deve executar a ramificação else para OTC_FCHI (reduzindo a 2.0)
    # E o lucro restante zera, fazendo o break acontecer na próxima chave (OTC_GDAXI continua 5.0)
    assert rm.pending_loss["OTC_FCHI"] == 2.0
    assert rm.pending_loss["OTC_GDAXI"] == 5.0
