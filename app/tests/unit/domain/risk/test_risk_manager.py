"""Testes unitários para o sistema RiskManager baseado em Critério de Kelly."""

import math

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
            "martingale_sizing_conviction": 0.60,
            "stop_win_kelly_enabled": False,
        },
        "params": {"payout_estimate": 0.95, "stake_min": 1.0, "entry_cooldown_ticks": 0},
    }


def test_kelly_calculation_standard(kelly_config):
    """Verifica o cálculo de Kelly com probabilidade e payout padrão."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "R_50", conviction=0.6)
    assert stake == pytest.approx(17.89, abs=0.1)


def test_kelly_negative_edge_returns_min_stake(kelly_config):
    """Verifica que agora forçamos a stake_min mesmo sem vantagem (No-Idle)."""
    kelly_config["params"]["payout_estimate"] = 0.01
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "R_50", conviction=0.5)
    assert stake == 1.0


def test_kelly_caps_stake_by_max_pct(kelly_config):
    """Verifica que a stake Kelly respeita max_stake_pct da configuracao."""
    kelly_config["kelly"]["fraction"] = 1.0
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "R_50", conviction=0.8)
    assert stake == pytest.approx(50.0, abs=0.1)


def test_kelly_high_conviction_cap(kelly_config):
    """Verifica que conviccoes altas usam max_stake_pct_high_conviction."""
    kelly_config["kelly"]["fraction"] = 1.0
    kelly_config["kelly"]["max_stake_pct"] = 0.02
    kelly_config["kelly"]["max_stake_pct_high_conviction"] = 0.04
    kelly_config["kelly"]["high_conviction_stake_threshold"] = 0.85
    rm = RiskManager(kelly_config)
    low = rm.calculate_stake(1000.0, "R_50", conviction=0.7)
    high = rm.calculate_stake(1000.0, "R_50", conviction=0.9)
    assert high > low
    assert low == pytest.approx(20.0, abs=0.1)
    assert high == pytest.approx(40.0, abs=0.1)


def test_kelly_dynamic_win_rate(kelly_config):
    """Verifica a ponderação do win rate dinâmico no cálculo de probabilidade."""
    rm = RiskManager(kelly_config)
    for _ in range(5):
        rm.active_contract_ids = [1]
        rm.register_result(10.0, 1, "R_50")

    p = rm.effective_win_rate("R_50", conviction=0.6)
    assert p == pytest.approx(0.72)


def test_kelly_respects_stake_min(kelly_config):
    """Verifica que a stake mínima é respeitada se houver edge."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(10.0, "R_50", conviction=0.6)
    assert stake == 1.0


def test_kelly_intelligent_recovery(kelly_config):
    """Verifica martingale apos perda: recupera loss + lucro alvo Kelly."""
    rm = RiskManager(kelly_config)

    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    assert rm.pending_loss["R_50"] == 10.0

    stake_low = rm.calculate_stake(1000.0, "R_50", conviction=0.6)
    assert stake_low == pytest.approx(20.53, abs=0.5)

    rm.last_martingale_stake = 0.0
    stake_high = rm.calculate_stake(1000.0, "R_50", conviction=0.8)
    assert stake_high == pytest.approx(stake_low, abs=0.02)

    rm.active_contract_ids = [2]
    rm.register_result(57.49, 2, "R_50")
    assert rm.pending_loss["R_50"] == 0.0


def test_martingale_after_partial_win(kelly_config):
    """Mantem martingale enquanto houver perda pendente apos win parcial."""
    kelly_config["kelly"]["max_stake_pct"] = 0.01
    rm = RiskManager(kelly_config)

    rm.pending_loss["R_75"] = 8.54
    rm.last_loss_stake = 100.0
    rm.consecutive_losses = 0

    stake = rm.calculate_stake(10000.0, "R_75", conviction=0.61)
    cover = (8.54 + 100.0 * 0.95) / 0.95
    assert stake == pytest.approx(math.ceil(cover * 100) / 100, abs=0.02)


def test_kelly_keeps_fraction_with_consecutive_losses_without_pending(kelly_config):
    rm = RiskManager(kelly_config)
    stake_base = rm.calculate_stake(1000.0, "R_50", conviction=0.6)
    rm.consecutive_losses = 2
    stake_same = rm.calculate_stake(1000.0, "R_50", conviction=0.6)
    assert stake_same == stake_base


def test_stake_zero_when_bankroll_below_min(kelly_config):
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(0.5, "R_50", conviction=0.4)
    assert stake == 0.0


def test_stake_zero_when_bankroll_below_stake_min_with_conviction(kelly_config):
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(0.5, "R_50", conviction=0.55)
    assert stake == 0.0


def test_kelly_stop_win_zero_stake(kelly_config):
    """Verifica que a stake é zero se o stop win foi atingido."""
    kelly_config["large_account_stop_win_pct"] = 3.0
    kelly_config["small_account_threshold"] = 0.0
    rm = RiskManager(kelly_config)
    rm.set_initial_bankroll(1000.0)
    rm.total_session_profit = 35.0

    stake = rm.calculate_stake(1000.0, "R_50", conviction=0.8)
    assert stake == 0.0


def test_risk_manager_consecutive_losses_reset_on_win(kelly_config):
    rm = RiskManager(kelly_config)

    assert rm.consecutive_losses == 0

    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "R_50")
    assert rm.consecutive_losses == 1

    rm.active_contract_ids = [2]
    rm.register_result(-10.0, 2, "R_50")
    assert rm.consecutive_losses == 2

    rm.active_contract_ids = [3]
    rm.register_result(15.0, 3, "R_50")
    assert rm.consecutive_losses == 0
    assert rm.is_on_cooldown(99) is False


def test_risk_manager_consecutive_losses_fraction_reduction(kelly_config):
    """Apos perda pendente, proxima entrada usa martingale em vez de reduzir Kelly."""
    rm = RiskManager(kelly_config)

    stake_base = rm.calculate_stake(1000.0, "R_50", conviction=0.6)

    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")

    stake_after_loss = rm.calculate_stake(1000.0, "R_75", conviction=0.6)
    assert stake_after_loss > stake_base


def test_risk_manager_martingale_same_stake_regardless_of_conviction(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    dl_metrics = {"execute": True, "trade_score": 0.60, "val_accuracy": 0.55}
    stake_first = rm.calculate_stake(1000.0, "R_75", conviction=0.55, dl_metrics=dl_metrics)
    rm.last_martingale_stake = 0.0
    stake_second = rm.calculate_stake(1000.0, "R_75", conviction=0.55, dl_metrics=dl_metrics)
    assert stake_second == stake_first
    assert stake_first > 17.0


def test_proposal_skip_cycles_expire(kelly_config):
    rm = RiskManager(kelly_config)
    rm.register_proposal_failure("R_10", cycles=2)
    assert "R_10" in rm.proposal_skip_symbols()
    rm.decay_proposal_skip_cycles()
    assert "R_10" in rm.proposal_skip_symbols()
    rm.decay_proposal_skip_cycles()
    assert "R_10" not in rm.proposal_skip_symbols()


def test_risk_manager_get_state_exports(kelly_config):
    """Verifica se get_state exporta corretamente as novas métricas de perdas consecutivas."""
    rm = RiskManager(kelly_config)
    rm.consecutive_losses = 3
    rm.current_cooldown_ticks = 80
    state = rm.get_state()
    assert state["consecutive_losses"] == 3
    assert state["current_cooldown_ticks"] == 80


def test_single_strike_stake_boost_toward_stop_win(kelly_config):
    """Verifica se Kelly escala para o lucro restante do stop win diario."""
    kelly_config["large_account_stop_win_pct"] = 10.0
    kelly_config["small_account_threshold"] = 50.0
    kelly_config["kelly"]["fraction"] = 0.001
    kelly_config["kelly"]["max_stake_pct"] = 0.05
    kelly_config["kelly"]["stop_win_kelly_enabled"] = True
    kelly_config["kelly"]["stop_win_kelly_max_fraction"] = 0.72
    kelly_config["kelly"]["stop_win_kelly_cycles_target"] = 1.0
    rm = RiskManager(kelly_config)
    rm.set_initial_bankroll(1000.0)
    rm.total_session_profit = 0.0
    stake = rm.calculate_stake(1000.0, "R_50", conviction=0.85)
    assert stake == pytest.approx((100.0 / 0.95) * 0.72, abs=0.1)


def test_register_result_late_settlement_clears_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.contract_to_symbol[999] = "R_75"
    rm.pending_loss = {"R_75": 10.99}
    rm.register_result(15.17, 999, "R_75")
    assert sum(rm.pending_loss.values()) == pytest.approx(0.0, abs=0.01)
    assert rm.total_session_profit == pytest.approx(15.17, abs=0.01)


def test_register_result_ignores_duplicate_settlement(kelly_config):
    rm = RiskManager(kelly_config)
    rm.cluster_results = {1: 5.0}
    rm.active_contract_ids = [1]
    rm.register_result(5.0, 1, "R_50")
    assert rm.total_session_profit == pytest.approx(0.0, abs=0.01)


def test_cross_symbol_recovery(kelly_config):
    """Recupera perda de um simbolo em operacao em outro via martingale."""
    rm = RiskManager(kelly_config)

    rm.active_contract_ids = [1]
    rm.record_contract_stake(1, 10.0)
    rm.register_result(-10.0, 1, "R_50")
    assert sum(rm.pending_loss.values()) == 10.0

    stake_b_high = rm.calculate_stake(1000.0, "R_75", conviction=0.8)
    assert stake_b_high == pytest.approx(20.53, abs=0.5)

    rm.active_contract_ids = [2]
    rm.register_result(12.0, 2, "R_75")
    assert sum(rm.pending_loss.values()) == 0.0


def test_partial_loss_recovery_and_break(kelly_config):
    """Verifica a redução parcial da perda pendente e o fluxo de break no loop de lucros."""
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1, 2]
    rm.register_result(-5.0, 1, "R_50")
    rm.register_result(-5.0, 2, "R_75")
    assert sum(rm.pending_loss.values()) == 10.0

    rm.active_contract_ids = [3]
    rm.register_result(3.0, 3, "R_75")

    assert rm.pending_loss["R_50"] == 2.0
    assert rm.pending_loss["R_75"] == 5.0


def test_martingale_allowed_rejects_weak_signal_even_with_always(kelly_config):
    kelly_config["kelly"]["martingale_min_val_accuracy"] = 0.50
    kelly_config["kelly"]["recovery_martingale_always"] = True
    kelly_config["kelly"]["martingale_sizing_conviction"] = 0.58
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 10.0
    dl_metrics = {"deploy_ok": True, "val_accuracy": 0.55, "trade_score": 0.50, "raw_prob": 0.50}
    assert rm._martingale_allowed("R_50", 0.50, dl_metrics=dl_metrics) is False


def test_martingale_allowed_fails_on_low_val_accuracy_without_always(kelly_config):
    kelly_config["kelly"]["martingale_min_val_accuracy"] = 0.50
    kelly_config["kelly"]["recovery_martingale_always"] = False
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 10.0
    dl_metrics = {"deploy_ok": True, "val_accuracy": 0.40, "trade_score": 0.50, "raw_prob": 0.50}
    assert rm._martingale_allowed("R_50", 0.50, dl_metrics=dl_metrics) is False
