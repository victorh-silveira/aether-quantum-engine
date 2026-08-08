"""Testes unitários para o sistema RiskManager baseado em Critério de Kelly."""

import pytest

from src.domain.risk.risk_manager import RiskManager


def test_recovery_dl_conviction_ok_wrapper(kelly_config):
    rm = RiskManager(kelly_config)
    assert rm._recovery_dl_conviction_ok(
        {"deploy_ok": True, "val_accuracy": 0.6, "trade_score": 0.65, "raw_prob": 0.65}
    )


def test_kelly_calculation_standard(kelly_config):
    """Verifica Kelly com piso explore M2 (neutral_bankroll_pct) e fraction do fixture."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "R_10", conviction=0.6)
    floor = 1000.0 * float(kelly_config["kelly"]["neutral_bankroll_pct"])
    assert stake >= floor - 1e-9
    assert stake == pytest.approx(7.64, abs=0.2)


def test_kelly_negative_edge_floors_p_and_sizes(kelly_config):
    """Sem edge bruto: piso de p garante f*>0 e stake>0 (sem kelly_no_edge)."""
    kelly_config["params"]["payout_estimate"] = 0.01
    kelly_config["kelly"]["kelly_p_floor"] = 0.55
    kelly_config["kelly"]["fraction"] = 0.5
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(10000.0, "R_10", conviction=0.5)
    assert stake > 0.0


def test_kelly_stake_capped_by_max_safe_bankroll_pct(kelly_config):
    """Verifica que a stake Kelly respeita o teto absoluto de 3.5% da banca."""
    kelly_config["kelly"]["fraction"] = 1.0
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(1000.0, "R_10", conviction=0.8)
    assert stake == pytest.approx(30.5, abs=0.2)


def test_kelly_high_conviction_scales_within_safe_cap(kelly_config):
    """Conviccoes altas escalam stake ate o teto seguro da banca."""
    kelly_config["kelly"]["fraction"] = 1.0
    kelly_config["kelly"]["max_stake_pct"] = 0.02
    kelly_config["kelly"]["max_stake_pct_high_conviction"] = 0.04
    kelly_config["kelly"]["max_bankroll_stake_fraction"] = 0.04
    kelly_config["kelly"]["high_conviction_stake_threshold"] = 0.85
    rm = RiskManager(kelly_config)
    low = rm.calculate_stake(1000.0, "R_10", conviction=0.7)
    high = rm.calculate_stake(1000.0, "R_10", conviction=0.9)
    assert high > low
    assert high == pytest.approx(24.4, abs=0.1)


def test_kelly_dynamic_win_rate(kelly_config):
    """Verifica a ponderação do win rate dinâmico no cálculo de probabilidade."""
    rm = RiskManager(kelly_config)
    for _ in range(10):
        rm.active_contract_ids = [1]
        rm.register_result(10.0, 1, "R_10")

    p = rm.effective_win_rate("R_10", conviction=0.6)
    assert p > 0.60
    assert p < 0.72


def test_kelly_respects_stake_min(kelly_config):
    """Verifica que a stake mínima é respeitada se houver edge."""
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(10.0, "R_10", conviction=0.6)
    assert stake == 1.0


def test_kelly_intelligent_recovery(kelly_config):
    """Verifica D'Alembert apos perda: Kelly + linear * U."""
    rm = RiskManager(kelly_config)

    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_10")
    assert rm.pending_loss["R_10"] == 10.0
    assert rm.consecutive_losses_linear == 1

    stake_low = rm.calculate_stake(
        10000.0,
        "R_10",
        conviction=0.6,
        dl_metrics={"execute": True, "trade_score": 0.65, "val_accuracy": 0.55},
    )
    stake_high = rm.calculate_stake(
        10000.0,
        "R_10",
        conviction=0.8,
        dl_metrics={"execute": True, "trade_score": 0.65, "val_accuracy": 0.55},
    )
    assert stake_high >= stake_low
    assert stake_low > 15.0

    rm.active_contract_ids = [2]
    rm.register_result(57.49, 2, "R_10")
    assert rm.pending_loss["R_10"] == 0.0


def test_dlambert_after_partial_win(kelly_config):
    """Mantem recovery linear enquanto houver perda pendente apos win parcial."""
    kelly_config["kelly"]["max_stake_pct"] = 0.01
    kelly_config["kelly"]["fraction"] = 0.005
    rm = RiskManager(kelly_config)

    rm.pending_loss["R_10"] = 8.54
    rm.last_loss_stake = 100.0
    rm.consecutive_losses_linear = 1
    rm.dlambert_unit = 10.0

    stake = rm.calculate_stake(
        10000.0,
        "R_10",
        conviction=0.61,
        dl_metrics={"execute": True, "trade_score": 0.65, "val_accuracy": 0.55},
    )
    assert stake > 10.0


def test_stake_zero_when_bankroll_below_min(kelly_config):
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(0.5, "R_10", conviction=0.4)
    assert stake == 0.0


def test_stake_zero_when_bankroll_below_stake_min_with_conviction(kelly_config):
    rm = RiskManager(kelly_config)
    stake = rm.calculate_stake(0.5, "R_10", conviction=0.55)
    assert stake == 0.0


def test_kelly_stop_win_zero_stake(kelly_config):
    """Verifica que a stake é zero se o stop win foi atingido."""
    kelly_config["large_account_stop_win_pct"] = 3.0
    kelly_config["small_account_threshold"] = 0.0
    rm = RiskManager(kelly_config)
    rm.set_initial_bankroll(1000.0)
    rm.total_session_profit = 35.0

    stake = rm.calculate_stake(1000.0, "R_10", conviction=0.8)
    assert stake == 0.0


def test_risk_manager_consecutive_losses_reset_on_win(kelly_config):
    rm = RiskManager(kelly_config)

    assert rm.consecutive_losses_linear == 0

    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "R_10")
    assert rm.consecutive_losses_linear == 1

    rm.active_contract_ids = [2]
    rm.register_result(-10.0, 2, "R_10")
    assert rm.consecutive_losses_linear == 2

    rm.active_contract_ids = [3]
    rm.register_result(15.0, 3, "R_10")
    assert rm.consecutive_losses_linear == 0
    assert rm.is_on_cooldown(99) is False


def test_risk_manager_consecutive_losses_fraction_reduction(kelly_config):
    """Apos perda pendente, proxima entrada usa D'Alembert em vez de reduzir Kelly."""
    rm = RiskManager(kelly_config)

    stake_base = rm.calculate_stake(1000.0, "R_10", conviction=0.6)

    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_10")

    stake_after_loss = rm.calculate_stake(1000.0, "R_10", conviction=0.6)
    assert stake_after_loss > stake_base


def test_risk_manager_dlambert_same_stake_with_same_linear(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_10")
    dl_metrics = {"execute": True, "trade_score": 0.60, "val_accuracy": 0.55}
    stake_first = rm.calculate_stake(1000.0, "R_10", conviction=0.55, dl_metrics=dl_metrics)
    stake_second = rm.calculate_stake(1000.0, "R_10", conviction=0.55, dl_metrics=dl_metrics)
    assert stake_second == stake_first
    assert stake_first > 1.5


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
    rm.consecutive_losses_linear = 3
    rm.current_cooldown_ticks = 80
    state = rm.get_state()
    assert state["consecutive_losses_linear"] == 3
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
    stake = rm.calculate_stake(
        1000.0,
        "R_10",
        conviction=0.85,
        dl_metrics={"execute": True, "live_n": 40, "live_wr": 0.55, "trade_score": 0.85},
    )
    assert stake in (15.0, 20.0, 25.0, 35.0)


def test_register_result_late_settlement_clears_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.contract_to_symbol[999] = "R_10"
    rm.pending_loss = {"R_10": 10.99}
    rm.register_result(15.17, 999, "R_10")
    assert sum(rm.pending_loss.values()) == pytest.approx(0.0, abs=0.01)
    assert rm.total_session_profit == pytest.approx(15.17, abs=0.01)


def test_register_result_ignores_duplicate_settlement(kelly_config):
    rm = RiskManager(kelly_config)
    rm.cluster_results = {1: 5.0}
    rm.active_contract_ids = [1]
    rm.register_result(5.0, 1, "R_10")
    assert rm.total_session_profit == pytest.approx(0.0, abs=0.01)


def test_cross_symbol_recovery(kelly_config):
    """Recupera perda de um simbolo em operacao em outro via D'Alembert."""
    rm = RiskManager(kelly_config)

    rm.active_contract_ids = [1]
    rm.record_contract_stake(1, 10.0)
    rm.register_result(-10.0, 1, "R_10")
    assert sum(rm.pending_loss.values()) == 10.0

    stake_b_high = rm.calculate_stake(
        1000.0,
        "R_10",
        conviction=0.8,
        dl_metrics={"execute": True, "trade_score": 0.65, "val_accuracy": 0.55},
    )
    assert stake_b_high > 1.5

    rm.active_contract_ids = [2]
    rm.register_result(12.0, 2, "R_10")
    assert sum(rm.pending_loss.values()) == 0.0


def test_partial_loss_recovery_and_break(kelly_config):
    """Verifica a redução parcial da perda pendente e o fluxo de break no loop de lucros."""
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1, 2]
    rm.register_result(-5.0, 1, "R_10")
    rm.register_result(-5.0, 2, "R_50")
    assert sum(rm.pending_loss.values()) == 10.0

    rm.active_contract_ids = [3]
    rm.register_result(3.0, 3, "R_10")

    assert rm.pending_loss["R_10"] == 2.0
    assert rm.pending_loss["R_50"] == 5.0


def test_recovery_allowed_rejects_weak_signal(kelly_config):
    kelly_config["dlambert"]["recovery_min_val_accuracy"] = 0.50
    kelly_config["dlambert"]["recovery_sizing_conviction"] = 0.58
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_10"] = 10.0
    dl_metrics = {"deploy_ok": True, "val_accuracy": 0.55, "trade_score": 0.50, "raw_prob": 0.50}
    assert rm._recovery_allowed("R_10", 0.50, dl_metrics=dl_metrics) is False


def test_recovery_allowed_fails_on_low_val_accuracy(kelly_config):
    kelly_config["dlambert"]["recovery_min_val_accuracy"] = 0.50
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_10"] = 10.0
    dl_metrics = {"deploy_ok": True, "val_accuracy": 0.40, "trade_score": 0.50, "raw_prob": 0.50}
    assert rm._recovery_allowed("R_10", 0.50, dl_metrics=dl_metrics) is False
