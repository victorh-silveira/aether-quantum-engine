from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_symbols_recovery import recovery_rank_score
from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap, neutral_edge_dynamic_unit
from src.infrastructure.state.state_manager import StateManager


def test_micro_capital_base_unit():
    # Banca de $100 -> Unidade base U deve ser 1.0% ($1.00)
    assert neutral_edge_dynamic_unit(100.0) == 1.0
    # Banca de $250 -> Unidade base U deve ser 1.0% ($2.50)
    assert neutral_edge_dynamic_unit(250.0) == 2.50
    # Banca de $1000 -> Unidade base U deve ser 0.15% ($1.50)
    assert neutral_edge_dynamic_unit(1000.0) == 1.50


def test_micro_capital_max_safe_stake_cap():
    # Para banca de $100, deve retornar no mínimo $10.00
    assert max_safe_stake_cap(100.0) == pytest.approx(10.0)
    # Para banca de $300, deve retornar o cálculo padrão de pct (3.5% = $10.50)
    assert max_safe_stake_cap(300.0) == pytest.approx(10.50)


def test_starvation_decay_skips_threshold_6():
    # Abaixo do limiar (5 skips) -> Sem decaimento (1.0)
    assert starvation_decay_factor(5) == 1.0
    # Exatamente no limiar (6 skips) -> 10% de decaimento (0.90)
    assert starvation_decay_factor(6) == 0.90
    # Acima do limiar (7 skips) -> 20% de decaimento (0.80)
    assert starvation_decay_factor(7) == 0.80


def test_state_manager_float_tolerance():
    mgr = StateManager()

    # Inicialização (snapshot = 0.0) -> Deve espelhar independentemente da diferença
    mgr.mirror_balance(100.0)
    assert mgr.read_cached_balance() == 100.0

    # Flutuação inferior a $0.02 (+$0.01) -> Deve ignorar e manter $100.0
    mgr.mirror_balance(100.01)
    assert mgr.read_cached_balance() == 100.0

    # Flutuação igual a $0.02 (+$0.02) -> Deve atualizar
    mgr.mirror_balance(100.02)
    assert mgr.read_cached_balance() == 100.02

    # Flutuação superior a $0.02 (+$0.05) -> Deve atualizar
    mgr.mirror_balance(100.07)
    assert mgr.read_cached_balance() == 100.07


def test_recovery_rank_score_drift_aligned_boost():
    item_aligned = ("RDBULL", TradeDirection.CALL, {})
    item_non_aligned = ("RDBULL", TradeDirection.PUT, {})

    score_aligned = recovery_rank_score(item_aligned, base_score=0.5)
    score_non_aligned = recovery_rank_score(item_non_aligned, base_score=0.5)

    # score_aligned deve receber +0.15 de boost por estar alinhado ao drift natural do RDBULL (CALL)
    assert score_aligned - score_non_aligned == pytest.approx(0.15)


def test_resolve_execution_direction_technical_discordance_veto():
    # DL indica CALL, mas indicadores têm 80% de PUT_VOTES (discordância >= 75%)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.60,
            "call_votes": 2,
            "put_votes": 8,
        },
    }
    res = resolve_execution_direction(entry, symbol="SYM")
    assert res is None
    assert entry["metrics"]["gate_reason"] == "technical_discordance"
    assert entry["metrics"]["quality_guard_reject"] is True


@patch("src.application.services.execution_direction_resolver.resolve_meta_payoff_edge")
@patch("src.application.services.execution_direction_resolver._reject_on_quality_gate")
def test_resolve_execution_direction_concordance_boost(mock_reject, mock_resolve_meta):
    mock_resolve_meta.return_value = (0.05, False)
    mock_reject.return_value = False

    # DL indica CALL, e indicadores têm 90% de CALL_VOTES (concordância >= 80%)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.60,
            "call_votes": 9,
            "put_votes": 1,
            "execute": True,
        },
    }
    res = resolve_execution_direction(entry, symbol="SYM")
    assert res is not None
    # prob 0.60 + 0.05 boost = 0.65
    assert entry["metrics"]["tcn_score"] == pytest.approx(0.65)
