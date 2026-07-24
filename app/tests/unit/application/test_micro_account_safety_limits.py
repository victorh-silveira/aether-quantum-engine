from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_symbols import candidate_execution_score
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
    assert max_safe_stake_cap(100.0) == pytest.approx(3.50)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=4) == pytest.approx(4.20)
    assert max_safe_stake_cap(300.0) == pytest.approx(10.50)
    soft = {"max_safe_stake_cap": 4.20}
    assert max_safe_stake_cap(80.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.0)
    assert max_safe_stake_cap(70.0, consecutive_losses_linear=5, soft_recovery=soft) == pytest.approx(3.5)


def test_starvation_decay_skips_threshold_8():
    assert starvation_decay_factor(7) == 1.0
    assert starvation_decay_factor(8) == pytest.approx(0.90)
    assert starvation_decay_factor(9) == pytest.approx(0.80)


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


def test_candidate_score_no_direction_bias_in_recovery():
    call_score = candidate_execution_score({"raw_prob": 0.62, "execute": True}, recovery_active=True, symbol="R_10")
    put_score = candidate_execution_score({"raw_prob": 0.38, "execute": True}, recovery_active=True, symbol="R_10")
    assert call_score == pytest.approx(put_score, abs=0.05)


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
    assert res is not None
    assert entry["metrics"].get("gate_reason") != "technical_discordance"


@patch("src.application.services.execution_direction_resolver.resolve_meta_payoff_edge")
@patch("src.application.services.execution_direction_resolver.reject_on_quality_gate")
def test_resolve_execution_direction_concordance_boost(mock_reject, mock_resolve_meta):
    mock_resolve_meta.return_value = (0.05, False)
    mock_reject.return_value = False
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
    assert entry["metrics"]["tcn_score"] == pytest.approx(0.66)


def test_align_direction_to_rsi_trend_mean_reversion():
    from src.application.services.execution_direction_discordance import align_direction_to_rsi_trend

    metrics = {
        "macro_indicators": {
            "rsi": 0.48,
            "adx": 0.15,
            "hurst": 0.35,
        }
    }
    dir_res = align_direction_to_rsi_trend(TradeDirection.PUT, metrics)
    assert dir_res == TradeDirection.PUT
    assert metrics.get("micro_regime_mean_reversion") is True
