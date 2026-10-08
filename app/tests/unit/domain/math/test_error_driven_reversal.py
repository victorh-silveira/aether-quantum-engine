"""Testes unitarios para calculo matematico de Brier score e Alpha Flip."""

import pytest

from src.domain.math.error_driven_reversal import (
    calculate_trade_brier_score,
    is_error_driven_reversal_armed,
    resolve_target_reversal_direction,
)


def test_calculate_trade_brier_score_call():
    """Verifica calculo de Brier score para CALL com vitoria e derrota."""
    brier_win, eps_win = calculate_trade_brier_score(0.65, won=True, direction="CALL")
    assert eps_win == pytest.approx(0.35)
    assert brier_win == pytest.approx(0.35**2)

    brier_loss, eps_loss = calculate_trade_brier_score(0.65, won=False, direction="CALL")
    assert eps_loss == pytest.approx(-0.65)
    assert brier_loss == pytest.approx(0.4225)


def test_calculate_trade_brier_score_put():
    """Verifica calculo de Brier score para PUT com vitoria e derrota."""
    brier_win, eps_win = calculate_trade_brier_score(0.35, won=True, direction="PUT")
    assert eps_win == pytest.approx(-0.35)
    assert brier_win == pytest.approx(0.35**2)

    brier_loss, eps_loss = calculate_trade_brier_score(0.35, won=False, direction="PUT")
    assert eps_loss == pytest.approx(0.65)
    assert brier_loss == pytest.approx(0.4225)


def test_calculate_trade_brier_score_clamping_and_unknown_direction():
    """Verifica clamping de limites e direcao desconhecida."""
    brier, eps = calculate_trade_brier_score(1.5, won=False, direction="UNKNOWN")
    assert eps == pytest.approx(-1.0)
    assert brier == pytest.approx(1.0)

    brier_neg, eps_neg = calculate_trade_brier_score(-0.2, won=True, direction="UNKNOWN")
    assert eps_neg == pytest.approx(1.0)
    assert brier_neg == pytest.approx(1.0)


def test_is_error_driven_reversal_armed():
    """Verifica condicoes de acionamento do Alpha Flip."""
    assert is_error_driven_reversal_armed(0.4225, 0.65, threshold=0.40, min_conviction=0.60) is True
    assert is_error_driven_reversal_armed(0.38, 0.65, threshold=0.40, min_conviction=0.60) is False
    assert is_error_driven_reversal_armed(0.4225, 0.52, threshold=0.40, min_conviction=0.60) is False
    assert is_error_driven_reversal_armed(0.4225, 0.35, threshold=0.40, min_conviction=0.60) is True


def test_resolve_target_reversal_direction():
    """Verifica inversao de direcao para contra-ataque adaptativo."""
    assert resolve_target_reversal_direction("CALL") == "PUT"
    assert resolve_target_reversal_direction("call") == "PUT"
    assert resolve_target_reversal_direction("PUT") == "CALL"
    assert resolve_target_reversal_direction("put") == "CALL"
    assert resolve_target_reversal_direction("UNKNOWN") == "PUT"
