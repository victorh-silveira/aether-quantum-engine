"""Configurações e fixtures compartilhadas para suíte de testes."""

import pytest

from src.infrastructure.state.trading_state import TradingState


@pytest.fixture(autouse=True)
def reset_trading_state():
    """Redefine o singleton TradingState antes de cada teste."""
    TradingState.reset()
