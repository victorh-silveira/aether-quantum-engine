"""Testes de sizing soft por discordancia de escala."""

import pytest

from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.domain.models.trade import TradeDirection


def test_scale_sizing_aligned_noop():
    metrics = {"kelly_fraction_scale": 1.0, "scale_discordance": False}
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.PUT, metrics)
    assert metrics["kelly_fraction_scale"] == 1.0
    assert metrics["scale_force_explore"] is False


def test_scale_sizing_discord_dampens_and_blocks_recover():
    metrics = {"kelly_fraction_scale": 1.0, "scale_discordance": True}
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.PUT, metrics)
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics["scale_max_stake_pct"] == pytest.approx(0.05)
    assert metrics["scale_force_explore"] is True
    assert "discord" in metrics["scale_sizing_reason"]
