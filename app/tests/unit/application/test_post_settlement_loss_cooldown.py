import pytest

from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    await_post_loss_cooldown,
    log_trading_cycle_cooldown_skip,
    orchestrator_cooldown_active,
    orchestrator_cooldown_remaining,
    orchestrator_cooldown_until,
    post_loss_cooldown_active,
    post_loss_cooldown_delay_seconds,
    schedule_post_loss_cooldown,
)


def test_post_loss_cooldown_behavior():
    assert post_loss_cooldown_delay_seconds(1) == 0.0
    assert post_loss_cooldown_delay_seconds(2) == 300.0
    assert post_loss_cooldown_delay_seconds(3) == 300.0
    assert post_loss_cooldown_active("LOSS", 1) is False
    assert post_loss_cooldown_active("LOSS", 2) is True
    assert post_loss_cooldown_active("WIN", 2) is False


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_is_noop(orch_ready):
    assert await await_post_loss_cooldown(orch_ready) == 0.0


def test_orchestrator_cooldown_helpers_active(orch_ready):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 3
    orch._last_settlement_outcome = "LOSS"
    assert schedule_post_loss_cooldown(orch) == 300.0
    assert orchestrator_cooldown_active(orch) is True
    assert orchestrator_cooldown_remaining(orch) > 0.0
    assert orchestrator_cooldown_until(orch) > 0.0
    log_trading_cycle_cooldown_skip(orch)
