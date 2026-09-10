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
    assert post_loss_cooldown_delay_seconds(0) == 0.0
    assert post_loss_cooldown_delay_seconds(1) == 300.0
    assert post_loss_cooldown_delay_seconds(2) == 300.0
    assert post_loss_cooldown_delay_seconds(3) == 600.0
    assert post_loss_cooldown_delay_seconds(4) == 900.0
    assert post_loss_cooldown_delay_seconds(5) == 900.0
    assert post_loss_cooldown_active("LOSS", 1) is True
    assert post_loss_cooldown_active("LOSS", 2) is True
    assert post_loss_cooldown_active("WIN", 2) is False


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_is_noop(orch_ready):
    assert await await_post_loss_cooldown(orch_ready) == 0.0


def test_orchestrator_cooldown_helpers_active(orch_ready):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 3
    orch._last_settlement_outcome = "LOSS"
    assert schedule_post_loss_cooldown(orch) == 600.0
    assert orchestrator_cooldown_active(orch) is True
    assert orchestrator_cooldown_remaining(orch) > 0.0
    assert orchestrator_cooldown_until(orch) > 0.0
    log_trading_cycle_cooldown_skip(orch)


def test_post_loss_cooldown_override_and_max_until(orch_ready):
    orch = orch_ready
    orch.config = {
        "orchestrator": {
            "execution": {
                "post_loss_cooldown": {
                    "lin_min": 2,
                    "delay_seconds_lin1": 300,
                    "delay_seconds_lin2": 300,
                    "delay_seconds_lin3": 600,
                    "delay_seconds_lin4": 900,
                }
            }
        }
    }
    assert post_loss_cooldown_delay_seconds(1, orch) == 0.0
    assert post_loss_cooldown_active("LOSS", 1, orch) is False
    assert post_loss_cooldown_delay_seconds(2, orch) == 300.0
    orch.risk_manager.consecutive_losses_linear = 1
    orch._last_settlement_outcome = "LOSS"
    assert schedule_post_loss_cooldown(orch) == 0.0
    orch._cooldown_until = 9_999_999_999.0
    orch.risk_manager.consecutive_losses_linear = 2
    assert schedule_post_loss_cooldown(orch) == 300.0
    assert orch._cooldown_until >= 9_999_999_999.0
    orch.config = None
    assert post_loss_cooldown_delay_seconds(1, orch) == 300.0
    orch.config = {"orchestrator": "x"}
    assert post_loss_cooldown_delay_seconds(3, orch) == 600.0
    from src.application.services.execution_runtime_config import resolve_post_loss_cooldown_config

    with pytest.raises(ValueError, match="post_loss_cooldown"):
        resolve_post_loss_cooldown_config({"post_loss_cooldown": "bad"})
