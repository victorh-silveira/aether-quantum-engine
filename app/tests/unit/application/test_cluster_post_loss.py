from unittest.mock import MagicMock

from src.application.services.llm.cluster_post_loss import (
    cluster_post_loss_block_reason,
    record_cluster_loss,
    tick_cluster_repeat_loss_cooldown,
)
from src.domain.models.trade import TradeDirection


def test_record_cluster_loss_sets_pause_and_last_setup():
    orch = MagicMock()
    orch.config = {"orchestrator": {"cluster_pause_after_loss_cycles": 3}}
    record_cluster_loss(orch, symbol="R_50", direction=TradeDirection.CALL)
    assert orch._cluster_pause_cycles_remaining == 3
    assert orch._last_loss_symbol == "R_50"
    assert orch._last_loss_direction == "CALL"


def test_pause_active_does_not_block_cluster_entries():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = True
    reason = cluster_post_loss_block_reason(
        orch,
        target_sym="R_25",
        target_direction=TradeDirection.CALL,
    )
    assert reason is None


def test_record_cluster_loss_sets_repeat_block_cycles_when_configured():
    orch = MagicMock()
    orch.config = {
        "orchestrator": {
            "cluster_pause_after_loss_cycles": 0,
            "cluster_repeat_loss_block_cycles": 2,
        }
    }
    record_cluster_loss(orch, symbol="R_50", direction=TradeDirection.PUT)
    assert orch._repeat_loss_block_cycles_remaining == 2


def test_tick_cluster_repeat_loss_cooldown_decrements_remaining():
    orch = MagicMock()
    orch._repeat_loss_block_cycles_remaining = 2
    orch._last_loss_symbol = "R_50"
    orch._last_loss_direction = "PUT"
    tick_cluster_repeat_loss_cooldown(orch)
    assert orch._repeat_loss_block_cycles_remaining == 1
    assert orch._last_loss_symbol == "R_50"


def test_tick_cluster_repeat_loss_cooldown_clears_last_loss():
    orch = MagicMock()
    orch._repeat_loss_block_cycles_remaining = 0
    orch._last_loss_symbol = "R_50"
    orch._last_loss_direction = "PUT"
    tick_cluster_repeat_loss_cooldown(orch)
    assert orch._last_loss_symbol == ""
    assert orch._last_loss_direction == ""
    assert orch._repeat_loss_block_cycles_remaining is None


def test_cluster_block_repeat_loss_setup_disabled():
    orch = MagicMock()
    orch.config = {"orchestrator": {"cluster_block_repeat_loss_setup": False}}
    orch._last_loss_symbol = "R_50"
    orch._last_loss_direction = "PUT"
    assert (
        cluster_post_loss_block_reason(
            orch,
            target_sym="R_50",
            target_direction=TradeDirection.PUT,
        )
        is None
    )


def test_repeat_loss_setup_blocks_same_symbol_direction():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = False
    orch._last_loss_symbol = "R_25"
    orch._last_loss_direction = "CALL"
    reason = cluster_post_loss_block_reason(
        orch,
        target_sym="R_25",
        target_direction=TradeDirection.CALL,
    )
    assert reason == "repeat_loss_setup"
