from unittest.mock import MagicMock

from src.application.services.llm.cluster_post_loss import cluster_post_loss_block_reason, record_cluster_loss
from src.domain.models.trade import TradeDirection


def test_record_cluster_loss_sets_pause_and_last_setup():
    orch = MagicMock()
    orch.config = {"orchestrator": {"cluster_pause_after_loss_cycles": 3}}
    record_cluster_loss(orch, symbol="OTC_NDX", direction=TradeDirection.CALL)
    assert orch._cluster_pause_cycles_remaining == 3
    assert orch._last_loss_symbol == "OTC_NDX"
    assert orch._last_loss_direction == "CALL"


def test_pause_active_only_when_strict_true():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = MagicMock()
    reason = cluster_post_loss_block_reason(
        orch,
        target_sym="OTC_SPC",
        target_direction=TradeDirection.CALL,
    )
    assert reason is None


def test_repeat_loss_setup_blocks_same_symbol_direction():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = False
    orch._last_loss_symbol = "OTC_SPC"
    orch._last_loss_direction = "CALL"
    reason = cluster_post_loss_block_reason(
        orch,
        target_sym="OTC_SPC",
        target_direction=TradeDirection.CALL,
    )
    assert reason == "repeat_loss_setup"
