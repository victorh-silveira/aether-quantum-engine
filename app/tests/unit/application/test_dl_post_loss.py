from types import SimpleNamespace

from src.application.services.deep_learning.dl_bridge_helpers import apply_symbol_loss_cooldown
from src.application.services.deep_learning.dl_post_loss import (
    post_loss_block_reason,
    register_post_loss_ban,
    tick_post_loss_bans,
)
from src.domain.models.trade import TradeDirection


def test_post_loss_ban_blocks_repeat_direction():
    orch = SimpleNamespace(config={"deep_learning": {}})
    register_post_loss_ban(orch, "R_75", TradeDirection.CALL, candle_cycles=2)
    assert post_loss_block_reason(orch, "R_75", TradeDirection.CALL) == "repeat_loss"


def test_post_loss_tick_clears_ban():
    orch = SimpleNamespace(config={"deep_learning": {}})
    register_post_loss_ban(orch, "R_75", TradeDirection.CALL, candle_cycles=1)
    tick_post_loss_bans(orch)
    assert post_loss_block_reason(orch, "R_75", TradeDirection.CALL) is None


def test_apply_cooldown_sets_repeat_loss_gate():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
    register_post_loss_ban(orch, "R_75", TradeDirection.PUT, candle_cycles=2)
    rm = SimpleNamespace(is_symbol_on_loss_cooldown=lambda _s: False)
    orch.risk_manager = rm
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": True, "raw_prob": 0.45},
    }
    out = apply_symbol_loss_cooldown(orch, "R_75", entry)
    assert out["metrics"]["execute"] is False
    assert out["metrics"]["gate_reason"] == "repeat_loss"
