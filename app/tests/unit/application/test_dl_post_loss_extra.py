from types import SimpleNamespace

from src.application.services.deep_learning.dl_post_loss import (
    last_post_loss_pair,
    post_loss_block_reason,
    register_post_loss_ban,
)
from src.domain.models.trade import TradeDirection


def test_post_loss_flip_unblocks_call():
    orch = SimpleNamespace(config={"deep_learning": {}})
    register_post_loss_ban(orch, "X", TradeDirection.CALL, candle_cycles=2)
    assert post_loss_block_reason(orch, "X", TradeDirection.CALL, raw_prob=0.40, flip_raw_min=0.58) is None


def test_post_loss_flip_unblocks_put():
    orch = SimpleNamespace(config={"deep_learning": {}})
    register_post_loss_ban(orch, "X", TradeDirection.PUT, candle_cycles=2)
    assert post_loss_block_reason(orch, "X", TradeDirection.PUT, raw_prob=0.62, flip_raw_min=0.58) is None


def test_register_zero_cycles_noop():
    orch = SimpleNamespace()
    register_post_loss_ban(orch, "X", TradeDirection.CALL, candle_cycles=0)
    assert not getattr(orch, "_dl_post_loss_bans", [])


def test_last_post_loss_pair_strips_empty():
    orch = SimpleNamespace()
    orch._last_loss_symbol = "  "
    orch._last_loss_direction = "   "
    sym, direction = last_post_loss_pair(orch)
    assert sym is None and direction is None


def test_last_post_loss_pair_from_risk_manager():
    orch = SimpleNamespace()
    orch.risk_manager = SimpleNamespace(last_loss_symbol="R_75", last_loss_direction="PUT")
    sym, direction = last_post_loss_pair(orch)
    assert sym == "R_75" and direction == "PUT"
