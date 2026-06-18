from src.application.services.execution_symbols_recovery import apply_recovery_direction_flip
from src.domain.models.trade import TradeDirection
from tests.market_symbols import HEDGE_PEER_SYMBOL, PAIR


def test_apply_recovery_direction_flip_inverts_same_symbol():
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True},
        },
    }
    best = (PAIR, TradeDirection.CALL, decisions[PAIR]["metrics"])
    flipped = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
    )
    assert flipped is not None
    assert flipped[1] == TradeDirection.PUT
    assert flipped[2].get("direction_inverted") is True


def test_apply_recovery_direction_flip_skips_when_conviction_strong():
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.62, "trade_score": 0.62, "deploy_ok": True},
        },
    }
    best = (PAIR, TradeDirection.CALL, decisions[PAIR]["metrics"])
    flipped = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
        flip_max_conviction=0.56,
    )
    assert flipped == best
    assert flipped[1] == TradeDirection.CALL


def test_apply_recovery_direction_flip_disabled_when_max_zero():
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True},
        },
    }
    best = (PAIR, TradeDirection.CALL, decisions[PAIR]["metrics"])
    flipped = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
        flip_max_conviction=0.0,
    )
    assert flipped == best
    assert flipped[1] == TradeDirection.CALL


def test_apply_recovery_direction_flip_noop_branches():
    base = (PAIR, TradeDirection.CALL, {"raw_prob": 0.62})
    assert (
        apply_recovery_direction_flip(
            None, {}, recovery_active=True, last_loss_symbol=PAIR, last_loss_direction="CALL", flip_enabled=True
        )
        is None
    )
    assert (
        apply_recovery_direction_flip(
            base, {}, recovery_active=False, last_loss_symbol=PAIR, last_loss_direction="CALL", flip_enabled=True
        )
        == base
    )
    assert (
        apply_recovery_direction_flip(
            base, {}, recovery_active=True, last_loss_symbol=PAIR, last_loss_direction="CALL", flip_enabled=False
        )
        == base
    )
    assert (
        apply_recovery_direction_flip(
            base, {}, recovery_active=True, last_loss_symbol=None, last_loss_direction="CALL", flip_enabled=True
        )
        == base
    )
    assert (
        apply_recovery_direction_flip(
            (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {}),
            {},
            recovery_active=True,
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            flip_enabled=True,
        )[0]
        == HEDGE_PEER_SYMBOL
    )
    assert apply_recovery_direction_flip(
        (PAIR, TradeDirection.PUT, {}),
        {},
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
    ) == (PAIR, TradeDirection.PUT, {})
    assert apply_recovery_direction_flip(
        (PAIR, TradeDirection.CALL, {}),
        {},
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
    ) == (PAIR, TradeDirection.CALL, {})


def test_apply_recovery_direction_flip_consecutive_losses_gt_1():
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True},
        },
    }
    best = (PAIR, TradeDirection.CALL, decisions[PAIR]["metrics"])
    flipped = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
        consecutive_losses=2,
    )
    assert flipped == best
    assert flipped[1] == TradeDirection.CALL
