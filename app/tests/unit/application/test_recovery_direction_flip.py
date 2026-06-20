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


def test_apply_recovery_direction_flip_allows_flip_when_consecutive_losses_gt_1():
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
    assert flipped is not None
    assert flipped[1] == TradeDirection.PUT
    assert flipped[2].get("direction_inverted") is True


def test_apply_recovery_direction_flip_with_trend_confirmation():
    # Caso 1: Tendencia confirma o flip (opposite.name == trend_direction)
    # Direcao original CALL -> Opposite seria PUT. Tendencia PUT confirma o flip.
    decisions_confirmed = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True, "trend_direction": "PUT"},
        },
    }
    best_confirmed = (PAIR, TradeDirection.CALL, decisions_confirmed[PAIR]["metrics"])
    flipped_confirmed = apply_recovery_direction_flip(
        best_confirmed,
        decisions_confirmed,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
        flip_use_trend=True,
    )
    assert flipped_confirmed is not None
    assert flipped_confirmed[1] == TradeDirection.PUT

    # Caso 2: Tendencia nao confirma o flip (opposite.name != trend_direction)
    # Direcao original CALL -> Opposite seria PUT. Tendencia CALL nao confirma o flip.
    decisions_rejected = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True, "trend_direction": "CALL"},
        },
    }
    best_rejected = (PAIR, TradeDirection.CALL, decisions_rejected[PAIR]["metrics"])
    flipped_rejected = apply_recovery_direction_flip(
        best_rejected,
        decisions_rejected,
        recovery_active=True,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        flip_enabled=True,
        flip_use_trend=True,
    )
    assert flipped_rejected == best_rejected
    assert flipped_rejected[1] == TradeDirection.CALL


def test_apply_recovery_direction_flip_coverage_branches():
    # Caso best is None
    assert (
        apply_recovery_direction_flip(
            None,
            {},
            recovery_active=True,
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            flip_enabled=True,
        )
        is None
    )

    # Caso symbol nao esta em decisions (linha 155-156)
    decisions_missing = {}
    best = (PAIR, TradeDirection.CALL, {"raw_prob": 0.54, "trade_score": 0.54, "deploy_ok": True})
    assert (
        apply_recovery_direction_flip(
            best,
            decisions_missing,
            recovery_active=True,
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            flip_enabled=True,
        )
        == best
    )

    # Caso flipped is None (linha 158)
    decisions_no_dir = {PAIR: {"metrics": {"trade_score": 0.54, "deploy_ok": True}}}
    assert (
        apply_recovery_direction_flip(
            best,
            decisions_no_dir,
            recovery_active=True,
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            flip_enabled=True,
        )
        == best
    )
