from src.application.services.execution_symbols_recovery import apply_recovery_direction_flip
from src.domain.models.trade import TradeDirection


def _best(symbol: str, direction: TradeDirection, raw_prob: float = 0.52):
    return symbol, direction, {"raw_prob": raw_prob, "dl_direction": direction.name}


def test_apply_recovery_direction_flip_keeps_dl_direction():
    best = _best("RDBEAR", TradeDirection.PUT, raw_prob=0.32)
    flipped = apply_recovery_direction_flip(
        best,
        {},
        recovery_active=True,
        last_loss_symbol="RDBEAR",
        last_loss_direction="PUT",
        flip_enabled=True,
        flip_max_conviction=0.56,
    )
    assert flipped == best
    assert flipped[1] == TradeDirection.PUT


def test_apply_recovery_direction_flip_noop_when_disabled():
    best = _best("RDBULL", TradeDirection.CALL)
    assert (
        apply_recovery_direction_flip(
            best,
            {},
            recovery_active=True,
            last_loss_symbol="RDBULL",
            last_loss_direction="CALL",
            flip_enabled=False,
        )
        == best
    )


def test_apply_recovery_direction_flip_noop_when_best_missing():
    assert apply_recovery_direction_flip(None, {}, recovery_active=True, flip_enabled=True) is None
