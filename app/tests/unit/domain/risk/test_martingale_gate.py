from src.domain.risk.martingale_gate import (
    apply_win_to_pending_loss,
    martingale_allowed,
    martingale_block_reason,
    martingale_pending_total,
    martingale_repeat_loss_blocked,
)


def test_martingale_allowed_with_pending_loss_native():
    assert martingale_allowed(pending_loss={"R_50": 50.0}, martingale_native=True)
    assert not martingale_allowed(pending_loss={}, martingale_native=True)


def test_martingale_native_ignores_dl_metrics():
    assert martingale_allowed(
        pending_loss={"R_50": 50.0},
        martingale_native=True,
        symbol="R_75",
        order_direction="CALL",
    )


def test_martingale_block_reason_repeat_loss_only_when_enabled():
    assert (
        martingale_block_reason(
            pending_loss={"R_50": 10.0},
            martingale_native=True,
            block_repeat_loss=True,
            symbol="R_50",
            order_direction="CALL",
            last_loss_symbol="R_50",
            last_loss_direction="CALL",
        )
        == "repeat_loss"
    )
    assert (
        martingale_block_reason(
            pending_loss={"R_50": 10.0},
            martingale_native=True,
            block_repeat_loss=False,
            symbol="R_50",
            order_direction="CALL",
            last_loss_symbol="R_50",
            last_loss_direction="CALL",
        )
        is None
    )


def test_martingale_repeat_loss_blocked_pair():
    assert martingale_repeat_loss_blocked("R_75", "PUT", "R_25", "PUT")
    assert martingale_repeat_loss_blocked("R_25", "CALL", "R_75", "CALL")
    assert not martingale_repeat_loss_blocked("R_25", "CALL", "R_75", "PUT")


def test_martingale_repeat_loss_blocked_without_direction():
    assert not martingale_repeat_loss_blocked("R_50", None, "R_50", "CALL")
    assert not martingale_repeat_loss_blocked("R_50", "CALL", "R_50", None)


def test_martingale_repeat_loss_same_symbol_and_side():
    assert martingale_repeat_loss_blocked("R_50", "CALL", "R_50", "CALL")


def test_martingale_block_reason_legacy_disabled():
    assert martingale_block_reason(pending_loss={"R_50": 5.0}, martingale_native=False) == "legacy_disabled"


def test_martingale_allowed_and_apply_win():
    pending = {"A": 10.0}
    apply_win_to_pending_loss(pending, 4.0)
    assert pending["A"] == 6.0
    apply_win_to_pending_loss(pending, 20.0)
    assert pending["A"] == 0.0


def test_martingale_repeat_loss_unrelated_symbol():
    assert not martingale_repeat_loss_blocked("FOO", "CALL", "R_50", "CALL")
    assert not martingale_repeat_loss_blocked("R_50", "CALL", "FOREX", "PUT")


def test_apply_win_stops_when_profit_exhausted():
    pending = {"A": 10.0, "B": 20.0}
    apply_win_to_pending_loss(pending, 10.0)
    assert pending["A"] == 0.0
    assert pending["B"] == 20.0


def test_martingale_pending_total():
    assert martingale_pending_total({"A": 1.5, "B": 2.5}) == 4.0
