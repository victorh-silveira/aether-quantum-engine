from src.domain.risk.martingale_gate import (
    apply_win_to_pending_loss,
    martingale_allowed,
    martingale_pending_total,
)


def test_martingale_allowed_with_pending_loss():
    assert martingale_allowed(pending_loss={"R_50": 50.0})
    assert not martingale_allowed(pending_loss={})


def test_martingale_allowed_and_apply_win():
    pending = {"A": 10.0}
    apply_win_to_pending_loss(pending, 4.0)
    assert pending["A"] == 6.0
    apply_win_to_pending_loss(pending, 20.0)
    assert pending["A"] == 0.0


def test_apply_win_stops_when_profit_exhausted():
    pending = {"A": 10.0, "B": 20.0}
    apply_win_to_pending_loss(pending, 10.0)
    assert pending["A"] == 0.0
    assert pending["B"] == 20.0


def test_martingale_pending_total():
    assert martingale_pending_total({"A": 1.5, "B": 2.5}) == 4.0
