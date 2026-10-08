"""Invariantes de compute_loss_clf_p_eff (shrink young + piso 0.55)."""

from hypothesis import (
    given,
    settings,
    strategies as st,
)

from src.application.services.loss_classifier_gate_support import compute_loss_clf_p_eff
from src.domain.models.trade import TradeDirection


_TRUST_N = 32
_SHRINK = 0.35
_HARD_FLOOR = 0.55
_YOUNG_FLOOR = 0.55
_PROP = settings(max_examples=40, deadline=200)


@_PROP
@given(
    p_loss=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    n_train=st.integers(min_value=0, max_value=128),
    tcn_ref=st.sampled_from((TradeDirection.CALL, TradeDirection.PUT)),
    tape=st.one_of(st.none(), st.sampled_from(("CALL", "PUT"))),
)
def test_p_eff_young_shrink_and_floor_ssot(p_loss: float, n_train: int, tcn_ref: TradeDirection, tape: str | None):
    p_eff, young, discord, floor = compute_loss_clf_p_eff(
        p_loss,
        n_train=n_train,
        flip_trust_n=_TRUST_N,
        flip_young_shrink=_SHRINK,
        hard_floor=_HARD_FLOOR,
        flip_young_p_eff_floor=_YOUNG_FLOOR,
        tcn_ref=tcn_ref,
        tape=tape,
    )
    expect_young = n_train < _TRUST_N
    expect_p_eff = 0.5 + (p_loss - 0.5) * _SHRINK if expect_young else p_loss
    expect_discord = bool(tape) and tape != tcn_ref.name
    assert young is expect_young
    assert p_eff == expect_p_eff
    assert floor == _YOUNG_FLOOR if expect_young else _HARD_FLOOR
    assert discord is expect_discord
    assert floor == 0.55
