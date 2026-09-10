"""Invariantes Kelly: breakeven, clamp de piso e f-star positivo."""

from hypothesis import (
    given,
    settings,
    strategies as st,
)

from src.domain.risk.kelly_p_align import (
    ensure_kelly_edge_p,
    kelly_breakeven_p,
    resolve_kelly_p_floor,
)


_PROP = settings(max_examples=40, deadline=200)


@_PROP
@given(payout=st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False))
def test_kelly_breakeven_p_is_one_over_one_plus_b(payout: float):
    be = kelly_breakeven_p(payout)
    assert be == 1.0 / (1.0 + payout)


@_PROP
@given(
    raw=st.one_of(
        st.none(),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
)
def test_resolve_kelly_p_floor_clamped_interval(raw: float | None):
    cfg = None if raw is None else {"kelly_p_floor": raw}
    floor = resolve_kelly_p_floor(cfg)
    assert 0.51 <= floor <= 0.65


@_PROP
@given(
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    payout=st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False),
    raw_floor=st.floats(min_value=0.40, max_value=0.80, allow_nan=False, allow_infinity=False),
)
def test_ensure_kelly_edge_p_beats_breakeven(p: float, payout: float, raw_floor: float):
    cfg = {"kelly_p_floor": raw_floor}
    out = ensure_kelly_edge_p(p, payout, cfg)
    floor = resolve_kelly_p_floor(cfg)
    be = kelly_breakeven_p(payout)
    assert out >= max(floor, be + 1e-4)
    kelly_f = (payout * out - (1.0 - out)) / payout
    assert kelly_f > 0.0
