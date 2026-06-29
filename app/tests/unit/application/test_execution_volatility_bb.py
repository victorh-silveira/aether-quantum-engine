"""Testes de volatilidade BB efetiva."""

from src.application.services.execution_volatility_bb import (
    bb_effective_width,
    squeeze_dynamic_min_edge,
    squeeze_extreme_regime,
)


def test_bb_effective_width_scales_with_implied_vol():
    base = bb_effective_width(bb_width=0.02, implied_vol_ratio=0.25, symbol="R_50", scale_enabled=True)
    assert base < 0.02
    raw = bb_effective_width(bb_width=0.02, implied_vol_ratio=0.5, symbol="R_50", scale_enabled=False)
    assert raw == 0.02


def test_squeeze_extreme_and_edge():
    extreme, norm = squeeze_extreme_regime(
        bb_effective=0.001,
        bb_width_history=[0.05, 0.04, 0.03],
        vol_ratio=0.8,
        implied_vol_ratio=0.5,
        symbol="R_75",
    )
    assert isinstance(extreme, bool)
    assert 0.0 <= norm <= 1.0
    edge = squeeze_dynamic_min_edge(base_edge=0.04, bb_norm=norm, squeeze_slope=0.025)
    assert edge >= 0.04
