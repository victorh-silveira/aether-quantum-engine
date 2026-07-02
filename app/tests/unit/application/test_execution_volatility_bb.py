"""Testes de volatilidade BB efetiva."""

from src.application.services.execution_volatility_bb import (
    bb_effective_width,
    squeeze_dynamic_min_edge,
    squeeze_exponential_min_edge,
    squeeze_extreme_regime,
    vol_compression_hyperbolic_edge,
)


def test_bb_effective_width_scales_with_implied_vol():
    base = bb_effective_width(bb_width=0.02, implied_vol_ratio=0.25, symbol="RDBULL", scale_enabled=True)
    assert base < 0.02
    raw = bb_effective_width(bb_width=0.02, implied_vol_ratio=0.5, symbol="RDBULL", scale_enabled=False)
    assert raw == 0.02


def test_squeeze_extreme_and_edge():
    extreme, norm = squeeze_extreme_regime(
        bb_effective=0.001,
        bb_width_history=[0.05, 0.04, 0.03],
        vol_ratio=0.8,
        implied_vol_ratio=0.5,
        symbol="RDBEAR",
    )
    assert isinstance(extreme, bool)
    assert 0.0 <= norm <= 1.0
    edge = squeeze_dynamic_min_edge(base_edge=0.04, bb_norm=norm, squeeze_slope=0.025)
    assert edge >= 0.04


def test_squeeze_exponential_edge_grows_faster_than_linear():
    bb_norm = 0.1
    linear = squeeze_dynamic_min_edge(base_edge=0.04, bb_norm=bb_norm, squeeze_slope=0.025)
    exponential = squeeze_exponential_min_edge(base_edge=0.04, bb_norm=bb_norm, squeeze_k=2.5)
    assert exponential > linear
    assert exponential > 0.04


def test_vol_compression_hyperbolic_edge_c0011_like():
    edge = vol_compression_hyperbolic_edge(base_edge=0.04, vol_ratio=0.41)
    assert edge > 0.04
    assert edge <= 0.12
    normal = vol_compression_hyperbolic_edge(base_edge=0.04, vol_ratio=0.70)
    assert normal == 0.04
