"""Unit tests for Medallion StatArb models, filters, and macro guards."""

import numpy as np
import pytest

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_bridge_guards import apply_macro_confluence_guard
from src.application.services.llm.medallion_statarb import (
    KalmanFilter,
    MarketHMMClassifier,
    compute_pca_cointegration_zscores,
    normal_pdf,
)
from src.domain.models.trade import TradeDirection


def test_kalman_filter_constant():
    """Test standard 1D Kalman filter on a constant sequence with noise."""
    kf = KalmanFilter(q=1e-5, r=1e-3)
    # A constant stream of 10.0
    measurements = [10.0] * 20
    filtered = kf.filter_series(measurements)

    assert len(filtered) == 20
    # Should converge quickly to approximately 10.0
    assert abs(filtered[-1] - 10.0) < 0.1


def test_kalman_filter_step():
    """Test Kalman filter transitions correctly on a step change."""
    kf = KalmanFilter(q=1e-3, r=1e-2)
    series = [5.0] * 10 + [15.0] * 10
    filtered = kf.filter_series(series)

    assert filtered[0] == 5.0
    # Kalman filter should adapt and move towards 15.0
    assert filtered[-1] > 14.0


def test_normal_pdf():
    """Test the manual Gaussian probability density function implementation."""
    # Peak at mean should be higher than tail
    p_peak = normal_pdf(0.0, 0.0, 1.0)
    p_tail = normal_pdf(2.0, 0.0, 1.0)
    assert p_peak > p_tail
    assert p_peak == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))

    # Test safety against extremely small sigma
    p_small_sig = normal_pdf(0.0, 0.0, 0.0)
    assert p_small_sig > 0.0


def test_hmm_classifier_regimes():
    """Test Bayesian Hidden Markov Model state transitions under simulated returns."""
    # Initialize with default parameters
    hmm = MarketHMMClassifier(sigma_low=0.0005, sigma_high=0.005)

    # 1. Low volatility returns should keep it in Mean Reversion (State 0)
    for _ in range(10):
        state, prob = hmm.update_regime(0.0001)

    assert state == 0
    assert prob > 0.5

    # 2. A sequence of massive volatility shocks should push it to Trending (State 1)
    for _ in range(15):
        state, prob = hmm.update_regime(0.015)

    assert state == 1
    assert prob > 0.5

    # 3. Returning back to zero volatility should eventually switch back to State 0
    for _ in range(25):
        state, prob = hmm.update_regime(0.0)

    assert state == 0
    assert prob > 0.5


def test_pca_cointegration_zscores_empty():
    """Test PCA cointegration returns empty/neutral values safely when data is missing."""
    # No symbols
    res = compute_pca_cointegration_zscores({}, [])
    assert res == {}

    # Too few elements in series
    res2 = compute_pca_cointegration_zscores({"A": [10.0], "B": [10.0]}, ["A", "B"])
    assert res2 == {"A": 0.0, "B": 0.0}


def test_pca_cointegration_zscores_calculation():
    """Test PCA cointegration Z-scores identify the deviating asset correctly."""
    # Three assets that share an upward trend (giving them variance),
    # but C suddenly diverges significantly above the trend at the last period.
    closes = {
        "A": [10.0 + 0.1 * i for i in range(15)],
        "B": [10.0 + 0.1 * i for i in range(15)],
        "C": [10.0 + 0.1 * i for i in range(14)] + [13.0],  # sudden divergence upwards
    }

    zscores = compute_pca_cointegration_zscores(closes, ["A", "B", "C"], lookback=15)

    assert "A" in zscores
    assert "B" in zscores
    assert "C" in zscores

    # C should have a positive Z-score indicating it's overvalued relative to the PCA spread
    assert zscores["C"] > 1.0
    # A and B should have negative Z-scores since they stayed flat relative to C's surge
    assert zscores["A"] < 0.0
    assert zscores["B"] < 0.0


def test_statarb_guard_confluence_boost_call():
    """Test that coaligned under-valued StatArb spread boosts conviction on CALL."""
    snap = MacroSnapshot(
        tag="risk_on",
        eurusd_bias="CALL",
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": -3.0},  # Under-valued relative to PCA spread (Z = -3.0)
        hmm_state=0,  # MEAN_REVERSION
        hmm_prob=0.9,
    )

    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.60,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": False},
        sym="OTC_GDAXI",
    )

    # Conviction boosted! Normally MCS is 0.80 + 0.15*1.0 + 0.04 = 0.99
    # The dynamic calculation: (0.60 * 0.3) + (0.99 * 0.7) = 0.873
    # Boosted by +0.15 is capped at 0.99
    assert conviction > 0.90
    assert direction == TradeDirection.CALL
    assert "STATARB_BOOST CALL" in note
    assert execute_ok is True


def test_statarb_guard_confluence_block_put():
    """Test that undervalued StatArb spread blocks conflicting PUT direction."""
    snap = MacroSnapshot(
        tag="risk_on",
        eurusd_bias="CALL",
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": -3.0},  # Under-valued relative to PCA spread (Z = -3.0)
        hmm_state=0,  # MEAN_REVERSION
        hmm_prob=0.9,
    )

    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.60,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": False},
        sym="OTC_GDAXI",
    )

    assert direction is None
    assert applied is True
    assert execute_ok is False
    assert "STATARB_BLOCK conflict PUT" in note


def test_statarb_guard_confluence_trending_regime():
    """Test that during a Trending regime (HMM State 1), extreme spreads block mean-reversion."""
    snap = MacroSnapshot(
        tag="risk_on",
        eurusd_bias="CALL",
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": 3.0},  # Over-valued relative to PCA spread (Z = 3.0)
        hmm_state=1,  # TRENDING
        hmm_prob=0.9,
    )

    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.60,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": False},
        sym="OTC_GDAXI",
    )

    assert direction is None
    assert applied is True
    assert execute_ok is False
    assert "STATARB_TREND_BLOCK" in note
