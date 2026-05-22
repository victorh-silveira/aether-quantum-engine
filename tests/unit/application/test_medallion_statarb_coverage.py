"""Additional coverage tests for Medallion StatArb models, filters, and macro guards."""

import numpy as np

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_bridge_guards import apply_macro_confluence_guard
from src.application.services.llm.macro_snapshot_build import build_macro_snapshot
from src.application.services.llm.medallion_statarb import (
    MarketHMMClassifier,
    compute_pca_cointegration_zscores,
)
from src.domain.models.trade import TradeDirection


def test_hmm_classifier_custom_matrix():
    """Test MarketHMMClassifier when initialized with custom transition matrix."""
    custom_a = np.array([[0.95, 0.05], [0.05, 0.95]])
    hmm = MarketHMMClassifier(transition_matrix=custom_a)
    assert np.allclose(hmm.A, custom_a)


def test_pca_cointegration_zscores_short_closes():
    """Test PCA cointegration returns neutral values when one series is too short."""
    closes = {
        "A": [10.0, 10.1, 10.2],
        "B": [11.0],  # too short (< 3 elements)
    }
    res = compute_pca_cointegration_zscores(closes, ["A", "B"])
    assert res == {"A": 0.0, "B": 0.0}


def test_pca_cointegration_zscores_flat_closes():
    """Test PCA cointegration returns neutral values when covariance matrix is flat/zero."""
    closes = {
        "A": [10.0] * 15,
        "B": [10.0] * 15,
    }
    res = compute_pca_cointegration_zscores(closes, ["A", "B"], lookback=15)
    assert res == {"A": 0.0, "B": 0.0}


def test_pca_cointegration_zscores_eigen_failure(monkeypatch):
    """Test PCA cointegration returns neutral values when eigh raises an exception."""

    def mock_eigh(*args, **kwargs):
        raise ValueError("simulated eigenvalues error")

    monkeypatch.setattr(np.linalg, "eigh", mock_eigh)

    closes = {
        "A": [10.0 + 0.1 * i for i in range(15)],
        "B": [10.0 - 0.1 * i for i in range(15)],
    }
    res = compute_pca_cointegration_zscores(closes, ["A", "B"], lookback=15)
    assert res == {"A": 0.0, "B": 0.0}


def test_pca_cointegration_zscores_zero_variance_residuals():
    """Test PCA cointegration returns 0.0 when residual variance is zero (e.g. perfect correlation)."""
    closes = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    res = compute_pca_cointegration_zscores(closes, ["A", "B"], lookback=5)
    assert res["A"] == 0.0
    assert res["B"] == 0.0


def test_statarb_guard_confluence_boost_put():
    """Test that coaligned over-valued StatArb spread boosts conviction on PUT."""
    snap = MacroSnapshot(
        tag="risk_off",
        eurusd_bias="PUT",
        us_dir="down",
        eu_dir="down",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": 3.0},  # Over-valued relative to PCA spread (Z = 3.0)
        hmm_state=0,  # MEAN_REVERSION
        hmm_prob=0.9,
    )

    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.60,
        snap,
        {"statarb_z_threshold": 2.5},
        sym="OTC_GDAXI",
    )

    assert conviction > 0.60
    assert direction == TradeDirection.PUT
    assert "STATARB_INTEL boost PUT" in note
    assert execute_ok is True


def test_statarb_guard_intelligence_preserves_conflicting_call():
    snap = MacroSnapshot(
        tag="risk_off",
        eurusd_bias="PUT",
        us_dir="down",
        eu_dir="down",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": 3.0},  # Over-valued (Z = 3.0)
        hmm_state=0,  # MEAN_REVERSION
        hmm_prob=0.9,
    )

    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.60,
        snap,
        {"statarb_z_threshold": 2.5},
        sym="OTC_GDAXI",
    )

    assert direction == TradeDirection.CALL
    assert execute_ok is True


def test_macro_snapshot_build_hmm_output():
    """Test that HMM regime information is correctly printed in macro snapshot formatting."""
    m15 = {
        "OTC_SPC": [100.0, 101.0],
        "OTC_FCHI": [100.0, 101.0],
    }
    # When hmm_prob < 1.0 or hmm_state > 0, it should output HMM regime details
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        m15,
        {"min_indices_for_vote": 1},
        hmm_state=1,  # TRENDING
        hmm_prob=0.85,
    )
    assert "HMM_regime=TRENDING" in snap.macro_block
    assert "85.0%" in snap.macro_block

    snap2 = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        m15,
        {"min_indices_for_vote": 1},
        hmm_state=0,  # MEAN_REVERSION
        hmm_prob=0.92,
    )
    assert "HMM_regime=MEAN_REVERSION" in snap2.macro_block
    assert "92.0%" in snap2.macro_block
