"""Testes unitarios para o monitor de drift de dados e conceito."""

import numpy as np

from src.domain.analytics.data_drift_monitor import DataDriftMonitor, DriftLevel


def test_data_drift_monitor_insufficient_reference():
    short_ref = np.array([0.01, -0.02, 0.005], dtype=np.float64)
    monitor = DataDriftMonitor(short_ref, num_bins=10)
    assert not monitor.is_ready

    report = monitor.evaluate(np.array([0.01, -0.01, 0.02]))
    assert report.psi == 0.0
    assert report.level == DriftLevel.NO_DRIFT
    assert report.recommendation == "insufficient_data"
    assert not report.is_actionable


def test_data_drift_monitor_insufficient_recent():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=0.01, size=1000)
    monitor = DataDriftMonitor(ref, num_bins=10)
    assert monitor.is_ready

    report = monitor.evaluate(np.array([0.01, -0.01]))
    assert report.recommendation == "insufficient_data"
    assert report.level == DriftLevel.NO_DRIFT


def test_data_drift_monitor_no_drift_identical_distribution():
    rng = np.random.default_rng(123)
    ref = rng.normal(loc=0.0, scale=0.01, size=2000)
    recent = rng.normal(loc=0.0, scale=0.01, size=500)

    monitor = DataDriftMonitor(ref, num_bins=10)
    report = monitor.evaluate(recent)

    assert report.level == DriftLevel.NO_DRIFT
    assert report.psi <= 0.10
    assert not report.is_actionable
    assert report.recommendation == "maintain_operational_regime"


def test_data_drift_monitor_moderate_drift():
    rng = np.random.default_rng(456)
    ref = rng.normal(loc=0.0, scale=0.01, size=2000)
    recent = rng.normal(loc=0.006, scale=0.012, size=500)

    monitor = DataDriftMonitor(ref, num_bins=10, psi_moderate_threshold=0.08, psi_severe_threshold=0.35)
    report = monitor.evaluate(recent)

    assert report.level == DriftLevel.MODERATE_DRIFT
    assert report.is_actionable
    assert report.recommendation == "reduce_sizing_moderate_regime_shift"


def test_data_drift_monitor_severe_drift():
    rng = np.random.default_rng(789)
    ref = rng.normal(loc=0.0, scale=0.01, size=2000)
    recent = rng.normal(loc=0.05, scale=0.05, size=500)

    monitor = DataDriftMonitor(ref, num_bins=10)
    report = monitor.evaluate(recent)

    assert report.level == DriftLevel.SEVERE_DRIFT
    assert report.psi > 0.25
    assert report.is_actionable
    assert report.recommendation == "pause_trading_severe_distribution_shift"


def test_data_drift_monitor_constant_series_moments():
    ref = np.zeros(100, dtype=np.float64)
    monitor = DataDriftMonitor(ref, num_bins=5)
    report = monitor.evaluate(np.zeros(50, dtype=np.float64))
    assert isinstance(report.kurtosis_delta, float)
    assert isinstance(report.skewness_delta, float)


def test_calculate_moments_short_data():
    from src.domain.analytics.data_drift_monitor import _calculate_moments

    skew, kurt = _calculate_moments(np.array([1.0, 2.0]))
    assert skew == 0.0
    assert kurt == 0.0
