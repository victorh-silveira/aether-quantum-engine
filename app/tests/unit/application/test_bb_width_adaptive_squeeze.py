import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_ANOMALY_RATIO,
    BB_WIDTH_HARMONIC_WINDOW,
    anomalous_bb_compression,
    bb_width_buffer_snapshot,
    evaluate_bb_width_squeeze,
    harmonic_mean_bb_width,
    record_bb_width,
    reset_bb_width_buffer,
)


@pytest.fixture(autouse=True)
def _clear_bb_buffer():
    reset_bb_width_buffer()
    yield
    reset_bb_width_buffer()


def _prime_width(value: float, count: int = BB_WIDTH_HARMONIC_WINDOW) -> None:
    for _ in range(count):
        record_bb_width(value)


def test_harmonic_mean_bb_width():
    assert harmonic_mean_bb_width(history=[0.04, 0.05]) == pytest.approx(2 / (1 / 0.04 + 1 / 0.05))


def test_rdbull_natural_width_not_anomalous_after_window():
    _prime_width(0.035)
    assert anomalous_bb_compression(0.035) is False


def test_anomalous_compression_below_fifty_five_percent_of_mean():
    _prime_width(0.050)
    threshold = harmonic_mean_bb_width() * BB_WIDTH_ANOMALY_RATIO
    assert anomalous_bb_compression(0.020) is True
    assert threshold > 0.020


def test_soft_compression_above_fifty_five_percent_ratio_not_anomalous():
    _prime_width(0.068)
    assert anomalous_bb_compression(0.041) is False
    assert (0.041 / harmonic_mean_bb_width()) + 1e-12 >= BB_WIDTH_ANOMALY_RATIO


def test_evaluate_bb_width_squeeze_records_and_flags():
    _prime_width(0.050)
    compressed, mean, width = evaluate_bb_width_squeeze(0.020)
    assert compressed is True
    assert width == pytest.approx(0.020)
    assert mean < 0.050
    assert len(bb_width_buffer_snapshot()) == BB_WIDTH_HARMONIC_WINDOW


def test_anomalous_compression_false_for_missing_mean():
    assert anomalous_bb_compression(0.03) is False


def test_anomalous_compression_false_for_non_positive_width():
    assert anomalous_bb_compression(0.0) is False
    assert anomalous_bb_compression(-0.02) is False


def test_harmonic_mean_bb_width_zero_reciprocal_sum_guard():
    assert harmonic_mean_bb_width(history=[float("inf")]) == 0.0


def test_evaluate_bb_width_squeeze_none_input():
    compressed, mean, width = evaluate_bb_width_squeeze(None)
    assert compressed is False
    assert mean == 0.0
    assert width == 0.0
