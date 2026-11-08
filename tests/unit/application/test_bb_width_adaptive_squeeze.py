import json

import pytest

from aether_paths import repo_path
from src.application.services.bb_width_adaptive_squeeze import (
    anomalous_bb_compression,
    bb_width_buffer_snapshot,
    evaluate_bb_width_squeeze,
    harmonic_mean_bb_width,
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


@pytest.fixture(autouse=True)
def _clear_bb_buffer():
    reset_bb_width_buffer()
    yield
    reset_bb_width_buffer()


def _anomaly_ratio() -> float:
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    return float(full["orchestrator"]["execution"]["bb_width_adaptive_squeeze"]["anomaly_ratio"])


def _harmonic_window() -> int:
    return int(load_indicator_config_from_settings()["windows"]["bb_width_harmonic_window"])


def _prime_width(value: float, count: int | None = None) -> None:
    for _ in range(count if count is not None else _harmonic_window()):
        record_bb_width(value)


def test_harmonic_mean_bb_width():
    assert harmonic_mean_bb_width(history=[0.04, 0.05]) == pytest.approx(2 / (1 / 0.04 + 1 / 0.05))


def test_anchor_natural_width_not_anomalous_after_window():
    _prime_width(0.035)
    assert anomalous_bb_compression(0.035, anomaly_ratio=_anomaly_ratio()) is False


def test_anomalous_compression_below_configured_ratio_of_mean():
    _prime_width(0.050)
    ratio = _anomaly_ratio()
    threshold = harmonic_mean_bb_width() * ratio
    assert anomalous_bb_compression(0.015, anomaly_ratio=ratio) is True
    assert threshold > 0.015


def test_soft_compression_above_configured_ratio_not_anomalous():
    _prime_width(0.068)
    ratio = _anomaly_ratio()
    assert anomalous_bb_compression(0.041, anomaly_ratio=ratio) is False
    assert (0.041 / harmonic_mean_bb_width()) + 1e-12 >= ratio


def test_evaluate_bb_width_squeeze_records_and_flags():
    _prime_width(0.050)
    ratio = _anomaly_ratio()
    compressed, mean, width = evaluate_bb_width_squeeze(0.015, anomaly_ratio=ratio)
    assert compressed is True
    assert width == pytest.approx(0.015)
    assert mean < 0.050
    assert len(bb_width_buffer_snapshot()) == _harmonic_window()


def test_anomalous_compression_false_for_missing_mean():
    assert anomalous_bb_compression(0.03, anomaly_ratio=_anomaly_ratio()) is False


def test_anomalous_compression_false_for_non_positive_width():
    ratio = _anomaly_ratio()
    assert anomalous_bb_compression(0.0, anomaly_ratio=ratio) is False
    assert anomalous_bb_compression(-0.02, anomaly_ratio=ratio) is False


def test_harmonic_mean_bb_width_zero_reciprocal_sum_guard():
    assert harmonic_mean_bb_width(history=[float("inf")]) == 0.0


def test_evaluate_bb_width_squeeze_none_input():
    compressed, mean, width = evaluate_bb_width_squeeze(None, anomaly_ratio=_anomaly_ratio())
    assert compressed is False
    assert mean == 0.0
    assert width == 0.0
