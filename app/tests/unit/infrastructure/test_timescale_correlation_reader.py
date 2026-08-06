import pytest

from src.infrastructure.market.timescale_correlation_reader import compute_correlation_matrix


def test_compute_correlation_matrix_identity_on_single_symbol():
    closes = {"OTC_SPC": [100.0, 101.0, 102.0, 101.5, 103.0]}
    matrix = compute_correlation_matrix(closes)
    assert matrix[("OTC_SPC", "OTC_SPC")] == pytest.approx(1.0)


def test_compute_correlation_matrix_two_symbols():
    closes = {
        "OTC_SPC": [100.0, 101.0, 102.0, 103.0, 104.0],
        "R_50": [200.0, 202.0, 204.0, 206.0, 208.0],
    }
    matrix = compute_correlation_matrix(closes)
    corr = matrix[("OTC_SPC", "OTC_SPC")]
    assert corr == pytest.approx(1.0, abs=0.01)
