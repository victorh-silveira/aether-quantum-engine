"""Validacao de calibracao e incerteza por blocos."""

import pytest

from indicator.validation import (
    block_bootstrap_lower,
    brier_predictions,
    expected_calibration_error,
    momentum_reference,
)


def test_brier_and_ece_measure_probability_without_future_data():
    assert brier_predictions([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    assert expected_calibration_error([0.8, 0.2], [1, 0]) == pytest.approx(0.2)
    assert expected_calibration_error([0.5, 0.5], [1, 0]) == 0


def test_block_bootstrap_is_reproducible_and_preserves_blocks():
    assert block_bootstrap_lower([0.1] * 100) == pytest.approx(0.1)
    assert block_bootstrap_lower([-0.1] * 100) == pytest.approx(-0.1)
    assert block_bootstrap_lower([0.1, -0.1] * 20) == block_bootstrap_lower([0.1, -0.1] * 20)
    with pytest.raises(ValueError, match="vazia"):
        block_bootstrap_lower([])


def test_momentum_reference_is_fixed_and_causal():
    assert momentum_reference([(0.0, 0.1), (0.0, -0.1), (0.0, 0.0)]) == [0.55, 0.45, 0.5]
