import numpy as np
import pytest

from src.application.services.deep_learning.dl_calibration import (
    apply_calibrator,
    brier_score,
    calibrate_trade_score,
    cap_calibrated_to_raw_band,
    expected_calibration_error,
    fit_calibrator,
    fit_temperature,
    logit_to_prob,
    raw_side_conviction,
    raw_to_logit,
    shrink_toward_fifty,
)
from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome, sample_weights_for_symbol
from src.application.services.deep_learning.dl_training import train_model_online, train_model_walkforward
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_calibration_helpers():
    assert raw_to_logit(0.5) == pytest.approx(0.0, abs=1e-5)
    assert logit_to_prob(0.0) == pytest.approx(0.5, abs=1e-5)
    assert shrink_toward_fifty(0.9, 0.75) == pytest.approx(0.9)
    assert shrink_toward_fifty(0.65, 0.48) > 0.56
    assert shrink_toward_fifty(0.65, 0.0) > 0.56
    cal = fit_calibrator([0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0])
    assert 0.75 <= cal.temperature <= 2.5
    assert fit_temperature([], []) == 1.0
    score = calibrate_trade_score(0.72, 0.54, cal)
    assert score >= 0.56
    assert raw_side_conviction(0.72) == pytest.approx(0.72)
    capped = cap_calibrated_to_raw_band(0.24, 0.96, 0.18)
    assert capped == pytest.approx(0.94, abs=1e-6)
    assert cap_calibrated_to_raw_band(0.55, 0.70, 0.0) == pytest.approx(0.70)
    capped_score = calibrate_trade_score(0.24, 0.55, cal, max_calibrated_raw_gap=0.18)
    assert capped_score <= cap_calibrated_to_raw_band(0.24, 1.0, 0.18) + 1e-6
    assert apply_calibrator(0.7, cal) > 0.0
    assert brier_score([0.8, 0.2], [1.0, 0.0]) < 0.5
    assert expected_calibration_error([0.8, 0.2], [1.0, 0.0]) >= 0.0


def test_outcome_weights():
    orch = type("O", (), {})()
    record_symbol_outcome(orch, "RDBULL", won=True, candle_epoch=100)
    assert hasattr(orch, "_dl_outcome_flags")
    assert sample_weights_for_symbol(orch, "RDBULL", 0) == []
    record_symbol_outcome(orch, "RDBULL", won=False, candle_epoch=101)
    record_symbol_outcome(orch, "RDBULL", won=False, candle_epoch=102)
    weights = sample_weights_for_symbol(orch, "RDBULL", 10)
    assert len(weights) == 10
    assert max(weights) > 1.0


def test_outcome_history_caps_at_eighty():
    orch = type("O", (), {})()
    for i in range(85):
        record_symbol_outcome(orch, "RDBULL", won=bool(i % 2), candle_epoch=i)
    assert len(orch._dl_outcome_flags["RDBULL"]) == 80
    assert len(orch._dl_outcome_epochs["RDBULL"]) == 80


def test_train_model_walkforward_weighted():
    prices = np.sin(np.linspace(0, 10, 120)) + 10.0
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    result = train_model_walkforward(
        model,
        prices,
        lookback=15,
        epochs=3,
        lr=0.001,
        validation_bars=12,
        sample_weights=[2.0] * 80,
        weight_decay=0.0001,
        label_smoothing=0.05,
        early_stopping_patience=2,
    )
    assert result is not None
    assert result.calibrator is not None
    assert result.val_brier <= 1.0


def test_train_model_online_returns_zero_without_samples():
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    loss = train_model_online(model, np.array([10.0, 11.0]), lookback=20, epochs=2, lr=0.01)
    assert loss == 0.0
