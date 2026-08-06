import numpy as np
import pytest

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator,
    apply_calibrator_stable,
    brier_score,
    calibrate_conviction,
    calibrate_trade_score,
    calibrator_from_dict,
    calibrator_to_dict,
    cap_calibrated_to_raw_band,
    expected_calibration_error,
    logit_to_prob,
    raw_side_conviction,
    raw_to_logit,
    shrink_toward_fifty,
)
from src.application.services.deep_learning.dl_calibration_fit import (
    fit_calibrator,
    fit_platt,
    fit_platt_logistic,
    fit_temperature,
)
from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome, sample_weights_for_symbol
from src.application.services.deep_learning.dl_splits import purged_temporal_splits, splits_valid
from src.application.services.deep_learning.dl_training import train_model_online, train_model_walkforward
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_apply_calibrator_stable_avoids_isotonic_ood_snap():
    cal = CalibratorState(
        method="isotonic",
        isotonic_x=(0.5019, 0.52, 0.53),
        isotonic_y=(0.4925, 0.55, 1.0),
    )
    raw = 0.486
    snapped = apply_calibrator(raw, cal)
    stable = apply_calibrator_stable(raw, cal)
    assert snapped == pytest.approx(0.4925)
    assert stable == pytest.approx(raw)


def test_calibration_helpers():
    assert raw_to_logit(0.5) == pytest.approx(0.0, abs=1e-5)
    assert logit_to_prob(0.0) == pytest.approx(0.5, abs=1e-5)
    assert shrink_toward_fifty(0.9, 0.75) == pytest.approx(0.9)
    assert shrink_toward_fifty(0.65, 0.48) > 0.56
    assert shrink_toward_fifty(0.65, 0.0) > 0.56
    cal = fit_calibrator([0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0])
    assert 0.75 <= cal.temperature <= 2.5 or cal.method == "isotonic"
    assert fit_temperature([], []) == 1.0
    a_val, b_val = fit_platt_logistic([0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0])
    assert isinstance(a_val, float)
    assert isinstance(b_val, float)
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
    payload = calibrator_to_dict(cal)
    restored = calibrator_from_dict(payload)
    assert restored.method == cal.method


def test_outcome_weights():
    orch = type("O", (), {})()
    record_symbol_outcome(orch, "OTC_SPC", won=True, candle_epoch=100)
    assert hasattr(orch, "_dl_outcome_flags")
    assert sample_weights_for_symbol(orch, "OTC_SPC", 0) == []
    record_symbol_outcome(orch, "OTC_SPC", won=False, candle_epoch=101)
    record_symbol_outcome(orch, "OTC_SPC", won=False, candle_epoch=102)
    weights = sample_weights_for_symbol(orch, "OTC_SPC", 10)
    assert len(weights) == 10
    assert max(weights) > 1.0


def test_outcome_history_caps_at_eighty():
    orch = type("O", (), {})()
    for i in range(85):
        record_symbol_outcome(orch, "OTC_SPC", won=bool(i % 2), candle_epoch=i)
    assert len(orch._dl_outcome_flags["OTC_SPC"]) == 80
    assert len(orch._dl_outcome_epochs["OTC_SPC"]) == 80


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
    )
    assert result is not None
    assert result.calibrator is not None
    assert result.val_brier <= 1.0


def test_train_model_online_returns_zero_without_samples():
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    loss = train_model_online(model, np.array([10.0, 11.0]), lookback=20, epochs=2, lr=0.01)
    assert loss == 0.0


def test_calibrate_conviction_legacy():
    assert calibrate_conviction(0.8, 0.6, 1.0) > 0.5
    assert calibrator_from_dict(None).temperature == 1.0
    assert expected_calibration_error([], []) == 1.0
    assert brier_score([], []) == 1.0
    assert fit_platt([], []) == (1.0, 0.0)


def test_outcome_weights_dampen_after_win_streak():
    orch = type("O", (), {})()
    for _ in range(6):
        record_symbol_outcome(orch, "OTC_SPC", won=True)
    weights = sample_weights_for_symbol(orch, "OTC_SPC", 12)
    assert min(weights[-4:]) < 1.0


def test_sample_weights_boost_labels_after_loss_direction():
    orch = type("O", (), {"_last_loss_direction": "CALL"})()
    orch._dl_outcome_flags = {"OTC_SPC": [False]}
    targets = [1.0, 0.0, 1.0]
    weights = sample_weights_for_symbol(orch, "OTC_SPC", 3, targets=targets)
    assert weights[1] > weights[0]
    orch_put = type("O", (), {"_last_loss_direction": "PUT"})()
    orch_put._dl_outcome_flags = {"OTC_SPC": [False]}
    weights_put = sample_weights_for_symbol(orch_put, "OTC_SPC", 2, targets=[0.0, 1.0])
    assert weights_put[1] > weights_put[0]


def test_purged_splits_edge_cases():
    assert purged_temporal_splits(15, 5) is None
    splits = purged_temporal_splits(500, 30, calib_ratio=0.1)
    assert splits is not None


def test_purged_splits_invalid_holdout_returns_none():
    assert purged_temporal_splits(26, 10) is not None
    assert purged_temporal_splits(30, 28, calib_ratio=0.5) is None
    assert splits_valid(10, 12, 20, 20) is False


def test_purged_splits_rejects_invalid_slice_ranges(monkeypatch):
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_splits.splits_valid",
        lambda *_args: False,
    )
    assert purged_temporal_splits(120, 20) is None


def test_fit_calibrator_identity_method():
    cal = fit_calibrator([0.9, 0.1], [1.0, 0.0], calibration_cfg={"method": "identity"})
    assert cal.method == "identity"
    assert cal.temperature == 1.0


def test_guard_sharpness_keeps_better_of_collapsed_pair(monkeypatch):
    from src.application.services.deep_learning import dl_calibration_fit as fit_mod
    from src.application.services.deep_learning.dl_calibration import CalibratorState

    preferred = CalibratorState(method="platt", platt_a=0.1, platt_b=0.0)
    scores = {
        id(preferred): (0.5, 0.2, 0.005),
        "identity": (0.5, 0.2, 0.001),
    }

    def fake_score(cal, probs, labels):
        if cal.method == "identity":
            return scores["identity"]
        return scores[id(preferred)]

    monkeypatch.setattr(fit_mod, "_candidate_score", fake_score)
    out = fit_mod._guard_sharpness(preferred, [0.9, 0.1], [1.0, 0.0], min_sharpness=0.03)
    assert out is preferred
