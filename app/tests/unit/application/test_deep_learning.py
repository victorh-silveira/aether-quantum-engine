import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_features import (
    FEATURE_DIM,
    build_feature_row,
    build_sequence_tensor,
    calculate_rsi,
    extract_features,
    extract_sequences,
)
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier, _Chomp1d
from src.application.services.deep_learning.dl_training import train_model_online, train_model_walkforward
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    FeatureNormStats,
    MarketDirectionClassifier,
    _accuracy,
    _precompute_price_series,
    create_direction_model,
    fit_norm_stats,
    load_model_checkpoint,
    model_accuracy,
    normalize_features,
    normalize_sequences,
    predict_next_direction,
    save_model_checkpoint,
)
from src.domain.models.trade import TradeDirection


def test_model_initialization():
    model = TemporalDirectionClassifier(input_dim=FEATURE_DIM)
    x = torch.randn(2, 24, FEATURE_DIM)
    out = model(x)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_calculate_rsi():
    prices = np.linspace(10, 20, 30)
    rsi = calculate_rsi(prices, period=14)
    assert len(rsi) == 30
    assert rsi[-1] > 70.0
    rsi_short = calculate_rsi(np.array([1.0, 2.0]), period=14)
    assert len(rsi_short) == 2
    assert np.all(rsi_short == 50.0)


def test_extract_features_and_sequences():
    prices = np.sin(np.linspace(0, 10, 120)) + 10.0
    features, targets = extract_features(prices, lookback=20)
    seqs, labels, masks = extract_sequences(prices, lookback=20)
    assert features.shape[0] == targets.shape[0]
    assert seqs.shape[1:] == (20, FEATURE_DIM)
    assert len(labels) == len(masks)
    f, t = extract_features(np.array([10.0, 11.0]), lookback=20)
    assert len(f) == 0
    assert len(t) == 0


def test_build_sequence_tensor_shape():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    tensor = build_sequence_tensor(prices, 16, len(prices) - 1)
    assert tensor.shape == (16, FEATURE_DIM)


def test_train_predict_features_aligned():
    prices = np.sin(np.linspace(0, 10, 120)) + 10.0
    series = _precompute_price_series(prices)
    features, _ = extract_features(prices, lookback=20)
    last_train_idx = len(prices) - 2
    assert np.allclose(build_feature_row(series, last_train_idx), features[-1], atol=1e-5)
    infer_row = build_feature_row(series, len(prices) - 1)
    assert infer_row.shape == (FEATURE_DIM,)


def test_train_and_predict():
    prices = np.sin(np.linspace(0, 10, 120)) + 10.0
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    result = train_model_walkforward(model, prices, lookback=20, epochs=2, lr=0.001, validation_bars=15)
    assert result is not None
    loss = train_model_online(model, prices, lookback=20, epochs=2, lr=0.001, validation_bars=15)
    assert isinstance(loss, float)
    loss_short = train_model_online(model, np.sin(np.linspace(0, 10, 25)) + 10.0, lookback=20, epochs=2, lr=0.01)
    assert loss_short == 0.0
    norm = result.norm_stats if result else fit_norm_stats(np.zeros((2, 20, INPUT_DIM), dtype=np.float32))
    calibrator = CalibratorState(temperature=1.0)
    with patch.object(model, "forward", return_value=torch.tensor([[0.8]])):
        direction, prob, trade_score, raw_prob = predict_next_direction(
            model, prices, lookback=20, norm_stats=norm, val_accuracy=0.75, calibrator=calibrator
        )
        assert direction == TradeDirection.CALL
        assert trade_score > 0.5
        assert raw_prob == pytest.approx(0.8)
    with patch.object(model, "forward", return_value=torch.tensor([[0.3]])):
        direction, prob, trade_score, raw_prob = predict_next_direction(
            model, prices, lookback=20, norm_stats=norm, val_accuracy=0.75, calibrator=calibrator
        )
        assert direction == TradeDirection.PUT
        assert trade_score > 0.5
        assert raw_prob == pytest.approx(0.3)
    dir_short, prob_short, score_short, raw_short = predict_next_direction(model, np.array([10.0]), lookback=20)
    assert dir_short is None
    assert prob_short == 0.5
    assert score_short == 0.5
    assert raw_short == 1.0


def test_purged_splits():
    splits = purged_temporal_splits(200, 20, calib_ratio=0.15)
    assert splits is not None
    train_sl, val_sl, calib_sl = splits
    assert train_sl.stop < val_sl.start
    assert val_sl.stop <= calib_sl.start
    assert purged_temporal_splits(10, 5) is None


def test_accuracy_wrapper_delegates_to_model_accuracy():
    model = create_direction_model(arch="tcn")
    x = np.random.randn(3, 8, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    assert _accuracy(model, x, y) == model_accuracy(model, x, y)


def test_accuracy_empty_and_legacy_checkpoint():
    model = MarketDirectionClassifier(input_dim=INPUT_DIM)
    assert _accuracy(model, np.array([]), np.array([])) == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy_only.pth"
        torch.save(model.state_dict(), path)
        assert load_model_checkpoint(path) is None
        bad = Path(tmp) / "bad.pth"
        torch.save("invalid", bad)
        assert load_model_checkpoint(bad) is None


def test_predict_without_norm_stats():
    prices = np.sin(np.linspace(0, 10, 120)) + 10.0
    model = create_direction_model(arch="tcn")
    direction, prob, trade_score, raw_prob = predict_next_direction(
        model, prices, lookback=20, norm_stats=None, min_direction_margin=0.0
    )
    assert direction in (TradeDirection.CALL, TradeDirection.PUT)
    assert 0.0 <= prob <= 1.0
    assert 0.0 <= raw_prob <= 1.0
    assert trade_score >= 0.5 - 0.01


def test_checkpoint_save_load():
    model = create_direction_model(arch="tcn")
    stats = fit_norm_stats(np.random.randn(5, 20, INPUT_DIM).astype(np.float32))
    calibrator = CalibratorState(temperature=1.25)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sym.pth"
        save_model_checkpoint(
            path,
            model,
            stats,
            last_candle_epoch=12345,
            lookback=32,
            calibrator=calibrator,
            arch="tcn",
        )
        loaded = load_model_checkpoint(path)
        assert loaded is not None
        m2, s2, epoch, cal, lookback, val_acc, val_brier, val_ece, deploy_ok, deploy_wr = loaded
        assert epoch == 12345
        assert cal.temperature == pytest.approx(1.25)
        assert lookback == 32
        assert val_acc == 0.0
        assert val_brier == pytest.approx(1.0)
        assert val_ece == pytest.approx(1.0)
        assert np.allclose(s2.mean, stats.mean)
        assert load_model_checkpoint(Path(tmp) / "missing.pth") is None


def test_normalize_features():
    raw = np.random.randn(4, 20, INPUT_DIM).astype(np.float32)
    stats = fit_norm_stats(raw)
    normed = normalize_features(raw.reshape(-1, INPUT_DIM), stats)
    assert normed.shape[0] == raw.shape[0] * raw.shape[1]
    assert np.all(normed <= 5.0)


def test_tcn_chomp_and_2d_input():
    chomp = _Chomp1d(0)
    x = torch.randn(2, 8, 10)
    assert chomp(x).shape == x.shape
    chomp_trim = _Chomp1d(2)
    assert chomp_trim(x).shape[2] == x.shape[2] - 2
    model = TemporalDirectionClassifier(input_dim=INPUT_DIM)
    out = model(torch.randn(3, INPUT_DIM))
    assert out.shape == (3, 1)
    out3d = model(torch.randn(2, 12, INPUT_DIM))
    assert out3d.shape == (2, 1)


def test_train_with_focal_loss():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    result = train_model_walkforward(
        model,
        prices,
        lookback=18,
        epochs=10,
        lr=0.001,
        validation_bars=14,
        focal_gamma=2.0,
        early_stopping_patience=1,
    )
    assert result is not None


def test_model_accuracy_mask_and_normalize_2d():
    model = create_direction_model(arch="legacy")
    assert isinstance(model, MarketDirectionClassifier)
    x = np.random.randn(4, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert model_accuracy(model, x, y) >= 0.0
    assert model_accuracy(model, x, y, np.zeros(4, dtype=np.float32)) == 0.0
    mask_sparse = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    acc_sparse = model_accuracy(model, x, y, mask_sparse)
    assert 0.0 <= acc_sparse <= 1.0
    flat = np.random.randn(3, INPUT_DIM).astype(np.float32)
    stats = FeatureNormStats(mean=np.zeros(INPUT_DIM, dtype=np.float32), std=np.ones(INPUT_DIM, dtype=np.float32))
    assert normalize_features(flat, stats).shape == flat.shape
    assert normalize_sequences(flat, stats).shape == flat.shape


def test_load_corrupted_checkpoint(tmp_path):
    bad = tmp_path / "corrupt.pth"
    bad.write_bytes(b"not-a-checkpoint")
    assert load_model_checkpoint(bad) is None


def test_early_stopping_trigger():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    accuracies = [1.0, 0.0, 0.0, 0.0]

    def mock_acc(*args, **kwargs):
        if accuracies:
            return accuracies.pop(0)
        return 0.0

    with patch("src.application.services.deep_learning.dl_training.model_accuracy", side_effect=mock_acc):
        result = train_model_walkforward(
            model,
            prices,
            lookback=18,
            epochs=5,
            lr=0.001,
            validation_bars=14,
            early_stopping_patience=1,
        )
    assert result is not None
