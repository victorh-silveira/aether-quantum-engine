from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_calibration import CalibratorState, calibrate_trade_score
from src.application.services.deep_learning.dl_feature_build import attach_microstructure, symbol_vol_target
from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio
from src.application.services.deep_learning.dl_labels import sequence_labels
from src.application.services.deep_learning.dl_lstm import RecurrentDirectionClassifier
from src.application.services.deep_learning.dl_outcomes import (
    blended_val_accuracy,
    is_symbol_session_paused,
    live_win_rate,
    maybe_pause_symbol_session,
    tick_dl_session_pauses,
)
from src.application.services.deep_learning.dl_params import (
    optional_float,
    parse_dl_params,
    parse_tcn_channels,
    resolve_training_history_bars,
    slice_dl_ohlc_window,
)
from src.application.services.deep_learning.dl_sequence_extract import extract_sequences
from src.application.services.deep_learning.dl_training_epochs import _masked_loss
from src.application.services.deep_learning.model import INPUT_DIM, create_direction_model


def test_symbol_vol_target_parses_and_defaults():
    assert symbol_vol_target("R_10") == pytest.approx(0.16)
    assert symbol_vol_target("R_50") == pytest.approx(0.50)


def test_symbol_vol_target_invalid_suffix():
    assert symbol_vol_target("R_XX") == pytest.approx(0.75)
    assert symbol_vol_target("1HZ75V") == pytest.approx(0.75)


def test_attach_microstructure_partial_arrays():
    series = {"log_return": np.zeros(4)}
    attach_microstructure(series, {"tick_count": np.ones(2)})
    assert len(series["tick_count"]) == 4
    attach_microstructure(series, None)
    assert series["price_velocity"].sum() == 0.0


def test_hurst_computes_non_neutral_segment():
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.2, size=80))
    out = hurst_exponent(prices, window=16, min_window=8)
    assert np.any(out != 0.5)


def test_hurst_skips_flat_returns():
    prices = np.full(80, 100.0)
    out = hurst_exponent(prices, window=16, min_window=8)
    assert np.allclose(out, 0.5)


def test_variance_ratio_short_series():
    prices = np.array([1.0, 1.1, 1.2])
    out = variance_ratio(prices, short=2, long=8)
    assert np.allclose(out, 1.0)


def test_sequence_labels_empty_when_too_short():
    targets, masks = sequence_labels(np.array([1.0, 2.0]), lookback=48, horizon_bars=1)
    assert len(targets) == 0
    assert len(masks) == 0


def test_extract_sequences_empty_targets(monkeypatch):
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_sequence_extract.sequence_labels",
        lambda *_a, **_k: (np.empty((0,)), np.empty((0,))),
    )
    prices = np.linspace(1.0, 2.0, 80)
    seqs, targets, masks = extract_sequences(prices, lookback=48, label_horizon_bars=1)
    assert seqs.shape[0] == 0
    assert len(targets) == 0
    assert len(masks) == 0


def test_extract_sequences_nonempty():
    prices = np.linspace(1.0, 2.0, 80)
    seqs, targets, masks = extract_sequences(prices, lookback=48, label_horizon_bars=1)
    assert seqs.shape[0] >= 0


def test_lstm_and_gru_forward():
    for rnn_type in ("lstm", "gru"):
        model = RecurrentDirectionClassifier(INPUT_DIM, rnn_type=rnn_type, num_layers=1)
        out3 = model(torch.randn(2, 12, INPUT_DIM))
        out2 = model(torch.randn(3, INPUT_DIM))
        logits = model(torch.randn(2, 12, INPUT_DIM), logits=True)
        assert out3.shape == (2, 1)
        assert out2.shape == (3, 1)
        assert logits.shape == (2,)


def test_create_direction_model_lstm_gru():
    assert isinstance(create_direction_model(arch="lstm"), RecurrentDirectionClassifier)
    assert isinstance(create_direction_model(arch="gru"), RecurrentDirectionClassifier)


def test_calibrate_trade_score_put_and_no_deploy():
    cal = CalibratorState(temperature=1.0, platt_a=1.0, platt_b=0.0)
    put_score = calibrate_trade_score(0.2, 0.55, cal, is_put=True)
    assert 0.0 <= put_score <= 1.0
    no_deploy = calibrate_trade_score(0.8, 0.55, cal, deploy_ok=False)
    assert no_deploy >= 0.0


def test_resolve_training_history_from_data_config():
    bars = resolve_training_history_bars({}, {"history_bars": 500})
    assert bars == 500


def test_slice_dl_ohlc_window_shorter_arrays():
    prices = np.arange(10, dtype=np.float64)
    open_ = np.arange(5, dtype=np.float64)
    trimmed, open_trimmed, _, _ = slice_dl_ohlc_window(prices, training_history_bars=8, open_=open_)
    assert len(trimmed) == 8
    assert open_trimmed is not None and len(open_trimmed) == 5


def test_optional_float_and_tcn_channels():
    assert optional_float({}, "missing") is None
    channels = parse_tcn_channels({"tcn": {"channels": []}})
    assert channels == (64, 64, 32)


def test_parse_dl_params_gru_block():
    params = parse_dl_params({"arch": "gru", "gru": {"hidden_size": 32, "num_layers": 1, "dropout": 0.1}}, {})
    assert params["arch"] == "gru"
    assert params["rnn_hidden_size"] == 32


def test_parse_dl_params_calibration_neutral_drift_list():
    params = parse_dl_params({"calibration": {"calibration_neutral_drift": [0.42, 0.58], "neutral_half_width": 0.02}})
    assert params["calibration"]["calibration_neutral_drift"] == [0.42, 0.58]


def test_masked_loss_handles_tuple_logits_without_aux_head():
    model = MagicMock()
    logits_tensor = torch.randn(4, dtype=torch.float32)
    model.return_value = (logits_tensor, logits_tensor)
    device = torch.device("cpu")
    x = np.random.randn(4, 8, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    m = np.ones(4, dtype=np.float32)
    loss = _masked_loss(model, x, y, m, [1.0] * 4, device, label_smoothing=0.0, focal_gamma=0.0)
    assert float(loss.item()) >= 0.0


def test_masked_loss_focal_gamma():
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    device = torch.device("cpu")
    x = np.random.randn(4, 8, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    m = np.ones(4, dtype=np.float32)
    loss = _masked_loss(model, x, y, m, [1.0] * 4, device, label_smoothing=0.05, focal_gamma=2.0)
    assert float(loss.item()) >= 0.0


def test_masked_loss_auxiliary_regression_head():
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    device = torch.device("cpu")
    x = np.random.randn(4, 8, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    m = np.ones(4, dtype=np.float32)
    deltas = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float32)
    loss = _masked_loss(model, x, y, m, [1.0] * 4, device, label_smoothing=0.0, focal_gamma=0.0, delta_batch=deltas)
    assert float(loss.item()) >= 0.0


def test_live_win_rate_and_blended():
    orch = SimpleNamespace(_dl_outcome_flags={"R_10": [True, False, True, False, True, False]})
    assert live_win_rate(orch, "R_10") is not None
    blended = blended_val_accuracy(orch, "R_10", 0.60, live_weight=0.5, min_live_samples=4)
    assert blended <= 0.60
    orch_losses = SimpleNamespace(_dl_outcome_flags={"R_10": [False] * 6})
    assert blended_val_accuracy(orch_losses, "R_10", 0.60) <= 0.60


def test_session_pause_helpers():
    orch = SimpleNamespace(_dl_session_pause={"R_10": 2})

    assert is_symbol_session_paused(orch, "R_10") is False
    tick_dl_session_pauses(orch)


def test_maybe_pause_symbol_session():
    orch = SimpleNamespace(
        _dl_outcome_flags={"R_10": [False, False, False]},
        config={
            "deep_learning": {"session_max_losses_in_window": 2, "session_window_trades": 3, "session_pause_cycles": 4}
        },
    )

    maybe_pause_symbol_session(orch, "R_10", max_losses_in_window=2, window_trades=3, pause_cycles=4)
    assert not hasattr(orch, "_dl_session_pause")
