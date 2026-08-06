import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pytest

from scripts.operations.train_meta_classifier import (
    build_training_summary,
    target_variance,
    validate_target_variance,
)
from scripts.operations.train_meta_data import (
    META_TRAIN_MAX_BARS,
    OhlcBundle,
    resolve_meta_train_bars,
)
from scripts.operations.train_meta_optuna import (
    LGBM_QUIET_PARAMS,
    LGBM_REGRESSION_OBJECTIVE,
    META_EXPORT_MAX_MAE_GAP,
    META_EXPORT_MIN_ZSCORE,
    OPTUNA_NEGATIVE_EDGE_PENALTY,
    _hygiene_for_bundle,
    assert_export_mae_gap,
    assert_export_zscore_floor,
    build_paired_training_dataset,
    configure_meta_train_logging,
    payoff_zscore_mean,
    train_lgbm_candidate,
)
from scripts.operations.train_meta_vector import (
    INNER_JOIN_MIN_SAMPLE_RATIO,
    TCN_CALL_PROXY_THRESHOLD,
    TCN_PUT_PROXY_THRESHOLD,
    _continuous_payoff_target,
    teacher_sample_weights,
)
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import meta_classifier_column_names


GRAY_KEEP_FLOOR = 96


def _synthetic_bundle(symbol: str = "OTC_SPC", *, n: int = 280, phase: float = 0.0) -> OhlcBundle:
    flat_head = np.full(120, 100.0)
    t = np.linspace(0, 14 * np.pi, n - 200)
    wiggle = 100.0 + 2.5 * np.sin(t + phase) + 0.2 * np.cos(3 * t)
    flat_tail = np.full(80, float(wiggle[-1]))
    closes = np.concatenate([flat_head, wiggle, flat_tail])
    open_ = closes - 0.05
    high = closes + 0.12
    low = closes - 0.12
    epochs = (np.arange(n, dtype=np.int64) + 1_700_000_000) * 60
    return OhlcBundle(
        symbol=symbol,
        granularity=60,
        closes=closes.astype(np.float64),
        open_=open_.astype(np.float64),
        high=high.astype(np.float64),
        low=low.astype(np.float64),
        epochs=epochs,
        source="test",
    )


def _decisive_teacher(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n)
    return np.where(bits == 1, 0.62, 0.38).astype(np.float32)


def test_build_paired_training_dataset_accepts_fetch_below_history():
    bundle = _synthetic_bundle(n=999)
    frame, y, proxy, _, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=5000,
        teacher_probs={"OTC_SPC": _decisive_teacher(len(bundle.closes))},
    )
    assert len(frame) >= GRAY_KEEP_FLOOR
    assert hygiene["n_kept"] == len(frame) == len(y) == len(proxy)
    assert hygiene["label_mode"] == 1
    assert hygiene["n_dropped_gray"] == 0
    validate_target_variance(y)


def test_build_paired_training_dataset_rejects_insufficient_history():
    closes = np.linspace(100.0, 101.0, 40)
    bundle = OhlcBundle(
        symbol="OTC_SPC",
        granularity=60,
        closes=closes,
        open_=closes - 0.01,
        high=closes + 0.02,
        low=closes - 0.02,
        epochs=(np.arange(40, dtype=np.int64) + 1_700_000_000) * 60,
        source="test",
    )
    with pytest.raises(RuntimeError, match="Historico insuficiente"):
        build_paired_training_dataset([bundle], micro_granularity=120, fetch_count=5000)


def test_build_paired_training_dataset_rejects_empty_bundles():
    with pytest.raises(RuntimeError, match="ao menos um bundle"):
        build_paired_training_dataset([], micro_granularity=120, fetch_count=280)


def test_build_paired_training_dataset_single_symbol_shape():
    bundle = _synthetic_bundle(n=5000)
    frame, y, proxy, _, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=5000,
        teacher_probs={"OTC_SPC": _decisive_teacher(len(bundle.closes), seed=3)},
    )
    assert len(frame) >= int(5000 * INNER_JOIN_MIN_SAMPLE_RATIO) - 40
    assert frame.shape[1] == META_FEATURE_DIM
    assert np.allclose(frame["cross_symbol_prob_delta"].to_numpy(), 0.0)
    assert hygiene["n_dropped_gray"] == 0
    assert hygiene["label_mode"] == 1
    validate_target_variance(y)


def test_build_paired_training_dataset_uses_forward_z_labels():
    n = 1200
    bundle = _synthetic_bundle(n=n)
    teacher = np.full(n, 0.50, dtype=np.float32)
    frame, y, proxy, pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=1200,
        teacher_probs={"OTC_SPC": teacher},
    )
    assert hygiene["label_mode"] in {1, 2}
    assert hygiene["n_dropped_gray"] == 0
    assert hygiene["n_kept"] == len(frame) == len(y) == len(proxy) == len(pnl)
    assert float(np.std(y)) > 0.0
    validate_target_variance(y, hygiene=hygiene)


def test_build_paired_flat_closes_uses_payoff_or_raises():
    n = 400
    closes = np.full(n, 100.0, dtype=np.float64)
    bundle = OhlcBundle(
        symbol="OTC_SPC",
        granularity=120,
        closes=closes,
        open_=closes.copy(),
        high=closes + 0.01,
        low=closes - 0.01,
        epochs=(np.arange(n, dtype=np.int64) + 1_700_000_000) * 120,
        source="test",
    )
    teacher = _decisive_teacher(n)
    frame, y, proxy, pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=400,
        teacher_probs={"OTC_SPC": teacher},
    )
    assert hygiene["label_mode"] == 2
    assert hygiene["close_nunique"] < 8 or float(hygiene["forward_var"]) <= 1e-12
    if float(np.var(y)) <= 1e-12:
        with pytest.raises(ValueError, match="variancia nula"):
            validate_target_variance(y, hygiene=hygiene)
    else:
        validate_target_variance(y, hygiene=hygiene)


def test_teacher_sample_weights_clip_confidence():
    proxy = np.array([0.38, 0.50, 0.62, 0.90], dtype=np.float32)
    weights = teacher_sample_weights(proxy)
    assert weights.shape == proxy.shape
    assert float(weights[0]) == pytest.approx(0.24)
    assert float(weights[1]) == pytest.approx(0.1)
    assert float(weights[2]) == pytest.approx(0.24)
    assert float(weights[3]) == pytest.approx(0.8)


def test_resolve_meta_train_bars_defaults_and_caps():
    assert resolve_meta_train_bars(1024) == 1024
    assert resolve_meta_train_bars(8000) == META_TRAIN_MAX_BARS
    assert resolve_meta_train_bars(10) >= 96


def test_validate_target_variance_rejects_flat_target():
    with pytest.raises(ValueError, match="variancia nula"):
        validate_target_variance(np.zeros(120, dtype=np.float32))
    with pytest.raises(ValueError, match="variancia nula"):
        validate_target_variance(np.full(8, 0.25, dtype=np.float32))


def test_validate_target_variance_accepts_dispersed_target():
    validate_target_variance(np.array([0.1, -0.2, 0.3, -0.1, 0.05], dtype=np.float32))


def test_target_variance_returns_float_dispersion():
    y = np.array([0.1, -0.2, 0.3, -0.1], dtype=np.float32)
    assert target_variance(y) == pytest.approx(float(np.var(y)))


def test_hygiene_for_bundle_preserves_mixed_types():
    out = _hygiene_for_bundle(
        {
            "n_kept": 100,
            "forward_var": 0.0125,
            "data_source": "timescale",
            "label_mode": np.int64(2),
            "z_collapse_pct": np.float64(1.5),
        }
    )
    assert out["n_kept"] == 100
    assert out["forward_var"] == pytest.approx(0.0125)
    assert out["data_source"] == "timescale"
    assert out["label_mode"] == 2
    assert out["z_collapse_pct"] == pytest.approx(1.5)


def test_build_training_summary_includes_continuous_telemetry():
    bundles = [_synthetic_bundle()]
    teacher = _decisive_teacher(len(bundles[0].closes))
    frame, y, _, _, _ = build_paired_training_dataset(
        bundles,
        micro_granularity=120,
        fetch_count=280,
        teacher_probs={"OTC_SPC": teacher},
    )
    summary = build_training_summary(
        frame=frame,
        y=y,
        train_mae=0.08,
        val_mae=0.11,
        bundle_meta={
            "feature_dim": len(meta_classifier_column_names()),
            "model_type": "regressor",
            "oos_payoff_zscore_mean": 0.31,
            "oos_information_ratio": 1.2,
            "oos_information_ratio_unit": 0.04,
            "n_val": 100,
            "optuna_objective_metric": "payoff_zscore",
            "n_dropped_gray": 12,
            "n_kept": len(frame),
        },
        output_path=Path("meta_lgbm.pkl"),
        symbols=["OTC_SPC"],
        bundles=bundles,
    )
    assert "cross_symbol_prob_delta_mean" in summary
    assert summary["train_mae"] == pytest.approx(0.08)
    assert summary["best_val_mae"] == pytest.approx(0.11)
    assert summary["target_variance"] == pytest.approx(target_variance(y))
    assert summary["model_type"] == "regressor"
    assert summary["oos_payoff_zscore_mean"] == pytest.approx(0.31)
    assert "class_balance_ratio" not in summary


def test_assert_export_zscore_floor_blocks_weak_models():
    with pytest.raises(RuntimeError, match="Export meta bloqueado"):
        assert_export_zscore_floor(
            {"oos_payoff_zscore_mean": 0.020, "oos_information_ratio": 0.50},
            floor=META_EXPORT_MIN_ZSCORE,
        )
    assert_export_zscore_floor(
        {"oos_payoff_zscore_mean": 0.048271, "oos_information_ratio": 1.316662},
        floor=META_EXPORT_MIN_ZSCORE,
    )
    assert_export_zscore_floor(
        {"oos_payoff_zscore_mean": 0.061, "oos_information_ratio": 0.80},
        floor=META_EXPORT_MIN_ZSCORE,
    )
    assert pytest.approx(0.04) == META_EXPORT_MIN_ZSCORE


def test_assert_export_mae_gap_blocks_overfit():
    assert_export_mae_gap(1.0, 1.85, max_gap=META_EXPORT_MAX_MAE_GAP)
    with pytest.raises(RuntimeError, match="val_mae/train_mae"):
        assert_export_mae_gap(1.0, 2.10, max_gap=META_EXPORT_MAX_MAE_GAP)
    assert pytest.approx(2.0) == META_EXPORT_MAX_MAE_GAP


def test_configure_meta_train_logging_silences_lightgbm_and_optuna():
    configure_meta_train_logging()
    assert logging.getLogger("lightgbm").level == logging.ERROR
    assert logging.getLogger("asyncio").level == logging.CRITICAL
    assert optuna.logging.get_verbosity() == optuna.logging.WARNING


def test_lgbm_quiet_params_include_verbose_and_warnings():
    assert LGBM_QUIET_PARAMS["verbose"] == -1
    assert LGBM_QUIET_PARAMS["warnings"] is False
    assert LGBM_REGRESSION_OBJECTIVE == "huber"


def test_negative_edge_penalty_matches_overfit_floor():
    assert pytest.approx(-1.0) == OPTUNA_NEGATIVE_EDGE_PENALTY
    assert payoff_zscore_mean(np.array([1.0, -1.0]), np.array([-0.5, 0.5])) <= 0.0


def test_build_paired_training_dataset_has_continuous_target_and_named_columns():
    bundle = _synthetic_bundle()
    frame, y, proxy, pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=280,
        teacher_probs={"OTC_SPC": _decisive_teacher(len(bundle.closes), seed=11)},
    )
    columns = meta_classifier_column_names()
    assert list(frame.columns) == columns
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[1] == META_FEATURE_DIM == len(columns) == 43
    assert len(frame) == len(y) == len(proxy) == len(pnl) == hygiene["n_kept"]
    assert np.issubdtype(y.dtype, np.floating)
    assert target_variance(y) > 0.0
    validate_target_variance(y)
    tail = columns[-5:]
    assert tail == [
        "cross_symbol_prob_delta",
        "cross_symbol_vol_ratio_diff",
        "cross_symbol_rsi_spread",
        "micro_tick_acceleration",
        "keltner_deviation_ratio",
    ]


def test_train_lgbm_candidate_uses_train_api():
    columns = meta_classifier_column_names()
    rows = 40
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.random((rows, len(columns))), columns=columns)
    split = 32
    x_train = frame.iloc[:split]
    x_val = frame.iloc[split:]
    y_train = rng.uniform(-0.5, 0.5, size=split).astype(np.float32)
    y_val = rng.uniform(-0.5, 0.5, size=rows - split).astype(np.float32)
    weights = np.linspace(0.2, 1.0, split, dtype=np.float64)
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        np.linspace(-0.1, 0.1, split, dtype=np.float32),
        np.linspace(-0.2, 0.2, rows - split, dtype=np.float32),
    ]
    with patch("scripts.operations.train_meta_optuna.lgb.train", return_value=mock_model) as mock_train:
        model, train_mae, val_mae = train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            {"max_depth": 4, "learning_rate": 0.05, "num_leaves": 24},
            sample_weight=weights,
        )
    call_args = mock_train.call_args
    params = call_args[0][0]
    train_set = call_args[0][1]
    assert params["objective"] == "huber"
    assert call_args.kwargs["num_boost_round"] == 1000
    assert isinstance(train_set, lgb.Dataset)
    assert len(call_args.kwargs["valid_sets"]) == 1
    assert len(call_args.kwargs["callbacks"]) == 1
    assert model is mock_model
    assert isinstance(train_mae, float)
    assert isinstance(val_mae, float)
    assert mock_model.predict.call_count == 2


def test_continuous_payoff_target_maps_call_to_bull_and_put_to_bear():
    proxy = np.array([0.60, 0.40, 0.51], dtype=np.float32)
    bull = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    bear = np.array([-0.5, 1.0, 0.3], dtype=np.float32)
    y = _continuous_payoff_target(proxy, bull, bear)
    assert y[0] == pytest.approx(1.0)
    assert y[1] == pytest.approx(1.0)
    assert y[2] == pytest.approx(-0.2)


def test_continuous_payoff_threshold_constants():
    assert pytest.approx(0.53) == TCN_CALL_PROXY_THRESHOLD
    assert pytest.approx(0.47) == TCN_PUT_PROXY_THRESHOLD
