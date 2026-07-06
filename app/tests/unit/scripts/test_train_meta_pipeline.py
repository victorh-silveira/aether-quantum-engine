import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    build_paired_training_dataset,
    configure_meta_train_logging,
    train_lgbm_candidate,
)
from scripts.operations.train_meta_vector import (
    INNER_JOIN_MIN_SAMPLE_RATIO,
    TCN_CALL_PROXY_THRESHOLD,
    TCN_PUT_PROXY_THRESHOLD,
    _continuous_payoff_target,
)
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import meta_classifier_column_names


def _synthetic_bundle(symbol: str, *, n: int = 280, phase: float = 0.0) -> OhlcBundle:
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


def _asymmetric_bundles(*, bull_n: int = 5000, bear_n: int = 999) -> list[OhlcBundle]:
    bull = _synthetic_bundle("RDBULL", n=bull_n)
    bear = _synthetic_bundle("RDBEAR", n=bear_n, phase=0.8)
    bear = OhlcBundle(
        symbol="RDBEAR",
        granularity=60,
        closes=bear.closes.astype(np.float64),
        open_=bear.open_.astype(np.float64),
        high=bear.high.astype(np.float64),
        low=bear.low.astype(np.float64),
        epochs=bull.epochs[:bear_n].copy(),
        source="test",
    )
    return [bull, bear]


def test_build_paired_training_dataset_accepts_paired_cap_below_planned_fetch():
    bundles = _asymmetric_bundles(bull_n=5000, bear_n=999)
    frame, y, _, _ = build_paired_training_dataset(bundles, micro_granularity=60, fetch_count=5000)
    assert len(frame) >= int(999 * INNER_JOIN_MIN_SAMPLE_RATIO) - 40
    validate_target_variance(y)


def test_build_paired_training_dataset_rejects_severely_short_bear():
    bull = _synthetic_bundle("RDBULL", n=500)
    bear_n = 120
    closes = np.linspace(100.0, 101.0, bear_n)
    bear = OhlcBundle(
        symbol="RDBEAR",
        granularity=60,
        closes=closes,
        open_=closes - 0.01,
        high=closes + 0.02,
        low=closes - 0.02,
        epochs=bull.epochs[:bear_n].copy(),
        source="test",
    )
    with pytest.raises(RuntimeError, match="paginacao assimétrica"):
        build_paired_training_dataset([bull, bear], micro_granularity=60, fetch_count=5000)


def test_build_paired_training_dataset_rejects_disjoint_epochs():
    bull = _synthetic_bundle("RDBULL", n=280)
    bear = _synthetic_bundle("RDBEAR", n=280, phase=0.8)
    bear = OhlcBundle(
        symbol="RDBEAR",
        granularity=60,
        closes=bear.closes,
        open_=bear.open_,
        high=bear.high,
        low=bear.low,
        epochs=bear.epochs + 9_999_999,
        source="test",
    )
    with pytest.raises(RuntimeError, match="Nenhuma epoch comum"):
        build_paired_training_dataset([bull, bear], micro_granularity=60, fetch_count=280)


def test_build_paired_training_dataset_accepts_high_overlap_inner_join():
    bundles = [_synthetic_bundle("RDBULL", n=5000), _synthetic_bundle("RDBEAR", n=5000, phase=0.3)]
    frame, y, _, _ = build_paired_training_dataset(bundles, micro_granularity=60, fetch_count=5000)
    assert len(frame) >= int(5000 * INNER_JOIN_MIN_SAMPLE_RATIO) - 40
    assert frame.shape[1] == META_FEATURE_DIM
    validate_target_variance(y)


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


def test_build_training_summary_includes_continuous_telemetry():
    bundles = [_synthetic_bundle("RDBULL"), _synthetic_bundle("RDBEAR", phase=0.8)]
    frame, y, _, _ = build_paired_training_dataset(bundles, micro_granularity=60, fetch_count=280)
    summary = build_training_summary(
        frame=frame,
        y=y,
        train_mae=0.08,
        val_mae=0.11,
        bundle_meta={"feature_dim": len(meta_classifier_column_names()), "model_type": "regressor"},
        output_path=Path("meta_lgbm.pkl"),
        symbols=["RDBULL", "RDBEAR"],
        bundles=bundles,
    )
    assert "cross_symbol_prob_delta_mean" in summary
    assert summary["train_mae"] == pytest.approx(0.08)
    assert summary["best_val_mae"] == pytest.approx(0.11)
    assert summary["target_variance"] == pytest.approx(target_variance(y))
    assert summary["model_type"] == "regressor"
    assert "class_balance_ratio" not in summary


def test_configure_meta_train_logging_silences_lightgbm_and_optuna():
    configure_meta_train_logging()
    assert logging.getLogger("lightgbm").level == logging.ERROR
    assert optuna.logging.get_verbosity() == optuna.logging.WARNING


def test_lgbm_quiet_params_include_verbose_and_warnings():
    assert LGBM_QUIET_PARAMS["verbose"] == -1
    assert LGBM_QUIET_PARAMS["warnings"] is False
    assert LGBM_REGRESSION_OBJECTIVE == "huber"


def test_build_paired_training_dataset_has_continuous_target_and_named_columns():
    bundles = [_synthetic_bundle("RDBULL"), _synthetic_bundle("RDBEAR", phase=0.8)]
    frame, y, proxy, pnl = build_paired_training_dataset(bundles, micro_granularity=60, fetch_count=280)
    columns = meta_classifier_column_names()
    assert list(frame.columns) == columns
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[1] == META_FEATURE_DIM == len(columns) == 39
    assert len(frame) == len(y) == len(proxy) == len(pnl)
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


def test_train_lgbm_candidate_uses_regressor_and_explicit_feature_name_columns():
    columns = meta_classifier_column_names()
    rows = 40
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.random((rows, len(columns))), columns=columns)
    split = 32
    x_train = frame.iloc[:split]
    x_val = frame.iloc[split:]
    y_train = rng.uniform(-0.5, 0.5, size=split).astype(np.float32)
    y_val = rng.uniform(-0.5, 0.5, size=rows - split).astype(np.float32)
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        np.linspace(-0.1, 0.1, split, dtype=np.float32),
        np.linspace(-0.2, 0.2, rows - split, dtype=np.float32),
    ]
    with patch("scripts.operations.train_meta_optuna.lgb.LGBMRegressor", return_value=mock_model) as mock_cls:
        model, train_mae, val_mae = train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            {"max_depth": 4, "learning_rate": 0.05, "num_leaves": 24},
        )
    assert mock_cls.call_args.kwargs["objective"] == "huber"
    assert mock_cls.call_args.kwargs["verbose"] == -1
    assert mock_cls.call_args.kwargs["warnings"] is False
    assert mock_cls.call_args.kwargs["n_jobs"] == 2
    assert model is mock_model
    assert isinstance(train_mae, float)
    assert isinstance(val_mae, float)
    fit_args, fit_kwargs = mock_model.fit.call_args
    assert fit_kwargs["feature_name"] == columns
    assert list(fit_args[0].columns) == columns
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
