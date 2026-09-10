import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import pytest

from scripts.operations.train_meta_classifier import (
    build_training_summary,
    target_variance,
    validate_target_variance,
)
from scripts.operations.train_meta_data import (
    META_TRAIN_MAX_BARS,
    OhlcBundle,
    meta_bars_meet_quality,
    resolve_meta_train_bars,
)
from scripts.operations.train_meta_optuna import (
    LGBM_METRIC,
    LGBM_N_ESTIMATORS_CV,
    LGBM_N_ESTIMATORS_LARGE,
    LGBM_QUIET_PARAMS,
    LGBM_REGRESSION_OBJECTIVE,
    META_EXPORT_MAX_MAE_GAP,
    META_EXPORT_MIN_ZSCORE,
    OPTUNA_NEGATIVE_EDGE_PENALTY,
    _boost_round_budget,
    _booster_best_iteration,
    _booster_trained_iterations,
    _eval_l1_histories,
    _hygiene_for_bundle,
    _lgbm_search_bounds,
    _mae_curves_from_booster,
    _predict_with_export,
    _purged_frame_split,
    _select_gap_feasible_iteration,
    assert_export_mae_gap,
    assert_export_zscore_floor,
    build_paired_training_dataset,
    configure_meta_train_logging,
    payoff_zscore_mean,
    run_optuna_study,
    train_lgbm_candidate,
)
from scripts.operations.train_meta_vector import (
    INNER_JOIN_MIN_SAMPLE_RATIO,
    LABEL_MODE_PAYOFF,
    TARGET_NULL_MAE_GAP_MAX,
    TCN_CALL_PROXY_THRESHOLD,
    TCN_PUT_PROXY_THRESHOLD,
    _apply_target_transform,
    _assert_train_val_target_scale,
    _continuous_payoff_target,
    _fit_train_target_transform,
    _null_mae_gap,
    _purged_split_arrays,
    _purged_split_stds,
    _resolve_training_labels,
    _scale_targets_from_train,
    _trim_degenerate_target_prefix,
    teacher_sample_weights,
)
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import meta_classifier_column_names


GRAY_KEEP_FLOOR = 96


def _synthetic_bundle(symbol: str = "R_10", *, n: int = 280, phase: float = 0.0) -> OhlcBundle:
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
        teacher_probs={"R_10": _decisive_teacher(len(bundle.closes))},
    )
    assert len(frame) >= GRAY_KEEP_FLOOR
    assert hygiene["n_kept"] == len(frame) == len(y) == len(proxy)
    assert hygiene["label_mode"] in {1, 2}
    assert hygiene["n_dropped_gray"] == 0
    assert float(hygiene["label_scale"]) > 0.0
    assert int(hygiene.get("n_dropped_flat_prefix", 0)) >= 0
    validate_target_variance(y)


def test_build_paired_training_dataset_rejects_insufficient_history():
    closes = np.linspace(100.0, 101.0, 40)
    bundle = OhlcBundle(
        symbol="R_10",
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
        teacher_probs={"R_10": _decisive_teacher(len(bundle.closes), seed=3)},
    )
    assert len(frame) >= int(5000 * INNER_JOIN_MIN_SAMPLE_RATIO) - 40
    assert frame.shape[1] == META_FEATURE_DIM
    assert np.allclose(frame["micro_price_velocity"].to_numpy(), 0.0)
    assert hygiene["n_dropped_gray"] == 0
    assert hygiene["label_mode"] in {1, 2}
    assert float(hygiene["label_scale"]) > 0.0
    assert int(hygiene["n_dropped_flat_prefix"]) >= 0
    validate_target_variance(y)


def test_build_paired_training_dataset_uses_forward_z_labels():
    n = 1200
    bundle = _synthetic_bundle(n=n)
    teacher = np.full(n, 0.50, dtype=np.float32)
    frame, y, proxy, pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=1200,
        teacher_probs={"R_10": teacher},
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
        symbol="R_10",
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
        teacher_probs={"R_10": teacher},
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
    assert resolve_meta_train_bars(10) >= 60


def test_meta_bars_meet_quality_accepts_api_shortfall():
    ok, soft = meta_bars_meet_quality(1984, 2000, shortfall_ratio=0.95)
    assert ok is True
    assert soft is True
    ok2, soft2 = meta_bars_meet_quality(1800, 2000, shortfall_ratio=0.95)
    assert ok2 is False
    assert soft2 is False
    ok3, soft3 = meta_bars_meet_quality(2000, 2000, shortfall_ratio=0.95)
    assert ok3 is True
    assert soft3 is False


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
            "label_scale": np.float64(42.5),
            "z_collapse_pct": np.float64(1.5),
        }
    )
    assert out["n_kept"] == 100
    assert out["forward_var"] == pytest.approx(0.0125)
    assert out["data_source"] == "timescale"
    assert out["label_mode"] == 2
    assert out["label_scale"] == pytest.approx(42.5)
    assert out["z_collapse_pct"] == pytest.approx(1.5)


def test_build_training_summary_includes_continuous_telemetry():
    bundles = [_synthetic_bundle()]
    teacher = _decisive_teacher(len(bundles[0].closes))
    frame, y, _, _, _ = build_paired_training_dataset(
        bundles,
        micro_granularity=120,
        fetch_count=280,
        teacher_probs={"R_10": teacher},
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
        symbols=["R_10"],
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
    assert LGBM_REGRESSION_OBJECTIVE == "regression_l1"


def test_negative_edge_penalty_matches_overfit_floor():
    assert pytest.approx(-1.0) == OPTUNA_NEGATIVE_EDGE_PENALTY
    assert payoff_zscore_mean(np.array([1.0, -1.0]), np.array([-0.5, 0.5])) <= 0.0


def test_build_paired_training_dataset_has_continuous_target_and_named_columns():
    bundle = _synthetic_bundle()
    frame, y, proxy, pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=280,
        teacher_probs={"R_10": _decisive_teacher(len(bundle.closes), seed=11)},
    )
    columns = meta_classifier_column_names()
    assert list(frame.columns) == columns
    assert isinstance(frame, pl.DataFrame)
    assert frame.shape[1] == META_FEATURE_DIM == len(columns) == 23
    assert frame.height == len(y) == len(proxy) == len(pnl) == hygiene["n_kept"]
    assert np.issubdtype(y.dtype, np.floating)
    assert target_variance(y) > 0.0
    validate_target_variance(y)
    tail = columns[-5:]
    assert tail == [
        "micro_price_velocity",
        "micro_tick_count_norm",
        "implied_vol_centered",
        "micro_tick_acceleration",
        "keltner_deviation_ratio",
    ]


def test_train_lgbm_candidate_uses_train_api():
    columns = meta_classifier_column_names()
    rows = 40
    rng = np.random.default_rng(7)
    frame = pl.DataFrame({name: rng.random(rows) for name in columns})
    split = 32
    x_train = frame.slice(0, split)
    x_val = frame.slice(split, rows - split)
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
    assert params["objective"] == LGBM_REGRESSION_OBJECTIVE == "regression_l1"
    assert params["metric"] == LGBM_METRIC == "l1"
    assert call_args.kwargs["num_boost_round"] == LGBM_N_ESTIMATORS_LARGE == 80
    assert isinstance(train_set, lgb.Dataset)
    assert len(call_args.kwargs["valid_sets"]) == 2
    assert call_args.kwargs["valid_names"] == ["train", "valid"]
    assert len(call_args.kwargs["callbacks"]) == 2
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
    assert pytest.approx(0.55) == TCN_CALL_PROXY_THRESHOLD
    assert pytest.approx(0.45) == TCN_PUT_PROXY_THRESHOLD


@pytest.mark.asyncio
async def test_resolve_training_bundles_skips_smoke_timescale_with_info(caplog):
    from scripts.operations.train_meta_data import resolve_training_bundles

    settings = {
        "data_handler": {"micro_granularity": 300, "granularity": 86400},
        "api_config": {"public_ws_url": "wss://example.test/ws"},
    }
    deriv_bundle = _synthetic_bundle("1HZ75V", n=280)

    async def _fake_inventory(*_a, **_k):
        return False

    async def _fake_deriv(*_a, **_k):
        return [deriv_bundle]

    async def _fake_persist(*_a, **_k):
        return 0

    with (
        patch(
            "scripts.operations.train_meta_data._timescale_inventory_meets_floor",
            new=_fake_inventory,
        ),
        patch(
            "scripts.operations.train_meta_data.load_bundles_from_deriv",
            new=_fake_deriv,
        ),
        patch(
            "scripts.operations.train_meta_data.persist_bundles_to_timescale",
            new=_fake_persist,
        ),
        patch(
            "scripts.operations.train_meta_data.assert_bundles_match_granularity",
            return_value=None,
        ),
        patch(
            "scripts.operations.train_meta_data.meta_bars_meet_quality",
            return_value=(True, False),
        ),
        caplog.at_level(logging.INFO, logger="AETH.meta"),
    ):
        out = await resolve_training_bundles(
            settings=settings,
            dsn="postgresql://aether:aether@localhost:5432/aether",
            symbols=["1HZ75V"],
            granularity=300,
            bars=120,
            source="auto",
            min_quality_bars=100,
            seed_timescale_on_deriv=False,
        )
    assert len(out) == 1
    assert out[0].symbol == "1HZ75V"
    assert any("Timescale smoke/curto" in r.message for r in caplog.records)
    assert not any("rejeitado" in r.message for r in caplog.records)


def test_resolve_training_labels_payoff_stays_raw():
    n = 80
    proxy = np.full(n, 0.62, dtype=np.float32)
    call_pnl = np.linspace(-80.0, 120.0, n).astype(np.float32)
    closes = 1000.0 + np.cumsum(np.linspace(0.5, 2.5, n))
    labels, meta = _resolve_training_labels(proxy, call_pnl, closes=closes)
    assert int(meta["label_mode"]) == LABEL_MODE_PAYOFF
    assert float(meta["label_scale"]) == pytest.approx(1.0)
    assert float(np.std(labels, ddof=0)) == pytest.approx(float(np.std(call_pnl, ddof=0)), rel=0.05)
    assert abs(float(np.std(labels, ddof=0)) - 1.0) > 0.5


def test_purged_split_keeps_teacher_sample_weights():
    columns = meta_classifier_column_names()
    n = 200
    frame = pl.DataFrame({name: np.ones(n, dtype=np.float32) for name in columns})
    y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    proxy = np.linspace(0.2, 0.9, n, dtype=np.float32)
    weights = teacher_sample_weights(proxy)
    _x_tr, _x_val, y_tr, _y_val, w_tr = _purged_frame_split(frame, y, sample_weight=weights)
    assert w_tr is not None
    assert len(w_tr) == len(y_tr)
    assert float(w_tr[0]) == pytest.approx(float(weights[0]))
    assert float(w_tr[-1]) == pytest.approx(float(weights[len(y_tr) - 1]))
    assert _x_val.height == 50
    assert _x_tr.height == 118


def test_lgbm_search_bounds_regularize_large_n():
    min_child_lo, min_child_hi, depth_hi, lambda_lo = _lgbm_search_bounds(4000, use_cv=False)
    assert depth_hi == 1
    assert lambda_lo >= 16.0
    assert min_child_lo >= 32
    assert min_child_lo <= min_child_hi
    cv_lo, cv_hi, cv_depth, cv_lambda = _lgbm_search_bounds(80, use_cv=True)
    assert cv_depth == 4
    assert cv_lambda == pytest.approx(0.1)
    assert cv_lo <= cv_hi
    assert _boost_round_budget(use_cv=False) == LGBM_N_ESTIMATORS_LARGE == 80
    assert _boost_round_budget(use_cv=True) == LGBM_N_ESTIMATORS_CV == 200


def test_run_optuna_study_overfit_message_includes_label_mode(monkeypatch):
    columns = meta_classifier_column_names()
    n = 200
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({name: rng.normal(size=n).astype(np.float32) for name in columns})
    y = rng.normal(size=n).astype(np.float32)
    captured: dict[str, object] = {}

    def fake_train(*args, sample_weight=None, num_boost_round=None, **_k):
        captured["w"] = sample_weight
        captured["params"] = args[4] if len(args) > 4 else {}
        captured["rounds"] = num_boost_round
        model = MagicMock()
        model.best_iteration = 12
        model.predict = lambda x: np.zeros(len(x), dtype=np.float64)
        return model, 0.01, 0.05

    monkeypatch.setattr("scripts.operations.train_meta_optuna.train_lgbm_candidate", fake_train)
    weights = teacher_sample_weights(np.linspace(0.2, 0.9, n).astype(np.float32))
    with pytest.raises(RuntimeError, match="label_mode=2") as exc:
        run_optuna_study(
            frame,
            y,
            trials=2,
            hygiene={"label_mode": 2},
            sample_weight=weights,
        )
    msg = str(exc.value)
    assert "y_std=" in msg
    assert "train_std=" in msg
    assert "val_std=" in msg
    assert "med_mae_gap=" in msg
    assert "med_train_mae=" in msg
    assert "med_val_mae=" in msg
    assert "med_best_iter=" in msg
    assert "med_naive_mae=" in msg
    assert "fold de treino" in msg
    assert captured["w"] is not None
    assert len(captured["w"]) > 0
    assert captured["rounds"] == LGBM_N_ESTIMATORS_LARGE
    params = captured["params"]
    assert isinstance(params, dict)
    assert "bagging_fraction" in params
    assert int(params["max_depth"]) <= 1
    assert int(params["num_leaves"]) <= 2
    assert float(params["learning_rate"]) <= 0.02 + 1e-12


def test_trim_prefix_aligns_train_val_scale():
    columns = meta_classifier_column_names()
    n = 800
    rng = np.random.default_rng(4)
    labels = np.concatenate([rng.normal(0.0, 1.0, 600), rng.normal(0.0, 80.0, 200)]).astype(np.float32)
    frame = pl.DataFrame({name: np.ones(n, dtype=np.float32) for name in columns})
    proxy = np.full(n, 0.62, dtype=np.float32)
    _out_frame, out_y, _p, _pnl, dropped = _trim_degenerate_target_prefix(
        frame,
        labels,
        proxy,
        labels.copy(),
        min_keep=150,
    )
    assert dropped > 0
    train_std, val_std = _purged_split_stds(out_y)
    _assert_train_val_target_scale(train_std, val_std)
    y_tr, y_va = _purged_split_arrays(out_y)
    assert _null_mae_gap(y_tr, y_va) <= TARGET_NULL_MAE_GAP_MAX + 1e-12


def test_trim_degenerate_target_prefix_drops_leading_zeros():
    columns = meta_classifier_column_names()
    n = 200
    frame = pl.DataFrame({name: np.ones(n, dtype=np.float32) for name in columns})
    labels = np.concatenate([np.zeros(80), np.linspace(-1.0, 1.0, 120)]).astype(np.float32)
    proxy = np.full(n, 0.62, dtype=np.float32)
    pnl = labels.copy()
    out_frame, out_y, out_proxy, out_pnl, dropped = _trim_degenerate_target_prefix(
        frame,
        labels,
        proxy,
        pnl,
        min_keep=40,
    )
    assert dropped == 80
    assert out_frame.height == 120
    assert len(out_y) == len(out_proxy) == len(out_pnl) == 120
    assert float(out_y[0]) != 0.0


def test_build_paired_training_dataset_trims_flat_prefix():
    bundle = _synthetic_bundle(n=999)
    _frame, y, _proxy, _pnl, hygiene = build_paired_training_dataset(
        [bundle],
        micro_granularity=120,
        fetch_count=5000,
        teacher_probs={"R_10": _decisive_teacher(len(bundle.closes), seed=5)},
    )
    assert int(hygiene["n_dropped_flat_prefix"]) > 0
    assert hygiene["n_kept"] == len(y)
    assert float(np.std(y, ddof=0)) > 0.0


def test_train_target_transform_unit_scales_train_fold():
    rng = np.random.default_rng(1)
    y_train = rng.normal(0.0, 4.0, size=400).astype(np.float64)
    y_val = rng.normal(0.0, 4.0, size=100).astype(np.float64)
    y_tr, y_va, scale = _scale_targets_from_train(y_train, y_val)
    assert float(np.std(y_tr, ddof=0)) == pytest.approx(1.0, abs=0.08)
    assert 0.4 <= float(np.std(y_va, ddof=0)) <= 1.6
    assert float(scale) > 1.0


def test_scale_targets_from_train_does_not_use_val_std():
    y_train = np.linspace(-1.0, 1.0, 80)
    y_val = np.linspace(-50.0, 50.0, 20)
    lo, hi, scale = _fit_train_target_transform(y_train)
    y_tr = _apply_target_transform(y_train, lo, hi, scale)
    y_va = _apply_target_transform(y_val, lo, hi, scale)
    train_std = float(np.std(y_train, ddof=0))
    val_std = float(np.std(y_val, ddof=0))
    assert float(scale) == pytest.approx(train_std, rel=0.15)
    assert abs(float(scale) - val_std) > 10.0
    assert float(np.max(np.abs(y_va))) <= float(np.max(np.abs(y_tr))) + 1e-6


def test_assert_train_val_target_scale_rejects_quiet_train():
    with pytest.raises(RuntimeError, match="degenerado"):
        _assert_train_val_target_scale(0.001, 1.8)
    with pytest.raises(RuntimeError, match="degenerado"):
        _assert_train_val_target_scale(0.0, 1.0)
    with pytest.raises(RuntimeError, match="degenerado"):
        _assert_train_val_target_scale(1.0, 2.1)
    with pytest.raises(RuntimeError, match="null_mae_gap"):
        _assert_train_val_target_scale(1.0, 1.4, null_mae_gap=2.2)
    _assert_train_val_target_scale(1.0, 1.4)
    _assert_train_val_target_scale(1.0, 1.4, null_mae_gap=1.9)


def test_run_optuna_study_rejects_degenerate_split():
    columns = meta_classifier_column_names()
    n = 200
    rng = np.random.default_rng(2)
    frame = pl.DataFrame({name: rng.normal(size=n).astype(np.float32) for name in columns})
    y = np.concatenate([np.zeros(150), np.linspace(-2.0, 2.0, 50)]).astype(np.float32)
    with pytest.raises(RuntimeError, match="degenerado"):
        run_optuna_study(frame, y, trials=2, hygiene={"label_mode": 2})


def test_predict_with_export_uses_label_location():
    model = MagicMock()
    model._aether_export_iteration = 0
    model._aether_label_location = 0.35
    out = _predict_with_export(model, np.zeros((4, 2)))
    assert np.allclose(out, 0.35)
    assert model.predict.call_count == 0
    model._aether_export_iteration = 3
    model.predict.return_value = np.ones(4, dtype=np.float64)
    out_trees = _predict_with_export(model, np.zeros((4, 2)))
    assert np.allclose(out_trees, 1.0)
    model.predict.assert_called_once()


def test_select_gap_feasible_iteration_picks_best_legal_val():
    assert _select_gap_feasible_iteration([], [0.5]) is None
    assert _select_gap_feasible_iteration([0.2, 0.1], [0.8, 0.5]) == 0
    train = [0.70, 0.55, 0.40]
    val = [1.20, 0.90, 1.00]
    assert _select_gap_feasible_iteration(train, val) == 1


def test_eval_l1_histories_reads_metric_aliases():
    train_l1, val_l1 = _eval_l1_histories({"train": {"l1": [0.4, 0.3]}, "valid": {"l1": [0.5, 0.6]}})
    assert train_l1 == [0.4, 0.3]
    assert val_l1 == [0.5, 0.6]
    train_reg, val_reg = _eval_l1_histories({"train": {"regression_l1": [0.15]}, "valid": {"regression_l1": [0.25]}})
    assert train_reg == [0.15]
    assert val_reg == [0.25]
    train_alt, val_alt = _eval_l1_histories({"train": {"other": [0.2]}, "valid": {"other": [0.3]}})
    assert train_alt == [0.2]
    assert val_alt == [0.3]
    empty_tr, empty_va = _eval_l1_histories("bad")
    assert empty_tr == []
    assert empty_va == []
    none_tr, none_va = _eval_l1_histories({})
    assert none_tr == []
    assert none_va == []


def test_booster_trained_iterations_reads_current_and_trees():
    model = MagicMock()
    model.current_iteration.return_value = 12
    assert _booster_trained_iterations(model) == 12
    model.current_iteration.return_value = True
    model.num_trees.return_value = 9
    assert _booster_trained_iterations(model) == 9
    model.current_iteration.return_value = 3.9
    assert _booster_trained_iterations(model) == 3
    model.current_iteration.return_value = object()
    model.num_trees.return_value = 0
    assert _booster_trained_iterations(model) == 0
    model.current_iteration.side_effect = TypeError
    model.num_trees.side_effect = TypeError
    assert _booster_trained_iterations(model) == 0
    model.current_iteration.side_effect = None
    model.current_iteration.return_value = 0
    model.num_trees.side_effect = None
    model.num_trees.return_value = 5
    assert _booster_trained_iterations(model) == 5


def test_mae_curves_from_booster_records_each_round():
    class _Booster:
        def predict(self, data, num_iteration=None, **_kwargs):
            scale = 1.0 / max(int(num_iteration or 1), 1)
            return np.full(len(data), scale, dtype=np.float64)

    train_hist, val_hist = _mae_curves_from_booster(
        _Booster(),
        np.ones((4, 2)),
        np.zeros(4),
        np.ones((3, 2)),
        np.zeros(3),
        n_iter=2,
    )
    assert len(train_hist) == 3
    assert len(val_hist) == 3
    assert train_hist[1] > train_hist[2]
    model = MagicMock()
    model._aether_export_iteration = 11
    model.best_iteration = 80
    assert _booster_best_iteration(model) == 11
    model._aether_export_iteration = "x"
    assert _booster_best_iteration(model) == 80
    model._aether_export_iteration = None
    model.best_iteration = object()
    assert _booster_best_iteration(model) == 0


def test_train_lgbm_candidate_skips_early_stop_when_disabled(monkeypatch):
    columns = meta_classifier_column_names()
    rows = 24
    rng = np.random.default_rng(3)
    frame = pl.DataFrame({name: rng.random(rows) for name in columns})
    x_train = frame.slice(0, 16)
    x_val = frame.slice(16, 8)
    y_train = rng.uniform(-0.4, 0.4, size=16).astype(np.float32)
    y_val = rng.uniform(-0.4, 0.4, size=8).astype(np.float32)
    monkeypatch.setattr(
        "scripts.operations.train_meta_optuna._eval_l1_histories",
        lambda _e: ([0.70, 0.55], [0.90, 0.80]),
    )

    class _Frozen:
        __slots__ = ()

        def predict(self, data, **_kwargs):
            return np.zeros(len(data), dtype=np.float64)

    frozen = _Frozen()
    with patch("scripts.operations.train_meta_optuna.lgb.train", return_value=frozen) as mock_train:
        train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            {"max_depth": 1, "learning_rate": 0.05, "num_leaves": 2},
            early_stopping=False,
        )
    assert len(mock_train.call_args.kwargs["callbacks"]) == 1
