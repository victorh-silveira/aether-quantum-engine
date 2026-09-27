"""Contratos matematicos de Touch: causalidade, trajetoria, purging e payout OOS."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.touch_model import predict_touch, select_touch_quote, validate_touch_bundle
from src.application.services.touch_training import (
    _unique_targets,
    build_touch_samples,
    evaluate_touch_policy,
    split_touch_samples,
    train_touch_model,
)
from src.domain.math.touch_ticks import (
    candidate_barriers,
    replay_touch_labels,
    tick_arrays,
    touch_expected_value,
    touch_features,
)
from src.domain.models.touch_policy import TouchPolicy, resolve_touch_policy, touch_enabled


def _ticks():
    epochs = np.arange(1000, 1000 + 1000 * 1000, 1000)
    prices = 100 + np.sin(np.arange(1000) / 20)
    return np.column_stack((epochs, prices))


def _quote(**kwargs):
    return {
        "proposal_id": "q",
        "group_ms": 400000,
        "decision_ms": 400000,
        "policy_hash": TouchPolicy().fingerprint(),
        "symbol": "1HZ75V",
        "barrier": 100.9,
        "contract_type": "ONETOUCH",
        "ask_price": 1.0,
        "payout": 2.0,
        "features": [1.0, 0.0, 0.0, 0.0, 1.0],
        **kwargs,
    }


def _bundle(**kwargs):
    return {
        "schema": "touch_ticks_v1",
        "policy_hash": TouchPolicy().fingerprint(),
        "qualified": True,
        "trained_ms": 1000000,
        "oos": {"source": "quoted_tick_replay", "trades": 120, "return_lcb90": 0.1},
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "coef": [1.0, 0.0, 0.0, 0.0, 0.0],
        "intercept": 0.0,
        "cal_x": [0.0, 1.0],
        "cal_y": [0.0, 1.0],
        **kwargs,
    }


def test_policy_strict_and_account_independent():
    assert not touch_enabled({})
    assert not touch_enabled({"touch": {"enabled": "true"}})
    config = {"touch": {"enabled": True, "latency_ms": [0, 2000], "barrier_sigma": [1.0]}}
    assert touch_enabled(config)
    assert resolve_touch_policy(config) == replace(TouchPolicy(), latency_ms=(0, 2000), barrier_sigma=(1.0,))
    for values in (
        {"symbol": "R_10"},
        {"duration_seconds": 60},
        {"min_oos_trades": 1},
        {"latency_ms": []},
        {"barrier_sigma": [float("nan")]},
        {"min_edge": 0},
    ):
        with pytest.raises(ValueError):
            resolve_touch_policy({"touch": values})


@pytest.mark.parametrize("ticks", [[], [[1, 1]], [[1, 1], [1, 2]], [[1, 1], [2, float("nan")]]])
def test_tick_validation(ticks):
    with pytest.raises(ValueError):
        tick_arrays(ticks)


def test_features_causal_and_barriers_shared():
    ticks = _ticks()[:400]
    features = touch_features(ticks, 101.0, TouchPolicy())
    assert len(features) == 5 and np.isfinite(features).all()
    assert len(candidate_barriers(ticks, TouchPolicy())) == 4
    for invalid in (ticks[:10], np.delete(ticks, 350, axis=0), np.column_stack((ticks[:, 0], np.ones(400)))):
        with pytest.raises(ValueError):
            touch_features(invalid, 101.0, TouchPolicy())


def test_touch_is_path_not_final_close_and_includes_equality():
    ticks = _ticks()
    ticks[:, 1] = 100.0
    ticks[450, 1] = 101.0
    labels = replay_touch_labels(ticks, _quote(barrier=101.0), TouchPolicy())
    assert labels == [1, 1, 1]
    assert replay_touch_labels(ticks, _quote(barrier=102.0), TouchPolicy()) == [0, 0, 0]
    ticks[450, 1] = 99.0
    assert replay_touch_labels(ticks, _quote(barrier=99.0, features=[-1.0] * 5), TouchPolicy()) == [1, 1, 1]
    with pytest.raises(ValueError, match="incompleta"):
        replay_touch_labels(ticks[:500], _quote(), TouchPolicy())
    with pytest.raises(ValueError, match="Lacuna"):
        replay_touch_labels(np.delete(ticks, 450, axis=0), _quote(), TouchPolicy())


def test_payout_and_complement():
    assert touch_expected_value(0.75, _quote()) == 0.5
    assert touch_expected_value(0.25, _quote(contract_type="NOTOUCH")) == 0.5
    for q in (_quote(ask_price=0.0), _quote(payout=float("nan")), _quote(contract_type="CALL")):
        with pytest.raises(ValueError):
            touch_expected_value(0.7, q)


def test_model_never_uses_directional_probability():
    bundle = _bundle()
    assert predict_touch(bundle, [0.0] * 5) == 0.5
    validate_touch_bundle(bundle, TouchPolicy(), 1000001)
    for invalid in (_bundle(qualified=False), _bundle(policy_hash="directional"), _bundle(trained_ms=2000000)):
        with pytest.raises(ValueError):
            validate_touch_bundle(invalid, TouchPolicy(), 1000001)
    for invalid in (_bundle(coef=[1.0]), _bundle(scale=[0.0] * 5), _bundle(cal_y=[float("nan")] * 2)):
        with pytest.raises(ValueError):
            predict_touch(invalid, [0.0] * 5)
    assert select_touch_quote(bundle, [_quote(features=[0.0] * 5)], TouchPolicy()) is None
    assert select_touch_quote(bundle, [_quote(features=[4.0] * 5)], TouchPolicy())[2] > 0.9


def test_samples_dedupe_and_missing_trajectories():
    quotes = [
        _quote(),
        _quote(),
        _quote(proposal_id="bad", symbol="OTHER"),
        _quote(proposal_id="late", decision_ms=999000),
    ]
    rows, rejected = build_touch_samples(quotes, _ticks(), TouchPolicy())
    assert len(rows) == 1 and rejected == 2
    assert "labels" in rows[0]


def _training_rows():
    rows = []
    for i in range(610):
        epoch = (i + 1) * 1000000
        y = i % 2
        features = [float(2 * y - 1), 0.0, -5.0, 1.0, 1.0]
        for kind in ("ONETOUCH", "NOTOUCH"):
            rows.append(
                _quote(
                    proposal_id=f"{i}-{kind}",
                    group_ms=epoch,
                    decision_ms=epoch,
                    features=features,
                    labels=[y] * 3,
                    contract_type=kind,
                )
            )
    return rows


def test_training_calibration_and_economic_holdout():
    rows = _training_rows()
    train, calibration, test = split_touch_samples(rows, TouchPolicy())
    assert max(r["decision_ms"] for r in train) + 602000 < min(r["group_ms"] for r in calibration)
    assert max(r["decision_ms"] for r in calibration) + 602000 < min(r["group_ms"] for r in test)
    bundle = train_touch_model(rows, TouchPolicy())
    assert bundle["oos"]["trades"] >= 120
    assert bundle["qualified"] is True
    assert bundle["oos"]["all_brier"] < bundle["oos"]["baseline_brier"]
    with pytest.raises(ValueError, match="insuficientes"):
        split_touch_samples(rows[:10], TouchPolicy())
    with pytest.raises(ValueError, match="purging"):
        split_touch_samples([{**r, "group_ms": i, "decision_ms": i} for i, r in enumerate(rows)], TouchPolicy())
    with pytest.raises(ValueError, match="classe unica"):
        _unique_targets([_quote(labels=[1] * 3)])


def test_oos_latency_worst_case_and_nonoverlap():
    bundle = _bundle()
    row = _quote(features=[5.0] * 5, labels=[1, 0, 1])
    report = evaluate_touch_policy(bundle, [row, {**row, "group_ms": 400001}], TouchPolicy())
    assert report["trades"] == 1 and report["mean_return"] == -1
    with patch("src.application.services.touch_training.select_touch_quote", return_value=None):
        assert evaluate_touch_policy(bundle, [row], TouchPolicy())["trades"] == 0
