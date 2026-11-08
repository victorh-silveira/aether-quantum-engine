"""Treino temporal e compatibilidade de artefatos."""

import json
import math

import pytest

from indicator.domain import Candle
from indicator.model import LOOKBACK, Model, assess, examples, features, load, ohlc_features, save, sigmoid, train
from indicator.optimizer import fit_logistic


def series(n=600):
    price = 100.0
    candles = []
    for index in range(n):
        price *= 1.01 if index % 8 < 4 else 0.99
        candles.append(Candle(index * 300, price, price, price, price))
    return candles


def test_features_sigmoid_and_examples_are_causal():
    candles = series(30)
    assert len(features(candles)) == LOOKBACK
    assert len(ohlc_features(candles)) == LOOKBACK * 3
    assert len(examples(candles, 300, "ohlc")[0]) == 21
    assert sigmoid(1000) == 1
    assert sigmoid(-1000) == 0
    assert len(examples(candles, 300)[0]) == 21
    with pytest.raises(ValueError):
        features(candles[:8])
    with pytest.raises(ValueError):
        features([Candle(index, 0, 0, 0, 0) for index in range(9)])
    with pytest.raises(ValueError, match="OHLC invalido"):
        ohlc_features([Candle(index, 1, 0, 2, 1) for index in range(9)])
    broken = list(candles)
    broken[10] = Candle(99999, 100, 100, 100, 100)
    assert len(examples(broken, 300)[0]) < 21


def test_optimizer_stops_on_temporal_plateau_and_keeps_best_weights():
    rows = [(0.0,) * LOOKBACK for _ in range(100)]
    labels = [index % 2 for index in range(100)]
    weights, bias, steps = fit_logistic(rows, labels, max_steps=200, patience=5, min_steps=30)
    assert steps == 30
    assert weights == [0.0] * LOOKBACK
    assert abs(bias) < 1e-8


def test_train_save_load_prediction_and_rejection(tmp_path):
    candles = series()
    model = train("R_75", 300, candles)
    assert model is not None
    accepted = assess("R_75", 300, candles)
    assert accepted.reason == "aprovado"
    assert accepted.validation_samples > 0
    assert 0 <= accepted.ev_eligible_samples <= accepted.validation_samples
    assert accepted.maximum_side_probability is not None
    assert 0.5 <= accepted.maximum_side_probability <= 1.0
    assert model.validation_accuracy > model.baseline_accuracy
    assert model.validation_brier < model.baseline_brier
    assert model.validation_protocol == "wf3_blockbootstrap_v1"
    assert model.validation_ece <= 0.05
    assert model.brier_improvement_lower95 > 0
    assert 0 < model.predict(candles) < 1
    path = save(model, tmp_path)
    assert load(tmp_path, "R_75", 300) == model
    assert load(tmp_path, "R_75", 900) is None
    assert train("R_75", 300, candles[:100]) is None
    assert assess("R_75", 300, candles[:100]).reason.startswith("amostras insuficientes")
    data = json.loads(path.read_text())
    data["weights"] = [math.inf] * 8
    path.write_text(json.dumps(data))
    assert load(tmp_path, "R_75", 300) is None
    path.write_text("invalid")
    assert load(tmp_path, "R_75", 300) is None


def test_save_retries_transient_windows_lock_and_cleans_temporary(tmp_path, monkeypatch):
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25)
    original = type(tmp_path).replace
    attempts = 0

    def locked_then_ready(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("arquivo em uso")
        return original(source, destination)

    monkeypatch.setattr(type(tmp_path), "replace", locked_then_ready)
    monkeypatch.setattr("indicator.model.time.sleep", lambda _: None)
    assert save(model, tmp_path).is_file()
    assert attempts == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_save_preserves_previous_model_when_lock_persists(tmp_path, monkeypatch):
    old = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "old", 0.6, 0.5, 0.2, 0.25)
    save(old, tmp_path)
    monkeypatch.setattr(type(tmp_path), "replace", lambda *_: (_ for _ in ()).throw(PermissionError("em uso")))
    monkeypatch.setattr("indicator.model.time.sleep", lambda _: None)
    with pytest.raises(PermissionError, match="em uso"):
        save(old, tmp_path)
    assert load(tmp_path, "R_75", 300) == old
    assert not list(tmp_path.glob("*.tmp"))


def test_model_rejects_majority_only_and_invalid_shape(tmp_path):
    candles = [Candle(index * 300, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(300)]
    assert train("R_75", 300, candles) is None
    rejected = assess("R_75", 300, candles)
    assert rejected.reason.startswith("experimental: sem ganho temporal")
    assert rejected.accuracy is not None
    assert f"brier={rejected.brier:.5f}" in rejected.reason
    assert rejected.model is not None
    assert rejected.model.status == "experimental"
    save(rejected.model, tmp_path)
    assert load(tmp_path, "R_75", 300) is None
    assert load(tmp_path, "R_75", 300, allow_experimental=True) == rejected.model
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "x", 0.6, 0.5, 0.2, 0.25)
    path = save(model, tmp_path)
    content = json.loads(path.read_text())
    content["scales"] = [0.0] * 8
    path.write_text(json.dumps(content))
    assert load(tmp_path, "R_75", 300) is None


def test_tie_label_is_excluded():
    candles = series(30)
    tied = list(candles)
    tied[10] = Candle(tied[10].epoch, tied[9].close, tied[9].close, tied[9].close, tied[9].close)
    assert len(examples(tied, 300)[0]) == 20


def test_invalid_price_window_is_excluded():
    candles = series(30)
    invalid = list(candles)
    invalid[10] = Candle(invalid[10].epoch, 0, 0, 0, 0)
    assert len(examples(invalid, 300)[0]) < len(examples(candles, 300)[0])


def test_unvalidated_artifact_is_not_loaded(tmp_path):
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.5, 0.5, 0.3, 0.25)
    save(model, tmp_path)
    assert load(tmp_path, "R_75", 300) is None


def test_unknown_model_status_is_rejected(tmp_path):
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25, "other")
    save(model, tmp_path)
    assert load(tmp_path, "R_75", 300, allow_experimental=True) is None


def test_tree_model_predicts_from_causal_body_and_validates_shape(tmp_path):
    model = Model(
        "R_25",
        300,
        (0.0,) * 24,
        0.0,
        (0.0,) * 24,
        (1.0,) * 24,
        "tree-v1",
        0.6,
        0.5,
        0.2,
        0.25,
        feature_set="tree",
        tree_feature=15,
        tree_left=0.2,
        tree_right=0.8,
    )
    candles = series(30)
    assert 0 < model.predict(candles) < 1
    save(model, tmp_path)
    assert load(tmp_path, "R_25", 300) == model
    path = tmp_path / "R_25_300.json"
    data = json.loads(path.read_text())
    data["tree_feature"] = 24
    path.write_text(json.dumps(data))
    assert load(tmp_path, "R_25", 300) is None


def test_research_rejects_ohlc_labels_that_do_not_align():
    candles = series(300)
    candles[20] = Candle(candles[20].epoch, candles[20].open, 1, 2, candles[20].close)
    result = assess("R_25", 300, candles)
    assert result.model is None
    assert result.reason == "features OHLC invalidas ou desalinhadas"
