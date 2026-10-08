"""Treino temporal e compatibilidade de artefatos."""

import json
import math

import pytest

from indicator.domain import Candle
from indicator.model import LOOKBACK, Model, examples, features, load, save, sigmoid, train


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
    assert sigmoid(1000) == 1
    assert sigmoid(-1000) == 0
    assert len(examples(candles, 300)[0]) == 21
    with pytest.raises(ValueError):
        features(candles[:8])
    with pytest.raises(ValueError):
        features([Candle(index, 0, 0, 0, 0) for index in range(9)])
    broken = list(candles)
    broken[10] = Candle(99999, 100, 100, 100, 100)
    assert len(examples(broken, 300)[0]) < 21


def test_train_save_load_prediction_and_rejection(tmp_path):
    candles = series()
    model = train("R_75", 300, candles)
    assert model is not None
    assert model.validation_accuracy > model.baseline_accuracy
    assert model.validation_brier < model.baseline_brier
    assert 0 < model.predict(candles) < 1
    path = save(model, tmp_path)
    assert load(tmp_path, "R_75", 300) == model
    assert load(tmp_path, "R_75", 900) is None
    assert train("R_75", 300, candles[:100]) is None
    data = json.loads(path.read_text())
    data["weights"] = [math.inf] * 8
    path.write_text(json.dumps(data))
    assert load(tmp_path, "R_75", 300) is None
    path.write_text("invalid")
    assert load(tmp_path, "R_75", 300) is None


def test_model_rejects_majority_only_and_invalid_shape(tmp_path):
    candles = [Candle(index * 300, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(300)]
    assert train("R_75", 300, candles) is None
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
