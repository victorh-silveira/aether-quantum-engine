"""Modelo direcional simples, versionado e validado no tempo."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from indicator.domain import Candle

LOOKBACK = 8


@dataclass(frozen=True)
class Model:
    """Pesos e proveniencia de um modelo para um par ativo/periodo."""

    symbol: str
    period: int
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    version: str
    validation_accuracy: float
    baseline_accuracy: float
    validation_brier: float
    baseline_brier: float

    def predict(self, candles: list[Candle]) -> float:
        """Retorna P(CALL) usando apenas velas fechadas."""
        values = features(candles)
        normalized = [
            (value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))
        return sigmoid(score)


def sigmoid(value: float) -> float:
    """Sigmoide numericamente estavel."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    positive = math.exp(value)
    return positive / (1 + positive)


def features(candles: list[Candle]) -> tuple[float, ...]:
    """Retornos causais das ultimas oito velas."""
    if len(candles) < LOOKBACK + 1:
        raise ValueError("historico insuficiente")
    closes = [candle.close for candle in candles[-LOOKBACK - 1 :]]
    if any(close <= 0 or not math.isfinite(close) for close in closes):
        raise ValueError("preco invalido")
    return tuple(math.log(closes[index + 1] / closes[index]) for index in range(LOOKBACK))


def examples(candles: list[Candle], period: int) -> tuple[list[tuple[float, ...]], list[int]]:
    """Monta amostras de uma vela a frente sem vazamento temporal."""
    inputs, labels = [], []
    for index in range(LOOKBACK, len(candles) - 1):
        window = candles[index - LOOKBACK : index + 1]
        try:
            vector = features(window)
        except ValueError:
            continue
        if (
            candles[index + 1].epoch - candles[index].epoch != period
            or any(window[offset + 1].epoch - window[offset].epoch != period for offset in range(LOOKBACK))
            or candles[index + 1].close == candles[index].close
        ):
            continue
        inputs.append(vector)
        labels.append(int(candles[index + 1].close > candles[index].close))
    return inputs, labels


def train(symbol: str, period: int, candles: list[Candle]) -> Model | None:
    """Aceita modelo somente se superar a maioria temporal em ACC e Brier."""
    inputs, labels = examples(candles, period)
    if len(labels) < 200:
        return None
    split = int(len(labels) * 0.7)
    train_x, val_x = inputs[:split], inputs[split:]
    train_y, val_y = labels[:split], labels[split:]
    means = tuple(sum(row[index] for row in train_x) / len(train_x) for index in range(LOOKBACK))
    scales = tuple(
        max((sum((row[index] - means[index]) ** 2 for row in train_x) / len(train_x)) ** 0.5, 1e-8)
        for index in range(LOOKBACK)
    )
    normalized = [tuple((value - means[index]) / scales[index] for index, value in enumerate(row)) for row in train_x]
    weights = [0.0] * LOOKBACK
    bias = 0.0
    for _ in range(200):
        gradient = [0.0] * LOOKBACK
        bias_gradient = 0.0
        for row, label in zip(normalized, train_y, strict=True):
            residual = sigmoid(bias + sum(weight * value for weight, value in zip(weights, row, strict=True))) - label
            bias_gradient += residual
            for index, value in enumerate(row):
                gradient[index] += residual * value
        bias -= 0.1 * bias_gradient / len(train_y)
        weights = [
            weight - 0.1 * (gradient[index] / len(train_y) + 0.001 * weight) for index, weight in enumerate(weights)
        ]
    baseline = sum(train_y) / len(train_y)
    predictions = [
        sigmoid(
            bias
            + sum(
                weight * (value - means[index]) / scales[index]
                for index, (weight, value) in enumerate(zip(weights, row, strict=True))
            )
        )
        for row in val_x
    ]
    accuracy = sum(
        (prediction >= 0.5) == bool(label) for prediction, label in zip(predictions, val_y, strict=True)
    ) / len(val_y)
    baseline_accuracy = sum((baseline >= 0.5) == bool(label) for label in val_y) / len(val_y)
    brier = sum((prediction - label) ** 2 for prediction, label in zip(predictions, val_y, strict=True)) / len(val_y)
    baseline_brier = sum((baseline - label) ** 2 for label in val_y) / len(val_y)
    if accuracy <= baseline_accuracy or brier >= baseline_brier:
        return None
    payload = f"{symbol}:{period}:{candles[-1].epoch}:{weights}:{bias}"
    version = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return Model(
        symbol, period, tuple(weights), bias, means, scales, version, accuracy, baseline_accuracy, brier, baseline_brier
    )


def save(model: Model, directory: Path) -> Path:
    """Persiste modelo por ativo e periodo com escrita atomica."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.symbol}_{model.period}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(model), sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def load(directory: Path, symbol: str, period: int) -> Model | None:
    """Carrega apenas artefato compativel e finito."""
    path = directory / f"{symbol}_{period}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = Model(
            **{
                **data,
                "weights": tuple(data["weights"]),
                "means": tuple(data["means"]),
                "scales": tuple(data["scales"]),
            }
        )
        numeric = (
            *model.weights,
            model.bias,
            *model.means,
            *model.scales,
            model.validation_accuracy,
            model.baseline_accuracy,
            model.validation_brier,
            model.baseline_brier,
        )
        if (
            model.symbol != symbol
            or model.period != period
            or len(model.weights) != LOOKBACK
            or len(model.means) != LOOKBACK
            or len(model.scales) != LOOKBACK
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0 for value in model.scales)
            or model.validation_accuracy <= model.baseline_accuracy
            or model.validation_brier >= model.baseline_brier
            or not model.version
        ):
            return None
        return model
    except (OSError, ValueError, TypeError, KeyError):
        return None
