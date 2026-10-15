"""Modelo direcional simples, versionado e validado no tempo."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from indicator.domain import Candle
from indicator.features import LOOKBACK, features, ohlc_features

REFERENCE_PAYOUT_RATE = 0.78
REFERENCE_PROBABILITY_MARGIN = 0.01


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
    status: str = "aprovado"
    calibration_slope: float = 1.0
    calibration_offset: float = 0.0
    regularization: float = 0.001
    feature_set: str = "close"
    tree_feature: int = -1
    tree_threshold: float = 0.0
    tree_left: float = 0.5
    tree_right: float = 0.5
    validation_protocol: str = "legacy"
    validation_ece: float = 1.0
    brier_improvement_lower95: float = 0.0

    def predict(self, candles: list[Candle]) -> float:
        """Retorna P(CALL) usando apenas velas fechadas."""
        values = ohlc_features(candles) if self.feature_set in {"ohlc", "tree"} else features(candles)
        if self.feature_set == "tree":
            probability = self.tree_left if values[self.tree_feature] <= self.tree_threshold else self.tree_right
            score = math.log(probability / (1.0 - probability))
            return sigmoid(self.calibration_slope * score + self.calibration_offset)
        normalized = [
            (value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))
        return sigmoid(self.calibration_slope * score + self.calibration_offset)


@dataclass(frozen=True)
class TrainingResult:
    """Decisao do treino e medidas da validacao temporal."""

    model: Model | None
    samples: int
    validation_samples: int = 0
    accuracy: float | None = None
    baseline_accuracy: float | None = None
    brier: float | None = None
    baseline_brier: float | None = None
    training_steps: int = 0
    ev_eligible_samples: int = 0
    maximum_side_probability: float = 0.5
    candidate_briers: tuple[tuple[float, float], ...] = ()
    calibration_ece: float | None = None
    improvement_lower95: float | None = None
    probability_quantiles: tuple[float, float, float] = (0.5, 0.5, 0.5)
    regime_briers: tuple[tuple[str, int, float], ...] = ()
    failure_reason: str | None = None

    @property
    def reason(self) -> str:
        if self.failure_reason is not None:
            return self.failure_reason
        if self.samples < 200:
            return f"amostras insuficientes ({self.samples}/200)"
        if self.model is not None and self.model.status == "experimental":
            return (
                f"experimental: sem ganho temporal (n_val={self.validation_samples} "
                f"acc={self.accuracy:.5f} ref={self.baseline_accuracy:.5f} "
                f"brier={self.brier:.5f} ref={self.baseline_brier:.5f} "
                f"ece={self.calibration_ece:.5f} lcb={self.improvement_lower95:.5f})"
            )
        return "aprovado"


def sigmoid(value: float) -> float:
    """Sigmoide numericamente estavel."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    positive = math.exp(value)
    return positive / (1 + positive)


def examples(
    candles: list[Candle], period: int, feature_set: str = "close"
) -> tuple[list[tuple[float, ...]], list[int]]:
    """Monta amostras de uma vela a frente sem vazamento temporal."""
    inputs, labels = [], []
    for index in range(LOOKBACK, len(candles) - 1):
        window = candles[index - LOOKBACK : index + 1]
        try:
            vector = ohlc_features(window) if feature_set in {"ohlc", "tree"} else features(window)
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
    model = assess(symbol, period, candles).model
    return model if model is not None and model.status == "aprovado" else None


def assess(symbol: str, period: int, candles: list[Candle]) -> TrainingResult:
    """Avalia o modelo e expoe as medidas mesmo quando ele e reprovado."""
    from indicator.research_pipeline import run_research

    inputs, labels = examples(candles, period)
    if len(labels) < 200:
        return TrainingResult(None, len(labels))
    ohlc_rows, ohlc_labels = examples(candles, period, "ohlc")
    if ohlc_labels != labels:
        return TrainingResult(None, len(ohlc_labels), failure_reason="features OHLC invalidas ou desalinhadas")
    research = run_research(inputs, ohlc_rows, labels)
    fitted = research.fitted
    side_probabilities = [max(prediction, 1.0 - prediction) for prediction in research.predictions]
    minimum_probability = 1.0 / (1.0 + REFERENCE_PAYOUT_RATE) + REFERENCE_PROBABILITY_MARGIN
    ev_eligible_samples = sum(probability > minimum_probability for probability in side_probabilities)
    status = (
        "aprovado"
        if research.brier < research.baseline_brier and research.improvement_lower95 > 0 and research.ece <= 0.05
        else "experimental"
    )
    payload = f"{symbol}:{period}:{candles[-1].epoch}:{fitted}:{status}:{research.slope}:{research.offset}"
    version = hashlib.sha256(payload.encode()).hexdigest()[:16]
    model = Model(
        symbol,
        period,
        fitted.weights,
        fitted.bias,
        fitted.means,
        fitted.scales,
        version,
        research.accuracy,
        research.baseline_accuracy,
        research.brier,
        research.baseline_brier,
        status,
        research.slope,
        research.offset,
        fitted.regularization,
        fitted.feature_set,
        fitted.tree_feature,
        fitted.tree_threshold,
        fitted.tree_left,
        fitted.tree_right,
        "wf3_blockbootstrap_v1",
        research.ece,
        research.improvement_lower95,
    )
    return TrainingResult(
        model,
        len(labels),
        research.validation_samples,
        research.accuracy,
        research.baseline_accuracy,
        research.brier,
        research.baseline_brier,
        fitted.steps,
        ev_eligible_samples,
        max(side_probabilities),
        research.candidate_briers,
        research.ece,
        research.improvement_lower95,
        research.probability_quantiles,
        research.regime_briers,
    )


def save(model: Model, directory: Path) -> Path:
    """Persiste modelo por ativo e periodo com escrita atomica."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.symbol}_{model.period}.json"
    temporary = directory / f".{model.symbol}_{model.period}_{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(asdict(model), sort_keys=True), encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.2)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load(directory: Path, symbol: str, period: int, *, allow_experimental: bool = False) -> Model | None:
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
            model.calibration_slope,
            model.calibration_offset,
            model.regularization,
            model.tree_threshold,
            model.tree_left,
            model.tree_right,
            model.validation_ece,
            model.brier_improvement_lower95,
        )
        if (
            model.symbol != symbol
            or model.period != period
            or model.feature_set not in {"close", "ohlc", "tree"}
            or len(model.weights) != (LOOKBACK if model.feature_set == "close" else LOOKBACK * 3)
            or len(model.means) != len(model.weights)
            or len(model.scales) != len(model.weights)
            or (model.feature_set == "tree" and not 0 <= model.tree_feature < LOOKBACK * 3)
            or (model.feature_set == "tree" and not 0 < model.tree_left < 1)
            or (model.feature_set == "tree" and not 0 < model.tree_right < 1)
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0 for value in model.scales)
            or not 0.0 <= model.calibration_slope <= 5.0
            or (model.feature_set != "tree" and model.regularization <= 0.0)
            or model.status not in {"aprovado", "experimental"}
            or (model.status == "experimental" and not allow_experimental)
            or (
                model.status == "aprovado"
                and model.validation_protocol == "wf3_blockbootstrap_v1"
                and (model.validation_ece > 0.05 or model.brier_improvement_lower95 <= 0)
            )
            or (
                model.status == "aprovado"
                and model.validation_protocol == "legacy"
                and model.validation_accuracy <= model.baseline_accuracy
            )
            or (model.status == "aprovado" and model.validation_brier >= model.baseline_brier)
            or not model.version
        ):
            return None
        return model
    except (OSError, ValueError, TypeError, KeyError):
        return None
