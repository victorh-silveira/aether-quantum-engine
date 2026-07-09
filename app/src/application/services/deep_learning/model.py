"""Fachada de modelos Deep Learning: TCN/LSTM, checkpoint e predicao."""

import logging

import numpy as np
import torch
from torch import nn

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator,
    brier_score,
    expected_calibration_error,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM, build_sequence_tensor
from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint, save_model_checkpoint
from src.application.services.deep_learning.dl_model_factory import create_direction_model
from src.application.services.deep_learning.dl_model_types import (
    CHECKPOINT_VERSION,
    DEFAULT_ARCH,
    INPUT_DIM,
    FeatureNormStats,
    TrainResult,
)
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.domain.models.trade import TradeDirection


__all__ = [
    "CHECKPOINT_VERSION",
    "DEFAULT_ARCH",
    "FeatureNormStats",
    "INPUT_DIM",
    "MarketDirectionClassifier",
    "TrainResult",
    "_accuracy",
    "create_direction_model",
    "evaluate_calibrated_metrics",
    "fit_norm_stats",
    "load_model_checkpoint",
    "model_accuracy",
    "normalize_features",
    "normalize_sequences",
    "predict_next_direction",
    "save_model_checkpoint",
]


torch.set_num_threads(1)

logger = logging.getLogger("AETH")


class MarketDirectionClassifier(nn.Module):
    """Alias legado apontando para o classificador TCN."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.inner = TemporalDirectionClassifier(input_dim=input_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        logits: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Delega forward para o classificador TCN interno."""
        return self.inner(x, logits=logits, return_aux=return_aux)


def fit_norm_stats(x: np.ndarray) -> FeatureNormStats:
    """Estima media e desvio por dimensao a partir das amostras de treino."""
    flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return FeatureNormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def normalize_features(x: np.ndarray, stats: FeatureNormStats) -> np.ndarray:
    """Aplica z-score com clip para limitar outliers nas features."""
    z = (x - stats.mean) / stats.std
    return np.clip(z, -5.0, 5.0).astype(np.float32)


def normalize_sequences(x: np.ndarray, stats: FeatureNormStats) -> np.ndarray:
    """Normaliza tensor (N, L, F) ou matriz (N, F)."""
    if x.ndim == 3:
        return normalize_features(x.reshape(-1, x.shape[-1]), stats).reshape(x.shape)
    return normalize_features(x, stats)


def _sanitize_feature_batch(batch: np.ndarray) -> np.ndarray:
    """Remove NaN e infinitos das features antes do forward."""
    arr = np.asarray(batch, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _model_raw_prob(model: nn.Module, batch: np.ndarray) -> np.ndarray:
    """Executa forward e retorna probabilidades brutas."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.as_tensor(_sanitize_feature_batch(batch), dtype=torch.float32, device=device)
        preds = model(tensor)
        flat = preds.squeeze(-1)
        flat = torch.nan_to_num(flat, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return flat.detach().cpu().numpy().astype(np.float32)


def model_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Calcula acuracia classica no conjunto informado."""
    if len(x) == 0:
        return 0.0
    probs = _model_raw_prob(model, x)
    predicted = (probs >= 0.5).astype(np.float32)
    if mask is not None and len(mask) == len(y):
        active = mask > 0.5
        n_active = int(active.sum())
        if n_active == 0:
            return 0.0
        return float((predicted[active] == y[active]).mean())
    return float((predicted == y).mean())


def predict_next_direction(
    model: nn.Module,
    prices: np.ndarray,
    lookback: int,
    norm_stats: FeatureNormStats | None = None,
    *,
    granularity: int = 60,
    symbol: str = "RDBULL",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    implied_vol_bars: int = 60,
    call_threshold: float = 0.75,
    put_threshold: float = 0.25,
    calibrator: CalibratorState | None = None,
) -> tuple[TradeDirection | None, float, float]:
    """Prediz direcao via threshold de confianca sobre probabilidade bruta."""
    n = len(prices)
    if n < lookback:
        return None, 0.5, 0.5
    seq = build_sequence_tensor(
        prices,
        lookback,
        n - 1,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=implied_vol_bars,
    ).reshape(1, lookback, FEATURE_DIM)
    if norm_stats is None:
        norm_stats = fit_norm_stats(seq)
    feat = normalize_sequences(seq, norm_stats)
    raw_prob = float(_model_raw_prob(model, feat)[0])
    prob = apply_calibrator(raw_prob, calibrator) if calibrator is not None else raw_prob
    if prob + 1e-9 >= float(call_threshold):
        return TradeDirection.CALL, prob, raw_prob
    if prob - 1e-9 <= float(put_threshold):
        return TradeDirection.PUT, prob, raw_prob
    return None, prob, raw_prob


def evaluate_calibrated_metrics(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    calibrator: CalibratorState,
) -> tuple[float, float]:
    """Calcula Brier e ECE apos calibracao Platt."""
    raw = _model_raw_prob(model, x)
    calibrated = [apply_calibrator(float(p), calibrator) for p in raw]
    labels = [float(v) for v in y]
    return brier_score(calibrated, labels), expected_calibration_error(calibrated, labels)


_accuracy = model_accuracy
