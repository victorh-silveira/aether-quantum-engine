"""Fachada de modelos Deep Learning: TCN, checkpoint v2 e predicao."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator,
    brier_score,
    calibrate_trade_score,
    calibrator_from_dict,
    calibrator_to_dict,
    expected_calibration_error,
)
from src.application.services.deep_learning.dl_features import (
    FEATURE_DIM,
    build_sequence_tensor,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.domain.models.trade import TradeDirection


torch.set_num_threads(1)

logger = logging.getLogger("AETH")

INPUT_DIM = FEATURE_DIM
DEFAULT_ARCH = "tcn"


@dataclass
class FeatureNormStats:
    """Media e desvio usados na normalizacao z-score das features."""

    mean: np.ndarray
    std: np.ndarray


@dataclass
class TrainResult:
    """Resultado de um ciclo de treino walk-forward com validacao."""

    avg_loss: float
    val_accuracy: float
    norm_stats: FeatureNormStats
    temperature: float = 1.0
    calibrator: CalibratorState | None = None
    val_brier: float = 1.0
    val_ece: float = 1.0


class MarketDirectionClassifier(nn.Module):
    """Alias legado apontando para o classificador TCN."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.inner = TemporalDirectionClassifier(input_dim=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delega forward para o classificador TCN interno."""
        return self.inner(x)


def create_direction_model(*, arch: str = DEFAULT_ARCH, input_dim: int = FEATURE_DIM) -> nn.Module:
    """Fabrica modelo de direcao conforme arquitetura configurada."""
    if arch == "tcn":
        return TemporalDirectionClassifier(input_dim=input_dim)
    return MarketDirectionClassifier(input_dim=input_dim)


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


def _model_raw_prob(model: nn.Module, batch: np.ndarray) -> np.ndarray:
    """Executa forward e retorna probabilidades brutas."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(batch)
        preds = model(tensor)
        return preds.squeeze(-1).numpy().astype(np.float32)


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
        if n_active >= 8:
            return float((predicted[active] == y[active]).mean())
        return float((predicted[active] == y[active]).mean())
    return float((predicted == y).mean())


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    norm_stats: FeatureNormStats,
    last_candle_epoch: int,
    *,
    lookback: int,
    calibrator: CalibratorState | None = None,
    arch: str = DEFAULT_ARCH,
    val_accuracy: float | None = None,
    val_brier: float | None = None,
    val_ece: float | None = None,
    deploy_ok: bool | None = None,
    deploy_win_rate: float | None = None,
    granularity: int | None = None,
) -> None:
    """Persiste checkpoint v2 com calibrador e metadados de arquitetura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = calibrator or CalibratorState()
    payload = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "norm_mean": norm_stats.mean,
        "norm_std": norm_stats.std,
        "feature_dim": FEATURE_DIM,
        "lookback": int(lookback),
        "last_candle_epoch": last_candle_epoch,
        "calibrator": calibrator_to_dict(cal),
    }
    if val_accuracy is not None:
        payload["val_accuracy"] = float(val_accuracy)
    if val_brier is not None:
        payload["val_brier"] = float(val_brier)
    if val_ece is not None:
        payload["val_ece"] = float(val_ece)
    if deploy_ok is not None:
        payload["deploy_ok"] = bool(deploy_ok)
    if deploy_win_rate is not None:
        payload["deploy_win_rate"] = float(deploy_win_rate)
    if granularity is not None:
        payload["granularity"] = int(granularity)
    torch.save(payload, path)


def load_model_checkpoint(
    path: Path,
) -> tuple[nn.Module, FeatureNormStats, int, CalibratorState, int, float, float, float, bool, float] | None:
    """Carrega checkpoint v2 ou descarta formatos incompativeis."""
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)  # nosec B614
    except Exception:
        logger.debug("DL: Checkpoint corrompido em %s; sera reiniciado.", path)
        return None
    if not isinstance(payload, dict) or "state_dict" not in payload:
        return None
    arch = str(payload.get("arch", "mlp"))
    feature_dim = int(payload.get("feature_dim", payload.get("input_dim", FEATURE_DIM)))
    if arch != DEFAULT_ARCH or feature_dim != FEATURE_DIM:
        logger.debug("DL: Checkpoint legado/incompativel em %s; sera reiniciado.", path)
        return None
    model = create_direction_model(arch=arch, input_dim=feature_dim)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError:
        logger.debug("DL: state_dict incompativel em %s; sera reiniciado.", path)
        return None
    norm_stats = FeatureNormStats(
        mean=np.asarray(payload["norm_mean"], dtype=np.float32),
        std=np.asarray(payload["norm_std"], dtype=np.float32),
    )
    epoch = int(payload.get("last_candle_epoch", 0))
    calibrator = calibrator_from_dict(payload.get("calibrator"))
    lookback = int(payload.get("lookback", 32))
    val_accuracy = float(payload.get("val_accuracy", 0.0))
    val_brier = float(payload.get("val_brier", 1.0))
    val_ece = float(payload.get("val_ece", 1.0))
    deploy_ok = bool(payload.get("deploy_ok", False))
    deploy_win_rate = float(payload.get("deploy_win_rate", 0.0))
    return model, norm_stats, epoch, calibrator, lookback, val_accuracy, val_brier, val_ece, deploy_ok, deploy_win_rate


def predict_next_direction(
    model: nn.Module,
    prices: np.ndarray,
    lookback: int,
    norm_stats: FeatureNormStats | None = None,
    *,
    val_accuracy: float = 0.5,
    calibrator: CalibratorState | None = None,
    temperature: float = 1.0,
    max_calibrated_raw_gap: float = 0.25,
    min_direction_margin: float = 0.10,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    deploy_ok: bool = False,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> tuple[TradeDirection | None, float, float, float]:
    """Prediz direcao a partir do raw com margem minima; score calibrado do lado escolhido."""
    n = len(prices)
    if n < lookback:
        return None, 0.5, 0.5, 1.0
    seq = build_sequence_tensor(
        prices,
        lookback,
        n - 1,
        granularity=granularity,
        pair_prices=pair_prices,
        open_=open_,
        high=high,
        low=low,
    ).reshape(1, lookback, FEATURE_DIM)
    if norm_stats is None:
        norm_stats = fit_norm_stats(seq)
    feat = normalize_sequences(seq, norm_stats)
    raw_prob = float(_model_raw_prob(model, feat)[0])
    margin = max(0.0, float(min_direction_margin))
    raw_side = max(raw_prob, 1.0 - raw_prob)
    if margin > 0 and raw_side + 1e-9 < 0.5 + margin:
        return None, 0.5, 0.5, raw_prob
    cal = calibrator or CalibratorState(temperature=temperature)
    gap_kw = {"max_calibrated_raw_gap": max_calibrated_raw_gap, "deploy_ok": deploy_ok}
    if raw_prob > 0.5:
        side_score = calibrate_trade_score(raw_prob, val_accuracy, cal, is_put=False, **gap_kw)
        return TradeDirection.CALL, side_score, side_score, raw_prob
    side_score = calibrate_trade_score(1.0 - raw_prob, val_accuracy, cal, is_put=True, **gap_kw)
    return TradeDirection.PUT, side_score, side_score, raw_prob


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


def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    """Wrapper legado para acuracia de classificacao."""
    return model_accuracy(model, x, y)


_precompute_price_series = precompute_price_series
