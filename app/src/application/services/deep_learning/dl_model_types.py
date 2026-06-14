"""Tipos e constantes compartilhados pelos modulos de modelo DL."""

from dataclasses import dataclass

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_features import FEATURE_DIM


INPUT_DIM = FEATURE_DIM
DEFAULT_ARCH = "tcn"
CHECKPOINT_VERSION = 3


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
