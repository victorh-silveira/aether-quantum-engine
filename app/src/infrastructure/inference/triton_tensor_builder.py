"""Montagem de tensores de inferencia para Triton a partir de OHLC."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_features import FEATURE_DIM, build_sequence_tensor
from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.model import normalize_sequences


class PartialInferenceHistoryError(ValueError):
    """Historico OHLC insuficiente para montar sequencia de inferencia."""


def resolve_sequence_end_index(prices_len: int, lookback: int) -> int:
    """Retorna indice final da sequencia quando o historico cobre o lookback."""
    if prices_len < lookback:
        raise PartialInferenceHistoryError(f"historico insuficiente: {prices_len} < {lookback}")
    return prices_len - 1


def inference_tensor_fingerprint(tensor: np.ndarray) -> bytes:
    """Gera fingerprint estavel do tensor normalizado de inferencia."""
    return np.asarray(tensor, dtype=np.float32).tobytes()


def build_inference_tensor(
    prices: np.ndarray,
    lookback: int,
    norm_stats: FeatureNormStats,
    *,
    granularity: int = 60,
    symbol: str = "OTC_SPC",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    implied_vol_bars: int = 60,
) -> np.ndarray:
    """Retorna tensor FP32 normalizado com shape [1, lookback, FEATURE_DIM]."""
    end_idx = resolve_sequence_end_index(len(prices), lookback)
    seq = build_sequence_tensor(
        prices,
        lookback,
        end_idx,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=implied_vol_bars,
    ).reshape(1, lookback, FEATURE_DIM)
    feat = normalize_sequences(seq, norm_stats)
    return np.asarray(feat, dtype=np.float32)
