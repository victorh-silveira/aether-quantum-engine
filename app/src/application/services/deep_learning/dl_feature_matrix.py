"""Montagem de linhas, matrizes e tensores de features DL ortogonais 14D."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_orthogonal import (
    FEATURE_DIM,
    build_orthogonal_feature_matrix,
    build_orthogonal_feature_row,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


def _causal_norm_knobs(indicator_cfg: dict | None = None) -> tuple[int, float]:
    cfg = indicator_cfg if isinstance(indicator_cfg, dict) else load_indicator_config_from_settings()
    norm = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
    window = int(norm.get("causal_norm_window", 288))
    clip = float(norm.get("causal_norm_clip", 3.0))
    return window, clip


def build_feature_row(
    series: dict[str, np.ndarray],
    index: int,
    *,
    macro_closes: np.ndarray | None = None,
    causal_norm_window: int | None = None,
    causal_norm_clip: float | None = None,
) -> np.ndarray:
    """Monta vetor ortogonal 14D na barra indicada."""
    window, clip = _causal_norm_knobs()
    if causal_norm_window is not None:
        window = int(causal_norm_window)
    if causal_norm_clip is not None:
        clip = float(causal_norm_clip)
    return build_orthogonal_feature_row(
        series,
        index,
        macro_closes=macro_closes,
        causal_norm_window=window,
        causal_norm_clip=clip,
    )


def build_feature_matrix(
    series: dict[str, np.ndarray],
    *,
    macro_closes: np.ndarray | None = None,
    causal_norm_window: int | None = None,
    causal_norm_clip: float | None = None,
) -> np.ndarray:
    """Monta matriz (N, 14) com uma linha de features por barra."""
    window, clip = _causal_norm_knobs()
    if causal_norm_window is not None:
        window = int(causal_norm_window)
    if causal_norm_clip is not None:
        clip = float(causal_norm_clip)
    return build_orthogonal_feature_matrix(
        series,
        macro_closes=macro_closes,
        causal_norm_window=window,
        causal_norm_clip=clip,
    )


def build_sequence_tensor(
    prices: np.ndarray,
    lookback: int,
    end_index: int,
    *,
    granularity: int = 60,
    symbol: str = "1HZ75V",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    implied_vol_bars: int = 60,
    macro_closes: np.ndarray | None = None,
) -> np.ndarray:
    """Monta janela (lookback, 14) terminando em end_index inclusive."""
    series = precompute_price_series(
        prices,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=implied_vol_bars,
        macro_closes=macro_closes,
    )
    matrix = build_feature_matrix(series, macro_closes=macro_closes)
    start = end_index - lookback + 1
    return matrix[start : end_index + 1].astype(np.float32)


__all__ = [
    "FEATURE_DIM",
    "build_feature_row",
    "build_feature_matrix",
    "build_sequence_tensor",
]
