"""Rotulos binarios alinhados a duracao do contrato Rise/Fall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LABEL_MODE_SPOT = "spot_forward"
LABEL_MODE_MA_TREND = "ma_trend"
LABEL_MODE_SUPERTREND_ATR = "supertrend_atr"


@dataclass(frozen=True)
class LabelSpec:
    """Contrato unico de label para treino, deploy e settlement."""

    horizon_bars: int = 1
    smooth_bars: int = 1
    label_mode: str = LABEL_MODE_SPOT
    ma_window: int = 5

    @classmethod
    def from_dl_config(cls, dl_cfg: dict | None) -> LabelSpec:
        """Monta LabelSpec a partir do bloco deep_learning da config."""
        cfg = dl_cfg if isinstance(dl_cfg, dict) else {}
        return cls(
            horizon_bars=max(1, int(cfg.get("label_horizon_bars", 1))),
            smooth_bars=max(1, int(cfg.get("label_smooth_bars", 1))),
            label_mode=str(cfg.get("label_mode", LABEL_MODE_SPOT)),
            ma_window=max(1, int(cfg.get("label_ma_window", 5))),
        )

    @property
    def embargo_bars(self) -> int:
        """Barras de embargo purged = horizon + smooth - 1."""
        return max(1, int(self.horizon_bars) + int(self.smooth_bars) - 1)


def _rolling_mean(prices: np.ndarray, index: int, window: int) -> float:
    """Media movel dos closes terminando na barra index."""
    span = max(1, int(window))
    start = max(0, index - span + 1)
    return float(np.mean(prices[start : index + 1]))


def _forward_mean(prices: np.ndarray, index: int, horizon_bars: int, smooth_bars: int) -> float | None:
    """Media dos closes forward apos horizon ou None se indice invalido."""
    smooth = max(1, int(smooth_bars))
    forward_start = index + max(1, int(horizon_bars))
    forward_end = forward_start + smooth
    if forward_end > len(prices):
        return None
    return float(np.mean(prices[forward_start:forward_end]))


def _regime_threshold(ma_window: int, horizon_bars: int) -> float:
    """Define um limiar minimo de deslocamento percentual para evitar ruido."""
    return max(0.0002, 0.0005 * float(horizon_bars) / float(max(1, ma_window)))


def _supertrend_direction(prices: np.ndarray, index: int, period: int = 10, multiplier: float = 2.0) -> int:
    """Calcula a direcao do SuperTrend na barra index (+1 CALL, -1 PUT)."""
    start = max(0, index - max(period * 3, 30))
    seg = prices[start : index + 1]
    if len(seg) < period + 1:
        return 1 if prices[index] >= prices[max(0, index - 1)] else -1
    diffs = np.abs(np.diff(seg))
    atr = float(np.mean(diffs[-period:]))
    diff_trend = float(seg[-1] - seg[0])
    if diff_trend > atr * multiplier * 0.5:
        return 1
    if diff_trend < -atr * multiplier * 0.5:
        return -1
    return 1 if seg[-1] >= seg[-2] else -1


def binary_label_at_index(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    smooth_bars: int = 1,
    label_mode: str = LABEL_MODE_SPOT,
    ma_window: int = 5,
) -> bool:
    """Retorna True para CALL conforme modo supertrend_atr, ma_trend ou spot_forward."""
    forward = _forward_mean(prices, index, horizon_bars, smooth_bars)
    if forward is None:
        return False
    mode = str(label_mode).strip().lower()
    if mode == LABEL_MODE_SUPERTREND_ATR:
        st_dir = _supertrend_direction(prices, index)
        diff = forward - float(prices[index])
        threshold = _regime_threshold(ma_window, horizon_bars) * float(prices[index])
        if st_dir == 1:
            return diff >= -threshold
        return diff > threshold
    if mode == LABEL_MODE_MA_TREND:
        current = _rolling_mean(prices, index, ma_window)
        threshold = _regime_threshold(ma_window, horizon_bars)
        return forward > current + threshold
    return forward > float(prices[index])


def sequence_labels(
    prices: np.ndarray,
    lookback: int,
    horizon_bars: int,
    *,
    smooth_bars: int = 1,
    label_mode: str = LABEL_MODE_SPOT,
    ma_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera targets binarios e mascara ativa para indices validos."""
    n = len(prices)
    horizon = max(1, int(horizon_bars))
    smooth = max(1, int(smooth_bars))
    tail = horizon + smooth
    last_i = n - horizon - smooth
    if n < lookback + tail or last_i < lookback:
        return np.empty((0,)), np.empty((0,))
    targets = []
    masks = []
    for i in range(lookback, last_i + 1):
        up = binary_label_at_index(
            prices,
            i,
            horizon,
            smooth_bars=smooth,
            label_mode=label_mode,
            ma_window=ma_window,
        )
        targets.append(1.0 if up else 0.0)
        masks.append(1.0)
    return np.array(targets, dtype=np.float32), np.array(masks, dtype=np.float32)
