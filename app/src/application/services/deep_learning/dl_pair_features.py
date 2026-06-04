"""Features de par Range R_* para tensor Deep Learning."""

import numpy as np


def align_pair_lengths(prices_a: np.ndarray, prices_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alinha series pelo menor comprimento."""
    n = min(len(prices_a), len(prices_b))
    if n <= 0:
        return np.array([]), np.array([])
    return np.asarray(prices_a[-n:], dtype=np.float64), np.asarray(prices_b[-n:], dtype=np.float64)


def precompute_pair_series(prices_bull: np.ndarray, prices_bear: np.ndarray) -> dict[str, np.ndarray]:
    """Precomputa spread, z-score e correlacao rolling bull/bear."""
    bull, bear = align_pair_lengths(prices_bull, prices_bear)
    n = len(bull)
    spread = np.zeros(n, dtype=np.float64)
    z_spread = np.zeros(n, dtype=np.float64)
    corr = np.zeros(n, dtype=np.float64)
    if n < 2:
        return {"spread": spread, "z_spread": z_spread, "corr": corr}
    spread[1:] = np.log((bull[1:] + 1e-10) / (bear[1:] + 1e-10))
    window = 20
    for i in range(window, n):
        seg = spread[max(0, i - window) : i + 1]
        mu = float(np.mean(seg))
        sd = float(np.std(seg)) + 1e-10
        z_spread[i] = (spread[i] - mu) / sd
        rb = bull[max(0, i - window) : i + 1]
        rk = bear[max(0, i - window) : i + 1]
        if len(rb) > 2 and np.std(rb) > 1e-12 and np.std(rk) > 1e-12:
            corr[i] = float(np.corrcoef(rb, rk)[0, 1])
    return {"spread": spread, "z_spread": z_spread, "corr": corr}


def pair_feature_row(pair_series: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Retorna vetor (3,) de features de par na barra index."""
    return np.array(
        [
            pair_series["spread"][index],
            pair_series["z_spread"][index],
            pair_series["corr"][index],
        ],
        dtype=np.float32,
    )


def spread_confirms_direction(
    prices_sym: np.ndarray,
    prices_peer: np.ndarray,
    index: int,
    *,
    target_up: bool,
    sym_is_bull: bool,
    horizon_bars: int = 1,
) -> bool:
    """Confirma label quando movimento do spread log(bull/bear) alinha ao simbolo treinado."""
    step = max(1, int(horizon_bars))
    j = index + step
    if j >= len(prices_sym) or j >= len(prices_peer):
        return True
    bull, bear = (prices_sym, prices_peer) if sym_is_bull else (prices_peer, prices_sym)
    spread_i = np.log((bull[index] + 1e-10) / (bear[index] + 1e-10))
    spread_j = np.log((bull[j] + 1e-10) / (bear[j] + 1e-10))
    spread_up = spread_j > spread_i
    want_spread_up = bool(target_up) if sym_is_bull else not bool(target_up)
    return spread_up == want_spread_up
