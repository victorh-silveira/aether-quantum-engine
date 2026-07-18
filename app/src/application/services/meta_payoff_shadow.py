"""Telemetria shadow: correlacao rolling entre Z de payoff e PnL realizado."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any


logger = logging.getLogger("AETH")

_SHADOW_WINDOW = 64
_MIN_PAIRS = 12
_SHADOW_READY_N = 64
_HARD_CORR_FLOOR = 0.15
_SOFT_ONLY_CORR_CEILING = 0.05
_z_pnl_pairs: deque[tuple[float, float]] = deque(maxlen=_SHADOW_WINDOW)


def reset_meta_payoff_shadow() -> None:
    """Limpa o buffer rolling de pares (z_score, pnl)."""
    _z_pnl_pairs.clear()


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Correlacao de Pearson ou None se amostra insuficiente."""
    n = len(xs)
    if n < _MIN_PAIRS or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    return float(cov / math.sqrt(var_x * var_y))


def shadow_pair_count() -> int:
    """Quantidade de pares shadow acumulados na janela."""
    return len(_z_pnl_pairs)


def shadow_correlation(orch: Any | None = None) -> float | None:
    """Correlacao shadow atual, preferindo cache no orquestrador."""
    if orch is not None:
        corr = getattr(orch, "_meta_payoff_shadow_corr", None)
        if corr is not None:
            return float(corr)
    xs = [pair[0] for pair in _z_pnl_pairs]
    ys = [pair[1] for pair in _z_pnl_pairs]
    return _pearson(xs, ys)


def shadow_ready(orch: Any | None = None) -> bool:
    """True quando ha N minimo e correlacao calculavel."""
    n = int(getattr(orch, "_meta_payoff_shadow_n", 0) or 0) if orch is not None else shadow_pair_count()
    if n < _SHADOW_READY_N:
        n = shadow_pair_count()
    corr = shadow_correlation(orch)
    return n >= _SHADOW_READY_N and corr is not None


def meta_hard_veto_allowed(orch: Any | None = None) -> bool:
    """True se correlacao shadow permite hard veto de payoff."""
    n = shadow_pair_count()
    if orch is not None:
        n = max(n, int(getattr(orch, "_meta_payoff_shadow_n", 0) or 0))
    corr = shadow_correlation(orch)
    if corr is None or n < _SHADOW_READY_N:
        return False
    return float(corr) >= _HARD_CORR_FLOOR


def record_meta_payoff_shadow_pair(
    *,
    z_score: float | None,
    profit: float,
    orch: Any | None = None,
) -> float | None:
    """Registra par (z, pnl) e atualiza correlacao/cache no orch."""
    if z_score is None:
        return None
    _z_pnl_pairs.append((float(z_score), float(profit)))
    xs = [pair[0] for pair in _z_pnl_pairs]
    ys = [pair[1] for pair in _z_pnl_pairs]
    corr = _pearson(xs, ys)
    if orch is not None:
        orch._meta_payoff_shadow_corr = corr
        orch._meta_payoff_shadow_n = len(_z_pnl_pairs)
        orch._meta_payoff_shadow_ready = shadow_ready(orch)
        orch._meta_payoff_hard_veto_allowed = meta_hard_veto_allowed(orch)
    if corr is not None and len(_z_pnl_pairs) >= _MIN_PAIRS and len(_z_pnl_pairs) % 8 == 0:
        logger.info(
            "META_SHADOW | corr(z,pnl)=%+.3f | n=%d | window=%d | hard=%s",
            corr,
            len(_z_pnl_pairs),
            _SHADOW_WINDOW,
            str(meta_hard_veto_allowed(orch)).lower(),
        )
    return corr
