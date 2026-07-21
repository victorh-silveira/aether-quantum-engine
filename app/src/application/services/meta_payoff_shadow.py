"""Telemetria shadow: correlacao rolling entre Z de payoff e PnL realizado."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from src.application.services.infra_timing_config import resolve_meta_classifier_infra_config


logger = logging.getLogger("AETH")


def _shadow() -> dict:
    """Resolve ou aplica  shadow."""
    return resolve_meta_classifier_infra_config()["shadow"]


_z_pnl_pairs: deque[tuple[float, float]] = deque(maxlen=64)


def reset_meta_payoff_shadow() -> None:
    """Resolve ou aplica reset meta payoff shadow."""
    _z_pnl_pairs.clear()


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Resolve ou aplica  pearson."""
    n = len(xs)
    if n < int(_shadow()["min_pairs"]) or n != len(ys):
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
    if n < int(_shadow()["ready_n"]):
        n = shadow_pair_count()
    corr = shadow_correlation(orch)
    return n >= int(_shadow()["ready_n"]) and corr is not None


def meta_hard_veto_allowed(orch: Any | None = None) -> bool:
    """True se correlacao shadow permite hard veto de payoff."""
    n = shadow_pair_count()
    if orch is not None:
        n = max(n, int(getattr(orch, "_meta_payoff_shadow_n", 0) or 0))
    corr = shadow_correlation(orch)
    if corr is None or n < int(_shadow()["ready_n"]):
        return False
    return float(corr) >= float(_shadow()["hard_corr_floor"])


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
    if corr is not None and len(_z_pnl_pairs) >= int(_shadow()["min_pairs"]) and len(_z_pnl_pairs) % 8 == 0:
        logger.info(
            "META_SHADOW | corr(z,pnl)=%+.3f | n=%d | window=%d | hard=%s",
            corr,
            len(_z_pnl_pairs),
            int(_shadow()["window"]),
            str(meta_hard_veto_allowed(orch)).lower(),
        )
    return corr
