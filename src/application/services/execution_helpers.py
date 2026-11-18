"""Helpers consolidados de execucao: cache EMA, banda neutra, payout, vol burst e recovery."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.deep_learning.dl_gating import MARKET_PAYOUT_SSOT
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.execution_runtime_config import resolve_volatility_booster_config
from src.domain.models.trade import TradeDirection


_STATE: dict[str, object] = {"cycle": None, "cache": {}}


def invalidate_ema_cache(cycle_id: int | None = None) -> None:
    """Limpa o cache de EMA; chamar no inicio de cada ciclo do orquestrador."""
    cache = _STATE["cache"]
    if isinstance(cache, dict):
        cache.clear()
    _STATE["cycle"] = cycle_id


def resolve_calibration_neutral_band(calibration_cfg: dict[str, Any] | None) -> tuple[float, float]:
    """Resolve banda neutra de calibracao a partir de drift ou half-width."""
    raw = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    drift = raw.get("calibration_neutral_drift")
    if isinstance(drift, (list, tuple)) and len(drift) >= 2:
        lo = float(drift[0])
        hi = float(drift[1])
        if hi > lo + 1e-12:
            return lo, hi
    half_raw = raw.get("neutral_half_width", raw.get("neutral_calibration_half_width", 0.04))
    half = float(half_raw)
    return 0.5 - half, 0.5 + half


def resolve_execution_payout(orch: Any | None) -> float:
    """Prioriza a ultima cotacao valida da sessao e falha para o SSOT configurado."""
    th = getattr(orch, "trade_handler", None)
    if th is not None:
        rate = getattr(th, "latest_payout_rate", None)
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate) and rate > 0.0:
            return float(rate)
    params = getattr(getattr(orch, "risk_manager", None), "risk_params", None)
    if isinstance(params, dict):
        try:
            payout = float(params.get("observed_payout_rate", params.get("payout_estimate")))
            if math.isfinite(payout) and payout > 0.0:
                return payout
        except (TypeError, ValueError):
            pass
    return MARKET_PAYOUT_SSOT


def _macro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio macro H1 com fallback para indicadores micro."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro.get("vol_ratio") is not None:
        return float(macro["vol_ratio"])
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", 1.0))


def _micro_bb_width(metrics: dict) -> float:
    """Le bb_width micro M2 dos indicadores de execucao."""
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("bb_width", 0.0))


def volatility_burst_active(metrics: dict, *, vol_burst: dict | None = None) -> bool:
    """True quando macro H1 e micro M2 indicam explosao de volatilidade direcional."""
    cfg = vol_burst if isinstance(vol_burst, dict) else load_indicator_config_from_settings()["vol_burst"]
    return _macro_vol_ratio(metrics) > float(cfg["macro_vol_ratio_min"]) and _micro_bb_width(metrics) > float(
        cfg["micro_bb_width_min"]
    )


def apply_volatility_vol_booster(
    metrics: dict,
    *,
    mandatory_min_trade_score: float,
    min_edge_execute: float,
) -> tuple[float, float]:
    """Afrouxa pisos de score e edge em regime de estouro de volatilidade."""
    if not volatility_burst_active(metrics):
        return float(mandatory_min_trade_score), float(min_edge_execute)
    metrics["volatility_vol_booster"] = True
    boost = resolve_volatility_booster_config()
    return (
        min(float(mandatory_min_trade_score), float(boost["mandatory_score"])),
        min(float(min_edge_execute), float(boost["min_edge"])),
    )


def pending_recovery_active(pending_loss: dict, consecutive_losses_linear: int = 0) -> bool:
    """True quando ha perda pendente ou linear D'Alembert ativo."""
    linear = (
        int(consecutive_losses_linear)
        if isinstance(consecutive_losses_linear, (int, float)) and not isinstance(consecutive_losses_linear, bool)
        else 0
    )
    if linear > 0:
        return True
    if not isinstance(pending_loss, dict):
        return False
    try:
        return sum(float(v) for v in pending_loss.values()) > 0.0
    except (TypeError, ValueError):
        return False


def recovery_blocked_symbols(risk_manager: Any, kelly_config: dict) -> frozenset[str]:
    """Simbolos bloqueados por rotacao apos perdas consecutivas."""
    linear = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    rotation_cycles = int(kelly_config.get("symbol_loss_rotation_cycles", 1))
    last = getattr(risk_manager, "last_loss_symbol", None)
    if linear >= rotation_cycles and last:
        return frozenset({str(last)})
    return frozenset()


def recovery_candidate_pool(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    recovery_active: bool,
    skip_symbols: frozenset[str] | None = None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Filtra o pool de candidatos sob recovery ativo."""
    _ = last_loss_symbol
    pool = list(candidates)
    if not recovery_active:
        return pool
    skip = skip_symbols or frozenset()
    if skip:
        pool = [item for item in pool if item[0] not in skip]
    return pool
