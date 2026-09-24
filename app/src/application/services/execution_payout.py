"""Resolve o payout usado no edge antes de enviar uma nova ordem."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_gating import MARKET_PAYOUT_SSOT


def resolve_execution_payout(orch: Any | None) -> float:
    """Prioriza a ultima cotacao valida da sessao e falha para o SSOT configurado."""
    params = getattr(getattr(orch, "risk_manager", None), "risk_params", None)
    if isinstance(params, dict):
        try:
            payout = float(params.get("observed_payout_rate", params.get("payout_estimate")))
            if payout > 0.0:
                return payout
        except (TypeError, ValueError):
            pass
    th = getattr(orch, "trade_handler", None)
    if th is not None:
        rate = getattr(th, "latest_payout_rate", None)
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0.0:
            return float(rate)
    return MARKET_PAYOUT_SSOT
