"""Predicao por simbolo no bridge Deep Learning."""

import asyncio
from typing import Any

from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async


def predict_symbol_decision(
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    train_loss: float | None,
    *,
    recovery_active: bool,
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> dict:
    """Gera predicao DL com indicadores e tendencia para resolucao direcional."""
    return asyncio.run(
        predict_symbol_decision_async(
            orch,
            symbol,
            model,
            prices,
            norm_stats,
            runtime,
            params,
            train_loss,
            recovery_active=recovery_active,
            granularity=granularity,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
    )
