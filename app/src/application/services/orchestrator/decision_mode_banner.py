"""Registro do modo de decisao no startup."""

from __future__ import annotations

import logging
from typing import Any


def emit_decision_engine_banner(
    logger: logging.Logger,
    config: dict[str, Any],
    *,
    decision_mode: str,
) -> None:
    """Emite uma linha CFG informando o modo de decisao ativo no ciclo."""
    dl_cfg = config.get("deep_learning") or {}
    if decision_mode == "deep_learning":
        risk_params = (config.get("risk_management") or {}).get("params") or {}
        data_cfg = config.get("data_handler") or {}
        ohlc_sec = data_cfg.get("granularity", 60)
        online = bool(dl_cfg.get("online_training", True))
        logger.info(
            "CFG | DL/%s | ohlc=%ss lb=%s | thr=%s/%s | contrato=%s%s | treino_online=%s | continuo",
            dl_cfg.get("arch", "tcn"),
            ohlc_sec,
            dl_cfg.get("lookback", ""),
            dl_cfg.get("confidence_call_threshold", 0.75),
            dl_cfg.get("confidence_put_threshold", 0.25),
            risk_params.get("duration", 300),
            risk_params.get("duration_unit", "s"),
            online,
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor exige deep_learning.enabled=true em config/settings.json",
        )
