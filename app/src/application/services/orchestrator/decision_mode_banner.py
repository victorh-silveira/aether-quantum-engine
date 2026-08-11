"""Registro do modo de decisao no startup."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.deep_learning.dl_horizon import contract_duration_seconds
from src.application.services.deep_learning.dl_params_timeframe import (
    resolve_dl_granularity,
    resolve_train_timeframe,
)


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
        data_cfg = config.get("data_handler") if isinstance(config.get("data_handler"), dict) else {}
        train_tf = resolve_train_timeframe(dl_cfg if isinstance(dl_cfg, dict) else {})
        ohlc_sec = resolve_dl_granularity(dl_cfg if isinstance(dl_cfg, dict) else {}, data_cfg)
        macro_sec = int(data_cfg.get("granularity") or ohlc_sec)
        micro_sec = int(data_cfg.get("micro_granularity") or ohlc_sec)
        contract_sec = contract_duration_seconds(risk_params if isinstance(risk_params, dict) else {})
        online = bool(dl_cfg.get("online_training", False))
        logger.info(
            "CFG | DL/%s | ohlc=%ss (%s) lb=%s | macro=%ss micro=%ss | thr=%s/%s | contrato=%ss | treino_online=%s | continuo",
            dl_cfg.get("arch", "tcn"),
            ohlc_sec,
            train_tf,
            dl_cfg.get("lookback", ""),
            macro_sec,
            micro_sec,
            dl_cfg.get("confidence_call_threshold", 0.75),
            dl_cfg.get("confidence_put_threshold", 0.25),
            contract_sec,
            online,
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor exige deep_learning.enabled=true em config/settings.json",
        )
