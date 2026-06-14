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
        exec_cfg = config.get("orchestrator", {}).get("execution", {})
        mandatory = bool(exec_cfg.get("mandatory_trade_each_cycle", False))
        hist = dl_cfg.get("training_history_bars")
        if hist is None:
            data_cfg = config.get("data_handler") or {}
            hist = data_cfg.get("history_bars", "")
        risk_params = (config.get("risk_management") or {}).get("params") or {}
        data_cfg = config.get("data_handler") or {}
        ohlc_sec = data_cfg.get("granularity", 60)
        logger.info(
            "CFG decisao | modo=DEEP_LEARNING | arch=%s | ohlc=%ss | lookback=%s | hist_treino=%s | "
            "label=%s | ma=%s | smooth=%s | threshold=%s/%s | contrato=%s%s | exec_obrigatoria=%s",
            dl_cfg.get("arch", "tcn"),
            ohlc_sec,
            dl_cfg.get("lookback", ""),
            hist,
            dl_cfg.get("label_mode", "ma_trend"),
            dl_cfg.get("label_ma_window", 5),
            dl_cfg.get("label_smooth_bars", 1),
            dl_cfg.get("confidence_call_threshold", 0.75),
            dl_cfg.get("confidence_put_threshold", 0.25),
            risk_params.get("duration", 60),
            risk_params.get("duration_unit", "s"),
            mandatory,
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor exige deep_learning.enabled=true em config/settings.json",
        )
