"""Registro do modo de decisao Medallion (LLM Gemini) no startup."""

from __future__ import annotations

import logging
from typing import Any


def emit_decision_engine_banner(logger: logging.Logger, config: dict[str, Any], *, dl_enabled: bool) -> None:
    """Emite uma linha CFG informando se o ciclo usara PyTorch Deep Learning."""
    dl_cfg = config.get("deep_learning") or {}
    if dl_enabled:
        exec_cfg = config.get("orchestrator", {}).get("execution", {})
        mandatory = bool(exec_cfg.get("mandatory_trade_each_cycle", True))
        hist = dl_cfg.get("training_history_bars")
        if hist is None:
            data_cfg = config.get("data_handler") or {}
            hist = data_cfg.get("history_bars", "")
        logger.info(
            "CFG decisao | modo=DEEP_LEARNING_PYTORCH | model_path=%s | lookback=%s | "
            "hist_treino=%s | exec_obrigatoria=%s",
            dl_cfg.get("model_path", ""),
            dl_cfg.get("lookback", ""),
            hist,
            mandatory,
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor exige deep_learning.enabled=true em config/settings.json",
        )
