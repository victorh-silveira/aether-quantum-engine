"""Registro do modo de decisao Medallion (LLM Gemini) no startup."""

from __future__ import annotations

import logging
from typing import Any


def emit_decision_engine_banner(logger: logging.Logger, config: dict[str, Any], *, dl_enabled: bool) -> None:
    """Emite uma linha CFG informando se o ciclo usara PyTorch Deep Learning."""
    dl_cfg = config.get("deep_learning") or {}
    if dl_enabled:
        logger.info(
            "CFG decisao | modo=DEEP_LEARNING_PYTORCH | model_path=%s | lookback=%s",
            dl_cfg.get("model_path", ""),
            dl_cfg.get("lookback", ""),
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor exige deep_learning.enabled=true em config/settings.json",
        )
