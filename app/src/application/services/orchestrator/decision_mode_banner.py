"""Registro do modo de decisao Medallion (LLM Gemini) no startup."""

from __future__ import annotations

import logging
from typing import Any


def emit_decision_engine_banner(logger: logging.Logger, config: dict[str, Any], *, llm_enabled: bool) -> None:
    """Emite uma linha CFG informando se o ciclo Medallion usara Gemini."""
    llm_cfg = config.get("llm") or {}
    if llm_enabled:
        logger.debug(
            "CFG decisao | modo=MEDALLION_LLM | gemini_modelo=%s | gemini_api_key_env=GEMINI_API_KEY",
            llm_cfg.get("model", ""),
        )
    else:
        logger.debug(
            "CFG decisao | modo=INATIVO | motor Medallion exige llm.enabled=true em config/settings.json",
        )
