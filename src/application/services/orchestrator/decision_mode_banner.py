"""Registro de modo de decisao (LLM vs simples) no startup."""

from __future__ import annotations

import logging
from typing import Any


def emit_decision_engine_banner(logger: logging.Logger, config: dict[str, Any], *, llm_enabled: bool) -> None:
    """Emite uma linha CFG informando se o ciclo usara Gemini ou modo simples."""
    llm_cfg = config.get("llm") or {}
    if llm_enabled:
        logger.debug(
            "CFG decisao | modo=LLM | gemini_modelo=%s | gemini_api_key_env=GEMINI_API_KEY",
            llm_cfg.get("model", ""),
        )
    else:
        logger.debug(
            "CFG decisao | modo=simples | sem Gemini | defina llm.enabled=true em config/settings.json",
        )
