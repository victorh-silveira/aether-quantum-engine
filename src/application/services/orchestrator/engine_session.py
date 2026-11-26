"""Bootstrap de configuracao e autenticacao para run.py (Laya continuo)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.application.services.auth_manager import AuthManager
from src.presentation.terminal.logger import setup_logger
from src.presentation.terminal.logging_config import resolve_logging_config
from src.shared.paths import REPO_ROOT, repo_path


def load_engine_config() -> tuple[dict[str, Any], logging.Logger]:
    """Carrega config/settings.json e inicializa o logger do motor."""
    os.chdir(REPO_ROOT)
    config_path = repo_path("config", "settings.json")
    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)
    log_cfg = resolve_logging_config(config)
    logger = setup_logger(
        "AETH",
        log_file=log_cfg["log_file"],
        level=int(log_cfg["level"]),
        quiet_channels=log_cfg["quiet_channels"],
    )
    return config, logger


def create_authenticated_auth(config: dict[str, Any], logger: logging.Logger) -> AuthManager | None:
    """Valida PAT Deriv e retorna AuthManager pronto ou None se falhar."""
    mode = str(config.get("trading", {}).get("mode", "demo"))
    auth = AuthManager(mode=mode, config=config)
    token = auth.get_pat()
    if not token:
        logger.error("Token Deriv ausente. Defina AETHER_DERIV_PAT no .env.")
        return None
    try:
        auth.rest_client()
    except Exception as exc:
        logger.error("%s", exc)
        return None
    return auth
