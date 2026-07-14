"""Bootstrap compartilhado de configuracao e autenticacao para run.py e train.py."""

import json
import logging
import os
from typing import Any

from aether_paths import REPO_ROOT, repo_path
from src.application.services.auth_manager import AuthManager
from src.application.services.orchestrator.engine_mode import (
    ENGINE_MODE_EXECUTE,
    apply_engine_mode,
)
from src.domain.risk.risk_policy import validate_engine_risk_config
from src.presentation.terminal.logger import setup_logger


def load_engine_config(*, engine_mode: str = ENGINE_MODE_EXECUTE) -> tuple[dict[str, Any], logging.Logger]:
    """Carrega settings.json, aplica engine_mode e inicializa o logger."""
    os.chdir(REPO_ROOT)
    config_path = repo_path("config", "settings.json")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    apply_engine_mode(config, engine_mode)
    log_file = config.get("logging", {}).get("log_file")
    logger = setup_logger("AETH", log_file=log_file)
    for issue in validate_engine_risk_config(config):
        logger.warning("CFG_RISK || %s", issue)
    return config, logger


def create_authenticated_auth(config: dict[str, Any], logger: logging.Logger) -> AuthManager | None:
    """Valida PAT Deriv e retorna AuthManager pronto ou None se falhar."""
    mode = str(config.get("trading", {}).get("mode", "demo"))
    auth = AuthManager(mode=mode, config=config)
    token = auth.get_pat()
    if not token:
        logger.error(
            "Token Deriv ausente. Defina AETHER_DERIV_PAT no .env (ou pat_...|APP_ID). "
            "Valide: python app/scripts/operations/deriv_pat_connect.py"
        )
        return None
    try:
        auth.rest_client()
    except Exception as exc:
        logger.error("%s", exc)
        return None
    return auth
