"""Ponto de entrada: carrega configuracao e executa o orquestrador."""

import asyncio
import json
import os

from aether_paths import REPO_ROOT, repo_path
from src.application.services.auth_manager import AuthManager
from src.application.services.orchestrator import Orchestrator
from src.presentation.terminal.logger import setup_logger


async def main():
    """Carrega JSON, autentica e inicia o ciclo ate encerramento."""
    os.chdir(REPO_ROOT)
    config_path = repo_path("config", "settings.json")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    log_file = config.get("logging", {}).get("log_file")
    logger = setup_logger("AETH", log_file=log_file)

    mode = str(config.get("trading", {}).get("mode", "demo"))
    auth = AuthManager(mode=mode, config=config)
    token = auth.get_pat()

    if not token:
        logger.error(
            "Token Deriv ausente. Defina AETHER_DERIV_PAT no .env (ou pat_...|APP_ID). "
            "Valide: python app/scripts/deriv_pat_connect.py"
        )
        raise SystemExit(1)
    try:
        auth.rest_client()
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    orchestrator = Orchestrator(config, auth)
    try:
        await orchestrator.run()
    except (
        asyncio.CancelledError,
        KeyboardInterrupt,
    ):
        await orchestrator.stop()
    if not orchestrator.running:
        logger.error(
            "Motor encerrou antes do loop principal. Veja INIT (PAT, OTP, stream) e %s",
            REPO_ROOT / ".env",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERRO fatal ao iniciar motor: {exc}", flush=True)
        raise SystemExit(1) from exc
