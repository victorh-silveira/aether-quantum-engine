"""Ponto de entrada: carrega configuracao e executa o orquestrador."""

import asyncio
import json
from pathlib import Path

from src.application.services.auth_manager import AuthManager
from src.application.services.orchestrator import Orchestrator
from src.presentation.terminal.logger import setup_logger


async def main():
    """Carrega JSON, autentica e inicia o ciclo ate encerramento."""
    config_path = Path("config/settings.json")
    with config_path.open() as f:
        config = json.load(f)

    log_file = config.get("logging", {}).get("log_file")
    logger = setup_logger("AETH", log_file=log_file)

    auth = AuthManager(mode=config["trading"]["mode"])
    token = auth.get_token()

    if not token:
        logger.error("Token nao encontrado no .env")
        return

    orchestrator = Orchestrator(config, token)
    try:
        await orchestrator.run()
    except (
        asyncio.CancelledError,
        KeyboardInterrupt,
    ):
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
