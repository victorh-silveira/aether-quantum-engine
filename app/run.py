"""Ponto de entrada: carrega configuracao e executa o orquestrador."""

import asyncio

from aether_paths import REPO_ROOT
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_EXECUTE
from src.application.services.orchestrator.engine_session import (
    create_authenticated_auth,
    load_engine_config,
)


async def main():
    """Carrega configuracao, autentica e executa o loop principal do motor."""
    config, logger = load_engine_config(engine_mode=ENGINE_MODE_EXECUTE)
    auth = create_authenticated_auth(config, logger)
    if auth is None:
        raise SystemExit(1)

    orchestrator = Orchestrator(config, auth)
    try:
        await orchestrator.run()
    except (
        asyncio.CancelledError,
        KeyboardInterrupt,
    ):
        await orchestrator.stop()
    if getattr(orchestrator, "shutdown_reason", None) == "stop_win":
        target = orchestrator.risk_manager.total_session_profit
        logger.info("STOP_WIN: meta diaria atingida (pnl_sessao=$%+.2f). Motor encerrado.", target)
        raise SystemExit(0)
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
