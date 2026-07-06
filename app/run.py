"""Ponto de entrada: carrega configuracao e executa o orquestrador."""

import asyncio
import sys

from aether_paths import REPO_ROOT
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_EXECUTE
from src.application.services.orchestrator.engine_session import (
    create_authenticated_auth,
    load_engine_config,
)
from src.application.services.orchestrator.graceful_shutdown import install_shutdown_excepthook


def _emit_fatal_startup_error(exc: BaseException) -> None:
    """Emite erro fatal no stderr quando streams ainda estao abertos."""
    try:
        stderr = getattr(sys, "stderr", None)
        if stderr is None or getattr(stderr, "closed", False):
            return
        print(f"ERRO fatal ao iniciar motor: {exc}", file=stderr, flush=True)
    except Exception:
        return


async def main() -> int:
    """Carrega configuracao, autentica e executa o loop principal do motor."""
    config, logger = load_engine_config(engine_mode=ENGINE_MODE_EXECUTE)
    auth = create_authenticated_auth(config, logger)
    if auth is None:
        return 1

    orchestrator = Orchestrator(config, auth)
    try:
        await orchestrator.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        return 130
    finally:
        await orchestrator.close_infrastructure_connections()

    reason = getattr(orchestrator, "shutdown_reason", None)
    if reason == "stop_win":
        target = orchestrator.risk_manager.total_session_profit
        logger.info("STOP_WIN: meta da sessao atingida (pnl_sessao=$%+.2f). Motor encerrado.", target)
        return 0
    if not orchestrator.running:
        logger.error(
            "Motor encerrou antes do loop principal. Veja INIT (PAT, OTP, stream) e %s",
            REPO_ROOT / ".env",
        )
        return 1
    return 0


if __name__ == "__main__":
    install_shutdown_excepthook()
    try:
        sys.exit(asyncio.run(main()))
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        _emit_fatal_startup_error(exc)
        sys.exit(1)
