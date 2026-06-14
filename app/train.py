"""Ponto de entrada exclusivo para treinamento Deep Learning."""

import asyncio

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN
from src.application.services.orchestrator.engine_session import (
    create_authenticated_auth,
    load_engine_config,
)


async def main() -> int:
    """Executa uma sessao completa de treino DL e encerra."""
    config, logger = load_engine_config(engine_mode=ENGINE_MODE_TRAIN)
    auth = create_authenticated_auth(config, logger)
    if auth is None:
        return 1
    orchestrator = Orchestrator(config, auth)
    try:
        ok = await orchestrator.run_training()
    except (asyncio.CancelledError, KeyboardInterrupt):
        await orchestrator.stop()
        return 130
    if not ok:
        logger.error("Treino encerrou antes de concluir. Veja logs e historicos de velas.")
        return 1
    logger.info("Treino concluido. Checkpoints em data/dl/. Execute python run.py para operar.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERRO fatal ao iniciar treino: {exc}", flush=True)
        raise SystemExit(1) from exc
