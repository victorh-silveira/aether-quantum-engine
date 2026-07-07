"""Sessao dedicada de treino DL acionada por train.py."""

from src.application.services.deep_learning.dl_bootstrap_train import run_dl_training_session
from src.application.services.deep_learning.dl_model_artifacts import upload_all_symbol_checkpoints
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner


async def run_orchestrator_training(orch) -> bool:
    """Conecta, sincroniza velas, treina modelos DL e encerra a sessao."""
    orch.logger.info("INIT: Treino DL | conectando Deriv")
    if not await orch._setup_session():
        orch.logger.error("INIT: Abortando treino (falha em infra, PAT, OTP ou WebSocket).")
        return False
    fetch_count = orch.stream._resolve_fetch_count()
    orch.logger.info(
        "INIT: Treino DL | sincronizando %d simbolos | alvo %d velas",
        len(orch.symbols),
        fetch_count,
    )
    if not await orch._start_streams():
        orch.logger.error("INIT: Abortando treino (falha ao sincronizar velas OHLC).")
        return False
    emit_decision_engine_banner(orch.logger, orch.config, decision_mode=orch._decision_mode())
    if orch._decision_mode() == "deep_learning":
        await run_dl_training_session(orch)
        orch._dl_bootstrap_completed = True
        await upload_all_symbol_checkpoints(orch)
    await orch._save_full_state()
    orch.logger.info("DL | sessao de treino finalizada")
    await orch.stop()
    return True
