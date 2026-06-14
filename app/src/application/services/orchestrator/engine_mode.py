"""Modos de operacao do motor: treino dedicado ou execucao de trades."""

ENGINE_MODE_TRAIN = "train"
ENGINE_MODE_EXECUTE = "execute"


def resolve_engine_mode(config: dict) -> str:
    """Retorna train ou execute a partir de orchestrator.engine_mode."""
    orch = config.get("orchestrator")
    if not isinstance(orch, dict):
        return ENGINE_MODE_EXECUTE
    raw = str(orch.get("engine_mode", ENGINE_MODE_EXECUTE)).strip().lower()
    if raw in ("train", "training"):
        return ENGINE_MODE_TRAIN
    return ENGINE_MODE_EXECUTE


def apply_engine_mode(config: dict, mode: str) -> dict:
    """Grava engine_mode em config e devolve o dict atualizado."""
    chunk = config.get("orchestrator")
    if not isinstance(chunk, dict):
        chunk = {}
        config["orchestrator"] = chunk
    chunk["engine_mode"] = mode
    return config


def training_enabled(orch) -> bool:
    """Indica se o orquestrador esta em modo de treino."""
    return resolve_engine_mode(orch.config) == ENGINE_MODE_TRAIN
