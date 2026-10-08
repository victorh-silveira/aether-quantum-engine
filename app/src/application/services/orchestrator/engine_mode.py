"""Modos de operacao do motor: treino dedicado ou execucao de trades."""

from src.application.services.deep_learning.dl_horizon import resolve_label_horizon_bars
from src.application.services.deep_learning.dl_params_timeframe import resolve_dl_granularity


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
    """Grava engine_mode e alinha label_horizon ao contrato na granularidade de treino."""
    chunk = config.get("orchestrator")
    if not isinstance(chunk, dict):
        chunk = {}
        config["orchestrator"] = chunk
    chunk["engine_mode"] = mode

    if mode == ENGINE_MODE_TRAIN:
        dl = config.get("deep_learning")
        if not isinstance(dl, dict):
            dl = {}
            config["deep_learning"] = dl
        data = config.get("data_handler") if isinstance(config.get("data_handler"), dict) else {}
        risk = config.get("risk_management") if isinstance(config.get("risk_management"), dict) else {}
        params = risk.get("params") if isinstance(risk.get("params"), dict) else {}
        gran = resolve_dl_granularity(dl, data)
        dl["label_horizon_bars"] = resolve_label_horizon_bars(gran, params, {})
    return config


def training_enabled(orch) -> bool:
    """Indica se o orquestrador esta em modo de treino."""
    return resolve_engine_mode(orch.config) == ENGINE_MODE_TRAIN
