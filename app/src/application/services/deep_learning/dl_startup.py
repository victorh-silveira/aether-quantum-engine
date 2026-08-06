"""Resolucao de fetch inicial e prontidao de checkpoint para execucao DL."""

from __future__ import annotations

from typing import Any

import torch

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_params import parse_dl_params, resolve_dl_granularity
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.application.services.deep_learning.dl_training_gate import min_dl_inference_len
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, resolve_engine_mode


def inference_startup_enabled(dl_config: dict[str, Any] | None) -> bool:
    """Indica se o motor deve iniciar em modo inferencia sem retreino."""
    dl_config = dl_config or {}
    return not bool(dl_config.get("online_training", True))


def all_symbols_have_checkpoints(
    symbols: list[str],
    dl_config: dict[str, Any],
    data_config: dict[str, Any] | None = None,
) -> bool:
    """Verifica se todos os simbolos possuem checkpoint PyTorch compativel no disco."""
    expected_lookback = int(dl_config.get("lookback", 0) or 0)
    expected_granularity = int(resolve_dl_granularity(dl_config, data_config))
    for symbol in symbols:
        path = resolve_dl_model_path(dl_config, str(symbol))
        if not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
            feat_dim = int(payload.get("feature_dim", payload.get("input_dim", 0)))
            if feat_dim != FEATURE_DIM:
                return False
            if expected_lookback > 0 and int(payload.get("lookback", 0) or 0) != expected_lookback:
                return False
            if (
                expected_granularity > 0
                and "granularity" in payload
                and int(payload["granularity"]) != expected_granularity
            ):
                return False
        except Exception:
            return False
    return bool(symbols)


def resolve_startup_fetch_bars(config: dict[str, Any], symbols: list[str]) -> tuple[int, str]:
    """Retorna barras a buscar no startup e rotulo do modo (inferencia ou treino)."""
    data_config = config.get("data_handler") or {}
    dl_config = config.get("deep_learning") or {}
    warmup = int(data_config.get("history_warmup_bars", 64))
    is_training_mode = resolve_engine_mode(config) == ENGINE_MODE_TRAIN
    if (
        is_training_mode
        or not inference_startup_enabled(dl_config)
        or not all_symbols_have_checkpoints(symbols, dl_config, data_config)
    ):
        if "fetch_count" in data_config:
            return max(1, int(data_config["fetch_count"])), "treino"
        history_bars = int(data_config.get("history_bars", 0))
        if history_bars > 0:
            return max(1, history_bars + warmup), "treino"
        return 500, "treino"
    risk_params = (config.get("risk_management") or {}).get("params") or {}
    params = parse_dl_params(dl_config, data_config, risk_params)
    floor = min_dl_inference_len(params) + max(0, warmup)
    if "startup_fetch_bars" in data_config:
        return max(floor, 1, int(data_config["startup_fetch_bars"])), "inferencia"
    return floor, "inferencia"


def prepare_inference_run_loop(orch: Any) -> bool:
    """Marca bootstrap concluido quando modelos em disco estao prontos para operar."""
    dl_config = orch.config.get("deep_learning") or {}
    data_config = orch.config.get("data_handler") or {}
    if not inference_startup_enabled(dl_config):
        return False
    if not all_symbols_have_checkpoints(orch.symbols, dl_config, data_config):
        return False
    orch._dl_bootstrap_completed = True
    return True
