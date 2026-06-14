"""Leitura de parametros do bloco deep_learning em settings."""

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_horizon import contract_duration_seconds, resolve_label_horizon_bars


def bars_per_day(granularity_seconds: int) -> int:
    """Quantidade de velas OHLC em um dia civil para a granularidade informada."""
    gran = max(1, int(granularity_seconds))
    return max(1, 86400 // gran)


def resolve_training_history_bars(dl_config: dict, data_config: dict | None = None) -> int:
    """Resolve quantas barras usar no treino."""
    data_config = data_config or {}
    if "training_history_bars" in dl_config:
        return max(1, int(dl_config["training_history_bars"]))
    if "history_bars" in data_config:
        return max(1, int(data_config["history_bars"]))
    gran = int(data_config.get("granularity") or dl_config.get("granularity") or 60)
    days = float(dl_config.get("training_history_days", 2.0))
    return max(1, int(bars_per_day(gran) * days))


def slice_dl_price_window(
    prices: np.ndarray,
    *,
    training_history_bars: int,
) -> np.ndarray:
    """Recorta fechamentos para a janela de historico usada no treino e na predicao."""
    n = len(prices)
    target = max(1, int(training_history_bars))
    if n <= target:
        return prices
    return prices[n - target :]


def _tail_array(arr: np.ndarray | None, n: int) -> np.ndarray | None:
    """Recorta array OHLC para o mesmo comprimento de prices."""
    if arr is None:
        return None
    if len(arr) >= n:
        return arr[-n:]
    return arr


def slice_dl_ohlc_window(
    prices: np.ndarray,
    *,
    training_history_bars: int,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Recorta precos e OHLC para a janela de historico do treino e inferencia."""
    trimmed = slice_dl_price_window(prices, training_history_bars=training_history_bars)
    n = len(trimmed)
    return trimmed, _tail_array(open_, n), _tail_array(high, n), _tail_array(low, n)


def optional_float(section: dict, key: str) -> float | None:
    """Le valor float de uma chave em configuracao aninhada ou None se ausente."""
    if key not in section:
        return None
    return float(section[key])


def parse_tcn_channels(dl_config: dict) -> tuple[int, ...]:
    """Extrai larguras dos blocos TCN da configuracao."""
    tcn = dl_config.get("tcn")
    if not isinstance(tcn, dict):
        return (64, 64, 32)
    raw = tcn.get("channels")
    if not isinstance(raw, (list, tuple)) or not raw:
        return (64, 64, 32)
    return tuple(max(1, int(ch)) for ch in raw)


def parse_tcn_dropout(dl_config: dict) -> float:
    """Extrai dropout do TCN da configuracao."""
    tcn = dl_config.get("tcn")
    if not isinstance(tcn, dict):
        return 0.2
    return float(tcn.get("dropout", 0.2))


def parse_rnn_config(dl_config: dict) -> dict[str, Any]:
    """Extrai parametros LSTM/GRU da configuracao."""
    block = dl_config.get("lstm")
    if not isinstance(block, dict):
        block = dl_config.get("gru")
    if not isinstance(block, dict):
        block = {}
    return {
        "hidden_size": int(block.get("hidden_size", 64)),
        "num_layers": int(block.get("num_layers", 2)),
        "dropout": float(block.get("dropout", 0.2)),
    }


def parse_dl_params(
    dl_config: dict,
    data_config: dict | None = None,
    risk_params: dict | None = None,
) -> dict[str, Any]:
    """Extrai parametros de treino, validacao e gating do bloco deep_learning."""
    data_config = data_config or {}
    risk_params = risk_params or {}
    gran = int(data_config.get("granularity") or dl_config.get("granularity") or 60)
    lookback = int(dl_config.get("lookback", 48))
    training_history_bars = resolve_training_history_bars(dl_config, data_config)
    label_horizon_bars = resolve_label_horizon_bars(gran, risk_params, dl_config)
    rnn = parse_rnn_config(dl_config)
    base = {
        "arch": str(dl_config.get("arch", "tcn")).strip().lower(),
        "tcn_channels": parse_tcn_channels(dl_config),
        "tcn_dropout": parse_tcn_dropout(dl_config),
        "rnn_hidden_size": rnn["hidden_size"],
        "rnn_num_layers": rnn["num_layers"],
        "rnn_dropout": rnn["dropout"],
        "lookback": lookback,
        "epochs": int(dl_config.get("training_epochs", 128)),
        "training_batch_size": int(dl_config.get("training_batch_size", 512)),
        "training_log_every_n_epochs": max(1, int(dl_config.get("training_log_every_n_epochs", 16))),
        "training_device": str(dl_config.get("training_device", "auto")).strip().lower(),
        "inference_device": str(dl_config.get("inference_device", "auto")).strip().lower(),
        "lr": float(dl_config.get("learning_rate", 0.0012)),
        "validation_bars": int(dl_config.get("validation_bars", 96)),
        "calib_ratio": float(dl_config.get("calib_ratio", 0.15)),
        "min_val_accuracy": float(dl_config.get("min_val_accuracy", 0.53)),
        "confidence_call_threshold": float(dl_config.get("confidence_call_threshold", 0.75)),
        "confidence_put_threshold": float(dl_config.get("confidence_put_threshold", 0.25)),
        "train_on_new_candle": bool(dl_config.get("train_on_new_candle_only", True)),
        "weight_decay": float(dl_config.get("weight_decay", 0.0002)),
        "granularity": gran,
        "contract_duration": max(1, int(risk_params.get("duration", 60))),
        "rolling_retrain_bars": int(dl_config.get("rolling_retrain_bars", 3)),
        "retrain_min_bars": int(dl_config.get("retrain_min_bars", 1)),
        "training_history_bars": training_history_bars,
        "bars_per_day": bars_per_day(gran),
        "label_horizon_bars": label_horizon_bars,
        "contract_seconds": contract_duration_seconds(risk_params),
        "val_acc_live_blend": float(dl_config.get("val_acc_live_blend", 0.35)),
    }
    gate = parse_deploy_gate_config(dl_config)
    min_eval_bars = lookback + 5
    gate = {**gate, "mini_bars": max(min_eval_bars, int(gate.get("mini_bars", 120)))}
    base["deploy_gate"] = gate
    return base
