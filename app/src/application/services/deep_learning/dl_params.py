"""Leitura de parametros do bloco deep_learning em settings."""

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_feature_indicators import feature_windows
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_horizon import (
    contract_duration_seconds,
    resolve_implied_vol_bars,
    resolve_label_horizon_bars,
    resolve_label_ma_window,
    resolve_label_mode,
    resolve_label_smooth_bars,
)
from src.application.services.deep_learning.dl_indicator_config import (
    load_indicator_config_from_settings,
    resolve_indicator_config,
)
from src.application.services.deep_learning.dl_params_blocks import (
    parse_calibration_config,
    parse_indicator_gating_config,
)
from src.application.services.deep_learning.dl_params_timeframe import (
    resolve_dl_granularity,
    resolve_train_timeframe,
)
from src.application.services.deep_learning.dl_sample_weighting import parse_sample_weighting_config


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
    days = float(dl_config.get("training_history_days", 90.0))
    return max(1, int(bars_per_day(gran) * days))


def resolve_validation_bars(
    dl_config: dict,
    *,
    training_history_bars: int,
    lookback: int,
    label_horizon_bars: int,
    label_smooth_bars: int,
) -> int:
    """Resolve tamanho da fatia de validacao temporal (fixo ou proporcional)."""
    if "validation_ratio" in dl_config:
        ratio = max(0.05, min(0.4, float(dl_config["validation_ratio"])))
        estimated_samples = max(20, training_history_bars - lookback - label_horizon_bars - label_smooth_bars)
        return max(5, int(estimated_samples * ratio))
    return max(5, int(dl_config.get("validation_bars", 96)))


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


def resolve_inference_history_bars(
    params: dict[str, Any],
    *,
    granularity: int | None = None,
) -> int:
    """Barras minimas de OHLC para montar features na inferencia (nao no treino)."""
    lookback = max(1, int(params.get("lookback", 30)))
    gran = max(1, int(granularity or params.get("granularity") or 60))
    implied = max(1, int(params.get("implied_vol_bars", 60)))
    indicators = params.get("indicators")
    if isinstance(indicators, dict) and isinstance(indicators.get("windows"), dict):
        win = feature_windows(gran, indicators["windows"])
    else:
        win = feature_windows(gran)
    warmup = max(
        lookback,
        implied,
        int(win["hurst_window"]),
        int(win["ema_50"]),
        int(win["rel_vol_span"]),
        int(win["bb_window"]),
        int(win["atr_window"]),
        int(win["rsi_period"]),
    )
    return warmup + lookback + 16


def parse_dl_params(
    dl_config: dict,
    data_config: dict | None = None,
    risk_params: dict | None = None,
) -> dict[str, Any]:
    """Extrai parametros de treino, validacao e gating do bloco deep_learning."""
    data_config = data_config or {}
    risk_params = risk_params or {}
    train_tf = resolve_train_timeframe(dl_config)
    gran = resolve_dl_granularity(dl_config, data_config)
    lookback = int(dl_config.get("lookback", 30))
    history_cfg = dict(data_config)
    if train_tf == "micro" and "training_history_bars" not in dl_config:
        micro_bars = int(data_config.get("micro_history_bars") or 0)
        if micro_bars > 0:
            history_cfg["history_bars"] = micro_bars
    training_history_bars = resolve_training_history_bars(dl_config, history_cfg)
    label_horizon_bars = resolve_label_horizon_bars(gran, risk_params, dl_config)
    label_smooth_bars = resolve_label_smooth_bars(dl_config)
    label_ma_window = resolve_label_ma_window(dl_config)
    label_mode = resolve_label_mode(dl_config)
    implied_vol_bars = resolve_implied_vol_bars(dl_config)
    validation_bars = resolve_validation_bars(
        dl_config,
        training_history_bars=training_history_bars,
        lookback=lookback,
        label_horizon_bars=label_horizon_bars,
        label_smooth_bars=label_smooth_bars,
    )
    rnn = parse_rnn_config(dl_config)
    base = {
        "arch": str(dl_config.get("arch", "tcn")).strip().lower(),
        "tcn_channels": parse_tcn_channels(dl_config),
        "tcn_dropout": parse_tcn_dropout(dl_config),
        "rnn_hidden_size": rnn["hidden_size"],
        "rnn_num_layers": rnn["num_layers"],
        "rnn_dropout": rnn["dropout"],
        "lookback": lookback,
        "epochs": int(dl_config.get("training_epochs", 50)),
        "early_stopping_patience": max(0, int(dl_config.get("early_stopping_patience", 6))),
        "min_epochs": max(0, int(dl_config.get("min_epochs", 0))),
        "training_batch_size": int(dl_config.get("training_batch_size", 512)),
        "training_log_every_n_epochs": max(1, int(dl_config.get("training_log_every_n_epochs", 5))),
        "training_device": str(dl_config.get("training_device", "auto")).strip().lower(),
        "inference_device": str(dl_config.get("inference_device", "auto")).strip().lower(),
        "lr": float(dl_config.get("learning_rate", 0.0012)),
        "label_smoothing": float(dl_config.get("label_smoothing", 0.0)),
        "focal_gamma": float(dl_config.get("focal_gamma", 1.5)),
        "lr_scheduler": str(dl_config.get("lr_scheduler", "cosine")).strip().lower(),
        "validation_bars": validation_bars,
        "validation_ratio": float(dl_config.get("validation_ratio", 0.15)),
        "calib_ratio": float(dl_config.get("calib_ratio", 0.15)),
        "min_val_accuracy": float(dl_config.get("min_val_accuracy", 0.53)),
        "min_edge_execute": float(dl_config.get("min_edge_execute", 0.0)),
        "confidence_call_threshold": float(dl_config.get("confidence_call_threshold", 0.75)),
        "confidence_put_threshold": float(dl_config.get("confidence_put_threshold", 0.25)),
        "train_on_new_candle": bool(dl_config.get("train_on_new_candle_only", True)),
        "online_training": bool(dl_config.get("online_training", True)),
        "weight_decay": float(dl_config.get("weight_decay", 0.0002)),
        "granularity": gran,
        "train_timeframe": train_tf,
        "micro_granularity": int(data_config.get("micro_granularity") or dl_config.get("micro_granularity") or gran),
        "contract_duration": max(1, int(risk_params.get("duration", 30))),
        "contract_duration_seconds": contract_duration_seconds(risk_params),
        "risk_params": dict(risk_params),
        "rolling_retrain_bars": int(dl_config.get("rolling_retrain_bars", 3)),
        "retrain_min_bars": int(dl_config.get("retrain_min_bars", 1)),
        "training_history_bars": training_history_bars,
        "inference_history_bars": (
            max(1, int(dl_config["inference_history_bars"]))
            if "inference_history_bars" in dl_config
            else resolve_inference_history_bars(
                {"lookback": lookback, "granularity": gran, "implied_vol_bars": implied_vol_bars},
                granularity=gran,
            )
        ),
        "bars_per_day": bars_per_day(gran),
        "label_horizon_bars": label_horizon_bars,
        "label_smooth_bars": label_smooth_bars,
        "label_ma_window": label_ma_window,
        "label_mode": label_mode,
        "implied_vol_bars": implied_vol_bars,
        "contract_seconds": contract_duration_seconds(risk_params),
        "val_acc_live_blend": float(dl_config.get("val_acc_live_blend", 0.35)),
        "trend_alignment_required": bool(dl_config.get("trend_alignment_required", False)),
    }
    indicators = (
        resolve_indicator_config(dl_config)
        if isinstance(dl_config.get("indicators"), dict)
        else load_indicator_config_from_settings()
    )
    exhaustion = indicators["exhaustion_filter"]
    base["indicators"] = indicators
    base["feature_windows"] = indicators["windows"]
    base["exhaustion_filter_enabled"] = bool(exhaustion["enabled"])
    base["exhaustion_rsi_lower"] = float(exhaustion["rsi_lower"])
    base["exhaustion_rsi_upper"] = float(exhaustion["rsi_upper"])
    base["exhaustion_keltner_lower"] = float(exhaustion["keltner_lower"])
    base["exhaustion_keltner_upper"] = float(exhaustion["keltner_upper"])
    gate = parse_deploy_gate_config(dl_config)
    min_eval_bars = lookback + 5
    gate = {**gate, "mini_bars": max(min_eval_bars, int(gate.get("mini_bars", 120)))}
    base["deploy_gate"] = gate
    base["indicator_gating"] = parse_indicator_gating_config(dl_config)
    base["calibration"] = parse_calibration_config(dl_config)
    base["sample_weighting"] = parse_sample_weighting_config(dl_config)
    return base
