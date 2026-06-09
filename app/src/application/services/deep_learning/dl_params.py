"""Leitura de parametros do bloco deep_learning em settings."""

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_horizon import resolve_label_horizon_bars


def bars_per_day(granularity_seconds: int) -> int:
    """Quantidade de velas OHLC em um dia civil para a granularidade informada."""
    gran = max(60, int(granularity_seconds))
    return max(1, 86400 // gran)


def resolve_training_history_bars(dl_config: dict, data_config: dict | None = None) -> int:
    """Resolve quantas barras usar no treino (padrao: um dia conforme granularidade)."""
    data_config = data_config or {}
    if "training_history_bars" in dl_config:
        return max(1, int(dl_config["training_history_bars"]))
    if "history_bars" in data_config:
        return max(1, int(data_config["history_bars"]))
    gran = int(data_config.get("granularity") or dl_config.get("granularity") or 300)
    days = float(dl_config.get("training_history_days", 1.0))
    return max(1, int(bars_per_day(gran) * days))


def slice_dl_price_window(
    prices: np.ndarray,
    pair_prices: np.ndarray | None,
    *,
    training_history_bars: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Recorta fechamentos para a janela de historico usada no treino e na predicao."""
    n = len(prices)
    target = max(1, int(training_history_bars))
    if n <= target:
        return prices, pair_prices
    start = n - target
    trimmed = prices[start:]
    peer = pair_prices
    if peer is not None and len(peer) >= n:
        peer = peer[start:]
    elif peer is not None and len(peer) > target:
        peer = peer[-target:]
    return trimmed, peer


def optional_float(section: dict, key: str) -> float | None:
    """Le valor float de uma chave em configuracao aninhada ou None se ausente."""
    if key not in section:
        return None
    return float(section[key])


def parse_binary_signal_params(dl_config: dict) -> dict[str, Any]:
    """Extrai thresholds do bloco binary_signal para gating CALL/PUT."""
    bs = dl_config.get("binary_signal", {})
    return {
        "min_rel_vol_execute": float(bs.get("min_rel_vol_execute", 0.28)),
        "sma_z_block_call": float(bs.get("sma_z_block_call", 0.003)),
        "sma_z_block_put": float(bs.get("sma_z_block_put", -0.003)),
        "sma_z_extreme": float(bs.get("sma_z_extreme", 0.005)),
        "weak_dl_override_margin": float(bs.get("weak_dl_override_margin", 0.05)),
        "pair_z_against_limit": float(bs.get("pair_z_against_limit", 1.0)),
        "variance_ratio_mean_rev_max": float(bs.get("variance_ratio_mean_rev_max", 0.88)),
        "wick_rejection_ratio": float(bs.get("wick_rejection_ratio", 1.8)),
        "require_pair_spread_confirm": bool(bs.get("require_pair_spread_confirm", True)),
        "require_candle_confirm": bool(bs.get("require_candle_confirm", True)),
        "min_close_loc_call": float(bs.get("min_close_loc_call", 0.48)),
        "max_close_loc_put": float(bs.get("max_close_loc_put", 0.52)),
        "rsi_block_call": float(bs.get("rsi_block_call", 0.72)),
        "rsi_block_put": float(bs.get("rsi_block_put", 0.28)),
    }


def parse_dl_params(
    dl_config: dict,
    data_config: dict | None = None,
    risk_params: dict | None = None,
) -> dict[str, Any]:
    """Extrai parametros de treino, validacao e gating do bloco deep_learning."""
    data_config = data_config or {}
    risk_params = risk_params or {}
    selection = dl_config.get("selection", {})
    gran = int(data_config.get("granularity") or dl_config.get("granularity") or 300)
    lookback = int(dl_config.get("lookback", 32))
    training_history_bars = resolve_training_history_bars(dl_config, data_config)
    label_horizon_bars = resolve_label_horizon_bars(gran, risk_params, dl_config)
    validation_bars = int(dl_config.get("validation_bars", 60))
    base = {
        "arch": str(dl_config.get("arch", "tcn")),
        "lookback": lookback,
        "epochs": int(dl_config.get("training_epochs", 20)),
        "lr": float(dl_config.get("learning_rate", 0.001)),
        "validation_bars": validation_bars,
        "label_min_move_pct": float(dl_config.get("label_min_move_pct", 0.0002)),
        "early_stopping_patience": int(dl_config.get("early_stopping_patience", 3)),
        "focal_gamma": float(dl_config.get("focal_gamma", 0.0)),
        "calib_ratio": float(dl_config.get("calib_ratio", 0.15)),
        "min_conviction": float(dl_config.get("min_conviction_execute", 0.58)),
        "min_edge_margin": float(dl_config.get("min_edge_margin", 0.06)),
        "min_val_accuracy": float(dl_config.get("min_val_accuracy", 0.50)),
        "train_on_new_candle": bool(dl_config.get("train_on_new_candle_only", True)),
        "recovery_min_conviction": float(dl_config.get("recovery_gating", {}).get("min_conviction_execute", 0.56)),
        "recovery_min_edge_margin": float(dl_config.get("recovery_gating", {}).get("min_edge_margin", 0.05)),
        "recovery_min_val_accuracy": float(dl_config.get("recovery_gating", {}).get("min_val_accuracy", 0.48)),
        "bypass_min_conviction": optional_float(dl_config.get("strong_signal_bypass", {}), "min_conviction_execute"),
        "bypass_min_edge": optional_float(dl_config.get("strong_signal_bypass", {}), "min_edge_margin"),
        "moderate_min_conviction": optional_float(
            dl_config.get("moderate_signal_bypass", {}), "min_conviction_execute"
        ),
        "moderate_min_edge": optional_float(dl_config.get("moderate_signal_bypass", {}), "min_edge_margin"),
        "moderate_min_val_accuracy": optional_float(dl_config.get("moderate_signal_bypass", {}), "min_val_accuracy"),
        "selection_min_conviction": float(
            selection.get("min_conviction_execute", dl_config.get("min_conviction_execute", 0.58))
        ),
        "selection_min_edge": float(selection.get("min_edge_margin", dl_config.get("min_edge_margin", 0.06))),
        "selection_min_val_accuracy": float(selection.get("min_val_accuracy", dl_config.get("min_val_accuracy", 0.50))),
        "selection_strong_raw": float(selection.get("strong_raw", 0.65)),
        "selection_strong_edge": float(selection.get("strong_edge", 0.12)),
        "max_calibrated_raw_gap": float(dl_config.get("max_calibrated_raw_gap", 0.18)),
        "max_calib_gap_execute": float(dl_config.get("max_calib_gap_execute", 0.18)),
        "min_raw_conviction_execute": float(dl_config.get("min_raw_conviction_execute", 0.52)),
        "max_val_brier_execute": float(dl_config.get("max_val_brier_execute", 0.36)),
        "recovery_allow_bypass": bool(dl_config.get("recovery_allow_bypass", False)),
        "recovery_min_raw_conviction": float(
            dl_config.get("recovery_gating", {}).get("min_raw_conviction_execute", 0.58)
        ),
        "min_direction_margin": float(dl_config.get("min_direction_margin", 0.06)),
        "max_raw_saturation": float(dl_config.get("max_raw_saturation", 0.97)),
        "saturation_min_trade_score": float(dl_config.get("saturation_min_trade_score", 0.58)),
        "require_regime_alignment": bool(dl_config.get("require_regime_alignment", False)),
        "high_val_acc_relax": float(dl_config.get("high_val_acc_relax", 0.68)),
        "relaxed_conviction": float(dl_config.get("relaxed_conviction", 0.54)),
        "brier_untrained_floor": float(dl_config.get("brier_untrained_floor", 0.99)),
        "min_regime_strength": float(dl_config.get("min_regime_strength", 0.0)),
        "val_acc_live_blend": float(dl_config.get("val_acc_live_blend", 0.55)),
        "min_live_win_rate": float(dl_config.get("min_live_win_rate", 0.42)),
        "bypass_min_val_accuracy": float(dl_config.get("strong_signal_bypass", {}).get("min_val_accuracy", 0.48)),
        "weight_decay": float(dl_config.get("weight_decay", 0.0001)),
        "label_smoothing": float(dl_config.get("label_smoothing", 0.02)),
        "granularity": gran,
        "rolling_retrain_bars": int(dl_config.get("rolling_retrain_bars", 12)),
        "retrain_min_bars": int(dl_config.get("retrain_min_bars", 0)),
        "training_history_bars": training_history_bars,
        "bars_per_day": bars_per_day(gran),
        "label_horizon_bars": label_horizon_bars,
    }
    gate = parse_deploy_gate_config(dl_config)
    min_eval_bars = lookback + 5
    gate = {**gate, "mini_bars": max(min_eval_bars, int(gate.get("mini_bars", 80)))}
    base["deploy_gate"] = gate
    base["binary_signal"] = parse_binary_signal_params(dl_config)
    return base
