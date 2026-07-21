"""Resolver unico de deep_learning.indicators a partir de settings."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path


_WINDOW_KEYS = (
    "adx_period",
    "atr_window",
    "bb_window",
    "cci_period",
    "ema_20",
    "ema_50",
    "ema_fast_crossover",
    "ema_slow_crossover",
    "hurst_window",
    "hurst_min_window",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "rel_vol_span",
    "roc_period",
    "rsi_period",
    "stoch_period",
    "stoch_smooth",
    "vol_ratio_short",
    "vol_ratio_long",
    "vol_window",
    "williams_period",
    "vr_short",
    "vr_long",
    "cmo_period",
    "kc_period",
    "kc_atr_period",
    "zscore_micro_window",
    "bb_width_harmonic_window",
)

_MULTIPLIER_KEYS = ("bb_std_mult", "kc_atr_mult", "cci_constant")
_NORMALIZATION_KEYS = ("series_z_clip", "bb_width_z_clip", "atr_z_clip")
_TREND_CONSENSUS_KEYS = ("min_bars", "rsi_call_above", "keltner_call_above", "di_call_above")
_CONGESTION_KEYS = ("min_bars", "adx_max", "bb_width_max")
_MARKET_RANK_KEYS = (
    "adx_weak_below",
    "adx_strong_at_or_above",
    "hurst_mean_revert_below",
    "hurst_trend_above",
)
_VOL_BURST_KEYS = ("macro_vol_ratio_min", "micro_bb_width_min")
_EDGE_ZSCORE_KEYS = (
    "hurst_trend_floor",
    "hurst_trend_ceiling",
    "bb_compression_max",
    "atr_transition_scale",
)
_EXHAUSTION_KEYS = ("enabled", "rsi_lower", "rsi_upper", "keltner_lower", "keltner_upper")

_CACHE: dict[str, Any] = {"indicators": None}


def _require_mapping(parent: dict[str, Any], key: str, required: tuple[str, ...]) -> dict[str, Any]:
    """Exige subbloco completo com todas as chaves obrigatorias."""
    raw = parent.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"deep_learning.indicators.{key} obrigatorio")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"deep_learning.indicators.{key} incompleto: {missing}")
    return raw


def resolve_indicator_config(dl_config: dict[str, Any] | None) -> dict[str, Any]:
    """Valida e normaliza deep_learning.indicators sem defaults locais."""
    cfg = dl_config if isinstance(dl_config, dict) else {}
    raw = cfg.get("indicators")
    if not isinstance(raw, dict):
        raise ValueError("deep_learning.indicators obrigatorio")
    windows_raw = _require_mapping(raw, "windows", _WINDOW_KEYS)
    multipliers_raw = _require_mapping(raw, "multipliers", _MULTIPLIER_KEYS)
    normalization_raw = _require_mapping(raw, "normalization", _NORMALIZATION_KEYS)
    trend_raw = _require_mapping(raw, "trend_consensus", _TREND_CONSENSUS_KEYS)
    congestion_raw = _require_mapping(raw, "congestion", _CONGESTION_KEYS)
    market_rank_raw = _require_mapping(raw, "market_rank", _MARKET_RANK_KEYS)
    vol_burst_raw = _require_mapping(raw, "vol_burst", _VOL_BURST_KEYS)
    edge_raw = _require_mapping(raw, "edge_zscore", _EDGE_ZSCORE_KEYS)
    exhaustion_raw = _require_mapping(raw, "exhaustion_filter", _EXHAUSTION_KEYS)
    return {
        "windows": {k: int(windows_raw[k]) for k in _WINDOW_KEYS},
        "multipliers": {k: float(multipliers_raw[k]) for k in _MULTIPLIER_KEYS},
        "normalization": {k: float(normalization_raw[k]) for k in _NORMALIZATION_KEYS},
        "trend_consensus": {
            "min_bars": int(trend_raw["min_bars"]),
            "rsi_call_above": float(trend_raw["rsi_call_above"]),
            "keltner_call_above": float(trend_raw["keltner_call_above"]),
            "di_call_above": float(trend_raw["di_call_above"]),
        },
        "congestion": {
            "min_bars": int(congestion_raw["min_bars"]),
            "adx_max": float(congestion_raw["adx_max"]),
            "bb_width_max": float(congestion_raw["bb_width_max"]),
        },
        "market_rank": {k: float(market_rank_raw[k]) for k in _MARKET_RANK_KEYS},
        "vol_burst": {k: float(vol_burst_raw[k]) for k in _VOL_BURST_KEYS},
        "edge_zscore": {k: float(edge_raw[k]) for k in _EDGE_ZSCORE_KEYS},
        "exhaustion_filter": {
            "enabled": bool(exhaustion_raw["enabled"]),
            "rsi_lower": float(exhaustion_raw["rsi_lower"]),
            "rsi_upper": float(exhaustion_raw["rsi_upper"]),
            "keltner_lower": float(exhaustion_raw["keltner_lower"]),
            "keltner_upper": float(exhaustion_raw["keltner_upper"]),
        },
    }


def indicator_windows(indicator_cfg: dict[str, Any]) -> dict[str, int]:
    """Extrai apenas o mapa de janelas/periodos."""
    windows = indicator_cfg.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("indicators.windows obrigatorio")
    return {k: int(windows[k]) for k in _WINDOW_KEYS}


def reset_indicator_config_cache() -> None:
    """Limpa cache de indicators carregado de settings.json."""
    _CACHE["indicators"] = None


def load_indicator_config_from_settings() -> dict[str, Any]:
    """Carrega e cacheia deep_learning.indicators de config/settings.json."""
    cached = _CACHE.get("indicators")
    if cached is not None:
        return cached
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") if isinstance(full, dict) else None
    resolved = resolve_indicator_config(dl if isinstance(dl, dict) else {})
    _CACHE["indicators"] = resolved
    return resolved


def indicators_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve indicators a partir do config completo do engine ou settings."""
    if isinstance(config, dict):
        dl = config.get("deep_learning")
        if isinstance(dl, dict) and isinstance(dl.get("indicators"), dict):
            return resolve_indicator_config(dl)
    return load_indicator_config_from_settings()


def load_bb_width_anomaly_ratio() -> float:
    """Le anomaly_ratio de orchestrator.execution.bb_width_adaptive_squeeze."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    squeeze = ((full.get("orchestrator") or {}).get("execution") or {}).get("bb_width_adaptive_squeeze") or {}
    if "anomaly_ratio" not in squeeze:
        raise KeyError("bb_width_adaptive_squeeze.anomaly_ratio obrigatorio")
    return float(squeeze["anomaly_ratio"])
