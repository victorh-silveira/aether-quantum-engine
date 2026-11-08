"""Resolver de deep_learning.live_signal_metrics."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.domain.config_knobs import require_float, require_int, require_keys


_LIVE_KEYS = (
    "window",
    "min_rank",
    "ece_bins",
    "ece_soft_threshold",
    "ece_rank_penalty",
    "drift_soft_penalty",
    "drift_soft_veto_n",
    "drift_score_factor",
    "drift_min_score",
    "wr_raw_inconsistent_gap",
)

_CACHE: dict[str, Any] = {"live": None}


def resolve_live_signal_metrics_config(dl_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve live signal metrics config."""
    cfg = dl_config if isinstance(dl_config, dict) else {}
    raw = cfg.get("live_signal_metrics")
    if not isinstance(raw, dict):
        return load_live_signal_metrics_from_settings()
    block = require_keys(raw, _LIVE_KEYS, "deep_learning.live_signal_metrics")
    return {
        "window": require_int(block, "window"),
        "min_rank": require_int(block, "min_rank"),
        "ece_bins": require_int(block, "ece_bins"),
        "ece_soft_threshold": require_float(block, "ece_soft_threshold"),
        "ece_rank_penalty": require_float(block, "ece_rank_penalty"),
        "drift_soft_penalty": require_float(block, "drift_soft_penalty"),
        "drift_soft_veto_n": require_int(block, "drift_soft_veto_n"),
        "drift_score_factor": require_float(block, "drift_score_factor"),
        "drift_min_score": require_float(block, "drift_min_score"),
        "wr_raw_inconsistent_gap": require_float(block, "wr_raw_inconsistent_gap"),
    }


def reset_live_signal_metrics_config_cache() -> None:
    """Resolve ou aplica reset live signal metrics config cache."""
    _CACHE["live"] = None


def load_live_signal_metrics_from_settings() -> dict[str, Any]:
    """Resolve ou aplica load live signal metrics from settings."""
    cached = _CACHE.get("live")
    if cached is not None:
        return cached
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") if isinstance(full, dict) else None
    raw = dl.get("live_signal_metrics") if isinstance(dl, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("deep_learning.live_signal_metrics obrigatorio")
    block = require_keys(raw, _LIVE_KEYS, "deep_learning.live_signal_metrics")
    resolved = {
        "window": require_int(block, "window"),
        "min_rank": require_int(block, "min_rank"),
        "ece_bins": require_int(block, "ece_bins"),
        "ece_soft_threshold": require_float(block, "ece_soft_threshold"),
        "ece_rank_penalty": require_float(block, "ece_rank_penalty"),
        "drift_soft_penalty": require_float(block, "drift_soft_penalty"),
        "drift_soft_veto_n": require_int(block, "drift_soft_veto_n"),
        "drift_score_factor": require_float(block, "drift_score_factor"),
        "drift_min_score": require_float(block, "drift_min_score"),
        "wr_raw_inconsistent_gap": require_float(block, "wr_raw_inconsistent_gap"),
    }
    _CACHE["live"] = resolved
    return resolved
