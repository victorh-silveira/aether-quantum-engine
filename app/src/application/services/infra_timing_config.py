"""Timeouts e knobs de timing/infra a partir de settings.json."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.domain.config_knobs import (
    merge_settings_block,
    require_bool,
    require_float,
    require_int,
    require_keys,
    require_mapping,
)


_ORCH_TIMING_KEYS = (
    "warm_up_live_data_timeout_seconds",
    "broker_handshake_timeout_seconds",
    "state_lock_acquire_timeout_seconds",
    "api_maintenance_fallback_seconds",
    "recovery_warm_up_delay_seconds",
    "recovery_pending_warm_up_max_seconds",
    "stream_warm_up_delay_seconds",
    "settlement_tolerance_window_seconds",
    "ws_connect",
)

_WS_CONNECT_KEYS = (
    "max_attempts",
    "open_timeout_seconds",
    "retry_delay_seconds",
    "retry_backoff",
    "subscribe_transaction_timeout_seconds",
)

_STREAM_RECONNECT_KEYS = ("max_attempts", "initial_backoff_seconds", "max_backoff_seconds")
_HISTORY_FETCH_KEYS = (
    "chunk",
    "delay_seconds",
    "symbol_delay_seconds",
    "rate_limit_retries",
    "rate_limit_backoff",
    "rate_limit_max_delay",
)
_META_SHADOW_KEYS = ("window", "min_pairs", "ready_n", "hard_corr_floor", "soft_only_corr_ceiling")
_META_CLIENT_KEYS = (
    "timeout_seconds",
    "max_connections",
    "max_keepalive_connections",
    "online_learn",
    "retrain_min_n",
    "max_buffer",
    "shadow",
)

_CACHE: dict[str, Any] = {}


def _load_settings() -> dict[str, Any]:
    """Resolve ou aplica  load settings."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    if not isinstance(full, dict):
        raise ValueError("settings.json invalido")
    return full


def resolve_orchestrator_timing_config(orch: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs de timing do orchestrator; overrides parciais mesclam no SSOT."""
    settings_orch = _load_settings().get("orchestrator")
    base = dict(settings_orch) if isinstance(settings_orch, dict) else {}
    override = dict(orch) if isinstance(orch, dict) else {}
    base_ws = base.get("ws_connect") if isinstance(base.get("ws_connect"), dict) else {}
    override_ws = override.get("ws_connect") if isinstance(override.get("ws_connect"), dict) else {}
    merged = {**base, **override}
    merged["ws_connect"] = {**base_ws, **override_ws}
    block = require_keys(merged, _ORCH_TIMING_KEYS, "orchestrator")
    ws = require_keys(block.get("ws_connect"), _WS_CONNECT_KEYS, "orchestrator.ws_connect")
    return {
        "warm_up_live_data_timeout_seconds": require_float(block, "warm_up_live_data_timeout_seconds"),
        "broker_handshake_timeout_seconds": require_float(block, "broker_handshake_timeout_seconds"),
        "state_lock_acquire_timeout_seconds": require_float(block, "state_lock_acquire_timeout_seconds"),
        "api_maintenance_fallback_seconds": require_float(block, "api_maintenance_fallback_seconds"),
        "recovery_warm_up_delay_seconds": require_float(block, "recovery_warm_up_delay_seconds"),
        "recovery_pending_warm_up_max_seconds": require_float(block, "recovery_pending_warm_up_max_seconds"),
        "stream_warm_up_delay_seconds": require_float(block, "stream_warm_up_delay_seconds"),
        "settlement_tolerance_window_seconds": require_float(block, "settlement_tolerance_window_seconds"),
        "ws_connect": {
            "max_attempts": require_int(ws, "max_attempts"),
            "open_timeout_seconds": require_float(ws, "open_timeout_seconds"),
            "retry_delay_seconds": require_float(ws, "retry_delay_seconds"),
            "retry_backoff": require_float(ws, "retry_backoff"),
            "subscribe_transaction_timeout_seconds": require_float(ws, "subscribe_transaction_timeout_seconds"),
        },
    }


def resolve_stream_reconnect_config(api_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve stream_reconnect com merge de override parcial sobre o SSOT."""
    override = None
    if isinstance(api_config, dict):
        nested = api_config.get("stream_reconnect")
        if isinstance(nested, dict):
            override = nested
        elif "max_attempts" in api_config:
            override = api_config
    raw = merge_settings_block(("api_config", "stream_reconnect"), override)
    block = require_keys(raw, _STREAM_RECONNECT_KEYS, "api_config.stream_reconnect")
    return {
        "max_attempts": require_int(block, "max_attempts"),
        "initial_backoff_seconds": require_float(block, "initial_backoff_seconds"),
        "max_backoff_seconds": require_float(block, "max_backoff_seconds"),
    }


def resolve_history_fetch_config(api_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve history_fetch com merge de override parcial sobre o SSOT."""
    override = None
    if isinstance(api_config, dict):
        nested = api_config.get("history_fetch")
        if isinstance(nested, dict):
            override = nested
        elif "chunk" in api_config:
            override = api_config
    raw = merge_settings_block(("api_config", "history_fetch"), override)
    block = require_keys(raw, _HISTORY_FETCH_KEYS, "api_config.history_fetch")
    return {
        "chunk": require_int(block, "chunk"),
        "delay_seconds": require_float(block, "delay_seconds"),
        "symbol_delay_seconds": require_float(block, "symbol_delay_seconds"),
        "rate_limit_retries": require_int(block, "rate_limit_retries"),
        "rate_limit_backoff": require_float(block, "rate_limit_backoff"),
        "rate_limit_max_delay": require_float(block, "rate_limit_max_delay"),
    }


def resolve_meta_classifier_infra_config(infra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve meta classifier infra config."""
    cfg = infra if isinstance(infra, dict) else _load_settings().get("infra")
    parent = cfg if isinstance(cfg, dict) else {}
    raw = require_mapping(parent, "meta_classifier", _META_CLIENT_KEYS, "infra")
    shadow = require_keys(raw.get("shadow"), _META_SHADOW_KEYS, "infra.meta_classifier.shadow")
    return {
        "timeout_seconds": require_float(raw, "timeout_seconds"),
        "max_connections": require_int(raw, "max_connections"),
        "max_keepalive_connections": require_int(raw, "max_keepalive_connections"),
        "online_learn": require_bool(raw, "online_learn"),
        "retrain_min_n": require_int(raw, "retrain_min_n"),
        "max_buffer": require_int(raw, "max_buffer"),
        "shadow": {
            "window": require_int(shadow, "window"),
            "min_pairs": require_int(shadow, "min_pairs"),
            "ready_n": require_int(shadow, "ready_n"),
            "hard_corr_floor": require_float(shadow, "hard_corr_floor"),
            "soft_only_corr_ceiling": require_float(shadow, "soft_only_corr_ceiling"),
        },
    }
