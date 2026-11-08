"""Helpers fail-closed para blocos obrigatorios de settings."""

from __future__ import annotations

from typing import Any

import settings_io


def load_settings_json() -> dict[str, Any]:
    """Carrega config/settings.json via bootstrap I/O (fora do dominio puro)."""
    full = settings_io.load_settings_json()
    if not isinstance(full, dict):
        raise ValueError("settings.json invalido")
    return full


def require_mapping(parent: dict[str, Any] | None, key: str, required: tuple[str, ...], path: str) -> dict[str, Any]:
    """Exige submapa completo sob parent[key]."""
    cfg = parent if isinstance(parent, dict) else {}
    raw = cfg.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}.{key} obrigatorio")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"{path}.{key} incompleto: {missing}")
    return raw


def require_keys(raw: dict[str, Any] | None, required: tuple[str, ...], path: str) -> dict[str, Any]:
    """Exige dict com todas as chaves obrigatorias."""
    if not isinstance(raw, dict):
        raise ValueError(f"{path} obrigatorio")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"{path} incompleto: {missing}")
    return raw


def require_float(raw: dict[str, Any], key: str) -> float:
    """Le float obrigatorio."""
    return float(raw[key])


def require_int(raw: dict[str, Any], key: str) -> int:
    """Le int obrigatorio."""
    return int(raw[key])


def require_bool(raw: dict[str, Any], key: str) -> bool:
    """Le bool obrigatorio."""
    return bool(raw[key])


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mescla override parcial sobre base sem perder subchaves."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def merge_settings_block(path_keys: tuple[str, ...], override: dict[str, Any] | None) -> dict[str, Any]:
    """Carrega bloco aninhado do SSOT e aplica override parcial."""
    cursor: Any = load_settings_json()
    trail: list[str] = []
    for key in path_keys:
        trail.append(key)
        if not isinstance(cursor, dict) or key not in cursor or not isinstance(cursor[key], dict):
            raise ValueError(".".join(trail) + " obrigatorio")
        cursor = cursor[key]
    base = dict(cursor)
    if isinstance(override, dict) and override:
        return deep_merge(base, override)
    return base
