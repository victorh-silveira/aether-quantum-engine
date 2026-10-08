"""Resolucao de App ID e persistencia de credenciais Deriv."""

import json
import os
from pathlib import Path
from typing import Any

from src.infrastructure.api.deriv_pat_binding import DerivPatBindingError, discover_app_id_for_pat


LEGACY_DERIV_APP_IDS = frozenset({"1089", "16929", "36544"})


def _load_settings_deriv_app_id(repo_root: Path | None) -> str:
    """Le deriv_app_id de config/settings.json quando disponivel."""
    if repo_root is None:
        return ""
    settings_path = repo_root / "config" / "settings.json"
    if not settings_path.is_file():
        return ""
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    api = data.get("api_config")
    if not isinstance(api, dict):
        return ""
    raw = api.get("deriv_app_id")
    return str(raw).strip() if raw else ""


def resolve_deriv_app_id(
    explicit: str | None = None,
    *,
    repo_root: Path | None = None,
    config: dict[str, Any] | None = None,
    pat: str | None = None,
) -> str:
    """Resolve App ID a partir de argumentos, PAT, env, config ou settings."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if pat and pat.strip().startswith("pat_") and repo_root is not None:
        try:
            return discover_app_id_for_pat(
                pat.strip(),
                repo_root,
                explicit=os.getenv("AETHER_DERIV_APP_ID") or None,
            )
        except DerivPatBindingError:
            pass
    for key in ("AETHER_DERIV_APP_ID", "DERIV_APP_ID"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()
    if config:
        api = config.get("api_config")
        if isinstance(api, dict):
            cfg_val = api.get("deriv_app_id")
            if cfg_val and str(cfg_val).strip():
                return str(cfg_val).strip()
    from_settings = _load_settings_deriv_app_id(repo_root)
    if from_settings:
        return from_settings
    return ""


def is_legacy_deriv_app_id(app_id: str) -> bool:
    """Indica App IDs legados que nao funcionam com PAT moderna."""
    return app_id in LEGACY_DERIV_APP_IDS


def looks_like_pat(value: str) -> bool:
    """Indica se o valor parece um token PAT Deriv."""
    return value.startswith("pat_")


def persist_deriv_app_id(repo_root: Path, app_id: str) -> Path:
    """Grava ou atualiza AETHER_DERIV_APP_ID no .env do repositorio."""
    env_path = repo_root / ".env"
    key = "AETHER_DERIV_APP_ID="
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.startswith(key):
            out.append(f"{key}{app_id}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}{app_id}")
    env_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return env_path
