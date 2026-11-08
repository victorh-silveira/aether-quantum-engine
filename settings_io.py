"""Leitura de config/settings.json fora das camadas DDD (I/O de bootstrap)."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path


def load_settings_json() -> dict[str, Any]:
    """Carrega config/settings.json como dict raiz."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    if not isinstance(full, dict):
        raise ValueError("settings.json invalido")
    return full
