"""Cache em disco de respostas Gemini por barra M15 no backtest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Carrega mapa bar_index -> payload LLM."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = str(row.get("bar_index", ""))
            payload = row.get("payload")
            if key and isinstance(payload, dict):
                out[key] = payload
    return out


def save_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Persiste cache JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(entries.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            f.write(json.dumps({"bar_index": int(key), "payload": entries[key]}, ensure_ascii=False))
            f.write("\n")


def append_cache_entry(path: Path, bar_index: int, payload: dict[str, Any]) -> None:
    """Acrescenta uma barra ao cache (seguro interromper e retomar depois)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"bar_index": int(bar_index), "payload": payload}, ensure_ascii=False))
        f.write("\n")
