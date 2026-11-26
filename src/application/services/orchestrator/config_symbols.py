"""Normalizacao de simbolo ancora e lista de simbolos ativos."""

from __future__ import annotations

from typing import Any


def normalize_symbols_and_anchor(config: dict[str, Any]) -> tuple[str, list[str]]:
    """Deriva ancora e lista de simbolos ativos a partir de config/settings.json."""
    anchor = str(config.get("anchor", "R_10"))
    raw = [str(s) for s in (config.get("symbols") or [anchor])]
    symbols = list(dict.fromkeys(raw))
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols
