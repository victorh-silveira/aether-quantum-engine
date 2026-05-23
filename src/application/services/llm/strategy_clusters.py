"""Listas US/EU do Medallion com exclusao configuravel de simbolos."""

from __future__ import annotations

from typing import Any


def resolve_cluster_lists(strategy: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """Retorna simbolos US e EU ativos apos excluded_symbols."""
    s = strategy if isinstance(strategy, dict) else {}
    clusters = s.get("clusters", {}) if isinstance(s.get("clusters"), dict) else {}
    excluded = {str(x) for x in (s.get("excluded_symbols") or [])}
    us = [str(x) for x in clusters.get("us", []) if str(x) not in excluded]
    eu = [str(x) for x in clusters.get("eu", []) if str(x) not in excluded]
    return us, eu
