"""Sentinela: feature flags temporarias nao podem estar vencidas."""

from __future__ import annotations

from datetime import date

from src.domain.config_flag_ttl import TEMPORARY_FLAGS


def test_temporary_feature_flags_not_expired():
    today = date.today()
    expired = [(name, expiry) for name, expiry in TEMPORARY_FLAGS if today > expiry]
    if expired:
        names = ", ".join(f"{n} (expiry={e.isoformat()})" for n, e in expired)
        raise AssertionError(f"Remover feature flags temporarias vencidas: {names}")
