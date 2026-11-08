"""Registro de feature flags temporarias com data de expiracao."""

from __future__ import annotations

from datetime import date


TEMPORARY_FLAGS: list[tuple[str, date]] = []
