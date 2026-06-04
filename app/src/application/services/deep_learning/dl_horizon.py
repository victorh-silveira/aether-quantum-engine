"""Horizonte de label alinhado ao contrato Deriv e granularidade OHLC."""

from typing import Any


def contract_duration_seconds(risk_params: dict[str, Any]) -> int:
    """Converte duration + duration_unit do risk_management em segundos."""
    unit = str(risk_params.get("duration_unit", "m")).strip().lower()
    duration = max(1, int(risk_params.get("duration", 1)))
    if unit in ("s", "sec", "second", "seconds"):
        return duration
    if unit in ("h", "hr", "hour", "hours"):
        return duration * 3600
    return duration * 60


def resolve_label_horizon_bars(
    granularity_seconds: int,
    risk_params: dict[str, Any],
    dl_config: dict[str, Any] | None = None,
) -> int:
    """Barras de lookahead do label igual a duracao do contrato na mesma granularidade."""
    dl_config = dl_config or {}
    explicit = dl_config.get("label_horizon_bars")
    if explicit is not None:
        return max(1, int(explicit))
    gran = max(60, int(granularity_seconds))
    contract_sec = contract_duration_seconds(risk_params)
    return max(1, int(round(contract_sec / float(gran))))
