"""Horizonte de label alinhado ao contrato Deriv e granularidade OHLC."""

from typing import Any


def contract_duration_seconds(risk_params: dict[str, Any]) -> int:
    """Converte duration + duration_unit do risk_management em segundos."""
    unit = str(risk_params.get("duration_unit", "m")).strip().lower()
    duration = max(1, int(risk_params.get("duration", 1)))
    if unit in ("s", "sec", "second", "seconds"):
        return duration
    if unit in ("t", "tick", "ticks"):
        return duration * 2
    if unit in ("h", "hr", "hour", "hours"):
        return duration * 3600
    if unit in ("d", "day", "days"):
        return duration * 86400
    return duration * 60


def resolve_label_smooth_bars(dl_config: dict[str, Any] | None) -> int:
    """Barras forward usadas na media movel do alvo (suaviza ruido do label)."""
    dl_config = dl_config or {}
    return max(1, int(dl_config.get("label_smooth_bars", 1)))


def resolve_label_ma_window(dl_config: dict[str, Any] | None) -> int:
    """Janela da media movel atual usada no modo ma_trend."""
    dl_config = dl_config or {}
    return max(1, int(dl_config.get("label_ma_window", 5)))


def resolve_label_mode(dl_config: dict[str, Any] | None) -> str:
    """Modo do rotulo: spot_forward ou ma_trend."""
    dl_config = dl_config or {}
    raw = str(dl_config.get("label_mode", "ma_trend")).strip().lower()
    if raw in ("spot", "spot_forward", "next_candle"):
        return "spot_forward"
    return "ma_trend"


def resolve_implied_vol_bars(dl_config: dict[str, Any] | None) -> int:
    """Barras para volatilidade realizada relativa ao alvo sintetico."""
    dl_config = dl_config or {}
    return max(12, int(dl_config.get("implied_vol_bars", 60)))


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
    gran = max(1, int(granularity_seconds))
    contract_sec = contract_duration_seconds(risk_params)
    return max(1, int(round(contract_sec / float(gran))))
