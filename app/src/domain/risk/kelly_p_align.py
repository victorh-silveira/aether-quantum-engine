"""Alinha p Kelly ao lado da ordem e garante f* positivo via piso de probabilidade."""

from typing import Any


def resolve_kelly_p_floor(kelly_config: dict[str, Any] | None) -> float:
    """Le kelly_p_floor SSOT e limita ao intervalo operacional [0.51, 0.65]."""
    cfg = kelly_config if isinstance(kelly_config, dict) else {}
    raw = cfg.get("kelly_p_floor", cfg.get("adapt_kelly_p_floor", 0.55))
    return max(0.51, min(0.65, float(raw)))


def kelly_breakeven_p(payout: float) -> float:
    """Probabilidade minima para Kelly bruto nao-negativo dado payout b."""
    b = float(payout)
    if b <= 0.0:
        return 1.0
    return 1.0 / (1.0 + b)


def ensure_kelly_edge_p(p: float, payout: float, kelly_config: dict[str, Any] | None) -> float:
    """Eleva p acima do breakeven e do piso SSOT para f* sempre positivo."""
    floor = resolve_kelly_p_floor(kelly_config)
    be = kelly_breakeven_p(payout)
    return max(float(p), floor, be + 1e-4)


def calculate_kelly_fraction(
    rm: Any,
    symbol: str,
    conviction: float,
    dl_metrics: dict | None,
    *,
    order_direction: Any = None,
) -> tuple[float, float, float]:
    """Calcula fracao Kelly bruta e retorna (f, payout b, probabilidade p)."""
    b = float(rm.risk_params.get("payout_estimate", 0.95))
    metrics = dl_metrics if isinstance(dl_metrics, dict) else None
    direction = order_direction.name if hasattr(order_direction, "name") else order_direction
    aligned = apply_kelly_side_p(
        metrics,
        order_direction=str(direction) if direction is not None else None,
        kelly_config=rm.kelly_config,
        conviction=conviction,
        payout=b,
    )
    p = rm.effective_win_rate(symbol, aligned, metrics=metrics)
    p = ensure_kelly_edge_p(p, b, rm.kelly_config)
    if metrics is not None:
        metrics["kelly_side_p"] = float(p)
        metrics["kelly_p_floored"] = True
        metrics["trade_score"] = float(p)
        metrics["conviction"] = float(p)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    return max(0.0, float(kelly_f)), b, p


def _read_call_prob(metrics: dict[str, Any]) -> float | None:
    """Extrai P(CALL) de calibrated_prob ou raw_prob."""
    for key in ("calibrated_prob", "raw_prob"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_direction(metrics: dict[str, Any] | None, order_direction: str | None) -> str | None:
    """Normaliza direcao da ordem ou das metricas."""
    side = str(order_direction or "").strip().upper()
    if side in ("CALL", "PUT"):
        return side
    if not isinstance(metrics, dict):
        return None
    for key in ("exec_direction", "resolved_direction", "order_direction"):
        side = str(metrics.get(key) or "").strip().upper()
        if side in ("CALL", "PUT"):
            return side
    return None


def side_confidence(cal_call: float, direction: str, *, metrics: dict[str, Any] | None = None) -> float:
    """Confianca do lado executado; em adapt usa magnitude do TCN no novo lado."""
    cal = float(cal_call)
    if isinstance(metrics, dict) and bool(metrics.get("scale_adapted")):
        tcn = str(metrics.get("tcn_direction") or "").upper()
        if tcn in ("CALL", "PUT") and tcn != direction:
            return (1.0 - cal) if tcn == "PUT" else cal
    return (1.0 - cal) if direction == "PUT" else cal


def apply_kelly_side_p(
    metrics: dict[str, Any] | None,
    *,
    order_direction: str | None,
    kelly_config: dict[str, Any] | None,
    conviction: float,
    payout: float | None = None,
) -> float:
    """Alinha confianca ao lado da ordem, aplica piso e opcionalmente breakeven do payout."""
    floor = resolve_kelly_p_floor(kelly_config)
    base = float(conviction)
    direction = _resolve_direction(metrics, order_direction)
    adapted_sync = False
    if isinstance(metrics, dict) and direction is not None:
        cal = _read_call_prob(metrics)
        if cal is not None:
            base = side_confidence(cal, direction, metrics=metrics)
            adapted_sync = bool(metrics.get("scale_adapted")) and str(
                metrics.get("tcn_direction") or ""
            ).upper() not in (
                "",
                direction,
            )
    p = max(base, floor)
    if payout is not None:
        p = ensure_kelly_edge_p(p, float(payout), kelly_config)
    if isinstance(metrics, dict):
        metrics["kelly_p_floored"] = bool(p > base + 1e-12)
        metrics["kelly_side_p"] = float(p)
        metrics["trade_score"] = float(p)
        metrics["conviction"] = float(p)
        if adapted_sync:
            metrics["scale_kelly_side_p"] = float(p)
            metrics["scale_kelly_side_synced"] = True
        else:
            metrics.setdefault("scale_kelly_side_synced", False)
    return float(p)
