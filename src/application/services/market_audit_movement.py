"""Formatadores de auditoria para vela ao-vivo e previsao de movimento futuro."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.market_forming_candle import LiveCandleSnapshot


def format_live_candle_line(snapshot: LiveCandleSnapshot, *, boundary_seconds: int = 300) -> str:
    """Monta linha compacta de auditoria da vela em formacao ao vivo."""
    delta = snapshot.spot_price - snapshot.open_price
    tf = f"M{max(1, int(boundary_seconds) // 60)}" if int(boundary_seconds) % 60 == 0 else f"{boundary_seconds}s"
    return (
        f"[LIVE_CANDLE] || {tf} || {snapshot.symbol}: {snapshot.current_side} ({delta:+.5f}) | "
        f"spot={snapshot.spot_price:.5f} o={snapshot.open_price:.5f} h={snapshot.high_price:.5f} l={snapshot.low_price:.5f} | "
        f"{snapshot.elapsed_seconds}s/{boundary_seconds}s ({snapshot.progress_pct * 100.0:.0f}%) | "
        f"wicks: U={snapshot.upper_wick_ratio * 100.0:.0f}% L={snapshot.lower_wick_ratio * 100.0:.0f}%"
    )


def format_next_movement_line(symbol: str, metrics: dict[str, Any]) -> str:
    """Monta linha de projecao quantitativa do proximo movimento do preco."""
    delta_raw = metrics.get("predicted_movement_delta")
    if delta_raw is None:
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    try:
        delta = float(delta_raw)
    except (TypeError, ValueError):
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    if not math.isfinite(delta):
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    delta_pct = delta * 100.0
    side = str(metrics.get("predicted_movement_side") or ("CALL" if delta > 0 else "PUT"))
    confluence = "ALIGNED" if bool(metrics.get("movement_confluence", True)) else "DIVERGENT"
    atr_r = metrics.get("movement_atr_ratio")
    atr_tok = f"{float(atr_r):.2f}x" if atr_r is not None and math.isfinite(float(atr_r)) else "n/a"
    return (
        f"[NEXT_MOVE] || {str(symbol).upper()} || EXP_DELTA: {delta_pct:+.4f}% | "
        f"SIDE: {side} | CONFLUENCE: {confluence} | ATR_RATIO: {atr_tok}"
    )
