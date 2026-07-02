"""Flip de reversao a media em exaustao com volatilidade em contracao."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def _flip_opposite(direction: TradeDirection) -> TradeDirection:
    """Inverte CALL/PUT."""
    return TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL


def _contraction_flip_target(
    dl_dir: TradeDirection,
    indicators: dict,
    *,
    exec_cfg: dict,
) -> TradeDirection | None:
    """Retorna direcao contra-tendencia quando exaustao e vol em contracao."""
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    vol_max = float(exec_cfg.get("mean_reversion_contraction_vol_ratio", 0.80))
    if vol_ratio >= vol_max:
        return None
    rsi = float(indicators.get("rsi", 0.5))
    cmo = float(indicators.get("cmo", 0.0))
    rsi_ob = float(exec_cfg.get("mean_reversion_rsi_overbought", 0.72))
    cmo_ob = float(exec_cfg.get("mean_reversion_cmo_overbought", 0.45))
    rsi_os = float(exec_cfg.get("mean_reversion_rsi_oversold", 0.28))
    cmo_os = float(exec_cfg.get("mean_reversion_cmo_oversold", -0.45))
    if rsi > rsi_ob and cmo > cmo_ob:
        return _flip_opposite(dl_dir)
    if rsi < rsi_os and cmo < cmo_os:
        return _flip_opposite(dl_dir)
    return None


def apply_contraction_mean_reversion_flip(
    exec_dir: TradeDirection,
    dl_dir: TradeDirection,
    hints: list[str],
    metrics: dict,
    *,
    exec_cfg: dict,
    clamp01,
) -> tuple[TradeDirection, list[str]]:
    """Forca flip contra DL em exaustao extrema com vol_ratio em contracao rapida."""
    indicators = metrics.get("indicators") or {}
    if not isinstance(indicators, dict):
        return exec_dir, hints
    if metrics.get("compression_trap_inverted"):
        return exec_dir, hints
    flipped = _contraction_flip_target(dl_dir, indicators, exec_cfg=exec_cfg)
    if flipped is None:
        return exec_dir, hints
    margin = float(exec_cfg.get("mean_reversion_flip_margin", 0.08))
    chosen = max(float(metrics.get("direction_call_score", 0.0)), float(metrics.get("direction_put_score", 0.0)))
    side_strength = clamp01(max(0.0, chosen - margin))
    metrics["mean_reversion_expansion_flip"] = True
    metrics["trade_score"] = side_strength
    metrics["resolved_conviction"] = side_strength
    metrics["exec_direction"] = flipped.name
    metrics["resolved_direction"] = flipped.name
    metrics["direction_inverted"] = flipped != dl_dir
    if "mean_reversion_expansion_flip" not in hints:
        hints = [*hints, "mean_reversion_expansion_flip"]
    return flipped, hints
