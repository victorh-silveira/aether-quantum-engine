"""Resolucao Softmax defensiva em conflito DL versus exaustao."""

from __future__ import annotations

import math

from src.domain.models.trade import TradeDirection


def _softmax_pair(a: float, b: float, temperature: float) -> tuple[float, float]:
    """Softmax estavel para par de scores."""
    temp = max(0.05, float(temperature))
    sa = float(a) / temp
    sb = float(b) / temp
    m = max(sa, sb)
    ea = math.exp(sa - m)
    eb = math.exp(sb - m)
    total = ea + eb
    if total <= 0.0:
        return 0.5, 0.5
    return ea / total, eb / total


def _vol_temperature(metrics: dict) -> float:
    """Temperatura Softmax escalonada pela volatilidade implicita."""
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("implied_vol_ratio", indicators.get("vol_ratio", 1.0)))
    return max(0.08, min(0.45, 0.22 / max(0.5, vol_ratio)))


def _severe_exhaustion_conflict(metrics: dict, dl_dir: TradeDirection, *, cfg: dict | None) -> bool:
    """Detecta contratendencia severa entre DL e bloco de exaustao."""
    if not metrics.get("exhaustion_conflict") and not metrics.get("exhaustion_hard_gate"):
        return False
    gate = (cfg or {}).get("exhaustion_gate") if isinstance(cfg, dict) else {}
    gate = gate if isinstance(gate, dict) else {}
    indicators = metrics.get("indicators") or {}
    rsi = float(indicators.get("rsi", 0.5))
    cmo = float(indicators.get("cmo", 0.0))
    rsi_ob = float(gate.get("rsi_overbought", 0.73))
    rsi_os = float(gate.get("rsi_oversold", 0.27))
    cmo_bull = float(gate.get("cmo_bull", 0.48))
    cmo_bear = float(gate.get("cmo_bear", -0.48))
    if dl_dir == TradeDirection.CALL and (rsi > rsi_ob or cmo > cmo_bull):
        return True
    if dl_dir == TradeDirection.PUT and (rsi < rsi_os or cmo < cmo_bear):
        return True
    return float(metrics.get("exhaustion_penalty", 0.0)) >= float(gate.get("min_penalty_skip", 0.12))


def resolve_softmax_exhaustion_conflict(
    exec_dir: TradeDirection,
    dl_dir: TradeDirection,
    metrics: dict,
    *,
    call_score: float,
    put_score: float,
    exec_cfg: dict | None = None,
) -> tuple[TradeDirection, dict]:
    """Converte conflito DL/exaustao em ordem defensiva sem rejeitar execucao."""
    if not _severe_exhaustion_conflict(metrics, dl_dir, cfg=exec_cfg):
        return exec_dir, metrics
    temp = _vol_temperature(metrics)
    exhaustion_call = put_score
    exhaustion_put = call_score
    if dl_dir == TradeDirection.CALL:
        p_dl, p_ex = _softmax_pair(call_score, exhaustion_put, temp)
        resolved = TradeDirection.PUT if p_ex + 1e-9 >= p_dl else TradeDirection.CALL
    else:
        p_dl, p_ex = _softmax_pair(put_score, exhaustion_call, temp)
        resolved = TradeDirection.CALL if p_ex + 1e-9 >= p_dl else TradeDirection.PUT
    vol_scale = max(0.35, min(1.0, 0.55 + 0.25 * float(metrics.get("indicators", {}).get("implied_vol_ratio", 1.0))))
    prev_scale = float(metrics.get("kelly_fraction_scale", 1.0))
    metrics["kelly_fraction_scale"] = prev_scale * vol_scale
    metrics["execution_mode"] = "EXEC_DIVERGENT"
    metrics["divergent_dl"] = dl_dir.name
    metrics["divergent_resolved"] = resolved.name
    metrics["divergent_vol_scale"] = vol_scale
    hints = list(metrics.get("direction_hints") or [])
    if "softmax_exhaustion_conflict" not in hints:
        hints.append("softmax_exhaustion_conflict")
    metrics["direction_hints"] = hints
    return resolved, metrics
