"""Fallback por menor entropia de Shannon na probabilidade calibrada."""

from __future__ import annotations

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked
from src.application.services.meta_classifier_cross_symbol import ANCHOR_BEAR, ANCHOR_BULL
from src.domain.math.probability_entropy import binary_entropy
from src.domain.models.trade import TradeDirection


def _direction_prob(entry: dict) -> float | None:
    """Retorna probabilidade calibrada ou bruta para calculo de entropia."""
    metrics = entry.get("metrics") or {}
    calibrated = metrics.get("calibrated_prob")
    if calibrated is not None:
        return float(calibrated)
    raw = metrics.get("raw_prob")
    if raw is None:
        return None
    return float(raw)


def _direction_pivot(metrics: dict) -> float:
    """Pivot CALL/PUT a partir dos thresholds dinamicos."""
    call_th = metrics.get("dynamic_call_threshold")
    put_th = metrics.get("dynamic_put_threshold")
    if call_th is not None and put_th is not None:
        return (float(call_th) + float(put_th)) * 0.5
    return 0.5


def pick_entropy_fallback_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    skip_symbols: frozenset[str] | None = None,
    recovery_active: bool = False,
    orch=None,
    cycle_id: int = 0,
) -> tuple[str, TradeDirection, dict] | None:
    """Seleciona simbolo com menor entropia e direcao pela maior assimetria residual."""
    skip = skip_symbols or frozenset()
    best_symbol: str | None = None
    best_entry: dict | None = None
    best_entropy = float("inf")
    best_asym = -1.0
    for symbol in trade_symbols:
        if symbol in skip:
            continue
        entry = decisions.get(symbol)
        if not entry or is_technically_blocked(entry):
            continue
        prob = _direction_prob(entry)
        if prob is None:
            continue
        ent = binary_entropy(float(prob))
        asym = abs(float(prob) - 0.5)
        if ent + 1e-12 < best_entropy or (abs(ent - best_entropy) < 1e-9 and asym > best_asym):
            best_entropy = ent
            best_asym = asym
            best_symbol = symbol
            best_entry = entry
    if best_symbol is None or best_entry is None:
        return None
    metrics = dict(best_entry.get("metrics") or {})
    pivot = _direction_pivot(metrics)
    prob = float(_direction_prob(best_entry) or 0.5)
    direction = infer_dl_direction(best_entry)
    if direction is None:
        if ANCHOR_BULL != ANCHOR_BEAR and best_symbol == ANCHOR_BULL:
            direction = TradeDirection.CALL
        elif ANCHOR_BULL != ANCHOR_BEAR and best_symbol == ANCHOR_BEAR:
            direction = TradeDirection.PUT
        else:
            direction = TradeDirection.CALL if prob > pivot else TradeDirection.PUT
    active_cycle = int(cycle_id or 0)
    if orch is not None:
        active_cycle = int(getattr(orch, "_active_cycle_id", 0) or active_cycle or 0)
    candidate = build_execution_candidate(
        best_symbol,
        {**best_entry, "direction": direction, "metrics": metrics},
        recovery_active=recovery_active,
        orch=orch,
        cycle_id=active_cycle,
        decisions=decisions,
        risk_manager=getattr(orch, "risk_manager", None) if orch is not None else None,
    )
    if candidate is None:
        return None
    sym, _, out_metrics = candidate
    out_metrics["execution_mode"] = "EXEC_FALLBACK"
    out_metrics["fallback_reason"] = "entropy_min"
    out_metrics["entropy_pick"] = best_entropy
    out_metrics["residual_asymmetry"] = best_asym
    return sym, direction, out_metrics
