"""Filtro de exaustao de barreira micro M1: veta compra de topo e venda de fundo saturados."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


MICRO_BOUNDARY_SCORE_CAP = 0.55
KELTNER_UPPER_EXHAUSTION = 1.10
KELTNER_LOWER_EXHAUSTION = -0.10
BB_UPPER_SATURATION = 0.95
BB_LOWER_SATURATION = 0.05


def _micro_boundary_levels(metrics: dict) -> tuple[float, float]:
    """Extrai keltner_pct_b e bollinger_pct_b micro M1 com fallback neutro."""
    indicators = metrics.get("indicators") or {}
    keltner = float(indicators.get("keltner", 0.5))
    bb_pct_b = float(indicators.get("bb_pct_b", 0.5))
    return keltner, bb_pct_b


def _is_upper_saturation(keltner: float, bb_pct_b: float) -> bool:
    """Topo saturado: acima do teto Keltner ou ultimo tick colado na banda superior de Bollinger."""
    return keltner > KELTNER_UPPER_EXHAUSTION or bb_pct_b + 1e-9 >= BB_UPPER_SATURATION


def _is_lower_saturation(keltner: float, bb_pct_b: float) -> bool:
    """Fundo saturado: abaixo do piso Keltner ou ultimo tick colado na banda inferior de Bollinger."""
    return keltner < KELTNER_LOWER_EXHAUSTION or bb_pct_b <= BB_LOWER_SATURATION + 1e-9


def validate_micro_boundary_exhaustion(exec_dir: TradeDirection, metrics: dict) -> bool:
    """Rebaixa o trade_score para 0.55 quando a direcao final compra topo ou vende fundo micro."""
    keltner, bb_pct_b = _micro_boundary_levels(metrics)
    if exec_dir == TradeDirection.CALL:
        saturated = _is_upper_saturation(keltner, bb_pct_b)
        side = "upper"
    else:
        saturated = _is_lower_saturation(keltner, bb_pct_b)
        side = "lower"
    if not saturated:
        return False
    current = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    metrics["trade_score"] = min(current, MICRO_BOUNDARY_SCORE_CAP)
    metrics["micro_boundary_exhaustion"] = True
    metrics["micro_boundary_side"] = side
    metrics["micro_boundary_keltner"] = keltner
    metrics["micro_boundary_bb_pct_b"] = bb_pct_b
    return True
