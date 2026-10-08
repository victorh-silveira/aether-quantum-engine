"""Validacao dos parametros de probabilidade da compra Rise/Fall."""

import math


def validate_quote_guard_params(params: dict, execution_config: dict) -> None:
    """Bloqueia compra sem probabilidade finita quando o quote guard e exigido."""
    raw = params.get("_quote_guard_side_probability")
    if raw is None:
        if bool(execution_config.get("require_quote_edge", False)):
            raise RuntimeError("Cotacao final Rise/Fall sem probabilidade valida; compra bloqueada")
        return
    try:
        probability = float(raw)
        min_edge = float(params.get("_quote_guard_min_edge", 0.0))
        margin = float(params.get("_quote_guard_safety_margin", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cotacao final Rise/Fall sem probabilidade valida; compra bloqueada") from exc
    if not all(math.isfinite(value) for value in (probability, min_edge, margin)) or not 0 <= probability <= 1:
        raise RuntimeError("Cotacao final Rise/Fall sem probabilidade valida; compra bloqueada")
