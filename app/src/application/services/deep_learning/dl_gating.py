"""Regras de gating e cálculo de edge para inferência do Deep Learning."""


def resolve_edge(prob: float, payout: float = 0.95) -> float:
    """Calcula o edge simples baseado na probabilidade e payout."""
    if prob is None:
        return 0.0
    p = float(prob)
    p_win = max(p, 1.0 - p) if p < 0.5 else p
    return float((p_win * (1.0 + payout)) - 1.0)


def resolve_calibrated_edge(calibrated_prob: float | None, raw_prob: float | None = 0.5, payout: float = 0.95) -> float:
    """Calcula o edge com base na probabilidade do lado dominante (win probability)."""
    if calibrated_prob is None:
        return resolve_edge(raw_prob, payout)

    p = float(calibrated_prob)
    p_win = max(p, 1.0 - p) if p < 0.5 else p
    return float((p_win * (1.0 + payout)) - 1.0)


def resolve_confidence_thresholds(params: dict) -> tuple[float, float]:
    if not isinstance(params, dict):
        return (0.51, 0.49)
    return (
        float(params.get("confidence_call_threshold", 0.51)),
        float(params.get("confidence_put_threshold", 0.49)),
    )

# Backwards compatibility re-export
