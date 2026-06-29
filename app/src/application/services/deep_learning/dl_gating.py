"""Utilitarios de probabilidade para execucao Deep Learning."""

from src.domain.models.trade import TradeDirection


def resolve_edge(raw_prob: float) -> float:
    """Margem da probabilidade bruta em relacao a incerteza maxima (0.5)."""
    return abs(float(raw_prob) - 0.5)


def resolve_calibrated_edge(calibrated_prob: float | None, *, raw_prob: float | None = None) -> float:
    """Margem de edge preferindo probabilidade calibrada."""
    if calibrated_prob is not None:
        return abs(float(calibrated_prob) - 0.5)
    if raw_prob is not None:
        return resolve_edge(raw_prob)
    return 0.0


def resolve_confidence_thresholds(params: dict) -> tuple[float, float]:
    """Retorna limiares CALL e PUT da configuracao."""
    return (
        float(params.get("confidence_call_threshold", 0.75)),
        float(params.get("confidence_put_threshold", 0.25)),
    )


def direction_from_raw_prob(
    raw_prob: float,
    *,
    call_threshold: float = 0.75,
    put_threshold: float = 0.25,
) -> TradeDirection | None:
    """Mapeia probabilidade bruta para CALL, PUT ou zona cinza."""
    prob = float(raw_prob)
    if prob + 1e-9 >= float(call_threshold):
        return TradeDirection.CALL
    if prob - 1e-9 <= float(put_threshold):
        return TradeDirection.PUT
    return None
