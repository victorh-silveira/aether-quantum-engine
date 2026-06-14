"""Regras de gating por threshold de confianca para execucao Deep Learning."""

from src.domain.models.trade import TradeDirection


def resolve_edge(raw_prob: float) -> float:
    """Margem da probabilidade bruta em relacao a incerteza maxima (0.5)."""
    return abs(float(raw_prob) - 0.5)


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
    """Mapeia probabilidade bruta para CALL, PUT ou abstencao."""
    prob = float(raw_prob)
    if prob + 1e-9 >= float(call_threshold):
        return TradeDirection.CALL
    if prob - 1e-9 <= float(put_threshold):
        return TradeDirection.PUT
    return None


def gating_block_reason(
    raw_prob: float,
    val_accuracy: float,
    *,
    min_val_accuracy: float = 0.53,
    call_threshold: float = 0.75,
    put_threshold: float = 0.25,
) -> str | None:
    """Retorna motivo de bloqueio ou None se executavel."""
    if val_accuracy + 1e-9 < float(min_val_accuracy):
        return "val_acc"
    if (
        direction_from_raw_prob(
            raw_prob,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
        )
        is None
    ):
        return "confidence"
    return None


def should_execute(
    raw_prob: float,
    val_accuracy: float,
    *,
    min_val_accuracy: float = 0.53,
    call_threshold: float = 0.75,
    put_threshold: float = 0.25,
) -> bool:
    """Indica se o candidato pode executar sob os limiares atuais."""
    return (
        gating_block_reason(
            raw_prob,
            val_accuracy,
            min_val_accuracy=min_val_accuracy,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
        )
        is None
    )
