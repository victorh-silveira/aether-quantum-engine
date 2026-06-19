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
    min_edge: float = 0.0,
) -> str | None:
    """Retorna motivo de bloqueio ou None se executavel."""
    if val_accuracy + 1e-9 < float(min_val_accuracy):
        return "val_acc"
    if resolve_edge(raw_prob) + 1e-9 < float(min_edge):
        return "edge"
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
    min_edge: float = 0.0,
) -> bool:
    """Indica se o candidato pode executar sob os limiares atuais."""
    return (
        gating_block_reason(
            raw_prob,
            val_accuracy,
            min_val_accuracy=min_val_accuracy,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
            min_edge=min_edge,
        )
        is None
    )


def check_indicator_gating_bounds(
    indicators: dict[str, float],
    indicator_cfg: dict,
) -> str | None:
    """Verifica se os indicadores tecnicos estao dentro dos limites configurados."""
    if not indicator_cfg.get("enabled", False):
        return None

    checks = [
        ("hurst", "hurst_min", "hurst_max", "indicator_hurst", 0.5, 0.0, 1.0),
        ("vol_ratio_short_long", "vol_ratio_min", "vol_ratio_max", "indicator_vol_ratio", 1.0, 0.0, 999.0),
        ("cmo", "cmo_min", "cmo_max", "indicator_cmo", 0.0, -1.0, 1.0),
        ("keltner_pct_b", "keltner_pct_b_min", "keltner_pct_b_max", "indicator_keltner", 0.5, -999.0, 999.0),
    ]

    for key, min_key, max_key, err_code, default_val, default_min, default_max in checks:
        val = float(indicators.get(key, default_val))
        if val < float(indicator_cfg.get(min_key, default_min)) or val > float(indicator_cfg.get(max_key, default_max)):
            return err_code

    adx = float(indicators.get("adx", 0.0))
    if adx < float(indicator_cfg.get("adx_min", 0.0)):
        return "indicator_adx"

    return None
