"""Granularidades OHLC aceitas pela API Deriv."""

DERIV_ALLOWED_GRANULARITY_SECONDS: tuple[int, ...] = (
    60,
    120,
    180,
    300,
    600,
    900,
    1800,
    3600,
    7200,
    14400,
    28800,
    86400,
)


def normalize_granularity_seconds(value: int | float) -> int:
    """Ajusta granularidade para o menor valor permitido pela Deriv >= ao pedido."""
    gran = max(1, int(value))
    if gran in DERIV_ALLOWED_GRANULARITY_SECONDS:
        return gran
    for allowed in DERIV_ALLOWED_GRANULARITY_SECONDS:
        if gran <= allowed:
            return allowed
    return DERIV_ALLOWED_GRANULARITY_SECONDS[-1]
