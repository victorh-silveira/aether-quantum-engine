"""Concordancia temporal entre o indicador e a direcao final do motor."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ApprovedSignal:
    """Sinal M5 aprovado e vinculado a uma versao ativa do modelo."""

    symbol: str
    side: str
    candle_epoch: int
    probability: float | None
    model_version: str
    period: int = 300


def approved_signal_from_row(row, *, expected_epoch: int) -> ApprovedSignal | None:
    """Aceita apenas a ultima leitura aprovada da vela fechada esperada."""
    if (
        row["reason"] not in ("ok", "setup_confirmado")
        or not row["model_active"]
        or row["side"] not in ("CALL", "PUT")
        or int(row["candle_epoch"]) != expected_epoch
        or not row["model_version"]
    ):
        return None
    probability = row["probability"]
    if probability is None:
        if not str(row["model_version"]).startswith("pullback-1-"):
            return None
    elif not isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        return None
    return ApprovedSignal(
        str(row["symbol"]),
        str(row["side"]),
        expected_epoch,
        float(probability) if probability is not None else None,
        str(row["model_version"]),
        int(row["period_seconds"]) if "period_seconds" in row else 300,
    )


def agrees_with_execution(signal: ApprovedSignal | None, symbol: str, direction: str, epoch: int) -> bool:
    """Impede usar sinal de outro ativo, outra vela ou lado divergente."""
    return signal is not None and signal.symbol == symbol and signal.side == direction and signal.candle_epoch == epoch
