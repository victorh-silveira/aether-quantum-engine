"""Concordancia temporal entre o indicador e a direcao final do motor."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ApprovedSignal:
    """Sinal M5 aprovado e vinculado a uma versao ativa do modelo."""

    symbol: str
    side: str
    candle_epoch: int
    probability: float
    model_version: str


def approved_signal_from_row(row, *, expected_epoch: int) -> ApprovedSignal | None:
    """Aceita apenas a ultima leitura aprovada da vela fechada esperada."""
    if (
        row["reason"] != "ok"
        or not row["model_active"]
        or row["side"] not in ("CALL", "PUT")
        or int(row["candle_epoch"]) != expected_epoch
        or not row["model_version"]
    ):
        return None
    probability = row["probability"]
    if probability is None or not isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        return None
    return ApprovedSignal(
        str(row["symbol"]),
        str(row["side"]),
        expected_epoch,
        float(probability),
        str(row["model_version"]),
    )


def agrees_with_execution(signal: ApprovedSignal | None, symbol: str, direction: str, epoch: int) -> bool:
    """Impede usar sinal de outro ativo, outra vela ou lado divergente."""
    return signal is not None and signal.symbol == symbol and signal.side == direction and signal.candle_epoch == epoch
