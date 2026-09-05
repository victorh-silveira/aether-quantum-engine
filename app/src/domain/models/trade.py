"""Modelos de execução de trade (Propostas, Contratos, Resultados)."""

from dataclasses import dataclass
from enum import Enum


class TradeDirection(Enum):
    """Enumeração de direções de trading possíveis."""

    CALL = "CALL"
    PUT = "PUT"
    MULTUP = "MULTUP"
    MULTDOWN = "MULTDOWN"


class TradeStatus(Enum):
    """Enumeração de status de contratos."""

    PROPOSED = "PROPOSED"
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"
    ERROR = "ERROR"


@dataclass(slots=True)
class Proposal:
    """Representa uma proposta de trade da corretora.

    Atributos:
        proposal_id (str): Identificador único da proposta.
        symbol (str): Símbolo de mercado.
        payout (float): Valor de payout potencial.
        spot (float): Preço atual no momento da proposta.
        expiry_time (int): Epoch timestamp da expiração.
    """

    proposal_id: str
    symbol: str
    direction: TradeDirection
    stake: float
    payout: float
    spot: float
    expiry_time: int


@dataclass(slots=True)
class Contract:
    """Representa um contrato de trading comprado."""

    contract_id: int
    proposal_id: str
    status: TradeStatus
    buy_price: float
    payout: float
    symbol: str
    direction: TradeDirection
    stake: float
    expiry_time: int
    profit: float | None = None
    longcode: str | None = None


@dataclass(frozen=True, slots=True)
class TradeResult:
    """Resultado de um trade finalizado.

    Atributos:
        contract_id (int): Referência ao contrato fechado.
        status (TradeStatus): Status final (WON/LOST/ERROR).
        profit (float): Lucro ou perda realizada.
        balance_after (float): Saldo da conta após a liquidação do trade.
    """

    contract_id: int
    status: TradeStatus
    profit: float
    balance_after: float
