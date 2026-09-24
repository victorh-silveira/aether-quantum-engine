"""Tipos de dominio e dataclasses para contratos de barreira (Touch / No-Touch)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContractFamily(Enum):
    """Familias de contratos digitais suportadas."""

    RISE_FALL = "RISE_FALL"
    TOUCH = "TOUCH"
    NO_TOUCH = "NO_TOUCH"


@dataclass(slots=True)
class BarrierContractConfig:
    """Configuracao de transicao para contratos de barreira."""

    enabled: bool = False
    min_atr: float = 1.20
    target_regimes: tuple[str, ...] = ("explosion", "expansion")
    barrier_multiplier: float = 0.80
    default_type: str = "ONETOUCH"


@dataclass(slots=True)
class ResolvedBarrierContract:
    """Resultado da selecao estrutural de contrato."""

    contract_type: str
    barrier: str | None
    duration: int
    duration_unit: str
    is_barrier: bool
