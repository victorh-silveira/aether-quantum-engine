"""Retry de proposta Deriv com reducao de stake."""

import math
from typing import Any


def is_proposal_runtime_error(exc: BaseException) -> bool:
    """Indica falha na etapa de proposal da API Deriv."""
    return isinstance(exc, RuntimeError) and "Erro na proposta:" in str(exc)


def is_retriable_proposal_error(exc: BaseException) -> bool:
    """Indica se a falha de proposta pode ser repetida com stake menor."""
    if not is_proposal_runtime_error(exc):
        return False
    msg = str(exc).lower()
    hard_fail = (
        "insufficient balance",
        "invalid symbol",
        "trading is not available",
        "market is closed",
        "id ausente",
        "resposta sem proposal",
    )
    return not any(token in msg for token in hard_fail)


def proposal_stake_attempts(stake: float, stake_min: float, scales: list[float] | None) -> list[float]:
    """Gera sequencia decrescente de stakes para tentativas de proposta."""
    base = max(float(stake), float(stake_min))
    factors = scales if scales else [0.85, 0.70, 0.55, 0.40]
    seen: set[float] = set()
    ordered: list[float] = []
    for factor in factors:
        attempt = max(float(stake_min), math.floor(base * float(factor) * 100.0) / 100.0)
        if attempt in seen:
            continue
        seen.add(attempt)
        ordered.append(attempt)
    if base not in seen:
        ordered.insert(0, base)
    return ordered or [base]


def proposal_retry_scales(execution_cfg: dict[str, Any] | None) -> list[float]:
    """Le fatores de retry da configuracao do orchestrator."""
    cfg = execution_cfg or {}
    raw = cfg.get("proposal_retry_scales")
    if not isinstance(raw, list) or not raw:
        return [0.85, 0.70, 0.55, 0.40]
    return [float(x) for x in raw if float(x) > 0.0]
