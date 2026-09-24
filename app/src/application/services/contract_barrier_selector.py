"""Selecao de contratos de barreira (ONETOUCH/NOTOUCH) baseada em volatilidade e ATR."""

from __future__ import annotations

import logging
from typing import Any

from src.domain.models.contract_barrier_types import BarrierContractConfig
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def load_barrier_config(config: dict[str, Any] | None) -> BarrierContractConfig:
    """Carrega configuracao de contratos de barreira a partir do config global."""
    if not isinstance(config, dict):
        return BarrierContractConfig()
    rm_cfg = config.get("risk_management", {})
    raw = rm_cfg.get("barrier_contracts") if isinstance(rm_cfg, dict) else None
    if not isinstance(raw, dict):
        return BarrierContractConfig()
    regimes_raw = raw.get("target_regimes", ("explosion", "expansion"))
    regimes_tuple = tuple(str(r).lower() for r in regimes_raw) if isinstance(regimes_raw, (list, tuple)) else ()
    return BarrierContractConfig(
        enabled=bool(raw.get("enabled", False)),
        min_atr=float(raw.get("min_atr", 1.20)),
        target_regimes=regimes_tuple or ("explosion", "expansion"),
        barrier_multiplier=float(raw.get("barrier_multiplier", 0.80)),
        default_type=str(raw.get("default_type", "ONETOUCH")).upper(),
    )


def resolve_barrier_offset(atr: float, direction: TradeDirection, multiplier: float = 0.80) -> str:
    """Calcula string de offset relativo da barreira (+X.XX ou -X.XX) com base no ATR."""
    safe_atr = max(0.01, float(atr))
    distance = round(safe_atr * max(0.1, float(multiplier)), 2)
    is_up = direction in {TradeDirection.CALL, TradeDirection.MULTUP}
    prefix = "+" if is_up else "-"
    return f"{prefix}{distance:.2f}"


def should_transition_to_barrier_contract(metrics: dict, cfg: BarrierContractConfig) -> bool:
    """Verifica se regime de volatilidade e ATR justificam migracao para contrato de barreira."""
    if not cfg.enabled:
        return False
    atr = metrics.get("atr")
    regime = str(metrics.get("volatility_regime") or metrics.get("regime") or "").lower()
    atr_val = float(atr) if isinstance(atr, (int, float)) else 0.0
    if atr_val >= cfg.min_atr:
        return True
    return bool(regime and any(target in regime for target in cfg.target_regimes))


def resolve_contract_barrier_structure(
    params: dict,
    metrics: dict,
    symbol: str,
    direction: TradeDirection,
    config: dict[str, Any] | None = None,
) -> dict:
    """Atualiza parametros de proposta para ONETOUCH se a condicao de volatilidade for atendida."""
    cfg = load_barrier_config(config)
    if not should_transition_to_barrier_contract(metrics, cfg):
        return params

    atr = float(metrics.get("atr") or 1.0)
    barrier_str = resolve_barrier_offset(atr, direction, multiplier=cfg.barrier_multiplier)
    new_params = dict(params)
    new_params["contract_type"] = cfg.default_type
    new_params["barrier"] = barrier_str

    metrics["barrier_contract_selected"] = True
    metrics["barrier_contract_type"] = cfg.default_type
    metrics["barrier_offset"] = barrier_str

    logger.info(
        "BARRIER_OPT || Selecionado %s para %s (dir=%s barreira=%s ATR=%.2f)",
        cfg.default_type,
        symbol,
        direction.name,
        barrier_str,
        atr,
    )
    return new_params
