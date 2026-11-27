"""Kelly fracionado e amortizacao por payout, adaptados da main 460c4f41."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StakeSizing:
    """Auditoria do dimensionamento, sem confundir piso com vantagem Kelly."""

    base_stake: float
    stake: float
    kelly_raw: float
    payout_rate: float
    edge: float
    pending_loss: float
    recovery_slice: float
    recovery_steps: int
    capped: bool
    source: str


def size_kelly_recovery(
    balance: float,
    probability: float,
    payout_rate: float,
    pending_loss: float = 0.0,
    *,
    stake_min: float = 1.0,
    min_pct: float = 0.01,
    max_pct: float = 0.05,
    fraction: float = 0.25,
    max_base_pct: float | None = None,
    recovery_enabled: bool = True,
) -> StakeSizing:
    """Aplica piso continuo e amortiza perdas sem multiplicacao geometrica."""
    eff_base_pct = max_pct if max_base_pct is None else max_base_pct
    values = (balance, probability, payout_rate, pending_loss, stake_min, min_pct, max_pct, fraction, eff_base_pct)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Dimensionamento exige valores finitos")
    if balance <= 0 or payout_rate <= 0 or pending_loss < 0 or not 0 <= probability <= 1:
        raise ValueError("Saldo, payout ou probabilidade invalidos")
    if (
        stake_min < 1
        or not 0.01 <= min_pct <= max_pct <= 0.05
        or not min_pct <= eff_base_pct <= max_pct
        or not 0 < fraction <= 1
    ):
        raise ValueError("Configuracao viola limites de dimensionamento")
    cap = math.floor(balance * max_pct * 100 + 1e-9) / 100
    base_cap = math.floor(balance * eff_base_pct * 100 + 1e-9) / 100
    floor = max(stake_min, math.ceil(balance * min_pct * 100 - 1e-9) / 100)
    if floor > cap:
        raise ValueError("Saldo insuficiente para piso e teto simultaneamente")
    edge = probability * (1 + payout_rate) - 1
    raw = edge / payout_rate
    kelly = max(0.0, raw) * fraction * balance
    base = min(base_cap, max(floor, math.floor(kelly * 100 + 1e-9) / 100))
    steps = max(1, math.ceil(pending_loss / (payout_rate * cap))) if recovery_enabled else 1
    portion = pending_loss / steps if recovery_enabled else 0.0
    target = max(base, portion / payout_rate)
    stake = min(cap, math.ceil(target * 100 - 1e-9) / 100)
    source = "RECOVERY" if stake > base else ("KELLY" if kelly > floor else "FLOOR")
    return StakeSizing(
        base,
        stake,
        raw,
        payout_rate,
        edge,
        pending_loss,
        portion,
        steps,
        kelly > base_cap or pending_loss > cap * payout_rate,
        source,
    )


def sizing_from_config(balance: float, probability: float, rate: float, pending: float, config: dict) -> StakeSizing:
    """Resolve exclusivamente parametros de risco do SSOT."""
    params = config.get("params", {})
    kelly_cfg = config.get("kelly", {})
    raw_base_cap = kelly_cfg.get("max_base_pct")
    return size_kelly_recovery(
        balance,
        probability,
        rate,
        pending,
        stake_min=float(params.get("stake_min", 1.0)),
        min_pct=float(params.get("min_stake_pct", 0.01)),
        max_pct=float(params.get("max_stake_pct", 0.05)),
        fraction=float(kelly_cfg.get("fraction", 0.25)),
        max_base_pct=None if raw_base_cap is None else float(raw_base_cap),
        recovery_enabled=bool(config.get("recovery", {}).get("enabled", True)),
    )
