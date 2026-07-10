"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
)
from src.domain.risk.super_concordance_kelly import is_unanimous_vote_alignment


_REGIME_TACTICAL_INVERT = frozenset({"CLIMAX_EXHAUSTION", "COMPRESSION_TRAP"})
_NEUTRAL_BANKROLL_PCT = 0.0015
_TURBO_EDGE_ZSCORE_THRESHOLD = 1.5
_TURBO_EDGE_STAKE_MULTIPLIER = 2.0
_PAYOUT_FALLBACK = 0.90
_MAX_SAFE_STAKE_BANKROLL_PCT = 0.035
_ADAPTIVE_RECOVERY_FACTOR_CAP = 2.50
_D_SQUEEZE_SOVEREIGN_TRADE_SCORE = 0.52


def _positive_float(value: object) -> float | None:
    """Converte valor numerico positivo ou retorna None quando invalido."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def resolve_contract_payout(
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
) -> float:
    """Resolve payout real do contrato com fallback estatico de seguranca."""
    candidates: list[float] = []
    if payout is not None:
        parsed = _positive_float(payout)
        if parsed is not None:
            candidates.append(parsed)
    if isinstance(risk_params, dict):
        for key in ("contract_payout", "payout", "payout_estimate"):
            parsed = _positive_float(risk_params.get(key))
            if parsed is not None:
                candidates.append(parsed)
    if candidates:
        return candidates[0]
    return _PAYOUT_FALLBACK


def adaptive_recovery_progression_factor(
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
) -> float:
    """Calcula fator adaptativo 1 + 1/payout_real com teto institucional de 2.50."""
    resolved = resolve_contract_payout(payout, risk_params)
    raw = 1.0 + (1.0 / resolved)
    return min(raw, _ADAPTIVE_RECOVERY_FACTOR_CAP)


def soft_recovery_progression_multiplier(
    consecutive_losses: int,
    *,
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
) -> float:
    """Retorna fator adaptativo^n para n perdas consecutivas na sessao."""
    losses = max(0, int(consecutive_losses))
    if losses <= 0:
        return 1.0
    factor = adaptive_recovery_progression_factor(payout, risk_params)
    return factor**losses


def resolve_session_base_unit(bankroll: float, base_unit: float, metrics: dict | None) -> float:
    """Resolve unidade base U como max(kelly, 0.15% banca) fora do D-SQUEEZE."""
    unit = max(float(base_unit), neutral_edge_dynamic_unit(bankroll))
    if isinstance(metrics, dict) and not _squeeze_floor_active(metrics):
        metrics["session_base_unit"] = unit
    return unit


def apply_soft_recovery_stake(
    *,
    pending_total: float,
    base_unit: float,
    consecutive_losses: int,
    previous_stake: float,
    bankroll: float,
    metrics: dict | None = None,
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
) -> float:
    """Aplica progressao adaptativa indexada ao payout real quando ha passivo pendente."""
    unit = resolve_session_base_unit(bankroll, base_unit, metrics)
    if float(pending_total) <= 0.0:
        return min(unit, max_safe_stake_cap(bankroll))
    factor = adaptive_recovery_progression_factor(payout, risk_params)
    resolved_payout = resolve_contract_payout(payout, risk_params)
    losses = max(0, int(consecutive_losses))
    anchor = float(previous_stake) if float(previous_stake) > 0.0 else unit
    if losses <= 0:
        stake = unit
    else:
        stake = unit * soft_recovery_progression_multiplier(
            losses,
            payout=payout,
            risk_params=risk_params,
        )
    if isinstance(metrics, dict):
        metrics["recovery_soft_progression"] = factor
        metrics["recovery_adaptive_payout"] = resolved_payout
        metrics["recovery_soft_losses"] = losses
        metrics["recovery_soft_anchor_stake"] = anchor
    cap = max_safe_stake_cap(bankroll)
    return min(stake, cap)


def max_safe_stake_cap(bankroll: float) -> float:
    """Retorna teto absoluto de exposicao: 3.5% da banca ativa."""
    return max(0.0, float(bankroll)) * _MAX_SAFE_STAKE_BANKROLL_PCT


def _squeeze_floor_active(metrics: dict) -> bool:
    """Indica disjuntor D-SQUEEZE ativo que bloqueia moduladores de edge."""
    return bool(metrics.get("meta_squeeze_downgrade") or metrics.get("consensus_stake_floor"))


def d_squeeze_sovereignty_active(metrics: dict | None) -> bool:
    """Indica barreira soberana D-SQUEEZE que revoga waiver de recovery no ciclo."""
    if not isinstance(metrics, dict):
        return False
    if _squeeze_floor_active(metrics):
        return True
    score = float(metrics.get("trade_score", metrics.get("conviction", -1.0)))
    return abs(score - _D_SQUEEZE_SOVEREIGN_TRADE_SCORE) < 1e-6


def enforce_d_squeeze_stake_floor(
    final_stake: float,
    stake_min: float,
    metrics: dict | None,
) -> float:
    """Comprime stake ao piso absoluto da API quando D-SQUEEZE revoga recovery."""
    if not d_squeeze_sovereignty_active(metrics):
        return final_stake
    if isinstance(metrics, dict):
        metrics["d_squeeze_recovery_waiver_revoked"] = True
    return float(stake_min)


def neutral_edge_dynamic_unit(bankroll: float) -> float:
    """Unidade base U em regime neutro: 0.15% da banca ativa."""
    return max(0.0, float(bankroll)) * _NEUTRAL_BANKROLL_PCT


def apply_neutral_edge_kelly_base(kelly_base: float, bankroll: float, metrics: dict | None) -> float:
    """Eleva kelly_base ao piso dinamico de 0.15% da banca fora do D-SQUEEZE."""
    if isinstance(metrics, dict) and _squeeze_floor_active(metrics):
        return kelly_base
    return resolve_session_base_unit(bankroll, float(kelly_base), metrics)


def turbo_edge_stake_multiplier(metrics: dict | None) -> float:
    """Super-alavancagem assimétrica quando Z_Edge extremo."""
    if not isinstance(metrics, dict) or _squeeze_floor_active(metrics):
        return 1.0
    if float(metrics.get("edge_zscore", 0.0)) + 1e-12 >= _TURBO_EDGE_ZSCORE_THRESHOLD:
        return _TURBO_EDGE_STAKE_MULTIPLIER
    return 1.0


def apply_turbo_edge_stake(final_stake: float, metrics: dict | None) -> float:
    """Aplica multiplicador turbo sobre stake final quando conviccao de cauda e extrema."""
    mult = turbo_edge_stake_multiplier(metrics)
    if mult > 1.0 and isinstance(metrics, dict):
        metrics["consensus_turbo_edge_active"] = True
    return float(final_stake) * mult


def _regime_tactical_inversion_active(metrics: dict) -> bool:
    """Indica inversao tatica forçada por CLIMAX_EXHAUSTION ou COMPRESSION_TRAP."""
    regime = str(metrics.get("universal_regime") or metrics.get("universal_regime_scenario") or "")
    if regime not in _REGIME_TACTICAL_INVERT:
        return False
    return bool(metrics.get("direction_inverted"))


def _recovery_waives_consensus_penalty(
    metrics: dict,
    kelly_config: dict[str, Any],
    *,
    consecutive_losses: int,
    pending_loss_total: float,
    order_direction: str | None,
) -> bool:
    """Suspende penalidade em recovery com inversao tatica, votos unanimes ou trade_score alto."""
    if d_squeeze_sovereignty_active(metrics):
        return False
    recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
    if not recovering:
        return False
    if _regime_tactical_inversion_active(metrics):
        return True
    if is_unanimous_vote_alignment(
        int(metrics.get("call_votes", 0)),
        int(metrics.get("put_votes", 0)),
        order_direction,
    ):
        return True
    cfg = kelly_config or {}
    score_min = float(cfg.get("penalty_smoothing_trade_score_min", 0.68))
    trade_score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    return trade_score + 1e-9 >= score_min


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
    consecutive_losses: int = 0,
    pending_loss_total: float = 0.0,
) -> float:
    """Retorna fator [floor, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
    if isinstance(metrics, dict) and _squeeze_floor_active(metrics):
        metrics["consensus_penalty_d_squeeze"] = True
        cfg = kelly_config if isinstance(kelly_config, dict) else {}
        return float(cfg.get("consensus_min_retention", 1.0 - float(cfg.get("consensus_max_cut", 0.50))))
    if isinstance(metrics, dict):
        recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
        if recovering and _recovery_waives_consensus_penalty(
            metrics,
            kelly_config or {},
            consecutive_losses=int(consecutive_losses),
            pending_loss_total=float(pending_loss_total),
            order_direction=order_direction,
        ):
            if _regime_tactical_inversion_active(metrics):
                metrics["consensus_penalty_regime_inversion_waived"] = True
            else:
                metrics["consensus_penalty_recovery_waived"] = True
            return 1.0
    return consensus_entropy_kelly_retention(metrics, order_direction, kelly_config=kelly_config)


__all__ = [
    "adaptive_recovery_progression_factor",
    "apply_neutral_edge_kelly_base",
    "apply_soft_recovery_stake",
    "apply_turbo_edge_stake",
    "consensus_entropy_applies_min_stake",
    "consensus_entropy_kelly_retention",
    "consensus_kelly_retention",
    "d_squeeze_sovereignty_active",
    "enforce_d_squeeze_stake_floor",
    "max_safe_stake_cap",
    "neutral_edge_dynamic_unit",
    "resolve_contract_payout",
    "resolve_session_base_unit",
    "soft_recovery_progression_multiplier",
    "turbo_edge_stake_multiplier",
]
