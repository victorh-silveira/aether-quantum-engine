"""Helpers de payout, turbo edge e waiver de consensus stake."""

from __future__ import annotations

from typing import Any

from src.domain.risk.soft_recovery_policy import fixed_step_progression_multiplier
from src.domain.risk.super_concordance_kelly import is_unanimous_vote_alignment


_REGIME_TACTICAL_INVERT = frozenset({"CLIMAX_EXHAUSTION", "COMPRESSION_TRAP"})
_NEUTRAL_BANKROLL_PCT = 0.0015
_TURBO_EDGE_ZSCORE_THRESHOLD = 1.5
_TURBO_EDGE_STAKE_MULTIPLIER = 2.0
_PAYOUT_FALLBACK = 0.90
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
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Retorna fator de progressao; niveis 3-4 usam passo fixo U+15%."""
    losses = max(0, int(consecutive_losses))
    if losses <= 0:
        return 1.0
    fixed = fixed_step_progression_multiplier(losses, soft_recovery=soft_recovery)
    return float(fixed) if fixed is not None else adaptive_recovery_progression_factor(payout, risk_params) ** losses


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


def neutral_edge_dynamic_unit(bankroll: float) -> float:
    """Unidade base U em regime neutro: 1.0% da banca para micro-capital ou 0.15%."""
    pct = 0.01 if bankroll <= 250.0 else _NEUTRAL_BANKROLL_PCT
    return max(0.0, float(bankroll)) * pct


def turbo_edge_stake_multiplier(metrics: dict | None) -> float:
    """Super-alavancagem assimétrica quando Z_Edge extremo e live saudavel."""
    if not isinstance(metrics, dict) or _squeeze_floor_active(metrics):
        return 1.0
    live_n = int(metrics.get("live_n", 0) or 0)
    if live_n < 20:
        return 1.0
    live_brier = metrics.get("live_brier")
    if live_brier is not None:
        try:
            if float(live_brier) > 0.22:
                return 1.0
        except (TypeError, ValueError):
            return 1.0
    if float(metrics.get("edge_zscore", 0.0)) + 1e-12 >= _TURBO_EDGE_ZSCORE_THRESHOLD:
        return _TURBO_EDGE_STAKE_MULTIPLIER
    return 1.0


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
