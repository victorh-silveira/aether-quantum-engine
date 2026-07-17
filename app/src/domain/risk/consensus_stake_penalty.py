"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.risk_recovery_state import (
    MICRO_BANKROLL_THRESHOLD,
    MICRO_TAIL_LINEAR_LEVEL,
    MICRO_TAIL_UNIT_MULTIPLIER,
)
from src.domain.risk.soft_recovery_policy import (
    apply_small_account_hard_floor,
    configured_max_safe_stake_cap,
    fixed_step_progression_multiplier,
    resolve_amort_cycles,
)
from src.domain.risk.stake_sizing import consensus_entropy_kelly_retention
from src.domain.risk.super_concordance_kelly import is_unanimous_vote_alignment


_REGIME_TACTICAL_INVERT = frozenset({"CLIMAX_EXHAUSTION", "COMPRESSION_TRAP"})
_NEUTRAL_BANKROLL_PCT = 0.0015
_TURBO_EDGE_ZSCORE_THRESHOLD = 1.5
_TURBO_EDGE_STAKE_MULTIPLIER = 2.0
_PAYOUT_FALLBACK = 0.90
_MAX_SAFE_STAKE_BANKROLL_PCT = 0.035
_MAX_SAFE_STAKE_BANKROLL_PCT_LINEAR2 = 0.025
_MAX_SAFE_STAKE_BANKROLL_PCT_LINEAR3 = 0.020
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
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Aplica progressao adaptativa ou passo fixo quando ha passivo pendente."""
    unit = resolve_session_base_unit(bankroll, base_unit, metrics)
    if float(pending_total) <= 0.0:
        return min(
            unit,
            max_safe_stake_cap(bankroll, consecutive_losses_linear=consecutive_losses, soft_recovery=soft_recovery),
        )
    factor = adaptive_recovery_progression_factor(payout, risk_params)
    resolved_payout = resolve_contract_payout(payout, risk_params)
    losses = max(0, int(consecutive_losses))
    progression = soft_recovery_progression_multiplier(
        losses, payout=payout, risk_params=risk_params, soft_recovery=soft_recovery
    )
    stake = unit if losses <= 0 else unit * progression
    amort = resolve_amort_cycles(losses, soft_recovery)
    cover = float(pending_total) / resolved_payout / float(amort)
    stake = max(stake, cover)
    if isinstance(metrics, dict):
        metrics["recovery_soft_progression"] = factor
        metrics["recovery_adaptive_payout"] = resolved_payout
        metrics["recovery_soft_losses"] = losses
        metrics["recovery_soft_anchor_stake"] = float(previous_stake) if float(previous_stake) > 0.0 else unit
        metrics["recovery_cover_need"] = cover
        metrics["recovery_amort_cycles"] = amort
        metrics["recovery_fixed_step"] = (
            fixed_step_progression_multiplier(losses, soft_recovery=soft_recovery) is not None
        )
        metrics["recovery_progression_multiplier"] = float(progression)
    cap = max_safe_stake_cap(bankroll, consecutive_losses_linear=consecutive_losses, soft_recovery=soft_recovery)
    return min(stake, cap)


def max_safe_stake_cap(
    bankroll: float,
    *,
    consecutive_losses_linear: int = 0,
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Retorna teto absoluto; micro-banca <$100 limita recovery a 5% do saldo."""
    linear = max(0, int(consecutive_losses_linear))
    bal = max(0.0, float(bankroll))
    if bal <= MICRO_BANKROLL_THRESHOLD and linear >= MICRO_TAIL_LINEAR_LEVEL:
        configured = configured_max_safe_stake_cap(soft_recovery)
        raw = configured if configured is not None else MICRO_TAIL_UNIT_MULTIPLIER * neutral_edge_dynamic_unit(bal)
        return apply_small_account_hard_floor(raw, bal, soft_recovery=soft_recovery)
    pct = _MAX_SAFE_STAKE_BANKROLL_PCT
    if linear >= 3:
        pct = min(pct, _MAX_SAFE_STAKE_BANKROLL_PCT_LINEAR3)
    elif linear >= 2:
        pct = min(pct, _MAX_SAFE_STAKE_BANKROLL_PCT_LINEAR2)
    return apply_small_account_hard_floor(bal * pct, bal, soft_recovery=soft_recovery)


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
    *,
    pending_total: float = 0.0,
) -> float:
    """Comprime stake ao piso absoluto da API quando D-SQUEEZE revoga recovery."""
    if not d_squeeze_sovereignty_active(metrics):
        return final_stake
    if float(pending_total) > 0.0:
        if isinstance(metrics, dict):
            metrics["d_squeeze_floor_waived_for_recovery"] = True
        return final_stake
    if isinstance(metrics, dict):
        metrics["d_squeeze_recovery_waiver_revoked"] = True
    return float(stake_min)


def neutral_edge_dynamic_unit(bankroll: float) -> float:
    """Unidade base U em regime neutro: 1.0% da banca para micro-capital ou 0.15%."""
    pct = 0.01 if bankroll <= 250.0 else _NEUTRAL_BANKROLL_PCT
    return max(0.0, float(bankroll)) * pct


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


def cross_veto_recovery_waiver_allowed(
    metrics: dict[str, Any] | None, *, direction: str | None, risk_manager: Any | None = None
) -> bool:
    """Verifica se o waiver de recovery para o veto cruzado esta ativo e permitido."""
    if metrics is None or direction is None:
        return False
    from src.domain.risk.risk_recovery_state import meta_payoff_veto_emergency_waiver  # noqa: PLC0415

    return meta_payoff_veto_emergency_waiver(metrics, direction=direction, risk_manager=risk_manager)
