"""Helpers de payout, turbo edge e waiver de consensus stake."""

from __future__ import annotations

from typing import Any

from src.domain.risk.kelly_runtime_config import kelly_runtime_from_config, load_kelly_runtime_from_settings
from src.domain.risk.soft_recovery_policy import fixed_step_progression_multiplier
from src.domain.risk.super_concordance_kelly import is_unanimous_vote_alignment


def _runtime(kelly_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica  runtime."""
    if isinstance(kelly_config, dict) and "neutral_bankroll_pct" in kelly_config:
        try:
            return kelly_runtime_from_config({"kelly": kelly_config})
        except ValueError:
            pass
    return load_kelly_runtime_from_settings()


def _positive_float(value: object) -> float | None:
    """Resolve ou aplica  positive float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def resolve_contract_payout(payout: float | None = None, risk_params: dict[str, Any] | None = None) -> float:
    """Resolve ou aplica resolve contract payout."""
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
    return float(_runtime()["payout_fallback"])


def adaptive_recovery_progression_factor(
    payout: float | None = None, risk_params: dict[str, Any] | None = None
) -> float:
    """Resolve ou aplica adaptive recovery progression factor."""
    resolved = resolve_contract_payout(payout, risk_params)
    raw = 1.0 + (1.0 / resolved)
    return min(raw, float(_runtime()["adaptive_recovery_factor_cap"]))


def soft_recovery_progression_multiplier(
    consecutive_losses: int,
    *,
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Resolve ou aplica soft recovery progression multiplier."""
    losses = max(0, int(consecutive_losses))
    if losses <= 0:
        return 1.0
    fixed = fixed_step_progression_multiplier(losses, soft_recovery=soft_recovery)
    return float(fixed) if fixed is not None else adaptive_recovery_progression_factor(payout, risk_params) ** losses


def _squeeze_floor_active(metrics: dict) -> bool:
    """Resolve ou aplica  squeeze floor active."""
    return bool(metrics.get("meta_squeeze_downgrade") or metrics.get("consensus_stake_floor"))


def d_squeeze_sovereignty_active(metrics: dict | None) -> bool:
    """Resolve ou aplica d squeeze sovereignty active."""
    if not isinstance(metrics, dict):
        return False
    if _squeeze_floor_active(metrics):
        return True
    score = float(metrics.get("trade_score", metrics.get("conviction", -1.0)))
    return abs(score - float(_runtime()["d_squeeze_sovereign_trade_score"])) < 1e-6


def neutral_edge_dynamic_unit(bankroll: float) -> float:
    """Resolve ou aplica neutral edge dynamic unit."""
    rt = _runtime()
    pct = (
        float(rt["micro_bankroll_pct"])
        if bankroll <= float(rt["micro_bankroll_threshold"])
        else float(rt["neutral_bankroll_pct"])
    )
    return max(0.0, float(bankroll)) * pct


def turbo_edge_stake_multiplier(metrics: dict | None) -> float:
    """Resolve ou aplica turbo edge stake multiplier."""
    if not isinstance(metrics, dict) or _squeeze_floor_active(metrics):
        return 1.0
    rt = _runtime()
    live_n = int(metrics.get("live_n", 0) or 0)
    if live_n < int(rt["turbo_live_n_min"]):
        return 1.0
    live_brier = metrics.get("live_brier")
    if live_brier is not None:
        try:
            if float(live_brier) > float(rt["turbo_live_brier_max"]):
                return 1.0
        except (TypeError, ValueError):
            return 1.0
    if float(metrics.get("edge_zscore", 0.0)) + 1e-12 >= float(rt["turbo_edge_zscore_threshold"]):
        return float(rt["turbo_edge_stake_multiplier"])
    return 1.0


def _recovery_waives_consensus_penalty(
    metrics: dict,
    kelly_config: dict[str, Any],
    *,
    consecutive_losses: int,
    pending_loss_total: float,
    order_direction: str | None,
) -> bool:
    """Resolve ou aplica  recovery waives consensus penalty."""
    if d_squeeze_sovereignty_active(metrics):
        return False
    recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
    if not recovering:
        return False
    if is_unanimous_vote_alignment(
        int(metrics.get("call_votes", 0)), int(metrics.get("put_votes", 0)), order_direction
    ):
        return True
    cfg = kelly_config or {}
    if "penalty_smoothing_trade_score_min" in cfg:
        score_min = float(cfg["penalty_smoothing_trade_score_min"])
    else:
        score_min = float(_runtime(cfg)["penalty_smoothing_trade_score_min"])
    trade_score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    return trade_score + 1e-9 >= score_min
