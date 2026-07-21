"""Ranking de mercado e resolucao de direcao para execucao."""

from __future__ import annotations

from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked
from src.application.services.execution_loss_protection import edge_conviction_disconnect_penalty
from src.application.services.execution_runtime_config import resolve_market_rank_composite
from src.application.services.force_trade_mode import force_trade_every_cycle, synthesize_force_direction
from src.application.services.meta_payoff_veto_gate import is_execution_signal_vetoed
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import metric_float


def _composite() -> dict:
    """Resolve ou aplica  composite."""
    return resolve_market_rank_composite()


def _trade_score(metrics: dict) -> float:
    """Resolve ou aplica  trade score."""
    return metric_float(metrics, "trade_score", "raw_prob", "conviction", default=0.0)


def _raw_side(metrics: dict) -> float:
    """Resolve ou aplica  raw side."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def mandatory_pool_eligible(entry: dict, **kwargs) -> bool:
    """Resolve ou aplica mandatory pool eligible."""
    if is_technically_blocked(entry):
        return False
    metrics = entry.get("metrics") or {}
    exec_cfg = kwargs.get("exec_cfg")
    force = force_trade_every_cycle(exec_cfg if isinstance(exec_cfg, dict) else None)
    if not force and is_execution_signal_vetoed(metrics):
        return False
    if infer_dl_direction(entry) is not None:
        return True
    return force and synthesize_force_direction(entry) is not None


def _recovery_score_adjustment(
    composite: float,
    *,
    recovery_active: bool,
    symbol: str | None,
    last_loss_symbol: str | None,
    metrics: dict | None = None,
) -> float:
    """Resolve ou aplica  recovery score adjustment."""
    cfg = _composite()
    if not recovery_active:
        if last_loss_symbol and symbol == last_loss_symbol:
            composite += float(cfg["last_loss_penalty"])
        return composite
    if last_loss_symbol and symbol == last_loss_symbol:
        composite += float(cfg["recovery_last_loss_penalty"])
    elif last_loss_symbol and symbol != last_loss_symbol:
        composite += float(cfg["rotate_bonus"])

    if metrics:
        rank = load_indicator_config_from_settings()["market_rank"]
        indicators = metrics.get("indicators") or {}
        defaults = cfg["indicator_defaults"]
        adx = float(indicators["adx"]) if "adx" in indicators else float(defaults["adx"])
        vol_ratio = float(indicators["vol_ratio"]) if "vol_ratio" in indicators else float(defaults["vol_ratio"])
        hurst = float(indicators["hurst"]) if "hurst" in indicators else float(defaults["hurst"])

        if adx < float(rank["adx_weak_below"]):
            composite += float(cfg["adx_weak_penalty"])
        elif adx >= float(rank["adx_strong_at_or_above"]) and vol_ratio >= 1.0:
            composite += float(cfg["adx_strong_bonus"])

        if hurst > float(rank["hurst_trend_above"]):
            composite += float(cfg["hurst_trend_bonus"])
        elif hurst < float(rank["hurst_mean_revert_below"]):
            composite += float(cfg["hurst_mean_revert_penalty"])

    return composite


def _apply_brier_ece_penalties(composite: float, metrics: dict, *, live_n: int) -> float:
    """Resolve ou aplica  apply brier ece penalties."""
    cfg = _composite()
    brier = metrics.get("val_brier")
    settlement_brier = metrics.get("deploy_settlement_brier")
    live_brier = metrics.get("live_brier")
    if live_n >= int(cfg["live_n_min"]) and live_brier is not None:
        effective_brier = float(live_brier)
    else:
        effective_brier = (
            float(settlement_brier) if settlement_brier is not None else (float(brier) if brier is not None else None)
        )
    if effective_brier is not None and effective_brier > float(cfg["live_brier_hard_above"]):
        composite += float(cfg["live_brier_hard_penalty"])
    elif effective_brier is not None and effective_brier > float(cfg["live_brier_soft_above"]):
        composite += float(cfg["live_brier_soft_penalty"])
    live_ece = metrics.get("live_ece")
    ece = live_ece if live_n >= int(cfg["live_n_min"]) and live_ece is not None else metrics.get("val_ece")
    if ece is not None and float(ece) > float(cfg["ece_hard_above"]):
        composite += float(cfg["ece_hard_penalty"])
    elif ece is not None and float(ece) > float(cfg["ece_soft_above"]):
        composite += float(cfg["ece_soft_penalty"])
    return composite


def market_decision_score(
    metrics: dict,
    *,
    exec_direction: TradeDirection | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    last_loss_symbol: str | None = None,
) -> float:
    """Resolve ou aplica market decision score."""
    override = metrics.get("market_decision_score_override")
    if override is not None:
        return float(override)
    cfg = _composite()
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(raw_side - 0.5)))
    live_n = int(metrics.get("live_n", 0) or 0)
    settlement_wr = metric_float(metrics, "deploy_settlement_win_rate", "deploy_win_rate", default=0.0)
    live_wr = metrics.get("live_wr")
    effective_wr = float(live_wr) if live_n >= int(cfg["live_n_min"]) and live_wr is not None else float(settlement_wr)
    composite = (
        raw_side * float(cfg["weight_trade_score"])
        + effective_wr * float(cfg["weight_edge"])
        + val * float(cfg["weight_margin"])
        + edge * float(cfg["weight_meta_z"])
    )
    resolved = metric_float(metrics, "resolved_conviction", default=0.0)
    if resolved > 0.0:
        composite = composite * float(cfg["blend_primary"]) + resolved * float(cfg["blend_secondary"])
    if metrics.get("execute"):
        composite += float(cfg["execute_bonus"])
    if metrics.get("deploy_ok"):
        composite += float(cfg["deploy_ok_bonus"])
    composite = _apply_brier_ece_penalties(composite, metrics, live_n=live_n)
    composite -= float(metrics.get("calib_drift_soft_penalty", 0.0))
    composite -= float(metrics.get("exhaustion_penalty", 0.0))
    composite -= float(metrics.get("loss_protection_penalty", 0.0))
    composite -= float(metrics.get("meta_soft_veto_penalty", 0.0))
    composite -= edge_conviction_disconnect_penalty(metrics, exec_direction=exec_direction)
    margin = float(metrics.get("direction_margin", 0.0))
    if margin + 1e-9 < float(cfg["thin_margin_below"]):
        composite += float(cfg["thin_margin_penalty"])
    if recovery_active and (metrics.get("meta_squeeze_downgrade") or metrics.get("consensus_stake_floor")):
        composite += float(cfg["squeeze_recovery_penalty"])
    composite *= float(metrics.get("universal_regime_score_factor", 1.0))
    return _recovery_score_adjustment(
        composite, recovery_active=recovery_active, symbol=symbol, last_loss_symbol=last_loss_symbol, metrics=metrics
    )


def build_market_execution_candidate(
    symbol: str,
    entry: dict,
    *,
    recovery_active: bool = False,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Resolve ou aplica build market execution candidate."""
    return build_execution_candidate(
        symbol, entry, exec_cfg=exec_cfg, calibration_cfg=calibration_cfg, recovery_active=recovery_active
    )
