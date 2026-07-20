"""Martingale classico: base stake_min, dobra apos LOSS, reset no WIN."""

from __future__ import annotations

from typing import Any

from src.domain.risk.risk_recovery_state import clear_dust_pending_loss
from src.domain.risk.risk_stake_flow import emit_cycle_stake_log
from src.domain.risk.stake_sizing import resolve_stake_conviction, resolve_stake_regime, round_stake


def resolve_martingale_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza bloco risk_management.martingale."""
    raw = config.get("martingale") if isinstance(config, dict) else None
    chunk = raw if isinstance(raw, dict) else {}
    try:
        multiplier = float(chunk.get("multiplier", 2.0))
    except (TypeError, ValueError):
        multiplier = 2.0
    if multiplier <= 1.0:
        multiplier = 2.0
    return {
        "enabled": bool(chunk.get("enabled", False)),
        "multiplier": multiplier,
    }


def martingale_enabled(rm: Any) -> bool:
    """Indica se o sizing Martingale esta ativo no RiskManager."""
    cfg = getattr(rm, "martingale_config", None)
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", False))
    risk = getattr(rm, "config", None)
    return bool(resolve_martingale_config(risk if isinstance(risk, dict) else None).get("enabled"))


def resolve_martingale_stake(rm: Any, bankroll: float) -> tuple[float, str]:
    """Calcula stake Martingale limitada apenas pela banca disponivel."""
    base = max(0.0, float((getattr(rm, "risk_params", {}) or {}).get("stake_min", 1.0)))
    bal = max(0.0, float(bankroll))
    if base <= 0.0 or bal + 1e-12 < base:
        return 0.0, "MARTINGALE"
    cfg = getattr(rm, "martingale_config", None)
    if not isinstance(cfg, dict):
        cfg = resolve_martingale_config(getattr(rm, "config", None))
    multiplier = float(cfg.get("multiplier", 2.0))
    linear = max(0, int(getattr(rm, "consecutive_losses_linear", 0) or 0))
    last_loss = max(0.0, float(getattr(rm, "last_loss_stake", 0.0) or 0.0))
    if linear <= 0:
        raw = base
    elif last_loss > 0.0:
        raw = last_loss * multiplier
    else:
        raw = base * (multiplier**linear)
    rounded = round_stake(raw, recovery_linear=True)
    return min(rounded, bal), "MARTINGALE"


def _metrics_for_conviction(dl_metrics: dict | None, conviction: float) -> dict:
    """Monta metricas minimas para resolve_stake_conviction no path Martingale."""
    if isinstance(dl_metrics, dict):
        merged = dict(dl_metrics)
        if "trade_score" not in merged and "conviction" not in merged:
            merged["trade_score"] = conviction
            merged["conviction"] = conviction
        return merged
    return {"trade_score": conviction, "conviction": conviction}


def calculate_martingale_stake_for_manager(
    rm: Any,
    bankroll: float,
    symbol: str,
    conviction: float,
    *,
    silent: bool,
    kwargs: dict,
) -> float:
    """Sizing Martingale classico com telemetria Kelly opcional no log."""
    dl_metrics = kwargs.get("dl_metrics")
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_metrics, conviction), rm.kelly_config)
    clear_dust_pending_loss(rm)
    loss_to_recover = sum(rm.pending_loss.values())
    linear_losses = int(getattr(rm, "consecutive_losses_linear", 0))
    stake_regime = resolve_stake_regime(pending_loss=loss_to_recover, consecutive_losses_linear=linear_losses)
    if isinstance(dl_metrics, dict):
        dl_metrics["stake_regime"] = stake_regime
    final_stake, mode_tag = resolve_martingale_stake(rm, bankroll)
    b = float(rm.risk_params.get("payout_estimate", 0.95))
    metrics = dl_metrics if isinstance(dl_metrics, dict) else None
    p = rm.effective_win_rate(symbol, conviction, metrics=metrics)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    live_wr = dl_metrics.get("live_wr") if isinstance(dl_metrics, dict) else None
    live_n = int(dl_metrics.get("live_n", 0) or 0) if isinstance(dl_metrics, dict) else 0
    if not silent:
        rm.logger.info(
            "KELLY | p=%.4f | live_wr=%s | live_n=%d | f*=%.6f | mode=%s",
            float(p),
            f"{float(live_wr):.4f}" if live_wr is not None else "n/a",
            int(live_n),
            float(max(0.0, kelly_f)),
            stake_regime.lower(),
        )
    emit_cycle_stake_log(
        rm,
        cycle_id=int(kwargs.get("cycle_id") or 0),
        silent=silent,
        mode_tag=mode_tag,
        final_stake=final_stake,
        f_star=0.0,
        p=p,
        b=b,
        bankroll=bankroll,
        loss_to_recover=loss_to_recover,
        linear_losses=linear_losses,
        symbol=symbol,
        rec_info="",
        stake_regime=stake_regime,
        safe_cap=float(bankroll),
        recovery_infeasible=False,
    )
    return final_stake
