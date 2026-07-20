"""Ranking de mercado e resolucao de direcao para execucao."""

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked
from src.application.services.execution_loss_protection import edge_conviction_disconnect_penalty
from src.application.services.force_trade_mode import force_trade_every_cycle, synthesize_force_direction
from src.application.services.meta_payoff_veto_gate import is_execution_signal_vetoed
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import metric_float


def _trade_score(metrics: dict) -> float:
    """Le score unificado de conviccao usado em selecao e ranking."""
    return metric_float(metrics, "trade_score", "raw_prob", "conviction", default=0.0)


def _raw_side(metrics: dict) -> float:
    """Conviccao lateralizada a partir de raw_prob."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def mandatory_pool_eligible(entry: dict, **kwargs) -> bool:
    """Indica se simbolo pode entrar no pool com direcao inferivel."""
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
    exec_direction: TradeDirection | None,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    metrics: dict | None = None,
) -> float:
    """Aplica bonus e penalidades de recovery ao score composto."""
    if not recovery_active:
        if last_loss_symbol and symbol == last_loss_symbol:
            composite -= 0.08
        return composite
    if last_loss_symbol and symbol == last_loss_symbol:
        composite -= 0.20
    elif last_loss_symbol and symbol != last_loss_symbol:
        composite += 0.06
    if last_loss_direction and exec_direction is not None:
        ld = str(last_loss_direction).upper()
        if exec_direction.name == ld:
            composite -= 0.18
        else:
            composite += 0.05

    if metrics:
        indicators = metrics.get("indicators") or {}
        adx = float(indicators.get("adx", 0.20))
        vol_ratio = float(indicators.get("vol_ratio", 0.90))
        hurst = float(indicators.get("hurst", 0.50))

        if adx < 0.18:
            composite -= 0.08
        elif adx >= 0.24 and vol_ratio >= 1.0:
            composite += 0.05

        if hurst > 0.58:
            composite += 0.03
        elif hurst < 0.45:
            composite -= 0.04

    return composite


def _apply_brier_ece_penalties(composite: float, metrics: dict, *, live_n: int) -> float:
    """Aplica penalidades de Brier/ECE no score composto de ranking."""
    brier = metrics.get("val_brier")
    settlement_brier = metrics.get("deploy_settlement_brier")
    live_brier = metrics.get("live_brier")
    if live_n >= 20 and live_brier is not None:
        effective_brier = float(live_brier)
    else:
        effective_brier = (
            float(settlement_brier) if settlement_brier is not None else (float(brier) if brier is not None else None)
        )
    if effective_brier is not None and effective_brier > 0.24:
        composite -= 0.06
    elif effective_brier is not None and effective_brier > 0.22:
        composite -= 0.03
    live_ece = metrics.get("live_ece")
    ece = live_ece if live_n >= 20 and live_ece is not None else metrics.get("val_ece")
    if ece is not None and float(ece) > 0.10:
        composite -= 0.04
    elif ece is not None and float(ece) > 0.08:
        composite -= 0.03
    return composite


def market_decision_score(
    metrics: dict,
    *,
    exec_direction: TradeDirection | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
) -> float:
    """Pontua candidato com probabilidade bruta, WR/Brier live e edge."""
    override = metrics.get("market_decision_score_override")
    if override is not None:
        return float(override)
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(raw_side - 0.5)))
    live_n = int(metrics.get("live_n", 0) or 0)
    settlement_wr = metric_float(metrics, "deploy_settlement_win_rate", "deploy_win_rate", default=0.0)
    live_wr = metrics.get("live_wr")
    effective_wr = float(live_wr) if live_n >= 20 and live_wr is not None else float(settlement_wr)
    composite = raw_side * 0.40 + effective_wr * 0.30 + val * 0.08 + edge * 0.22
    resolved = metric_float(metrics, "resolved_conviction", default=0.0)
    if resolved > 0.0:
        composite = composite * 0.6 + resolved * 0.4
    if metrics.get("execute"):
        composite += 0.05
    if metrics.get("deploy_ok"):
        composite += 0.03
    composite = _apply_brier_ece_penalties(composite, metrics, live_n=live_n)
    composite -= float(metrics.get("calib_drift_soft_penalty", 0.0))
    if metrics.get("direction_inverted"):
        composite -= 0.10
    composite -= float(metrics.get("exhaustion_penalty", 0.0))
    composite -= float(metrics.get("loss_protection_penalty", 0.0))
    composite -= float(metrics.get("meta_soft_veto_penalty", 0.0))
    composite -= edge_conviction_disconnect_penalty(metrics, exec_direction=exec_direction)
    margin = float(metrics.get("direction_margin", 0.0))
    if margin + 1e-9 < 0.05:
        composite -= 0.08
    if recovery_active and (metrics.get("meta_squeeze_downgrade") or metrics.get("consensus_stake_floor")):
        composite -= 0.25
    composite *= float(metrics.get("universal_regime_score_factor", 1.0))
    return _recovery_score_adjustment(
        composite,
        recovery_active=recovery_active,
        symbol=symbol,
        exec_direction=exec_direction,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        metrics=metrics,
    )


def build_market_execution_candidate(
    symbol: str,
    entry: dict,
    *,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao resolvida pelo motor unificado."""
    _ = (consecutive_losses, mean_reversion_enabled, low_accuracy_enabled)
    return build_execution_candidate(
        symbol,
        entry,
        exec_cfg=exec_cfg,
        calibration_cfg=calibration_cfg,
        recovery_active=recovery_active,
    )
