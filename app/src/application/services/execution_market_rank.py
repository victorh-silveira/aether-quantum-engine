"""Ranking de mercado e resolucao de direcao para execucao."""

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked
from src.application.services.execution_loss_protection import edge_conviction_disconnect_penalty
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


def mandatory_pool_eligible(entry: dict, **_) -> bool:
    """Indica se simbolo pode entrar no pool com direcao inferivel."""
    if is_technically_blocked(entry):
        return False
    metrics = entry.get("metrics") or {}
    if is_execution_signal_vetoed(metrics):
        return False
    return infer_dl_direction(entry) is not None


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


def market_decision_score(
    metrics: dict,
    *,
    exec_direction: TradeDirection | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
) -> float:
    """Pontua candidato com probabilidade bruta, val_acc e edge."""
    override = metrics.get("market_decision_score_override")
    if override is not None:
        return float(override)
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(raw_side - 0.5)))
    composite = raw_side * 0.45 + val * 0.35 + edge * 0.20
    resolved = metric_float(metrics, "resolved_conviction", default=0.0)
    if resolved > 0.0:
        composite = composite * 0.6 + resolved * 0.4
    if metrics.get("execute"):
        composite += 0.05
    if metrics.get("deploy_ok"):
        composite += 0.03
    brier = metrics.get("val_brier")
    if brier is not None and float(brier) > 0.28:
        composite -= 0.04
    if metrics.get("direction_inverted"):
        composite -= 0.10
    composite -= float(metrics.get("exhaustion_penalty", 0.0))
    composite -= float(metrics.get("loss_protection_penalty", 0.0))
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
