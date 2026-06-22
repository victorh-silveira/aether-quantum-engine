"""Ranking de mercado e resolucao de direcao para execucao."""

import contextlib

from src.application.services.execution_direction import infer_dl_direction
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction


_ABSOLUTE_HARD_BLOCKS = frozenset({"data", "predict_error", "training", "cooldown", "session_pause", "trend_conflict"})
_CLUSTER_CORE = frozenset({"R_50", "R_75"})


def _raw_side(metrics: dict) -> float:
    """Conviccao lateralizada a partir de raw_prob."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def _trade_score(metrics: dict) -> float:
    """Score unificado do candidato."""
    return float(metrics.get("trade_score", metrics.get("conviction", metrics.get("raw_prob", 0.0))))


def mandatory_pool_eligible(entry: dict) -> bool:
    """Indica se simbolo pode entrar no pool com direcao inferivel."""
    metrics = entry.get("metrics") or {}
    gate = str(metrics.get("gate_reason") or "")
    if gate in _ABSOLUTE_HARD_BLOCKS:
        return False
    if metrics.get("deploy_ok") is False:
        return False
    return resolve_market_direction(entry) is not None


def _resolve_low_accuracy(
    dl_dir: TradeDirection, metrics: dict, trend_str: str | None, val_acc: float
) -> TradeDirection | None:
    """Aplica inversao inteligente se val_accuracy < 0.50."""
    if val_acc >= 0.50:
        return None
    if trend_str:
        with contextlib.suppress(KeyError, ValueError):
            trend_dir = TradeDirection[trend_str.upper()]
            metrics["direction_inverted"] = dl_dir != trend_dir
            return trend_dir
    metrics["direction_inverted"] = True
    return TradeDirection.PUT if dl_dir == TradeDirection.CALL else TradeDirection.CALL


def _resolve_mean_reversion(dl_dir: TradeDirection, metrics: dict) -> TradeDirection | None:
    """Aplica inversao por reversao a media."""
    indicators = metrics.get("indicators") or {}
    hurst = float(indicators.get("hurst", 0.5))
    adx = float(indicators.get("adx", 0.5))
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    rsi = float(indicators.get("rsi", 0.5))

    if hurst < 0.48 and adx < 0.25 and vol_ratio < 1.0:
        if dl_dir == TradeDirection.PUT and rsi < 0.45:
            metrics["direction_inverted"] = True
            return TradeDirection.CALL
        if dl_dir == TradeDirection.CALL and rsi > 0.55:
            metrics["direction_inverted"] = True
            return TradeDirection.PUT
    return None


def resolve_market_direction(
    entry: dict,
    *,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
) -> TradeDirection | None:
    """Resolve CALL/PUT a partir da predicao DL com inversao inteligente e tendencia."""
    _ = consecutive_losses
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = entry.get("metrics") or {}
    val_acc = float(metrics.get("val_accuracy", 0.50))
    trend_str = metrics.get("trend_direction")

    resolved = _resolve_low_accuracy(dl_dir, metrics, trend_str, val_acc)
    if resolved is not None:
        return resolved

    resolved = _resolve_mean_reversion(dl_dir, metrics)
    if resolved is not None:
        return resolved

    # Verifica se a tendência é forte
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    adx = float(indicators.get("adx", 0.5))
    trend_strong = vol_ratio >= 0.85 or adx >= 0.25

    # Alinhamento com tendência no recovery
    if recovery_active and trend_str and trend_strong:
        with contextlib.suppress(KeyError, ValueError):
            trend_dir = TradeDirection[trend_str.upper()]
            metrics["direction_inverted"] = dl_dir != trend_dir
            return trend_dir

    # Fallback para tendência em gating bloqueado
    if trend_str and not metrics.get("execute", True) and trend_strong:
        with contextlib.suppress(KeyError, ValueError):
            return TradeDirection[trend_str.upper()]

    return dl_dir


def _recovery_score_adjustment(
    composite: float,
    *,
    recovery_active: bool,
    symbol: str | None,
    exec_direction: TradeDirection | None,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> float:
    """Aplica bonus e penalidades de recovery ao score composto."""
    if not recovery_active:
        return composite
    if symbol in _CLUSTER_CORE:
        composite += 0.03
    if last_loss_symbol and symbol != last_loss_symbol:
        composite += 0.04
    if last_loss_direction and exec_direction is not None:
        ld = str(last_loss_direction).upper()
        composite += 0.03 if exec_direction.name != ld else -0.12
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
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(raw_side - 0.5)))
    composite = raw_side * 0.55 + val * 0.25 + edge * 0.20
    if metrics.get("execute"):
        composite += 0.05
    if metrics.get("deploy_ok"):
        composite += 0.03
    brier = metrics.get("val_brier")
    if brier is not None and float(brier) > 0.28:
        composite -= 0.04
    return _recovery_score_adjustment(
        composite,
        recovery_active=recovery_active,
        symbol=symbol,
        exec_direction=exec_direction,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
    )


def build_market_execution_candidate(
    symbol: str,
    entry: dict,
    *,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao resolvida pelo ranking de mercado."""
    direction = resolve_market_direction(entry, recovery_active=recovery_active, consecutive_losses=consecutive_losses)
    if direction is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    enrich_metrics_conviction(metrics)
    dl_dir = infer_dl_direction(entry)
    metrics["dl_direction"] = dl_dir.name if dl_dir else direction.name
    metrics["exec_direction"] = direction.name
    if "direction_inverted" not in metrics:
        metrics["direction_inverted"] = False
    return symbol, direction, metrics
