"""Ranking de mercado e resolucao de direcao para execucao obrigatoria."""

from src.application.services.execution_direction import infer_dl_direction
from src.domain.models.trade import TradeDirection


_ABSOLUTE_HARD_BLOCKS = frozenset({"data", "predict_error", "training", "cooldown", "session_pause"})
_CLUSTER_CORE = frozenset({"R_50", "R_75"})
_SMA_Z_EXTREME = 0.004
_WEAK_RAW_MARGIN = 0.08


def _raw_side(metrics: dict) -> float:
    """Conviccao lateralizada a partir de raw_prob."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def _trade_score(metrics: dict) -> float:
    """Score calibrado unificado do candidato."""
    return float(metrics.get("trade_score", metrics.get("conviction", 0.0)))


def mandatory_pool_eligible(entry: dict) -> bool:
    """Indica se simbolo pode entrar no pool obrigatorio com direcao inferivel."""
    metrics = entry.get("metrics") or {}
    gate = str(metrics.get("gate_reason") or "")
    if gate in _ABSOLUTE_HARD_BLOCKS:
        return False
    if metrics.get("deploy_ok") is False:
        return False
    return resolve_market_direction(entry) is not None


def resolve_market_direction(entry: dict) -> TradeDirection | None:
    """Resolve CALL/PUT combinando DL, raw_prob e contexto binario da barra."""
    direction = infer_dl_direction(entry)
    if direction is None:
        return None
    metrics = entry.get("metrics") or {}
    raw = metrics.get("raw_prob")
    if raw is None:
        return direction
    ctx = metrics.get("binary_ctx") or {}
    if not ctx:
        return direction
    raw_side = _raw_side(metrics)
    if raw_side > 0.5 + _WEAK_RAW_MARGIN:
        return direction
    sma_z = float(ctx.get("sma_z", 0.0))
    if sma_z >= _SMA_Z_EXTREME:
        direction = TradeDirection.PUT
    elif sma_z <= -_SMA_Z_EXTREME:
        direction = TradeDirection.CALL
    return direction


def _binary_alignment_bonus(direction: TradeDirection, ctx: dict) -> float:
    """Bonus quando direcao escolhida alinha com indicadores da ultima barra."""
    bonus = 0.0
    rsi = float(ctx.get("rsi", 0.5))
    sma_z = float(ctx.get("sma_z", 0.0))
    close_loc = float(ctx.get("close_loc", 0.5))
    vr = float(ctx.get("variance_ratio", 1.0))
    rel_vol = float(ctx.get("rel_vol", 0.0))
    z_spread = float(ctx.get("z_spread", 0.0))
    if direction == TradeDirection.CALL:
        if rsi < 0.45:
            bonus += 0.02
        if sma_z < 0.0:
            bonus += 0.02
        if close_loc > 0.52:
            bonus += 0.015
        if z_spread <= 0.0:
            bonus += 0.01
    else:
        if rsi > 0.55:
            bonus += 0.02
        if sma_z > 0.0:
            bonus += 0.02
        if close_loc < 0.48:
            bonus += 0.015
        if z_spread >= 0.0:
            bonus += 0.01
    if vr < 0.85:
        bonus += 0.01
    if rel_vol >= 0.28:
        bonus += 0.01
    return min(bonus, 0.09)


def market_decision_score(
    metrics: dict,
    *,
    exec_direction: TradeDirection | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
) -> float:
    """Pontua candidato com todos os indicadores DL e contexto de mercado."""
    score = _trade_score(metrics)
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(score - 0.5) if score > 0.0 else abs(raw_side - 0.5)))
    composite = score * 0.38 + raw_side * 0.32 + val * 0.12 + edge * 0.10
    if metrics.get("execute"):
        composite += 0.06
    if metrics.get("deploy_ok"):
        composite += 0.04
    live_wr = metrics.get("live_win_rate")
    if live_wr is not None and float(live_wr) >= 0.5:
        composite += 0.03
    brier = metrics.get("val_brier")
    if brier is not None and float(brier) > 0.28:
        composite -= 0.04
    ctx = metrics.get("binary_ctx") or {}
    if exec_direction is not None and ctx:
        composite += _binary_alignment_bonus(exec_direction, ctx)
    if score + 1e-9 < 0.45 and raw_side + 1e-9 < 0.52:
        composite *= 0.45
    if recovery_active:
        if symbol in _CLUSTER_CORE:
            composite += 0.03
        if last_loss_symbol and symbol != last_loss_symbol:
            composite += 0.04
        if (
            last_loss_direction
            and exec_direction is not None
            and exec_direction.name == str(last_loss_direction).upper()
        ):
            composite += 0.05
    return composite


def build_market_execution_candidate(
    symbol: str,
    entry: dict,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao resolvida pelo ranking de mercado."""
    direction = resolve_market_direction(entry)
    if direction is None:
        return None
    dl_dir = infer_dl_direction(entry)
    metrics = dict(entry.get("metrics") or {})
    metrics["dl_direction"] = dl_dir.name if dl_dir else direction.name
    metrics["exec_direction"] = direction.name
    metrics["direction_inverted"] = dl_dir is not None and dl_dir != direction
    return symbol, direction, metrics
