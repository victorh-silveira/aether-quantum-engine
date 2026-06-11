"""Ranking de mercado e resolucao de direcao para execucao obrigatoria."""

from src.application.services.deep_learning.dl_candle_flow import (
    flow_alignment_bonus,
    flow_implied_direction,
    flow_strength,
)
from src.application.services.deep_learning.dl_direction_consensus import (
    dl_direction_from_raw,
    resolve_consensus_direction,
)
from src.application.services.execution_direction import infer_dl_direction
from src.domain.models.trade import TradeDirection


_ABSOLUTE_HARD_BLOCKS = frozenset({"data", "predict_error", "training", "cooldown", "session_pause"})
_CLUSTER_CORE = frozenset({"R_50", "R_75"})
_WEAK_GATES = frozenset(
    {
        "raw_conviction",
        "conviction",
        "direction_margin",
        "mean_reversion",
        "candle_reject",
        "random_walk",
        "rsi_exhaust",
        "wick_reject",
        "pair_spread",
        "noise_floor",
    }
)
_MIN_CONSENSUS_STRENGTH = 0.22


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
    """Resolve CALL/PUT por consenso entre DL, fluxo, vela e indicadores."""
    metrics = entry.get("metrics") or {}
    ctx = metrics.get("binary_ctx") or {}
    dl_dir = infer_dl_direction(entry)
    direction, strength = resolve_consensus_direction(metrics, ctx, dl_dir)
    if direction is None:
        return None
    if strength + 1e-9 < _MIN_CONSENSUS_STRENGTH and _trade_score(metrics) + 1e-9 < 0.50:
        return dl_dir or dl_direction_from_raw(metrics)
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


def _weak_signal_multiplier(score: float, raw_side: float, exec_direction: TradeDirection | None, ctx: dict) -> float:
    """Retorna fator multiplicador quando sinal DL e fraco ou conflita com fluxo."""
    if score + 1e-9 < 0.45 and raw_side + 1e-9 < 0.52:
        return 0.45
    if exec_direction is None or not ctx or score + 1e-9 >= 0.55:
        return 1.0
    flow_dir = flow_implied_direction(ctx)
    if flow_dir is not None and flow_dir != exec_direction and flow_strength(ctx) + 1e-9 >= 0.22:
        return 0.55
    return 1.0


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
        composite += 0.03 if exec_direction.name != ld else -0.04
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
    """Pontua candidato com todos os indicadores DL e contexto de mercado."""
    score = _trade_score(metrics)
    raw_side = _raw_side(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(score - 0.5) if score > 0.0 else abs(raw_side - 0.5)))
    consensus = float(metrics.get("consensus_strength", 0.0))
    composite = score * 0.45 + raw_side * 0.28 + val * 0.14 + edge * 0.08 + consensus * 0.12
    if metrics.get("execute"):
        composite += 0.04
    if metrics.get("deploy_ok"):
        composite += 0.03
    live_wr = metrics.get("live_win_rate")
    if live_wr is not None and float(live_wr) >= 0.5:
        composite += 0.03
    brier = metrics.get("val_brier")
    if brier is not None and float(brier) > 0.28:
        composite -= 0.04
    if consensus + 1e-9 < 0.28 and score + 1e-9 < 0.55:
        composite *= 0.50
    ctx = metrics.get("binary_ctx") or {}
    if exec_direction is not None and ctx:
        composite += flow_alignment_bonus(exec_direction, ctx)
        composite += _binary_alignment_bonus(exec_direction, ctx)
    dl_pure = dl_direction_from_raw(metrics)
    if dl_pure is not None and exec_direction is not None and dl_pure != exec_direction:
        if score + 1e-9 >= 0.52:
            composite -= 0.12
        elif score + 1e-9 >= 0.45:
            composite -= 0.06
    composite *= _weak_signal_multiplier(score, raw_side, exec_direction, ctx)
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
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao resolvida pelo ranking de mercado."""
    metrics = dict(entry.get("metrics") or {})
    ctx = metrics.get("binary_ctx") or {}
    dl_dir = dl_direction_from_raw(metrics) or infer_dl_direction(entry)
    direction, strength = resolve_consensus_direction(metrics, ctx, dl_dir)
    if direction is None:
        direction = resolve_market_direction(entry)
    if direction is None:
        return None
    metrics["consensus_strength"] = strength
    metrics["dl_direction"] = dl_dir.name if dl_dir else direction.name
    metrics["exec_direction"] = direction.name
    metrics["direction_inverted"] = dl_dir is not None and dl_dir != direction
    return symbol, direction, metrics
