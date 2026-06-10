"""Ranking de mercado e resolucao de direcao para execucao obrigatoria."""

from src.application.services.deep_learning.dl_candle_flow import (
    binary_direction_vote,
    flow_alignment_bonus,
    flow_implied_direction,
    flow_strength,
)
from src.application.services.execution_direction import infer_dl_direction
from src.domain.models.trade import TradeDirection


_ABSOLUTE_HARD_BLOCKS = frozenset({"data", "predict_error", "training", "cooldown", "session_pause"})
_CLUSTER_CORE = frozenset({"R_50", "R_75"})
_WEAK_SCORE_LIMIT = 0.53
_STRONG_RAW_SIDE = 0.58
_MIN_CLOSE_LOC_CALL = 0.48
_MAX_CLOSE_LOC_PUT = 0.52
_STRONG_DL_SCORE = 0.68
_FLOW_OVERRIDE_MIN = 0.50
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


def _raw_side(metrics: dict) -> float:
    """Conviccao lateralizada a partir de raw_prob."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def _trade_score(metrics: dict) -> float:
    """Score calibrado unificado do candidato."""
    return float(metrics.get("trade_score", metrics.get("conviction", 0.0)))


def _weak_execution_signal(metrics: dict) -> bool:
    """Indica sinal DL insuficiente para confiar na direcao prevista."""
    score = _trade_score(metrics)
    raw_side = _raw_side(metrics)
    if raw_side + 1e-9 >= _STRONG_RAW_SIDE:
        return False
    gate = str(metrics.get("gate_reason") or "")
    if score + 1e-9 < 0.01:
        return True
    if gate in _WEAK_GATES:
        return True
    if metrics.get("execute"):
        return score + 1e-9 < _WEAK_SCORE_LIMIT
    return score + 1e-9 < _WEAK_SCORE_LIMIT or raw_side + 1e-9 < _STRONG_RAW_SIDE


def _candle_implied_direction(ctx: dict) -> TradeDirection | None:
    """Infere direcao a partir do corpo e fechamento da ultima vela."""
    if not ctx:
        return None
    body = float(ctx.get("body", 0.0))
    close_loc = float(ctx.get("close_loc", 0.5))
    if body > 0.0 and close_loc + 1e-9 >= _MIN_CLOSE_LOC_CALL:
        return TradeDirection.CALL
    if body < 0.0 and close_loc <= _MAX_CLOSE_LOC_PUT + 1e-9:
        return TradeDirection.PUT
    return None


def _strong_dl_signal(metrics: dict) -> bool:
    """Indica predicao DL forte o suficiente para confiar sem fluxo oposto extremo."""
    score = _trade_score(metrics)
    return score + 1e-9 >= _STRONG_DL_SCORE and bool(metrics.get("execute")) and metrics.get("deploy_ok") is not False


def _resolve_mandatory_weak_direction(dl: TradeDirection, ctx: dict) -> TradeDirection:
    """Escolhe direcao por fluxo, vela e contexto quando o DL nao e confiavel."""
    flow_dir = flow_implied_direction(ctx)
    strength = flow_strength(ctx)
    if flow_dir is not None and strength + 1e-9 >= 0.15:
        return flow_dir
    candle = _candle_implied_direction(ctx)
    if candle is not None:
        return candle
    voted = binary_direction_vote(ctx)
    if voted is not None:
        return voted
    return dl


def _flow_overrides_dl(direction: TradeDirection, flow_dir: TradeDirection, strength: float, metrics: dict) -> bool:
    """Indica se o fluxo de mercado deve substituir a direcao prevista pelo DL."""
    if _strong_dl_signal(metrics) and _raw_side(metrics) + 1e-9 >= _STRONG_RAW_SIDE:
        return flow_dir != direction and strength + 1e-9 >= _FLOW_OVERRIDE_MIN
    if (_weak_execution_signal(metrics) or strength + 1e-9 >= 0.20) and strength + 1e-9 >= 0.18:
        return True
    return flow_dir != direction and strength + 1e-9 >= 0.32


def _resolve_flow_aware_direction(direction: TradeDirection, ctx: dict, metrics: dict) -> TradeDirection:
    """Combina DL com fluxo de velas e regime de mercado."""
    flow_dir = flow_implied_direction(ctx)
    if flow_dir is None:
        return _resolve_mandatory_weak_direction(direction, ctx) if _weak_execution_signal(metrics) else direction
    strength = flow_strength(ctx)
    return flow_dir if _flow_overrides_dl(direction, flow_dir, strength, metrics) else direction


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
    metrics = entry.get("metrics") or {}
    ctx = metrics.get("binary_ctx") or {}
    direction = infer_dl_direction(entry)
    if direction is None:
        direction = flow_implied_direction(ctx) or _candle_implied_direction(ctx) or binary_direction_vote(ctx)
    if direction is None:
        return None
    if ctx:
        return _resolve_flow_aware_direction(direction, ctx, metrics)
    if _weak_execution_signal(metrics):
        return _resolve_mandatory_weak_direction(direction, ctx)
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
        composite += flow_alignment_bonus(exec_direction, ctx)
        composite += _binary_alignment_bonus(exec_direction, ctx)
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
    direction = resolve_market_direction(entry)
    if direction is None:
        return None
    dl_dir = infer_dl_direction(entry)
    metrics = dict(entry.get("metrics") or {})
    metrics["dl_direction"] = dl_dir.name if dl_dir else direction.name
    metrics["exec_direction"] = direction.name
    metrics["direction_inverted"] = dl_dir is not None and dl_dir != direction
    return symbol, direction, metrics
