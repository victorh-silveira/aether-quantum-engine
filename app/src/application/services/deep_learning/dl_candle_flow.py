"""Fluxo de velas e momentum de curto prazo para direcao binaria."""

from src.domain.models.trade import TradeDirection


def augment_binary_context(ctx: dict, series: dict, idx: int) -> dict:
    """Enriquece contexto binario com fluxo multi-barra e momentum."""
    bodies = series["body"]
    start = max(0, idx - 2)
    body_sum_3 = float(bodies[start : idx + 1].sum())
    ret_3 = 0.0
    prices = series.get("_prices_ref")
    if prices is not None and idx >= 3 and len(prices) > idx:
        base = float(prices[idx - 3])
        ret_3 = (float(prices[idx]) - base) / (base + 1e-10)
    vr = float(ctx.get("variance_ratio", 1.0))
    ctx["body_sum_3"] = body_sum_3
    ctx["ema_spread"] = float(series["ema_spread"][idx])
    ctx["ret_5"] = float(series["ret_5"][idx])
    ctx["ret_3"] = ret_3
    ctx["rsi_slope"] = float(series["rsi_slope"][idx])
    ctx["market_trend"] = 1.0 if vr > 0.88 else 0.0 if vr <= 0.82 else 0.5
    streak = 0
    if idx >= 0:
        last_sign = 0
        for j in range(idx, max(-1, idx - 4), -1):
            b = float(bodies[j])
            sign = 1 if b > 0 else -1 if b < 0 else 0
            if sign == 0:
                break
            if last_sign == 0:
                last_sign = sign
                streak = 1
            elif sign == last_sign:
                streak += 1
            else:
                break
    ctx["body_streak"] = float(streak)
    return ctx


def _body_flow_votes(body: float, body_sum: float, streak: int) -> tuple[int, int]:
    """Acumula votos CALL/PUT a partir do corpo e sequencia de velas."""
    call_votes = 0
    put_votes = 0
    if body > 0.0:
        call_votes += 2
    elif body < 0.0:
        put_votes += 2
    if body_sum > 0.0008:
        call_votes += 3
    elif body_sum < -0.0008:
        put_votes += 3
    if streak >= 2:
        if body > 0.0:
            call_votes += 2
        elif body < 0.0:
            put_votes += 2
    return call_votes, put_votes


def _regime_flow_votes(vr: float, ema: float, ret5: float, ret3: float, sma_z: float) -> tuple[int, int]:
    """Acumula votos conforme regime de mercado e momentum."""
    call_votes = 0
    put_votes = 0
    if vr > 0.88:
        if ema > 0.0004:
            call_votes += 2
        elif ema < -0.0004:
            put_votes += 2
        if ret5 > 0.0008:
            call_votes += 1
        elif ret5 < -0.0008:
            put_votes += 1
        if ret3 > 0.0005:
            call_votes += 1
        elif ret3 < -0.0005:
            put_votes += 1
    elif vr <= 0.82:
        if sma_z >= 0.003:
            put_votes += 3
        elif sma_z <= -0.003:
            call_votes += 3
    elif ema > 0.0006:
        call_votes += 1
    elif ema < -0.0006:
        put_votes += 1
    return call_votes, put_votes


def _aux_flow_votes(close_loc: float, rsi_slope: float) -> tuple[int, int]:
    """Acumula votos a partir de fechamento relativo e inclinacao do RSI."""
    call_votes = 0
    put_votes = 0
    if close_loc > 0.54:
        call_votes += 1
    elif close_loc < 0.46:
        put_votes += 1
    if rsi_slope > 0.01:
        call_votes += 1
    elif rsi_slope < -0.01:
        put_votes += 1
    return call_votes, put_votes


def _finalize_flow_votes(call_votes: int, put_votes: int, close_loc: float) -> TradeDirection | None:
    """Resolve empate de votos usando fechamento relativo da barra."""
    if call_votes > put_votes:
        return TradeDirection.CALL
    if put_votes > call_votes:
        return TradeDirection.PUT
    if close_loc > 0.52:
        return TradeDirection.CALL
    if close_loc < 0.48:
        return TradeDirection.PUT
    return None


def flow_implied_direction(ctx: dict) -> TradeDirection | None:
    """Infere CALL/PUT pelo fluxo de velas, momentum e regime de mercado."""
    if not ctx:
        return None
    body = float(ctx.get("body", 0.0))
    body_sum = float(ctx.get("body_sum_3", body))
    ema = float(ctx.get("ema_spread", 0.0))
    ret5 = float(ctx.get("ret_5", 0.0))
    ret3 = float(ctx.get("ret_3", 0.0))
    close_loc = float(ctx.get("close_loc", 0.5))
    sma_z = float(ctx.get("sma_z", 0.0))
    vr = float(ctx.get("variance_ratio", 1.0))
    streak = int(float(ctx.get("body_streak", 0.0)))
    rsi_slope = float(ctx.get("rsi_slope", 0.0))
    call_votes, put_votes = _body_flow_votes(body, body_sum, streak)
    rc, rp = _regime_flow_votes(vr, ema, ret5, ret3, sma_z)
    ac, ap = _aux_flow_votes(close_loc, rsi_slope)
    return _finalize_flow_votes(call_votes + rc + ac, put_votes + rp + ap, close_loc)


def flow_strength(ctx: dict) -> float:
    """Mede intensidade do fluxo de mercado entre 0 e 1."""
    if not ctx:
        return 0.0
    body = abs(float(ctx.get("body", 0.0)))
    body_sum = abs(float(ctx.get("body_sum_3", 0.0)))
    ema = abs(float(ctx.get("ema_spread", 0.0)))
    ret5 = abs(float(ctx.get("ret_5", 0.0)))
    streak = float(ctx.get("body_streak", 0.0))
    rel_vol = float(ctx.get("rel_vol", 0.0))
    raw = body * 120.0 + body_sum * 80.0 + ema * 400.0 + ret5 * 150.0 + streak * 0.12
    if rel_vol >= 0.28:
        raw += 0.08
    return min(1.0, raw)


def apply_candle_flow_override(
    direction: TradeDirection,
    raw_prob: float,
    context: dict[str, float],
    params: dict,
) -> tuple[TradeDirection, bool, float]:
    """Substitui direcao ambigua do DL quando fluxo de velas e forte e oposto."""
    bs = params.get("binary_signal") or {}
    weak_margin = float(bs.get("weak_dl_override_margin", 0.08))
    raw_side = max(float(raw_prob), 1.0 - float(raw_prob))
    if raw_side > 0.5 + weak_margin:
        return direction, False, raw_prob
    flow_dir = flow_implied_direction(context)
    strength = flow_strength(context)
    if flow_dir is None or flow_dir == direction or strength + 1e-9 < 0.24:
        return direction, False, raw_prob
    flipped = float(raw_prob) if flow_dir == TradeDirection.CALL else 1.0 - float(raw_prob)
    return flow_dir, True, flipped


def flow_aligns_with(direction: TradeDirection, ctx: dict) -> bool:
    """Indica se a direcao proposta segue o fluxo dominante da barra."""
    flow_dir = flow_implied_direction(ctx)
    if flow_dir is None:
        return True
    strength = flow_strength(ctx)
    if strength + 1e-9 < 0.18:
        return True
    return direction == flow_dir


def flow_alignment_bonus(direction: TradeDirection, ctx: dict) -> float:
    """Bonus ou penalidade conforme alinhamento com fluxo de velas."""
    flow_dir = flow_implied_direction(ctx)
    strength = flow_strength(ctx)
    if flow_dir is None or strength + 1e-9 < 0.12:
        return 0.0
    if direction == flow_dir:
        return min(0.14, 0.06 + strength * 0.10)
    return -min(0.16, 0.08 + strength * 0.12)


_SMA_Z_EXTREME = 0.004


def _accumulate_binary_votes(ctx: dict) -> tuple[int, int, float]:
    """Conta votos CALL/PUT a partir de indicadores estatisticos da barra."""
    sma_z = float(ctx.get("sma_z", 0.0))
    rsi = float(ctx.get("rsi", 0.5))
    close_loc = float(ctx.get("close_loc", 0.5))
    z_spread = float(ctx.get("z_spread", 0.0))
    call_votes = 0
    put_votes = 0
    if sma_z <= -0.002:
        call_votes += 2
    elif sma_z >= 0.002:
        put_votes += 2
    if rsi < 0.45:
        call_votes += 1
    elif rsi > 0.55:
        put_votes += 1
    if close_loc > 0.52:
        call_votes += 1
    elif close_loc < 0.48:
        put_votes += 1
    if z_spread <= -0.2:
        call_votes += 1
    elif z_spread >= 0.2:
        put_votes += 1
    return call_votes, put_votes, sma_z


def binary_direction_vote(ctx: dict) -> TradeDirection | None:
    """Infere CALL/PUT por votos estatisticos quando fluxo e inconclusivo."""
    if not ctx:
        return None
    call_votes, put_votes, sma_z = _accumulate_binary_votes(ctx)
    if call_votes > put_votes:
        return TradeDirection.CALL
    if put_votes > call_votes:
        return TradeDirection.PUT
    if sma_z >= _SMA_Z_EXTREME:
        return TradeDirection.PUT
    if sma_z <= -_SMA_Z_EXTREME:
        return TradeDirection.CALL
    return None


def sma_extreme_direction(sma_z: float) -> TradeDirection | None:
    """Retorna direcao quando sma_z esta em extremo estatistico."""
    if sma_z >= _SMA_Z_EXTREME:
        return TradeDirection.PUT
    if sma_z <= -_SMA_Z_EXTREME:
        return TradeDirection.CALL
    return None
