"""Adapt SCALE por maioria de votos entre fita, MILI, MINI e RSI (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _rsi_side(metrics: dict[str, Any], *, neutral: float) -> str | None:
    """Deriva CALL/PUT do RSI; None se neutro/ausente."""
    raw = metrics.get("rsi")
    if raw is None and isinstance(metrics.get("indicators"), dict):
        raw = metrics["indicators"].get("rsi")
    try:
        rsi = float(raw)
    except (TypeError, ValueError):
        return None
    if rsi + 1e-12 < float(neutral):
        return TradeDirection.PUT.name
    if rsi - 1e-12 > float(neutral):
        return TradeDirection.CALL.name
    return None


def collect_scale_side_votes(
    metrics: dict[str, Any],
    tcn_dir: TradeDirection,
    *,
    include_rsi: bool = True,
    include_micro_bar: bool = False,
    rsi_neutral: float = 0.5,
) -> dict[str, Any]:
    """Conta votos CALL/PUT das fontes de telemetria SCALE + RSI."""
    votes: list[tuple[str, str]] = [("tcn", tcn_dir.name)]
    tape = _side(metrics.get("scale_tape_consensus"))
    if tape is not None:
        votes.append(("tape", tape))
    mili = _side(metrics.get("scale_mili_dir"))
    if mili is not None:
        votes.append(("mili", mili))
    mini_prev = _side(metrics.get("scale_mini_prev_bar_dir"))
    mini_cur = _side(metrics.get("scale_mini_bar_dir"))
    if mini_prev is not None and mini_cur is not None and mini_prev == mini_cur:
        votes.append(("mini_pair", mini_cur))
    if include_micro_bar:
        micro = _side(
            metrics.get("closed_micro_candle_dir")
            or metrics.get("scale_micro_prev_bar_dir")
            or metrics.get("scale_micro_bar_dir")
        )
        if micro is not None:
            votes.append(("micro_bar", micro))
    if include_rsi:
        rsi_side = _rsi_side(metrics, neutral=rsi_neutral)
        if rsi_side is not None:
            votes.append(("rsi", rsi_side))
    call_n = sum(1 for _, side in votes if side == TradeDirection.CALL.name)
    put_n = sum(1 for _, side in votes if side == TradeDirection.PUT.name)
    winner: str | None = None
    if call_n > put_n:
        winner = TradeDirection.CALL.name
    elif put_n > call_n:
        winner = TradeDirection.PUT.name
    payload = {
        "scale_vote_call_n": int(call_n),
        "scale_vote_put_n": int(put_n),
        "scale_vote_n": int(call_n + put_n),
        "scale_vote_winner": winner,
        "scale_vote_sources": ",".join(f"{name}:{side}" for name, side in votes),
    }
    metrics.update(payload)
    return payload


def adapt_on_majority_votes(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    cfg: dict[str, Any],
) -> TradeDirection | None:
    """Adapta ao lado com mais votos quando a lideranca minima e atingida."""
    if not bool(cfg.get("adapt_on_majority_votes", True)):
        return None
    include_rsi = bool(cfg.get("adapt_majority_include_rsi", True))
    include_micro = bool(cfg.get("adapt_majority_include_micro_bar", False))
    rsi_neutral = float(cfg.get("adapt_majority_rsi_neutral", 0.5))
    payload = collect_scale_side_votes(
        metrics,
        exec_dir,
        include_rsi=include_rsi,
        include_micro_bar=include_micro,
        rsi_neutral=rsi_neutral,
    )
    winner = payload.get("scale_vote_winner")
    if winner is None or winner == exec_dir.name:
        return None
    call_n = int(payload["scale_vote_call_n"])
    put_n = int(payload["scale_vote_put_n"])
    lead = abs(call_n - put_n)
    min_lead = max(1, int(cfg.get("adapt_majority_min_lead", 1)))
    if lead < min_lead:
        return None
    min_votes = max(2, int(cfg.get("adapt_majority_min_votes", 3)))
    if int(payload["scale_vote_n"]) < min_votes:
        return None
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = "majority_votes"
    return TradeDirection[str(winner)]
