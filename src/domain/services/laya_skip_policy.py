"""Politica pura de dominio para SKIP configuravel sobre decisoes do Laya."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.indicators.technical_indicators import IndicatorSnapshot
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import TradeDirection


@dataclass(frozen=True, slots=True)
class SkipVerdict:
    """Resultado da avaliacao de SKIP para o ciclo atual."""

    should_skip: bool
    reason: str


def evaluate_laya_skip(
    decision: LayaDecision,
    snapshot: IndicatorSnapshot,
    config: dict[str, Any],
    *,
    entry_delay_seconds: float | None = None,
) -> SkipVerdict:
    """Avalia filtros configuraveis de protecao sem alterar probabilidades do Laya."""
    policy = (config or {}).get("laya", {}).get("skip_policy", {})
    if not bool(policy.get("enabled", False)):
        return SkipVerdict(should_skip=False, reason="")
    max_delay = float(policy.get("max_entry_delay_seconds", 0.0) or 0.0)
    if max_delay > 0.0 and entry_delay_seconds is not None and entry_delay_seconds > max_delay:
        return SkipVerdict(should_skip=True, reason=f"atraso_vela({entry_delay_seconds:.0f}s>{max_delay:.0f}s)")
    sign = 1.0 if decision.direction == TradeDirection.CALL else -1.0
    min_prob = float(policy.get("min_probability", 0.0) or 0.0)
    if min_prob > 0.5 and decision.probability < min_prob:
        return SkipVerdict(
            should_skip=True,
            reason=f"prob_abaixo_minimo({decision.probability * 100.0:.2f}%<{min_prob * 100.0:.2f}%)",
        )
    min_body = float(policy.get("min_body_atr", 0.0) or 0.0)
    if min_body > 0.0 and snapshot.body_atr < min_body:
        return SkipVerdict(
            should_skip=True,
            reason=f"vela_doji(body={snapshot.body_atr:.2f}ATR<{min_body:.2f})",
        )
    if (
        bool(policy.get("block_counter_momentum", False))
        and (snapshot.candle_dir * sign) < 0.0
        and ((snapshot.macd_hist_delta * sign) < 0.0 or (snapshot.macd_hist * sign) < 0.0)
    ):
        return SkipVerdict(should_skip=True, reason="contra_momentum_vela_e_macd")
    bb_margin = float(policy.get("bb_extreme_margin", 0.0) or 0.0)
    if bb_margin > 0.0:
        is_overbought_call = sign > 0.0 and snapshot.bollinger_pct_b >= (1.0 - bb_margin)
        is_oversold_put = sign < 0.0 and snapshot.bollinger_pct_b <= bb_margin
        if is_overbought_call or is_oversold_put:
            return SkipVerdict(
                should_skip=True,
                reason=f"extremo_banda(BB%B={snapshot.bollinger_pct_b:.2f})",
            )
    if (
        bool(policy.get("block_strong_counter_trend", False))
        and (snapshot.candle_dir * sign) < 0.0
        and (snapshot.macd_hist_delta * sign) < 0.0
        and (snapshot.di_diff_14 * sign) < 0.0
    ):
        return SkipVerdict(should_skip=True, reason="contra_fluxo_triplo(vela+macd+di)")
    max_streak = int(policy.get("max_counter_streak", 0) or 0)
    if max_streak > 0 and (snapshot.streak * sign) <= -float(max_streak):
        return SkipVerdict(should_skip=True, reason=f"contra_streak({snapshot.streak:+.0f})")
    max_wick = float(policy.get("max_counter_wick_atr", 0.0) or 0.0)
    if max_wick > 0.0:
        counter_wick = snapshot.upper_wick_atr if sign > 0.0 else snapshot.lower_wick_atr
        if counter_wick >= max_wick and counter_wick > snapshot.body_atr:
            return SkipVerdict(should_skip=True, reason=f"pavio_rejeicao({counter_wick:.2f}ATR>={max_wick:.2f})")
    min_ratio = float(policy.get("min_atr_ratio", 0.0) or 0.0)
    max_ratio = float(policy.get("max_atr_ratio", 0.0) or 0.0)
    if (min_ratio > 0.0 and snapshot.atr_ratio < min_ratio) or (max_ratio > 0.0 and snapshot.atr_ratio > max_ratio):
        return SkipVerdict(should_skip=True, reason=f"volatilidade_anomala(ratio={snapshot.atr_ratio:.2f})")
    return SkipVerdict(should_skip=False, reason="")
