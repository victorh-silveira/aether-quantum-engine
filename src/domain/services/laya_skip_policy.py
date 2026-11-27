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
) -> SkipVerdict:
    """Avalia filtros configuraveis de protecao sem alterar probabilidades do Laya."""
    policy = (config or {}).get("laya", {}).get("skip_policy", {})
    if not bool(policy.get("enabled", False)):
        return SkipVerdict(should_skip=False, reason="")
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
        and (snapshot.macd_hist_delta * sign) < 0.0
    ):
        return SkipVerdict(should_skip=True, reason="contra_momentum_vela_e_macd")
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
    return SkipVerdict(should_skip=False, reason="")
