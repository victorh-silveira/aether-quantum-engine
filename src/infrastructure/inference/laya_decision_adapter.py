"""Adapter de inferencia continua via Laya System 1 em GPU CUDA ou CPU."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import warnings
from typing import Any

from src.domain.models.laya_decision import LayaDecision
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.laya_checkpoint import resolve_approved_checkpoint
from src.infrastructure.inference.laya_context import encode_laya_context, validate_laya_usage


_SENSITIVE_KEYS = frozenset({"token", "api_token", "pat", "secret", "password", "dsn", "key", "app_id"})


def _finite_probability(value: Any, default: float = 0.5) -> float:
    """Preserva zero legitimo e sinaliza fallback para valores invalidos."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        logging.getLogger("AETH").warning("LAYA_RESPONSE | probabilidade invalida; fallback=%.2f", default)
        return default
    return number


_LAYA_QUESTIONS = {
    "direction": {
        "type": "choice",
        "instructions": (
            "Evaluate the closed M5 OHLC candles, all technical indicators, and bot settings "
            "to predict the next 5-minute binary contract direction (CALL or PUT)."
        ),
        "criteria": {
            "CALL": "Price closes higher in 5 minutes: bullish trend, upward momentum, or oversold bounce.",
            "PUT": "Price closes lower in 5 minutes: bearish trend, downward momentum, or overbought reversal.",
        },
    }
}


def sanitize_settings_context(config: dict[str, Any]) -> dict[str, Any]:
    """Remove credenciais e campos sensiveis do settings.json antes de enviar ao Laya."""

    def _clean(obj: Any) -> Any:
        """Percorre dicionarios e listas removendo chaves sensiveis."""
        if isinstance(obj, dict):
            return {
                k: _clean(v) for k, v in obj.items() if not any(secret in str(k).lower() for secret in _SENSITIVE_KEYS)
            }
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    return _clean(config)


def _format_candles_summary(candles: list[Candle], limit: int = 12) -> list[dict[str, float | int]]:
    """Formata ultimas velas fechadas de forma compacta para o estado do Laya."""
    tail = candles[-max(1, limit) :]
    return [
        {
            "epoch": bar.epoch,
            "open": round(bar.open, 5),
            "high": round(bar.high, 5),
            "low": round(bar.low, 5),
            "close": round(bar.close, 5),
        }
        for bar in tail
    ]


def build_laya_state(symbol: str, candles: list[Candle], indicators: dict, settings: dict, limit: int = 12) -> dict:
    """Monta a entrada causal comum ao dataset e ao motor de execucao."""
    return {
        "symbol": symbol,
        "timeframe_seconds": 300,
        "indicators": {k: round(float(v), 6) for k, v in indicators.items()},
        "recent_candles": _format_candles_summary(candles, limit),
        "settings": sanitize_settings_context(settings),
    }


def _extract_choice_probs(
    raw: dict[str, Any], indicators: dict[str, float]
) -> tuple[TradeDirection, float, float, float]:
    """Extrai direcao binaria continua (nunca SKIP) e probabilidades normalizadas."""
    answers = raw.get("answers") if isinstance(raw.get("answers"), dict) else {}
    entry = answers.get("direction") if isinstance(answers.get("direction"), dict) else {}
    dir_probs = entry.get("probabilities") if isinstance(entry.get("probabilities"), dict) else {}
    if not dir_probs:
        probs_map = raw.get("probabilities") or {}
        dir_probs = probs_map.get("direction") if isinstance(probs_map, dict) else {}
    if not isinstance(dir_probs, dict):
        dir_probs = {}
    p_call = _finite_probability(dir_probs.get("CALL"))
    p_put = _finite_probability(dir_probs.get("PUT"))
    total = p_call + p_put
    if total <= 1e-12:
        logging.getLogger("AETH").warning("LAYA_RESPONSE | soma nula; fallback=0.5/0.5")
        p_call, p_put = 0.5, 0.5
    else:
        p_call /= total
        p_put /= total
    raw_conf = entry.get("answer_confidence", entry.get("confidence"))
    if raw_conf is None:
        conf_map = raw.get("confidence") or {}
        raw_conf = conf_map.get("direction") if isinstance(conf_map, dict) else None
    conf = _finite_probability(raw_conf, max(p_call, p_put)) if raw_conf is not None else max(p_call, p_put)
    if p_call == p_put:
        ema20 = float(indicators.get("ema_20", 0.0))
        ema50 = float(indicators.get("ema_50", 0.0))
        direction = TradeDirection.CALL if ema20 >= ema50 else TradeDirection.PUT
    else:
        direction = TradeDirection.CALL if p_call > p_put else TradeDirection.PUT
    return direction, p_call, p_put, conf


class LayaDecisionAdapter:
    """Adapter de decisao binaria continua usando laya.load(..., vegaml=False)."""

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        device: str = "auto",
        context_candles: int = 12,
        agent: Any | None = None,
        approved_checkpoint: str = "",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.context_candles = max(1, int(context_candles))
        self._agent = agent
        self.approved_checkpoint = approved_checkpoint
        self.model_version = model_id

    def _ensure_agent(self) -> Any:
        """Carrega o modelo Laya pre-treinado sob demanda com suporte a CUDA."""
        if self._agent is not None:
            return self._agent
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        warnings.filterwarnings(
            "ignore",
            message=r"laya: this checkpoint ships invalid temperatures.*",
            category=RuntimeWarning,
        )
        import laya  # noqa: PLC0415
        import torch  # noqa: PLC0415

        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        checkpoint, self.model_version = resolve_approved_checkpoint(self.model_id, self.approved_checkpoint)
        self._agent = laya.load(checkpoint, device=resolved_device, vegaml=False)
        return self._agent

    def _predict_sync(
        self,
        symbol: str,
        candles: list[Candle],
        indicators: dict[str, float],
        settings_context: dict[str, Any],
    ) -> LayaDecision:
        """Executa inferencia sincrona em um unico forward pass System 1."""
        agent = self._ensure_agent()
        state = build_laya_state(symbol, candles, indicators, settings_context, self.context_candles)
        context, max_len, signature = encode_laya_context(agent, state, _LAYA_QUESTIONS)
        t0 = time.perf_counter()
        raw = agent.system_one(context, _LAYA_QUESTIONS, max_len=max_len)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = validate_laya_usage(raw)
        logging.getLogger("AETH").info(
            "LAYA_CONTEXT | %s g=300s bars=%d epoch=%d sig=%s tokens=%s dropped=%s max_len=%d",
            symbol,
            len(candles),
            candles[-1].epoch,
            signature,
            usage.get("state_tokens"),
            usage.get("state_tokens_dropped", 0),
            max_len,
        )
        direction, p_call, p_put, conf = _extract_choice_probs(raw if isinstance(raw, dict) else {}, indicators)
        win_prob = p_call if direction is TradeDirection.CALL else p_put
        return LayaDecision(
            symbol=symbol,
            direction=direction,
            probability=win_prob,
            prob_call=p_call,
            prob_put=p_put,
            confidence=conf,
            latency_ms=latency_ms,
            indicators=dict(indicators),
            model_version=self.model_version,
        )

    async def decide(
        self,
        symbol: str,
        candles: list[Candle],
        indicators: dict[str, float],
        settings_context: dict[str, Any],
    ) -> LayaDecision:
        """Executa decisao do Laya fora do event loop asyncio."""
        return await asyncio.to_thread(self._predict_sync, symbol, candles, indicators, settings_context)
