"""Helpers de parametros, series multi-timeframe e extras de contexto LLM."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm import (
    IndicatorConfig,
    bundle_llm_indicators_for_log,
    effective_indicator_config_log,
    ema_distance_guard_line,
    min_bars_for_indicators,
    mtf_confluence_line,
)
from src.application.services.llm.deriv_timeframes import deriv_tf_compact_numeric_tag, deriv_tf_label
from src.application.services.llm.gemini_constants import GEMINI_HTTP_CEILING_SEC
from src.application.services.llm.indicators import _hurst_exponent, _vol_range_pct
from src.application.services.llm.llm_config_merge import risk_limits_section
from src.application.services.llm.narrative_context import (
    describe_micro_structure,
    describe_session_context,
    describe_volatility_regime,
)
from src.application.services.llm.regime import classify_regime, sigma_pct_m5
from src.application.services.llm.sniper_payload import build_mtf_metrics_matrix


def tail_closes(closes: list[float], bars: int) -> list[float]:
    """Recorta fechamentos para no maximo ``bars`` pontos."""
    if not closes:
        return []
    tail = closes[-bars:] if bars > 0 else closes
    n = min(len(tail), bars) if bars > 0 else len(tail)
    return [float(x) for x in tail[-n:]]


def min_conviction_execute(cfg: dict[str, Any], orch_config: dict[str, Any]) -> float:
    """Resolve limiar minimo de conviccao para executar ordens."""
    limits = risk_limits_section(orch_config)
    mc_lim = limits.get("min_conviction_execute")
    value = max(0.0, min(0.99, float(cfg.get("min_conviction_execute", 0.80))))
    with suppress(TypeError, ValueError):
        if mc_lim is not None:
            value = max(0.0, min(0.99, float(mc_lim)))
    return value


def llm_timing_fields(cfg: dict[str, Any]) -> tuple[float, float, float, int]:
    """Calcula timeouts HTTP, envelope assincrono e tentativas Gemini."""
    timeout_http = float(cfg.get("timeout_seconds", 120))
    outer_margin = max(2.0, float(cfg.get("llm_http_outer_margin_seconds", 5.0)))
    gemini_retries = max(0, int(cfg.get("llm_retry_attempts", 3)))
    attempts = max(1, gemini_retries + 1)
    deadline_cap = min(float(GEMINI_HTTP_CEILING_SEC), max(1.0, timeout_http))
    per_attempt = deadline_cap + 0.5
    llm_async_outer = max(timeout_http + outer_margin, float(attempts) * per_attempt + 1.0)
    max_lat_cfg = max(1.0, float(cfg.get("max_decision_latency_seconds", 15.0)))
    extra_wall = max(0.0, float(cfg.get("llm_extra_wall_clock_seconds", 25.0)))
    max_decision_latency = max(max_lat_cfg, llm_async_outer + extra_wall)
    return timeout_http, llm_async_outer, max_decision_latency, gemini_retries


def timeframe_runtime_fields(cfg: dict[str, Any], ic: IndicatorConfig) -> dict[str, int]:
    """Normaliza granularidades e barras das seis camadas de timeframe."""
    min_bars = min_bars_for_indicators(ic)
    trigger_gran = int(
        cfg.get(
            "trigger_granularity_seconds", cfg.get("m1_granularity_seconds", cfg.get("m3_granularity_seconds", 300))
        )
    )
    macro_gran = int(cfg.get("macro_granularity_seconds", cfg.get("m30_granularity_seconds", 14400)))
    structure_gran = int(cfg.get("structure_granularity_seconds", cfg.get("m15_granularity_seconds", 3600)))
    swing_gran = int(cfg.get("swing_granularity_seconds", cfg.get("analysis_granularity_seconds", 900)))
    micro_swing_gran = int(cfg.get("micro_swing_granularity_seconds", 300))
    micro_trigger_gran = int(cfg.get("micro_trigger_granularity_seconds", 60))
    trigger_bars = max(2, int(cfg.get("trigger_bars", cfg.get("m1_bars", cfg.get("m3_bars", 120)))), min_bars)
    macro_bars = max(2, int(cfg.get("macro_bars", cfg.get("m30_bars", 120))), min_bars)
    structure_bars = max(2, int(cfg.get("structure_bars", cfg.get("m15_bars", 120))), min_bars)
    swing_bars = max(2, int(cfg.get("swing_bars", cfg.get("analysis_bars", 120))), min_bars)
    micro_swing_bars = max(2, int(cfg.get("micro_swing_bars", 120)), min_bars)
    micro_trigger_bars = max(2, int(cfg.get("micro_trigger_bars", 120)), min_bars)
    return {
        "tf_macro_gran": macro_gran,
        "tf_macro_bars": macro_bars,
        "tf_structure_gran": structure_gran,
        "tf_structure_bars": structure_bars,
        "tf_swing_gran": swing_gran,
        "tf_swing_bars": swing_bars,
        "tf_trigger_gran": trigger_gran,
        "tf_trigger_bars": trigger_bars,
        "tf_micro_swing_gran": micro_swing_gran,
        "tf_micro_swing_bars": micro_swing_bars,
        "tf_micro_trigger_gran": micro_trigger_gran,
        "tf_micro_trigger_bars": micro_trigger_bars,
        "m1_gran": trigger_gran,
        "m1_bars": trigger_bars,
        "m30_gran": macro_gran,
        "m30_bars": macro_bars,
        "m3_gran": trigger_gran,
        "m3_bars": trigger_bars,
        "m15_gran": structure_gran,
        "m15_bars": structure_bars,
        "m5_gran": swing_gran,
        "m5_bars": swing_bars,
    }


async def fetch_six_layer_series(
    orch: Any,
    sym: str,
    runtime: dict[str, Any],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[tuple[float, float, float, float]],
    list[float],
    list[float],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Busca fechamentos e OHLC das seis camadas de timeframe para um simbolo."""
    macro_gran = int(runtime.get("tf_macro_gran", runtime.get("m30_gran", 14400)))
    structure_gran = int(runtime.get("tf_structure_gran", runtime.get("m15_gran", 3600)))
    swing_gran = int(runtime.get("tf_swing_gran", runtime.get("m5_gran", 900)))
    trigger_gran = int(runtime.get("tf_trigger_gran", runtime.get("m1_gran", 60)))
    micro_swing_gran = int(runtime.get("tf_micro_swing_gran", 300))
    micro_trigger_gran = int(runtime.get("tf_micro_trigger_gran", 60))
    macro_bars = int(runtime.get("tf_macro_bars", runtime.get("m30_bars", 120)))
    structure_bars = int(runtime.get("tf_structure_bars", runtime.get("m15_bars", 120)))
    swing_bars = int(runtime.get("tf_swing_bars", runtime.get("m5_bars", 120)))
    trigger_bars = int(runtime.get("tf_trigger_bars", runtime.get("m1_bars", 120)))
    micro_swing_bars = int(runtime.get("tf_micro_swing_bars", 120))
    micro_trigger_bars = int(runtime.get("tf_micro_trigger_bars", 120))
    (
        macro_closes,
        structure_closes,
        swing_closes,
        micro_swing_closes,
        trigger_ohlc_rows,
        micro_trigger_closes,
    ) = await asyncio.gather(
        orch.stream.fetch_candle_closes(sym, macro_gran, macro_bars),
        orch.stream.fetch_candle_closes(sym, structure_gran, structure_bars),
        orch.stream.fetch_candle_closes(sym, swing_gran, swing_bars),
        orch.stream.fetch_candle_closes(sym, micro_swing_gran, micro_swing_bars),
        orch.stream.fetch_candle_ohlc(sym, trigger_gran, trigger_bars),
        orch.stream.fetch_candle_closes(sym, micro_trigger_gran, micro_trigger_bars),
    )
    if trigger_ohlc_rows:
        trigger_closes = [float(row[3]) for row in trigger_ohlc_rows]
        trigger_ohlc_use = [tuple(float(x) for x in row) for row in trigger_ohlc_rows[-10:]]
    else:
        trigger_closes = await orch.stream.fetch_candle_closes(sym, trigger_gran, trigger_bars)
        trigger_ohlc_use = []
    return (
        tail_closes(macro_closes, macro_bars),
        tail_closes(structure_closes, structure_bars),
        tail_closes(swing_closes, swing_bars),
        tail_closes(trigger_closes, trigger_bars),
        trigger_ohlc_use,
        tail_closes(micro_swing_closes, micro_swing_bars),
        tail_closes(micro_trigger_closes, micro_trigger_bars),
        macro_gran,
        structure_gran,
        swing_gran,
        trigger_gran,
        micro_swing_gran,
        micro_trigger_gran,
    )


def build_context_extra(
    *,
    macro_use: list[float],
    structure_use: list[float],
    swing_use: list[float],
    trigger_use: list[float],
    micro_swing_use: list[float],
    micro_trigger_use: list[float],
    trigger_ohlc_use: list[tuple[float, float, float, float]],
    macro_gran: int,
    structure_gran: int,
    swing_gran: int,
    trigger_gran: int,
    micro_swing_gran: int,
    micro_trigger_gran: int,
    micro_swing_desc: str,
    micro_trigger_desc: str,
    ic: IndicatorConfig,
    sp: Any,
    sniper_tok: dict[str, str],
) -> dict[str, Any]:
    """Monta metricas, tags MTF e bundles de indicadores para o prompt."""
    lm = deriv_tf_label(macro_gran)
    ls = deriv_tf_label(structure_gran)
    lw = deriv_tf_label(swing_gran)
    lt = deriv_tf_label(trigger_gran)
    l5 = deriv_tf_label(micro_swing_gran)
    l1 = deriv_tf_label(micro_trigger_gran)
    regime_label = classify_regime(structure_use, swing_use, ic)
    entropy_swing = sigma_pct_m5(swing_use, ic)
    swing_arr = np.asarray(swing_use, dtype=np.float64)
    trigger_arr = np.asarray(trigger_use, dtype=np.float64)
    vol_range = _vol_range_pct(swing_arr, ic.volatility_window)
    hurst_trigger = _hurst_exponent(trigger_arr, ic.hurst_window)
    z_trigger = ti._z_score_last(trigger_arr, ic.zscore_window)
    entropy_trigger = ti._shannon_entropy(trigger_arr, ic.entropy_bins, ic.entropy_window)
    mtf_matrix = build_mtf_metrics_matrix(
        (
            (lm, macro_use),
            (ls, structure_use),
            (lw, swing_use),
            (lt, trigger_use),
            (l5, micro_swing_use),
            (l1, micro_trigger_use),
        ),
        ic,
        sp,
    )
    ema_raw = ema_distance_guard_line(lt, trigger_use, ic)
    ema_guard = "EXTREMO" if "EXTREMO_ESTATISTICO_ALERTA" in ema_raw else "OK"
    return {
        "regime_label": regime_label,
        "atr_m5_pct": entropy_swing,
        "entropy_swing": entropy_swing,
        "vol_range_pct": vol_range,
        "hurst_value": hurst_trigger if hurst_trigger is not None else 0.5,
        "zscore_value": z_trigger if z_trigger is not None else 0.0,
        "entropy_trigger": entropy_trigger,
        "regime_line": describe_volatility_regime(structure_use, swing_use, ic),
        "session_line": describe_session_context(),
        "micro_line": describe_micro_structure(micro_trigger_use, ic, l1),
        "micro_swing_desc": micro_swing_desc,
        "micro_trigger_desc": micro_trigger_desc,
        "mtf_matrix": mtf_matrix,
        "ema_guard": ema_guard,
        "llm_macro_closes": macro_use,
        "llm_structure_closes": structure_use,
        "llm_swing_closes": swing_use,
        "llm_trigger_closes": trigger_use,
        "llm_micro_swing_closes": micro_swing_use,
        "llm_micro_trigger_closes": micro_trigger_use,
        "llm_trigger_ohlc": trigger_ohlc_use,
        "m15_closes": macro_use,
        "m5_closes": swing_use,
        "m3_closes": trigger_use,
        "llm_tf_labels": (lm, ls, lw, lt, l5, l1),
        "llm_tf_numeric_tags": (
            deriv_tf_compact_numeric_tag(macro_gran),
            deriv_tf_compact_numeric_tag(structure_gran),
            deriv_tf_compact_numeric_tag(swing_gran),
            deriv_tf_compact_numeric_tag(trigger_gran),
            deriv_tf_compact_numeric_tag(micro_swing_gran),
            deriv_tf_compact_numeric_tag(micro_trigger_gran),
        ),
        "llm_indicator_cfg": effective_indicator_config_log(ic),
        "llm_indicator_bundle": bundle_llm_indicators_for_log(
            macro_use,
            structure_use,
            swing_use,
            trigger_use,
            ic,
            lm,
            ls,
            lw,
            lt,
            micro_swing=micro_swing_use,
            micro_trigger=micro_trigger_use,
            l5=l5,
            l1=l1,
        ),
        "llm_mtf_confluence": mtf_confluence_line(
            swing_use,
            trigger_use,
            ic,
            higher_label=lw,
            lower_label=lt,
        ),
        "llm_mtf_confluence_m30_m5": mtf_confluence_line(
            macro_use,
            structure_use,
            ic,
            higher_label=lm,
            lower_label=ls,
        ),
        "sniper_tokens": sniper_tok,
        "sniper_tokens_audit": dict(sniper_tok),
    }
