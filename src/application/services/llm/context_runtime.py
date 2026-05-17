"""Resolve runtime e monta blocos de contexto para a decisao LLM Profunda."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import Any

import numpy as np
from dotenv import load_dotenv

import src.application.services.llm.indicators as ti
from src.application.services.llm import (
    bundle_llm_indicators_for_log,
    effective_indicator_config_log,
    min_bars_for_indicators,
    mtf_confluence_line,
    resolve_indicator_config,
)
from src.application.services.llm.deriv_timeframes import deriv_tf_compact_numeric_tag, deriv_tf_label
from src.application.services.llm.gemini_constants import GEMINI_DEFAULT_MODEL, GEMINI_HTTP_CEILING_SEC
from src.application.services.llm.indicators import _hurst_exponent
from src.application.services.llm.llm_config_merge import (
    effective_llm_section,
    merge_execution_section,
    risk_limits_section,
)
from src.application.services.llm.narrative_context import (
    describe_m3_trigger,
    describe_m5_filter,
    describe_m15_map,
    describe_micro_structure,
    describe_mtf_alignment,
    describe_session_context,
    describe_volatility_regime,
)
from src.application.services.llm.regime import classify_regime, sigma_pct_m5
from src.application.services.llm.sniper_payload import build_sniper_tokens
from src.application.services.llm.sovereign_system import SOVEREIGN_SYSTEM
from src.application.services.llm.strategy_payload_config import resolve_strategy_payload_config


def _tail_closes(closes: list[float], bars: int) -> list[float]:
    """Recorta fechamentos para no maximo ``bars`` pontos."""
    if not closes:
        return []
    tail = closes[-bars:] if bars > 0 else closes
    n = min(len(tail), bars) if bars > 0 else len(tail)
    return [float(x) for x in tail[-n:]]


def runtime_for_gemini_call(runtime: dict[str, Any], _ctx: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Retorna o runtime de analise profunda (Modo Rapido Eliminado)."""
    return dict(runtime), "normal"


def resolve_llm_runtime(orch: Any) -> dict[str, Any]:
    """Extrai e normaliza parametros de runtime LLM Profunda."""
    load_dotenv()
    merge_execution_section(orch.config)
    cfg = effective_llm_section(orch.config)
    params = orch.config.get("risk_management", {}).get("params") or {}
    raw_np = cfg.get("max_predict_tokens")
    opt = cfg.get("options") if isinstance(cfg.get("options"), dict) else {}
    keep_alive = str(cfg.get("keep_alive") or opt.get("keep_alive") or "60m")
    llm_temperature = float(opt.get("temperature", 0.0))

    trigger_gran = int(
        cfg.get(
            "trigger_granularity_seconds", cfg.get("m1_granularity_seconds", cfg.get("m3_granularity_seconds", 300))
        )
    )
    trigger_bars_raw = max(
        2, int(cfg.get("trigger_bars", cfg.get("m1_bars", cfg.get("m3_bars", cfg.get("ohlc_bars", 120)))))
    )
    macro_gran = int(cfg.get("macro_granularity_seconds", cfg.get("m30_granularity_seconds", 14400)))
    macro_bars_raw = max(2, int(cfg.get("macro_bars", cfg.get("m30_bars", 120))))
    structure_gran = int(cfg.get("structure_granularity_seconds", cfg.get("m15_granularity_seconds", 3600)))
    structure_bars_raw = max(2, int(cfg.get("structure_bars", cfg.get("m15_bars", 120))))
    swing_gran = int(cfg.get("swing_granularity_seconds", cfg.get("analysis_granularity_seconds", 900)))
    swing_bars_raw = max(2, int(cfg.get("swing_bars", cfg.get("analysis_bars", 120))))

    timeout_http = float(cfg.get("timeout_seconds", 120))
    outer_margin = max(2.0, float(cfg.get("llm_http_outer_margin_seconds", 5.0)))
    gemini_retries = max(0, int(cfg.get("llm_retry_attempts", 3)))

    gemini_attempts = max(1, gemini_retries + 1)
    gemini_deadline_cap = min(float(GEMINI_HTTP_CEILING_SEC), max(1.0, timeout_http))
    gemini_per_attempt_async_cap = gemini_deadline_cap + 0.5
    min_outer_gemini_sequence = float(gemini_attempts) * gemini_per_attempt_async_cap + 1.0
    llm_async_outer = max(timeout_http + outer_margin, min_outer_gemini_sequence)

    max_lat_cfg = max(1.0, float(cfg.get("max_decision_latency_seconds", 15.0)))
    extra_wall = max(0.0, float(cfg.get("llm_extra_wall_clock_seconds", 25.0)))
    max_decision_latency = max(max_lat_cfg, llm_async_outer + extra_wall)

    raw_generation = cfg.get("generation_config")
    generation_config = dict(raw_generation) if isinstance(raw_generation, dict) else {}
    limits = risk_limits_section(orch.config)
    mc_lim = limits.get("min_conviction_execute")
    min_conviction_execute = max(0.0, min(0.99, float(cfg.get("min_conviction_execute", 0.80))))
    with suppress(TypeError, ValueError):
        if mc_lim is not None:
            min_conviction_execute = max(0.0, min(0.99, float(mc_lim)))

    indicator_config = resolve_indicator_config(cfg.get("indicator_config"))
    strategy_payload = resolve_strategy_payload_config(orch.config)
    min_bars = min_bars_for_indicators(indicator_config)
    trigger_bars = max(trigger_bars_raw, min_bars)
    need_align = min_bars_for_indicators(indicator_config)
    macro_bars = max(macro_bars_raw, need_align)
    structure_bars = max(structure_bars_raw, need_align)
    swing_bars = max(swing_bars_raw, need_align)
    ind_cfg_raw = cfg.get("indicator_config") if isinstance(cfg.get("indicator_config"), dict) else {}
    inversion_threshold = float(ind_cfg_raw.get("inversion_threshold", 0.0))
    follow_threshold = float(ind_cfg_raw.get("follow_threshold", 0.0))

    return {
        "base_url": str(cfg.get("base_url", "")),
        "model": str(cfg.get("model", GEMINI_DEFAULT_MODEL)),
        "timeout": timeout_http,
        "llm_async_outer_seconds": llm_async_outer,
        "num_predict": max(32, int(raw_np)) if raw_np is not None else 1024,
        "keep_alive": keep_alive,
        "llm_system": str(cfg.get("system_prompt") or SOVEREIGN_SYSTEM),
        "gemini_api_key": str(os.getenv("GEMINI_API_KEY") or "").strip(),
        "generation_config": generation_config,
        "force_call_put_when_llm_wait": bool(cfg.get("force_call_put_when_llm_wait", True)),
        "llm_temperature": llm_temperature,
        "strategy_payload": strategy_payload,
        "min_conviction_execute": min_conviction_execute,
        "inversion_threshold": inversion_threshold,
        "follow_threshold": follow_threshold,
        "tf_macro_gran": macro_gran,
        "tf_macro_bars": macro_bars,
        "tf_structure_gran": structure_gran,
        "tf_structure_bars": structure_bars,
        "tf_swing_gran": swing_gran,
        "tf_swing_bars": swing_bars,
        "tf_trigger_gran": trigger_gran,
        "tf_trigger_bars": trigger_bars,
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
        "m3_max_ema_distance_pct": 1.0,
        "min_payout_accept": max(0.8, min(0.99, float(cfg.get("min_payout_accept", 0.85)))),
        "payout_estimate": max(0.5, min(0.99, float(params.get("payout_estimate", 0.95)))),
        "duration": params.get("duration", "MULT"),
        "du": str(params.get("duration_unit", "")),
        "parse_retry_attempts": max(1, int(cfg.get("parse_retry_attempts", 2))),
        "max_decision_latency_seconds": max_decision_latency,
        "logic_line_max_chars": max(60, min(300, int(cfg.get("logic_line_max_chars", 140)))),
        "indicator_config": indicator_config,
        "same_direction_strict_enabled": bool(cfg.get("same_direction_strict_enabled", True)),
        "max_same_direction_streak": max(0, int(cfg.get("max_same_direction_streak", 1))),
        "llm_retry_attempts": gemini_retries,
        "llm_wait_fallback_mode": str(cfg.get("llm_wait_fallback_mode") or "").strip().lower(),
    }


async def fetch_context_blocks(
    orch: Any, sym: str, runtime: dict[str, Any]
) -> tuple[str, str, str, str, str, dict[str, Any]]:
    """Monta narrativas macro, estrutura, swing, gatilho, alinhamento MTF e extras."""
    ic = runtime["indicator_config"]
    macro_gran = int(runtime.get("tf_macro_gran", runtime.get("m30_gran", 3600)))
    macro_bars = int(runtime.get("tf_macro_bars", runtime.get("m30_bars", 120)))
    structure_gran = int(runtime.get("tf_structure_gran", runtime.get("m15_gran", 900)))
    structure_bars = int(runtime.get("tf_structure_bars", runtime.get("m15_bars", 120)))
    swing_gran = int(runtime.get("tf_swing_gran", runtime.get("m5_gran", 300)))
    swing_bars = int(runtime.get("tf_swing_bars", runtime.get("m5_bars", 120)))
    trigger_gran = int(runtime.get("tf_trigger_gran", runtime.get("m1_gran", 60)))
    trigger_bars = int(runtime.get("tf_trigger_bars", runtime.get("m1_bars", 120)))
    lm = deriv_tf_label(macro_gran)
    ls = deriv_tf_label(structure_gran)
    lw = deriv_tf_label(swing_gran)
    lt = deriv_tf_label(trigger_gran)
    macro_closes, structure_closes, swing_closes, trigger_ohlc_rows = await asyncio.gather(
        orch.stream.fetch_candle_closes(sym, macro_gran, macro_bars),
        orch.stream.fetch_candle_closes(sym, structure_gran, structure_bars),
        orch.stream.fetch_candle_closes(sym, swing_gran, swing_bars),
        orch.stream.fetch_candle_ohlc(sym, trigger_gran, trigger_bars),
    )
    trigger_ohlc_use = []
    if trigger_ohlc_rows:
        trigger_closes = [float(row[3]) for row in trigger_ohlc_rows]
        trigger_ohlc_use = [tuple(float(x) for x in row) for row in trigger_ohlc_rows[-10:]]
    else:
        trigger_closes = await orch.stream.fetch_candle_closes(sym, trigger_gran, trigger_bars)
    macro_use = _tail_closes(macro_closes, macro_bars)
    structure_use = _tail_closes(structure_closes, structure_bars)
    swing_use = _tail_closes(swing_closes, swing_bars)
    trigger_use = _tail_closes(trigger_closes, trigger_bars)
    tag_m = deriv_tf_compact_numeric_tag(macro_gran)
    tag_s = deriv_tf_compact_numeric_tag(structure_gran)
    tag_w = deriv_tf_compact_numeric_tag(swing_gran)
    tag_t = deriv_tf_compact_numeric_tag(trigger_gran)
    sp = runtime.get("strategy_payload")
    sniper_tok = build_sniper_tokens(trigger_use, ic, sp)
    macro_desc = describe_m15_map(macro_use, ic, lm)
    structure_desc = describe_m5_filter(structure_use, ic, ls)
    swing_desc = describe_m5_filter(swing_use, ic, lw)
    trigger_desc = describe_m3_trigger(trigger_use, ic, lt)
    mtf_align = describe_mtf_alignment(macro_use, structure_use, swing_use, trigger_use, ic, lm, ls, lw, lt)
    regime_label = classify_regime(structure_use, swing_use, ic)
    atr_m5 = sigma_pct_m5(swing_use, ic)
    hurst_val = _hurst_exponent(np.asarray(swing_use), ic.hurst_window)
    z_val = ti._z_score_last(np.asarray(trigger_use), ic.zscore_window)
    extra: dict[str, Any] = {
        "regime_label": regime_label,
        "atr_m5_pct": atr_m5,
        "hurst_value": hurst_val if hurst_val is not None else 0.5,
        "zscore_value": z_val if z_val is not None else 0.0,
        "regime_line": describe_volatility_regime(structure_use, swing_use, ic),
        "session_line": describe_session_context(),
        "micro_line": describe_micro_structure(trigger_use, ic, lt),
        "llm_macro_closes": macro_use,
        "llm_structure_closes": structure_use,
        "llm_swing_closes": swing_use,
        "llm_trigger_closes": trigger_use,
        "llm_trigger_ohlc": trigger_ohlc_use,
        "m15_closes": macro_use,
        "m5_closes": swing_use,
        "m3_closes": trigger_use,
        "llm_tf_numeric_tags": (tag_m, tag_s, tag_w, tag_t),
        "llm_indicator_cfg": effective_indicator_config_log(ic),
        "llm_indicator_bundle": bundle_llm_indicators_for_log(
            macro_use, structure_use, swing_use, trigger_use, ic, lm, ls, lw, lt
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
    return macro_desc, structure_desc, swing_desc, trigger_desc, mtf_align, extra
