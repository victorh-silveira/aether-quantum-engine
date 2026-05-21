"""Resolve runtime e monta blocos de contexto para a decisao LLM Profunda."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.application.services.llm import resolve_indicator_config
from src.application.services.llm.context_runtime_support import (
    build_context_extra,
    fetch_six_layer_series,
    llm_timing_fields,
    min_conviction_execute,
    timeframe_runtime_fields,
)
from src.application.services.llm.deriv_timeframes import deriv_tf_label
from src.application.services.llm.gemini_constants import GEMINI_DEFAULT_MODEL
from src.application.services.llm.llm_config_merge import effective_llm_section, merge_execution_section
from src.application.services.llm.narrative_context import (
    describe_m3_trigger,
    describe_m5_filter,
    describe_m15_map,
    describe_mtf_alignment_six,
)
from src.application.services.llm.sniper_payload import build_sniper_tokens
from src.application.services.llm.sovereign_system import SOVEREIGN_SYSTEM
from src.application.services.llm.strategy_payload_config import resolve_strategy_payload_config


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
    timeout_http, llm_async_outer, max_decision_latency, gemini_retries = llm_timing_fields(cfg)
    raw_generation = cfg.get("generation_config")
    generation_config = dict(raw_generation) if isinstance(raw_generation, dict) else {}
    indicator_config = resolve_indicator_config(cfg.get("indicator_config"))
    strategy_payload = resolve_strategy_payload_config(orch.config)
    tf_fields = timeframe_runtime_fields(cfg, indicator_config)
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
        "min_conviction_execute": min_conviction_execute(cfg, orch.config),
        "inversion_threshold": inversion_threshold,
        "follow_threshold": follow_threshold,
        **tf_fields,
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
        "llm_fallback_model": str(cfg.get("llm_fallback_model") or "").strip(),
        "llm_wait_fallback_mode": str(cfg.get("llm_wait_fallback_mode") or "").strip().lower(),
    }


async def fetch_context_blocks(
    orch: Any, sym: str, runtime: dict[str, Any]
) -> tuple[str, str, str, str, str, dict[str, Any]]:
    """Monta narrativas macro, estrutura, swing, gatilho, alinhamento MTF e extras."""
    ic = runtime["indicator_config"]
    sp = runtime.get("strategy_payload")
    (
        macro_use,
        structure_use,
        swing_use,
        trigger_use,
        trigger_ohlc_use,
        micro_swing_use,
        micro_trigger_use,
        macro_gran,
        structure_gran,
        swing_gran,
        trigger_gran,
        micro_swing_gran,
        micro_trigger_gran,
    ) = await fetch_six_layer_series(orch, sym, runtime)
    lm = deriv_tf_label(macro_gran)
    ls = deriv_tf_label(structure_gran)
    lw = deriv_tf_label(swing_gran)
    lt = deriv_tf_label(trigger_gran)
    l5 = deriv_tf_label(micro_swing_gran)
    l1 = deriv_tf_label(micro_trigger_gran)
    sniper_tok = build_sniper_tokens(trigger_use, ic, sp)
    macro_desc = describe_m15_map(macro_use, ic, lm)
    structure_desc = describe_m5_filter(structure_use, ic, ls)
    swing_desc = describe_m5_filter(swing_use, ic, lw)
    trigger_desc = describe_m3_trigger(trigger_use, ic, lt)
    micro_swing_desc = describe_m5_filter(micro_swing_use, ic, l5)
    micro_trigger_desc = describe_m3_trigger(micro_trigger_use, ic, l1)
    mtf_align = describe_mtf_alignment_six(
        macro_use,
        structure_use,
        swing_use,
        trigger_use,
        micro_swing_use,
        micro_trigger_use,
        ic,
        lm,
        ls,
        lw,
        lt,
        l5,
        l1,
    )
    extra = build_context_extra(
        macro_use=macro_use,
        structure_use=structure_use,
        swing_use=swing_use,
        trigger_use=trigger_use,
        micro_swing_use=micro_swing_use,
        micro_trigger_use=micro_trigger_use,
        trigger_ohlc_use=trigger_ohlc_use,
        macro_gran=macro_gran,
        structure_gran=structure_gran,
        swing_gran=swing_gran,
        trigger_gran=trigger_gran,
        micro_swing_gran=micro_swing_gran,
        micro_trigger_gran=micro_trigger_gran,
        micro_swing_desc=micro_swing_desc,
        micro_trigger_desc=micro_trigger_desc,
        ic=ic,
        sp=sp,
        sniper_tok=sniper_tok,
    )
    return macro_desc, structure_desc, swing_desc, trigger_desc, mtf_align, extra
