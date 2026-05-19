"""Utilitarios de processamento para decisoes de simbolos via LLM."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.llm import (
    dual_confluence_prompt_fragment,
    format_numeric_indicators_tight_line,
)
from src.application.services.llm.context_runtime import fetch_context_blocks
from src.application.services.llm.prompt_utils import (
    build_institutional_pa_bundle,
    build_sniper_trading_prompt as _build_prompt_impl,
)
from src.application.services.llm.sniper_payload import coerce_sniper_tokens


async def _fetch_cluster_status(orch: Any, runtime: dict[str, Any]) -> str:
    """Fetch realtime candle data for US and EU target index clusters to compute trends."""
    is_real = (
        hasattr(orch, "stream")
        and hasattr(orch.stream, "fetch_candle_closes")
        and not hasattr(orch.stream, "mock_calls")
        and not hasattr(orch.stream.fetch_candle_closes, "mock_calls")
        and (
            asyncio.iscoroutinefunction(orch.stream.fetch_candle_closes)
            or hasattr(orch.stream.fetch_candle_closes, "_is_coroutine")
        )
    )
    if not is_real:
        return ""

    try:
        us_symbols = ["OTC_SPC", "OTC_NDX", "OTC_DJI"]
        eu_symbols = ["OTC_FCHI", "OTC_GDAXI", "OTC_FTSE"]
        all_syms = us_symbols + eu_symbols
        swing_gran = int(runtime.get("tf_swing_gran", 300))

        results = await asyncio.gather(
            *[orch.stream.fetch_candle_closes(s, swing_gran, 6) for s in all_syms], return_exceptions=True
        )

        us_parts = []
        eu_parts = []

        for i, s in enumerate(all_syms):
            closes = results[i]
            if isinstance(closes, list) and len(closes) >= 2:
                last_px = float(closes[-1])
                ret = ((closes[-1] - closes[0]) / closes[0]) * 100.0
                direction = "Up" if ret > 0.02 else ("Down" if ret < -0.02 else "Flat")
                part = f"{s.replace('OTC_', '')}: {last_px:.2f} ({direction} {ret:+.2f}%)"
            else:
                part = f"{s.replace('OTC_', '')}: N/A"

            if s in us_symbols:
                us_parts.append(part)
            else:
                eu_parts.append(part)

        return f"US_CLUSTER [{', '.join(us_parts)}] || EU_CLUSTER [{', '.join(eu_parts)}]"
    except Exception as e:
        if hasattr(orch, "logger") and orch.logger:
            orch.logger.warning(f"Error fetching cluster status: {e}")
        return ""


async def build_symbol_prompt(
    orch: Any,
    sym: str,
    runtime: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], float, str, str, str, str, str, str, str, list[float]]:
    """Extrai contexto e constroi o prompt final para a LLM."""
    macro_d, struct_d, swing_d, trig_d, mtf_d, ctx = await fetch_context_blocks(orch, sym, runtime)
    regime_label = str(ctx.get("regime_label", "range"))
    line_macro_structure = str(ctx.get("llm_mtf_confluence_m30_m5") or "")
    line_swing_trigger = str(ctx.get("llm_mtf_confluence") or "")
    macro_c = list(ctx.get("llm_macro_closes") or ctx.get("m15_closes") or [])
    structure_c = list(ctx.get("llm_structure_closes") or [])
    swing_c = list(ctx.get("llm_swing_closes") or ctx.get("m5_closes") or [])
    trigger_c = list(ctx.get("llm_trigger_closes") or ctx.get("m3_closes") or [])
    tf_tags = ctx.get("llm_tf_numeric_tags")
    if not isinstance(tf_tags, tuple) or len(tf_tags) != 4:
        tf_tags = ("60", "15", "5", "1")

    indicators_numeric_line = format_numeric_indicators_tight_line(
        macro_c,
        structure_c,
        swing_c,
        trigger_c,
        runtime["indicator_config"],
        ctx.get("atr_m5_pct"),
        mtf_d,
        line_swing_trigger,
        tf_tags=tf_tags,
    )
    atr_pt = ctx.get("atr_m5_pct")
    cf_dual = dual_confluence_prompt_fragment(line_macro_structure, line_swing_trigger)

    institutional_pa_bundle = build_institutional_pa_bundle(
        regime_label=regime_label,
        atr_m5_pct=float(atr_pt) if atr_pt is not None else None,
        indicators_numeric_line=indicators_numeric_line,
        cf_dual=cf_dual,
        line_macro_structure=line_macro_structure,
        line_swing_trigger=line_swing_trigger,
        compact=False,
    )
    sniper_tok = coerce_sniper_tokens(ctx.get("sniper_tokens"))
    bundle_txt = str(ctx.get("llm_indicator_bundle") or "")
    wr_v, wr_n = None, 0
    rm = getattr(orch, "risk_manager", None)
    if rm is not None and hasattr(rm, "get_wr_rolling_stats"):
        raw_wr = rm.get_wr_rolling_stats(sym)
        if isinstance(raw_wr, tuple) and len(raw_wr) == 2:
            wr_v, wr_n = raw_wr[0], int(raw_wr[1])

    cluster_status = await _fetch_cluster_status(orch, runtime)

    prompt = _build_prompt_impl(
        sym,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        sniper_tok,
        mtf_d,
        str(ctx.get("regime_line", "")),
        str(ctx.get("session_line", "")),
        str(ctx.get("micro_line", "")),
        ctx.get("atr_m5_pct"),
        trigger_c,
        runtime["payout_estimate"],
        runtime["min_payout_accept"],
        runtime["duration"],
        runtime["du"],
        trigger_ohlc=ctx.get("llm_trigger_ohlc"),
        strategy_payload=runtime.get("strategy_payload"),
        institutional_pa_bundle=institutional_pa_bundle,
        indicator_bundle_line=bundle_txt,
        wr_rolling=wr_v,
        wr_samples=wr_n,
        cluster_status=cluster_status,
    )
    return (
        prompt,
        ctx,
        sniper_tok,
        0.5,
        indicators_numeric_line,
        institutional_pa_bundle,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        mtf_d,
        swing_c,
    )
