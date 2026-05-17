"""Utilitarios de processamento para decisoes de simbolos via LLM."""

from __future__ import annotations

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
