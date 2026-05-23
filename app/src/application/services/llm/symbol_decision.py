"""Pipeline de contexto, prompt e decisao LLM por simbolo (Analyse Profunda)."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.context_runtime import runtime_for_gemini_call
from src.application.services.llm.duration_logic import calculate_adaptive_duration, enforce_minimum_duration
from src.application.services.llm.llm_bridge_debug import emit_direction_debug as _emit_direction_debug
from src.application.services.llm.llm_bridge_telemetry import (
    attach_decision_telemetry,
    build_metrics_for_decision as _build_metrics_for_decision_core,
    emit_llm_decision_log,
    emit_llm_http_snapshot,
    format_llm_runtime_thresholds,
)
from src.application.services.llm.llm_decision import resolved_system_instruction
from src.application.services.llm.llm_symbol_io import last_reference_price, request_llm_payload
from src.application.services.llm.prompt_utils import (
    iter_llm_prompt_audit_sections,
)
from src.application.services.llm.symbol_decision_post import (
    append_entropy_high_note,
    patch_final_symbol_metrics,
)
from src.application.services.llm.symbol_decision_utils import (
    anchor_llm_decision_complete,
    apply_macro_post_parse,
    build_symbol_prompt,
    decision_from_payload,
)
from src.domain.models.trade import TradeDirection


_request_payload = request_llm_payload


def _emit_symbol_decision_audit(orch: Any, **p: Any) -> None:
    """Registra auditoria LLM_RESPOSTA apos metricas finais do simbolo."""
    sym, runtime, ctx, metrics = p["sym"], p["runtime"], p["ctx"], p["metrics"]
    ms = p["macro_snapshot"]
    statarb_z = float(ms.statarb_spreads[sym]) if ms.statarb_spreads and sym in ms.statarb_spreads else None
    hmm_state = int(ms.hmm_state) if hasattr(ms, "hmm_state") else None
    hmm_prob = float(ms.hmm_prob) if hasattr(ms, "hmm_prob") else None

    audit_sections = iter_llm_prompt_audit_sections(
        sym,
        p["macro_d"],
        p["struct_d"],
        p["swing_d"],
        p["trig_d"],
        p["sniper_tok"],
        p["mtf_d"],
        str(ctx.get("regime_line", "")),
        str(ctx.get("session_line", "")),
        str(ctx.get("micro_line", "")),
        list(ctx.get("llm_trigger_closes") or []),
        runtime["payout_estimate"],
        runtime["min_payout_accept"],
        runtime["duration"],
        runtime["du"],
        trigger_ohlc=ctx.get("llm_trigger_ohlc"),
        strategy_payload=runtime.get("strategy_payload"),
        institutional_pa_bundle=p["institutional_pa_bundle"],
        indicator_bundle_line=str(ctx.get("llm_indicator_bundle") or ""),
        wr_rolling=metrics.get("wr_rolling"),
        wr_samples=int(metrics.get("wr_samples") or 0),
        macro_confluence=ms.macro_block,
        fx_reference_line=ms.fx_reference_line,
        macro_sentiment=ms.tag,
        statarb_z=statarb_z,
        hmm_state=hmm_state,
        hmm_prob=hmm_prob,
    )
    emit_llm_decision_log(
        orch.logger,
        sym,
        cycle_id=orch._active_cycle_id,
        logic_line_max_chars=int(runtime.get("logic_line_max_chars", 140)),
        direction=p["direction"],
        conviction=float(metrics.get("conviction", p["conviction"])),
        ref_px=p["ref_px"],
        model=runtime["model"],
        mtf_alignment=p["mtf_effective"],
        justification=p["note"],
        regime_label=str(ctx.get("regime_label", "range")),
        atr_m5_pct=ctx.get("atr_m5_pct"),
        baseline_prob=float(p["baseline_prob"]),
        wr_rolling=metrics.get("wr_rolling"),
        wr_samples=int(metrics.get("wr_samples") or 0),
        decision_source=str(metrics.get("decision_source", "llm")),
        indicator_cfg=str(ctx.get("llm_indicator_cfg") or ""),
        indicators_numeric_line=p["indicators_numeric_line"],
        runtime_thresholds=format_llm_runtime_thresholds(runtime),
        prompt_char_count=len(p["prompt"]),
        prompt_audit_sections=audit_sections,
        engine_runtime=runtime,
        motor_note=p["note"] if p["inverted"] else "",
        llm_http_ms=p["llm_http_ms"],
        llm_response_chars=p["llm_resp_chars"],
        llm_direction_from_api=p["llm_direction_from_api"],
        us_cluster=p["us_dir"].name if p["us_dir"] else None,
        eu_cluster=p["eu_dir"].name if p["eu_dir"] else None,
        macro_sentiment=ms.tag,
    )


async def collect_symbol_llm_decision(
    orch: Any,
    *,
    sym: str,
    runtime: dict[str, Any],
    llm_metrics: Any,
) -> tuple[TradeDirection | None, dict[str, Any]]:
    """Orquestra prompt, chamada Gemini e metricas por simbolo (Sempre Modo Profundo)."""
    (
        prompt,
        ctx,
        sniper_tok,
        baseline_prob,
        indicators_numeric_line,
        institutional_pa_bundle,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        mtf_d,
        swing_c,
        macro_snapshot,
    ) = await build_symbol_prompt(orch, sym, runtime)

    call_runtime, _ = runtime_for_gemini_call(runtime, ctx)
    cfg_root = getattr(orch, "config", None)
    raw_llm = cfg_root.get("llm") if isinstance(cfg_root, dict) else None
    symbols = list(getattr(orch, "symbols", ()) or ())
    leading_cycle_blank = int(orch._active_cycle_id) > 1 and (not symbols or sym == symbols[0])
    sys_extra = str(runtime.get("llm_system") or "")
    emit_llm_http_snapshot(
        orch.logger,
        sym,
        cycle_id=orch._active_cycle_id,
        http_user=prompt,
        http_system=sys_extra,
        http_system_resolved=resolved_system_instruction(sys_extra),
        sniper_tokens=dict(sniper_tok),
        llm_config=raw_llm if isinstance(raw_llm, dict) else {},
        leading_cycle_blank=leading_cycle_blank,
        mtf_matrix=str(ctx.get("mtf_matrix", "")),
        indicators_numeric_line=indicators_numeric_line,
        institutional_pa_bundle=institutional_pa_bundle,
        indicator_bundle_line=str(ctx.get("llm_indicator_bundle", "")),
        tf_labels=ctx.get("llm_tf_labels") or (),
        macro_confluence=macro_snapshot.macro_block,
        fx_reference_line=macro_snapshot.fx_reference_line,
        macro_sentiment=macro_snapshot.tag,
    )
    ref_px = last_reference_price(orch.stream, sym)
    payload = await _request_payload(
        orch,
        sym,
        call_runtime,
        prompt,
        system=str(runtime.get("llm_system") or ""),
        cycle_id=int(orch._active_cycle_id),
    )
    llm_http_ms = float(payload.pop("_llm_latency_ms", 0) or 0)
    llm_resp_chars = int(payload.pop("_llm_raw_chars", 0) or 0)
    llm_direction_from_api = bool(payload.pop("_llm_direction_from_api", False))
    mtf_effective = str(mtf_d or "").strip() or "-"
    direction, conviction, note, us_dir, eu_dir = decision_from_payload(payload)

    note = append_entropy_high_note(note, conviction, swing_c, runtime, orch.logger)
    inverted = False

    macro_cfg = ctx.get("macro_cfg") if isinstance(ctx.get("macro_cfg"), dict) else None
    direction, conviction, note, us_dir, eu_dir, macro_guard, macro_execute = apply_macro_post_parse(
        direction,
        conviction,
        note,
        us_dir,
        eu_dir,
        macro_snapshot,
        macro_cfg,
        sym=sym,
    )

    llm_ok, llm_fail_tag = anchor_llm_decision_complete(
        orch, sym, direction, us_dir, eu_dir, macro_snapshot=macro_snapshot
    )
    if not llm_ok:
        direction = None
        macro_execute = False
        note = f"{note} | {llm_fail_tag}".strip()

    direction, metrics = _build_metrics_for_decision_core(
        runtime,
        direction,
        conviction,
        note,
        ref_px,
        mtf_effective,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        llm_metrics,
        closes_m5=swing_c,
    )
    metrics["duration"] = enforce_minimum_duration(
        sym, calculate_adaptive_duration(runtime, ctx, base_duration=runtime.get("duration", 1))
    )
    attach_decision_telemetry(
        metrics,
        ctx,
        str(ctx.get("regime_label", "range")),
        baseline_prob,
        str(metrics.get("decision_source", "llm")),
        orch,
        sym,
    )
    _emit_direction_debug(
        orch,
        runtime=runtime,
        direction_base=direction,
        direction_final=direction,
        conviction=float(metrics.get("conviction", conviction)),
        macro_desc=macro_d,
        structure_desc=struct_d,
        swing_desc=swing_d,
        trigger_desc=trig_d,
        mtf_alignment=mtf_effective,
        decision_source=str(metrics.get("decision_source", "")),
        adjusted=inverted,
        exec_inverted=inverted,
    )
    patch_final_symbol_metrics(
        metrics,
        execute_flag=bool(metrics.get("execute", False)) and macro_execute,
        inverted=inverted,
        llm_http_ms=llm_http_ms,
        llm_resp_chars=llm_resp_chars,
        llm_direction_from_api=llm_direction_from_api,
        us_dir=us_dir,
        eu_dir=eu_dir,
        macro_snapshot=macro_snapshot,
        macro_guard=macro_guard,
    )
    _emit_symbol_decision_audit(
        orch,
        sym=sym,
        runtime=runtime,
        prompt=prompt,
        ctx=ctx,
        direction=direction,
        conviction=conviction,
        note=note,
        ref_px=ref_px,
        mtf_effective=mtf_effective,
        baseline_prob=baseline_prob,
        indicators_numeric_line=indicators_numeric_line,
        institutional_pa_bundle=institutional_pa_bundle,
        macro_snapshot=macro_snapshot,
        macro_d=macro_d,
        struct_d=struct_d,
        swing_d=swing_d,
        trig_d=trig_d,
        sniper_tok=sniper_tok,
        mtf_d=mtf_d,
        metrics=metrics,
        inverted=inverted,
        llm_http_ms=llm_http_ms,
        llm_resp_chars=llm_resp_chars,
        llm_direction_from_api=llm_direction_from_api,
        us_dir=us_dir,
        eu_dir=eu_dir,
    )
    return direction, metrics
